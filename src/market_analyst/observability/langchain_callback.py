"""LangChain callback handler that emits OpenTelemetry GenAI spans.

LangChain notifies registered handlers on every LLM call, tool call, and chain
step. We use those notifications to:

1. Open a ``chat {model}`` span on ``on_llm_start`` and close it on
   ``on_llm_end`` / ``on_llm_error``, stamping the standard ``gen_ai.*``
   attributes (provider, model, conversation id, agent name, token counts).
2. Open an ``execute_tool {tool_name}`` span on ``on_tool_start`` and close it
   on ``on_tool_end`` / ``on_tool_error``, updating the Prometheus tool-call
   counter + error counter.

Why a callback instead of explicit ``llm_span()`` wrappers everywhere: a single
``executor_node`` invocation can spawn many LLM and tool calls inside the
``create_react_agent`` loop. Wrapping the outer call loses the inner detail.
Callbacks attach to every step, so a six-step ReAct trace becomes six spans
plus their tool children, not one opaque "executor" span.

Usage:

    from market_analyst.observability.langchain_callback import make_callbacks

    callbacks = make_callbacks(
        agent_name="market-analyst.executor",
        workflow_name="research_then_write",
        conversation_id=thread_id,
        model="claude-sonnet-4-5",
    )
    result = react_agent.invoke({"messages": [...]}, config={"callbacks": callbacks})
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from market_analyst.observability import metrics
from market_analyst.observability.genai_attrs import (
    SPAN_NAME_CHAT,
    SPAN_NAME_EXECUTE_TOOL,
    GenAIAttrs,
)
from market_analyst.observability.tracing import get_tracer

if TYPE_CHECKING:
    pass


class GenAICallbackHandler(BaseCallbackHandler):
    """Emit GenAI-spec spans for every LangChain LLM and tool call.

    Each ``run_id`` (the UUID LangChain assigns to a step) maps to one open
    span + a start timestamp held in ``self._spans``. The handler is NOT
    thread-safe by design: pass one handler per ``invoke`` call (the
    ``make_callbacks`` helper below builds a fresh handler per call).
    """

    def __init__(
        self,
        *,
        agent_name: str,
        workflow_name: str | None = None,
        conversation_id: str | None = None,
        provider: str = "anthropic",
        model: str | None = None,
    ):
        self.agent_name = agent_name
        self.workflow_name = workflow_name
        self.conversation_id = conversation_id
        self.provider = provider
        self.model = model
        # run_id -> (span, started_monotonic, kind, label)
        self._spans: dict[UUID, tuple[Any, float, str, str]] = {}
        self._tracer = get_tracer("market_analyst.langchain")

    # --- LLM lifecycle ---------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],  # noqa: ARG002 - required by interface, unused
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._open_llm_span(serialized, run_id, kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],  # noqa: ARG002 - required by interface, unused
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._open_llm_span(serialized, run_id, kwargs)

    def _open_llm_span(self, serialized: dict[str, Any], run_id: UUID, kwargs: dict[str, Any]) -> None:
        # Best-effort model resolution: callback metadata > constructor default.
        model = (
            (kwargs.get("invocation_params") or {}).get("model")
            or (serialized.get("kwargs") or {}).get("model")
            or self.model
            or "unknown"
        )
        span_name = f"{SPAN_NAME_CHAT} {model}"
        span = self._tracer.start_span(span_name)
        span.set_attribute(GenAIAttrs.OPERATION_NAME, SPAN_NAME_CHAT)
        span.set_attribute(GenAIAttrs.PROVIDER_NAME, self.provider)
        span.set_attribute(GenAIAttrs.REQUEST_MODEL, model)
        span.set_attribute(GenAIAttrs.AGENT_NAME, self.agent_name)
        if self.workflow_name:
            span.set_attribute(GenAIAttrs.WORKFLOW_NAME, self.workflow_name)
        if self.conversation_id:
            span.set_attribute(GenAIAttrs.CONVERSATION_ID, self.conversation_id)
        self._spans[run_id] = (span, time.monotonic(), "llm", model)

    def on_llm_end(self, response: Any, *, run_id: UUID, **_: Any) -> None:
        entry = self._spans.pop(run_id, None)
        if entry is None:
            return
        span, started, _, model = entry
        try:
            usage = self._extract_token_usage(response)
            if usage is not None:
                input_tokens, output_tokens = usage
                if input_tokens:
                    span.set_attribute(GenAIAttrs.INPUT_TOKENS, int(input_tokens))
                if output_tokens:
                    span.set_attribute(GenAIAttrs.OUTPUT_TOKENS, int(output_tokens))
                if input_tokens or output_tokens:
                    span.set_attribute(
                        GenAIAttrs.TOTAL_TOKENS, int(input_tokens) + int(output_tokens)
                    )
                metrics.record_tokens(
                    self.agent_name, model, int(input_tokens or 0), int(output_tokens or 0)
                )
        finally:
            metrics.operation_duration.labels(
                SPAN_NAME_CHAT, model or "unknown", self.agent_name
            ).observe(time.monotonic() - started)
            span.end()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        self._fail_span(run_id, error)

    # --- Tool lifecycle --------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,  # noqa: ARG002 - required by interface, unused
        *,
        run_id: UUID,
        **_: Any,
    ) -> None:
        tool_name = serialized.get("name") or "unknown_tool"
        span = self._tracer.start_span(f"{SPAN_NAME_EXECUTE_TOOL} {tool_name}")
        span.set_attribute(GenAIAttrs.OPERATION_NAME, SPAN_NAME_EXECUTE_TOOL)
        span.set_attribute(GenAIAttrs.TOOL_NAME, tool_name)
        span.set_attribute(GenAIAttrs.AGENT_NAME, self.agent_name)
        if self.conversation_id:
            span.set_attribute(GenAIAttrs.CONVERSATION_ID, self.conversation_id)
        self._spans[run_id] = (span, time.monotonic(), "tool", tool_name)

    def on_tool_end(self, output: Any, *, run_id: UUID, **_: Any) -> None:  # noqa: ARG002 - output unused
        entry = self._spans.pop(run_id, None)
        if entry is None:
            return
        span, started, _, tool_name = entry
        metrics.operation_duration.labels(
            SPAN_NAME_EXECUTE_TOOL, "n/a", self.agent_name
        ).observe(time.monotonic() - started)
        metrics.record_tool_call(tool_name, "ok")
        span.end()

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        entry = self._spans.pop(run_id, None)
        if entry is not None:
            _, started, _, tool_name = entry
            metrics.operation_duration.labels(
                SPAN_NAME_EXECUTE_TOOL, "n/a", self.agent_name
            ).observe(time.monotonic() - started)
            metrics.record_tool_call(tool_name, "error")
            metrics.record_tool_error(tool_name, type(error).__name__)
        self._fail_span(run_id, error, _entry=entry)

    # --- Helpers ---------------------------------------------------------

    def _fail_span(
        self,
        run_id: UUID,
        error: BaseException,
        *,
        _entry: tuple[Any, float, str, str] | None = None,
    ) -> None:
        entry = _entry if _entry is not None else self._spans.pop(run_id, None)
        if entry is None:
            return
        span, *_ = entry
        try:
            from opentelemetry.trace import Status, StatusCode

            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
        finally:
            span.end()

    @staticmethod
    def _extract_token_usage(response: Any) -> tuple[int, int] | None:
        """Pull (input, output) token counts from a LangChain LLMResult.

        LangChain's ``llm_output`` shape varies per provider. For Anthropic via
        ``langchain_anthropic`` we usually get ``llm_output["usage"]`` or an
        ``AIMessage`` with ``usage_metadata`` on the first generation. Try a
        couple of common shapes and return ``None`` if nothing matches; the
        span still ends without token attrs, just without numbers.
        """
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        if usage:
            input_t = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            output_t = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            if input_t or output_t:
                return int(input_t), int(output_t)

        generations = getattr(response, "generations", None) or []
        for gen_group in generations:
            for gen in gen_group:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
        return None


def make_callbacks(
    *,
    agent_name: str,
    workflow_name: str | None = None,
    conversation_id: str | None = None,
    model: str | None = None,
    provider: str = "anthropic",
) -> list[BaseCallbackHandler]:
    """Build the callback list to pass via ``config={"callbacks": ...}``.

    Returning a list (not the handler directly) keeps the call sites uniform
    even if we later add more handlers (e.g. a LangSmith handler in CI).
    """
    return [
        GenAICallbackHandler(
            agent_name=agent_name,
            workflow_name=workflow_name,
            conversation_id=conversation_id,
            model=model,
            provider=provider,
        )
    ]
