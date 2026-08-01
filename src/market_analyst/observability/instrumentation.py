"""Context managers that wrap LLM and tool calls with GenAI spans + metrics.

The shapes are intentionally narrow: one for chat completions, one for tool
executions. Each:

1. opens a span named per the GenAI semantic conventions
2. sets the standard ``gen_ai.*`` attributes
3. on exit, records duration + token counters into Prometheus
4. propagates exceptions, marking the span as errored

Usage:

    with llm_span(
        model="claude-sonnet-4-5",
        provider="anthropic",
        agent_name="market-analyst.planner",
        workflow_name="research_then_write",
        conversation_id=thread_id,
    ) as span:
        response = llm.invoke(...)
        record_tokens(span, "market-analyst.planner", "claude-sonnet-4-5",
                      response.usage_metadata.input_tokens,
                      response.usage_metadata.output_tokens)

The ``record_tokens`` helper is split from the context manager so callers can
ignore tokens (e.g. tool spans) without paying the noise of an Optional.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from market_analyst.observability import metrics
from market_analyst.observability.genai_attrs import (
    SPAN_NAME_CHAT,
    SPAN_NAME_EXECUTE_TOOL,
    GenAIAttrs,
)
from market_analyst.observability.tracing import get_tracer


def _safe_set(span: Any, key: str, value: Any) -> None:
    """Only set attributes that have a value; skip None/empty cleanly."""
    if value is None or value == "":
        return
    span.set_attribute(key, value)


@contextmanager
def llm_span(
    *,
    model: str,
    provider: str = "anthropic",
    agent_name: str,
    workflow_name: str | None = None,
    conversation_id: str | None = None,
    operation: str = SPAN_NAME_CHAT,
) -> Iterator[Any]:
    """Open a ``gen_ai.{operation}`` span for an LLM call.

    Caller is responsible for calling :func:`record_tokens` once the response
    is available (token counts aren't known until then).
    """
    tracer = get_tracer("market_analyst.llm")
    span_name = f"{operation} {model}" if model else operation
    started = time.monotonic()

    with tracer.start_as_current_span(span_name) as span:
        _safe_set(span, GenAIAttrs.OPERATION_NAME, operation)
        _safe_set(span, GenAIAttrs.PROVIDER_NAME, provider)
        _safe_set(span, GenAIAttrs.REQUEST_MODEL, model)
        _safe_set(span, GenAIAttrs.AGENT_NAME, agent_name)
        _safe_set(span, GenAIAttrs.WORKFLOW_NAME, workflow_name)
        _safe_set(span, GenAIAttrs.CONVERSATION_ID, conversation_id)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            metrics.operation_duration.labels(operation, model or "unknown", agent_name).observe(time.monotonic() - started)


@contextmanager
def tool_span(
    *,
    tool_name: str,
    agent_name: str,
    conversation_id: str | None = None,
    tool_type: str = "function",
    tool_call_id: str | None = None,
) -> Iterator[Any]:
    """Open a ``gen_ai.execute_tool`` span for a tool invocation.

    Records the tool-call counter (with outcome ``ok`` / ``error``) and the
    error counter (by exception class) on exit. The caller does nothing.
    """
    tracer = get_tracer("market_analyst.tool")
    started = time.monotonic()
    outcome = "ok"
    err_type: str | None = None

    with tracer.start_as_current_span(f"{SPAN_NAME_EXECUTE_TOOL} {tool_name}") as span:
        _safe_set(span, GenAIAttrs.OPERATION_NAME, SPAN_NAME_EXECUTE_TOOL)
        _safe_set(span, GenAIAttrs.TOOL_NAME, tool_name)
        _safe_set(span, GenAIAttrs.TOOL_TYPE, tool_type)
        _safe_set(span, GenAIAttrs.TOOL_CALL_ID, tool_call_id)
        _safe_set(span, GenAIAttrs.AGENT_NAME, agent_name)
        _safe_set(span, GenAIAttrs.CONVERSATION_ID, conversation_id)
        try:
            yield span
        except Exception as exc:
            outcome = "error"
            err_type = type(exc).__name__
            span.record_exception(exc)
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            metrics.operation_duration.labels(SPAN_NAME_EXECUTE_TOOL, "n/a", agent_name).observe(time.monotonic() - started)
            metrics.record_tool_call(tool_name, outcome)
            if err_type:
                metrics.record_tool_error(tool_name, err_type)


def record_tokens(
    span: Any,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Stamp token counts on a span and the Prometheus counter together.

    Why both: the span attributes are what Tempo/Honeycomb queries read; the
    Prometheus counter is what Alertmanager watches for runaway spend.
    """
    if input_tokens:
        span.set_attribute(GenAIAttrs.INPUT_TOKENS, int(input_tokens))
    if output_tokens:
        span.set_attribute(GenAIAttrs.OUTPUT_TOKENS, int(output_tokens))
    total = (input_tokens or 0) + (output_tokens or 0)
    if total:
        span.set_attribute(GenAIAttrs.TOTAL_TOKENS, int(total))
    metrics.record_tokens(agent_name, model, input_tokens or 0, output_tokens or 0)
