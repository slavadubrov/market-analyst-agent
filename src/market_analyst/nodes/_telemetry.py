"""Shared telemetry helpers for LangGraph nodes.

Three concerns:

1. ``get_conversation_id(config)`` -- pull the LangGraph ``thread_id`` out of
   the runtime config so spans can stamp ``gen_ai.conversation.id``.
2. ``node_callbacks(...)`` -- thin wrapper around
   :func:`market_analyst.observability.make_callbacks` that fills in the model
   name from the same env var the nodes already read.
3. ``WORKFLOW_NAME`` / ``AGENT_NS`` -- string constants so the GenAI
   ``workflow.name`` and ``agent.name`` attributes stay consistent across
   nodes (otherwise Grafana panels keyed on ``gen_ai.agent.name`` get noisy
   from typos).

This module imports nothing from ``langgraph`` at module scope so it stays
test-friendly.
"""

from __future__ import annotations

from typing import cast

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from market_analyst.llm import resolve_model
from market_analyst.observability.langchain_callback import make_callbacks

WORKFLOW_NAME = "market_analyst.research_then_write"
AGENT_NS = "market-analyst"


def get_conversation_id(config: RunnableConfig | None) -> str | None:
    """Extract the LangGraph ``thread_id`` from the runtime config, if present.

    LangGraph passes config as ``{"configurable": {"thread_id": "..."}}`` when
    a checkpointer is in use. In the ``--no-persist`` path there is no thread
    id; return ``None`` and let the span omit ``gen_ai.conversation.id``.
    """
    if not config:
        return None
    return cast(str | None, (config.get("configurable") or {}).get("thread_id"))


def node_callbacks(
    *,
    node_name: str,
    config: RunnableConfig | None,
    workflow_name: str = WORKFLOW_NAME,
) -> list[BaseCallbackHandler]:
    """Build the callback list for a node's LLM ``invoke`` call.

    Picks the model up from the same env var the node uses, so the span
    attribute ``gen_ai.request.model`` always matches the actual model called.
    """
    provider, model = resolve_model(config=config)
    return make_callbacks(
        agent_name=f"{AGENT_NS}.{node_name}",
        workflow_name=workflow_name,
        conversation_id=get_conversation_id(config),
        model=model,
        provider=provider,
    )
