"""Observability primitives for long-running agents (Part 5: The Habitat).

Exposes the things the rest of the codebase needs:

- :func:`setup_telemetry` -- initialize OpenTelemetry tracer + Prometheus metrics.
  Idempotent and a no-op when no OTLP endpoint is configured.
- :func:`llm_span` / :func:`tool_span` -- context managers that emit OpenTelemetry
  spans following the GenAI semantic conventions (``gen_ai.*`` attributes).
- :class:`BudgetTracker` -- per-run token / tool-call budget with a kill switch
  and a circuit breaker on consecutive tool errors.
- :class:`BudgetExceeded` -- raised when a budget is exhausted; callers convert
  this into a graceful run abort.

See ``docs/habitat.md`` for the design notes and the article reference.
"""

from market_analyst.observability.budget import (
    BudgetExceeded,
    BudgetTracker,
    CircuitBreakerOpen,
)
from market_analyst.observability.genai_attrs import GenAIAttrs
from market_analyst.observability.instrumentation import (
    llm_span,
    record_tokens,
    tool_span,
)
from market_analyst.observability.langchain_callback import (
    GenAICallbackHandler,
    make_callbacks,
)
from market_analyst.observability.metrics import (
    mount_metrics_http_server,
    record_budget_kill,
    record_tool_call,
    record_tool_error,
)
from market_analyst.observability.tracing import setup_telemetry

__all__ = [
    "BudgetExceeded",
    "BudgetTracker",
    "CircuitBreakerOpen",
    "GenAIAttrs",
    "GenAICallbackHandler",
    "llm_span",
    "make_callbacks",
    "mount_metrics_http_server",
    "record_budget_kill",
    "record_tokens",
    "record_tool_call",
    "record_tool_error",
    "setup_telemetry",
    "tool_span",
]
