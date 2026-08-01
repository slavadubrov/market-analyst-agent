"""Prometheus metrics for long-running runs.

These metrics give Grafana / Alertmanager something to watch. They cover the
three failure modes that the article calls out as cost-sensitive:

- token usage per agent (``gen_ai_client_token_usage_total``)
- tool-call latency per tool (``gen_ai_client_operation_duration_seconds``)
- tool-error rate (``market_analyst_tool_errors_total``)

Plus a counter for budget kill-switches firing, which the kill-switch alert in
the article keys off (``market_analyst_budget_kill_total``).

Why a custom Prometheus registry: ``prometheus_client`` registers metrics on
the default registry at import time. With pytest re-importing modules between
tests, you can get ``Duplicated timeseries`` errors. The accessor helpers below
register lazily into a private registry that ``mount_metrics_http_server``
exposes on ``/metrics``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    start_http_server,
)

if TYPE_CHECKING:
    pass

REGISTRY = CollectorRegistry(auto_describe=True)

# --- Token usage --------------------------------------------------------------

token_usage = Counter(
    "gen_ai_client_token_usage_total",
    "Tokens consumed per (agent, model, token_type).",
    labelnames=("gen_ai_agent_name", "gen_ai_request_model", "token_type"),
    registry=REGISTRY,
)

operation_duration = Histogram(
    "gen_ai_client_operation_duration_seconds",
    "Wall-clock duration of a gen_ai operation.",
    labelnames=("gen_ai_operation_name", "gen_ai_request_model", "gen_ai_agent_name"),
    registry=REGISTRY,
    # Buckets tuned for LLM calls (0.5s..120s) and tool calls (10ms..30s).
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# --- Tool calls ---------------------------------------------------------------

tool_calls = Counter(
    "market_analyst_tool_calls_total",
    "Tool invocations, regardless of outcome.",
    labelnames=("tool_name", "outcome"),  # outcome: ok | error
    registry=REGISTRY,
)

tool_errors = Counter(
    "market_analyst_tool_errors_total",
    "Tool invocations that raised an exception.",
    labelnames=("tool_name", "error_type"),
    registry=REGISTRY,
)

# --- Budget / kill switch -----------------------------------------------------

budget_kills = Counter(
    "market_analyst_budget_kill_total",
    "Runs aborted by the budget kill-switch.",
    labelnames=("reason",),  # token_budget | tool_call_budget | circuit_breaker
    registry=REGISTRY,
)


# --- Public helpers -----------------------------------------------------------


def record_tokens(
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Increment token counters; labels match the GenAI metric conventions."""
    if input_tokens:
        token_usage.labels(agent_name, model, "input").inc(input_tokens)
    if output_tokens:
        token_usage.labels(agent_name, model, "output").inc(output_tokens)


def record_tool_call(tool_name: str, outcome: str) -> None:
    """Increment the tool-call counter. ``outcome`` is ``ok`` or ``error``."""
    tool_calls.labels(tool_name, outcome).inc()


def record_tool_error(tool_name: str, error_type: str) -> None:
    """Increment the tool-error counter, keyed by the exception class name."""
    tool_errors.labels(tool_name, error_type).inc()


def record_budget_kill(reason: str) -> None:
    """Increment the budget-kill counter."""
    budget_kills.labels(reason).inc()


def mount_metrics_http_server(port: int | None = None) -> None:
    """Expose ``/metrics`` on the given port (defaults to ``METRICS_PORT`` or 9464).

    Safe to call multiple times; only the first call binds a socket. The default
    9464 matches the OpenTelemetry collector's prometheus receiver convention.
    """
    bind_port = port if port is not None else int(os.getenv("METRICS_PORT", "9464"))
    try:
        start_http_server(bind_port, registry=REGISTRY)
    except OSError:
        # Already bound (e.g. second worker on the same host). Not fatal.
        pass
