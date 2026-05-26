"""OpenTelemetry SDK bootstrap.

Reads OTEL configuration from environment variables (the standard
``OTEL_EXPORTER_OTLP_*`` set plus ``OTEL_SERVICE_NAME``). If no OTLP endpoint
is configured we install a noop tracer so the rest of the codebase can call
``trace.get_tracer(...)`` unconditionally.

Why an explicit bootstrap (not auto-instrumentation): the LangGraph nodes and
the tool wrappers in :mod:`market_analyst.observability.instrumentation` emit
spans that follow the GenAI semantic conventions. Auto-instrumentation would
double-instrument the underlying ``langchain`` / ``langchain_anthropic`` calls
with attributes that don't match the GenAI spec, which is harder to debug than
silence.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Module-level flag so setup_telemetry() is idempotent.
_initialized = False


def _otlp_endpoint() -> str | None:
    """Return the configured OTLP endpoint, or None when telemetry should noop."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    # Treat empty string the same as unset to keep the "off" path simple.
    return endpoint or None


def setup_telemetry(service_name: str | None = None) -> Any:
    """Initialize OpenTelemetry once. Returns the configured tracer provider.

    Args:
        service_name: Overrides ``OTEL_SERVICE_NAME``. Defaults to
            ``market-analyst-agent`` for parity with the article's queries
            (``{service_name="market-analyst-agent"}``).

    Returns:
        The active ``TracerProvider``. When no OTLP endpoint is configured the
        returned provider is a no-op suitable for development and tests.
    """
    global _initialized

    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if _initialized:
        return trace.get_tracer_provider()

    resource = Resource.create(
        {
            SERVICE_NAME: service_name or os.getenv("OTEL_SERVICE_NAME", "market-analyst-agent"),
        }
    )
    provider = TracerProvider(resource=resource)

    endpoint = _otlp_endpoint()
    if endpoint:
        # Lazy import so the gRPC exporter never loads when telemetry is off.
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("OpenTelemetry tracing enabled → %s", endpoint)
    else:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set; tracing is a no-op")

    trace.set_tracer_provider(provider)
    _initialized = True
    # Return the global provider (which may differ from ``provider`` if a prior
    # process already set one — OpenTelemetry refuses overrides). Returning the
    # global guarantees idempotent callers see the same object every time.
    return trace.get_tracer_provider()


def get_tracer(name: str = "market_analyst") -> Any:
    """Return a tracer, initializing the provider on first use."""
    from opentelemetry import trace

    if not _initialized:
        setup_telemetry()
    return trace.get_tracer(name)
