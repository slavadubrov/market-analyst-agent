"""Harness boot/teardown helpers.

The article's "Healthy Run Lifecycle" step 1 (Boot) and the "Debug Bundle
Pattern" section both want the same surface on the harness:

    with harness_run(thread_id) as ctx:
        ...

Where ``ctx`` carries the per-thread workspace, the initializer context (so
the caller can detect resumes), and a list of tool-call records the debug
bundle will pick up if the block raises.

Three concerns this module encapsulates so the CLI and the worker can both
use it:

1. ``setup_telemetry()`` — initialized once per process.
2. Per-thread workspace + ``PROGRESS.md`` boot hook (writes "started" /
   "resumed" / "completed" / "failed" lines).
3. Debug-bundle writer on exception.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from market_analyst.observability import (
    mount_metrics_http_server,
    setup_telemetry,
)
from market_analyst.runtime.debug_bundle import write_debug_bundle
from market_analyst.runtime.initializer import InitializerContext, write_progress
from market_analyst.runtime.workspace import ensure_thread_workspace

logger = logging.getLogger(__name__)


@dataclass
class HarnessContext:
    """Per-run context handed to the caller by ``harness_run``.

    Attributes:
        thread_id: The LangGraph thread id.
        workspace: Path to the per-thread workspace (created on entry).
        initializer: Snapshot of any prior progress in the workspace.
        tool_calls: Append-only log of ``{tool, status, latency_ms, error}``
            dicts. Populated by callers that want them in the debug bundle.
        final_state: Last known state, set by the caller before exiting so the
            debug bundle has something to serialize on failure.
    """

    thread_id: str
    workspace: Path
    initializer: InitializerContext
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_state: Any | None = None


def _maybe_start_metrics_server() -> None:
    """Bring up the Prometheus ``/metrics`` server if METRICS_ENABLED=true."""
    if os.getenv("METRICS_ENABLED", "false").lower() != "true":
        return
    mount_metrics_http_server()


@contextmanager
def harness_run(thread_id: str, *, workflow_name: str | None = None) -> Iterator[HarnessContext]:
    """Boot a run, yield a ``HarnessContext``, and write a debug bundle on error.

    ``setup_telemetry`` is idempotent so calling ``harness_run`` once per
    invocation is fine even when nested inside other contexts. The metrics
    server only binds a port when explicitly opted in via ``METRICS_ENABLED``,
    so unit tests don't get socket conflicts.
    """
    setup_telemetry()
    _maybe_start_metrics_server()

    workspace = ensure_thread_workspace(thread_id)
    initializer = InitializerContext.load(workspace)

    if initializer.is_resume():
        write_progress(
            workspace,
            line=f"RESUMED at {datetime.now(timezone.utc).isoformat()} "
            f"(prior progress lines: {initializer.progress.count(chr(10))})",
        )
    else:
        write_progress(
            workspace,
            line=f"STARTED workflow={workflow_name or 'analysis'} thread_id={thread_id}",
        )

    ctx = HarnessContext(thread_id=thread_id, workspace=workspace, initializer=initializer)
    try:
        yield ctx
        write_progress(workspace, line="COMPLETED")
    except Exception as exc:
        write_progress(workspace, line=f"FAILED: {type(exc).__name__}: {exc}")
        try:
            debug_dir = write_debug_bundle(
                thread_id=thread_id,
                state=ctx.final_state if ctx.final_state is not None else {},
                exception=exc,
                tool_calls=ctx.tool_calls or None,
            )
            logger.warning("Debug bundle written to %s", debug_dir)
        except Exception as bundle_exc:  # pragma: no cover - defensive
            logger.exception("Failed to write debug bundle: %s", bundle_exc)
        raise
