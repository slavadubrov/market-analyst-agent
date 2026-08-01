"""Worker loop for the queue + worker + checkpoint DB shape.

Run with ``python -m market_analyst.runtime.worker``. The loop:

1. Reads a job off the Redis Stream.
2. Calls ``run_analysis`` (the same entry point the CLI uses).
3. Acks the message once the run returns (or fails — the harness's debug
   bundle captures everything needed to investigate later).

Crash semantics: a worker that dies between pull and ack will see the same
message redelivered. The checkpoint DB + idempotency store make that
re-delivery safe — the new run resumes from the last checkpoint and replays
any side-effecting tool calls instead of re-running them.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
from typing import Any

from market_analyst.memory import get_checkpointer
from market_analyst.runtime.harness import harness_run
from market_analyst.runtime.queue import (
    RunJob,
    ack,
    ensure_consumer_group,
    pull_one,
)
from market_analyst.schemas import ExecutionMode

logger = logging.getLogger(__name__)


_shutdown_requested = False


def _install_signal_handlers() -> None:
    def _stop(*_: Any) -> None:
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("Shutdown signal received; finishing current job and exiting.")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)


def _mode_from_string(mode: str) -> ExecutionMode | None:
    if mode == "deep":
        return ExecutionMode.DEEP_RESEARCH
    if mode == "flash":
        return ExecutionMode.FLASH_BRIEFING
    return None


def _process_job(job: RunJob) -> None:
    """Run one job through the analysis workflow."""
    from market_analyst.workflows.analysis_workflow import run_analysis

    try:
        checkpointer = get_checkpointer()
    except Exception as exc:  # pragma: no cover - depends on real Postgres
        logger.error("Worker could not connect to checkpointer: %s", exc)
        raise

    with harness_run(job.thread_id, workflow_name="analysis") as ctx:
        result = run_analysis(
            query=job.query,
            user_id=job.user_id,
            thread_id=job.thread_id,
            checkpointer=checkpointer,
            force_mode=_mode_from_string(job.mode),
        )
        ctx.final_state = result.get("state")
        logger.info(
            "Job %s complete (thread=%s, requires_approval=%s)",
            job.message_id,
            job.thread_id,
            result.get("requires_approval"),
        )


def run_forever(consumer_name: str | None = None) -> None:
    """Block in a pull/process/ack loop until SIGINT/SIGTERM.

    ``consumer_name`` defaults to the pod hostname plus the PID, which is the
    convention Kubernetes / Cloud Run uses to make every worker uniquely
    addressable in Redis's consumer group.
    """
    consumer_name = consumer_name or f"{socket.gethostname()}-{os.getpid()}"
    _install_signal_handlers()
    ensure_consumer_group()
    logger.info("Worker %s ready; polling stream", consumer_name)

    while not _shutdown_requested:
        try:
            job = pull_one(consumer_name=consumer_name, block_ms=5000)
        except Exception as exc:  # pragma: no cover - depends on Redis
            logger.exception("pull_one failed: %s", exc)
            continue
        if job is None:
            continue
        try:
            _process_job(job)
        except Exception as exc:
            # The harness wrote a debug bundle; log + continue so one bad job
            # doesn't take the worker down.
            logger.exception("Job %s failed: %s", job.message_id, exc)
        ack(job.message_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        run_forever()
    except KeyboardInterrupt:
        sys.exit(0)
