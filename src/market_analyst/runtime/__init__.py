"""Runtime primitives for long-running agents (Part 5: The Habitat).

Each submodule implements one of the failure-mode mitigations from the article:

- :mod:`workspace` -- per-``thread_id`` workspace directories.
- :mod:`debug_bundle` -- the postmortem bundle written on failure.
- :mod:`idempotency` -- idempotency-key store for tool calls with side effects.
- :mod:`initializer` -- the ``PROGRESS.md`` / ``feature-list.json`` boot hook.
- :mod:`evaluator` -- fresh-context evaluator subagent ("default-FAIL").

These pieces are deliberately small and filesystem-backed so they work locally
on ``docker compose up`` without any extra infrastructure. The article's
"What Has to Change Before This Goes to Production" notes call out the
production-grade swaps (Vault for secrets, S3 for the bundle store, etc.).
"""

from market_analyst.runtime.debug_bundle import write_debug_bundle
from market_analyst.runtime.harness import HarnessContext, harness_run
from market_analyst.runtime.idempotency import (
    IdempotencyStore,
    get_idempotency_store,
)
from market_analyst.runtime.initializer import (
    InitializerContext,
    read_progress,
    write_progress,
)
from market_analyst.runtime.workspace import (
    ensure_thread_workspace,
    workspace_path_for,
)

__all__ = [
    "HarnessContext",
    "IdempotencyStore",
    "InitializerContext",
    "ensure_thread_workspace",
    "get_idempotency_store",
    "harness_run",
    "read_progress",
    "workspace_path_for",
    "write_debug_bundle",
    "write_progress",
]
