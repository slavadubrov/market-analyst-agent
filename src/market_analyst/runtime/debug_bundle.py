"""Failure debug bundle.

The article's "Debug Bundle Pattern" section lists exactly what an operator
(or another agent) needs after a six-hour run dies: the event log, the last
state snapshot, the trace, tool-call CSV, workspace tarball, screenshots,
and the env. This module writes that bundle into
``<workspace>/_debug/`` so the failure is debuggable later.

Design notes:

- Inputs are kept narrow. The caller passes the state dict (already on hand)
  and an optional list of tool-call records; we don't reach into any global
  state to assemble the bundle, which keeps the function easy to test and
  hard to misuse.
- ``workspace.tar.zst`` would ship the whole workspace if we had ``zstandard``
  available. For the reference repo we use ``tar`` + ``gzip`` (always in the
  stdlib) and name it ``workspace.tar.gz`` — closer to "this is shippable"
  than dropping the file.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from market_analyst.runtime.workspace import ensure_thread_workspace


def _serialize(value: Any) -> Any:
    """Best-effort JSON serializer for state values (Pydantic, datetime, etc.)."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _safe_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=_serialize, sort_keys=True)


def _write_last_state(debug_dir: Path, state: Mapping[str, Any] | Any) -> None:
    state_payload = state.model_dump() if hasattr(state, "model_dump") else dict(state) if isinstance(state, Mapping) else {"value": _serialize(state)}
    (debug_dir / "last_state.json").write_text(_safe_dump(state_payload))


def _write_error(debug_dir: Path, exception: BaseException) -> None:
    formatted = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    (debug_dir / "error.txt").write_text(formatted)


def _write_tool_calls_csv(debug_dir: Path, tool_calls: list[dict[str, Any]]) -> None:
    cols = ["ts", "tool", "input_hash", "latency_ms", "status", "error"]
    lines = [",".join(cols)]
    for row in tool_calls:
        lines.append(",".join(str(row.get(c, "")).replace(",", " ") for c in cols))
    (debug_dir / "tool_calls.csv").write_text("\n".join(lines) + "\n")


def _write_env(debug_dir: Path, thread_id: str, extra_env: Mapping[str, str] | None) -> None:
    env_lines = [
        f"python_version={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"thread_id={thread_id}",
        f"bundle_created_at={datetime.now(timezone.utc).isoformat()}",
    ]
    for key in ("MARKET_ANALYST_MODEL", "ANTHROPIC_MODEL", "GIT_COMMIT", "IMAGE_TAG"):
        if os.environ.get(key):
            env_lines.append(f"{key}={os.environ[key]}")
    if extra_env:
        for key, value in extra_env.items():
            env_lines.append(f"{key}={value}")
    (debug_dir / "env.txt").write_text("\n".join(env_lines) + "\n")


def _write_workspace_tarball(debug_dir: Path, workspace: Path) -> None:
    """Pack the workspace (skipping ``_debug`` to avoid recursion)."""
    tarball = debug_dir / "workspace.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for entry in workspace.iterdir():
            if entry.name == "_debug":
                continue
            tar.add(entry, arcname=entry.name)


def write_debug_bundle(
    *,
    thread_id: str,
    state: Mapping[str, Any] | Any,
    exception: BaseException | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    workspace_root: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> Path:
    """Write the debug bundle under ``<workspace>/_debug/`` and return that path.

    Args:
        thread_id: LangGraph ``thread_id`` for the failed run.
        state: The final ``AgentState`` (or its ``model_dump()``) the run died
            with. Persisted to ``last_state.json``.
        exception: If supplied, the formatted traceback ends up in ``error.txt``.
        tool_calls: Optional list of ``{ts, tool, input_hash, latency_ms, status, error}``
            dicts. Written to ``tool_calls.csv``.
        workspace_root: Override the workspace root (used by tests).
        extra_env: Additional env vars to include in ``env.txt`` (e.g. the
            image tag the worker is running).

    Returns:
        Path to the ``_debug/`` directory containing the bundle.
    """
    workspace = ensure_thread_workspace(thread_id, root=workspace_root)
    debug_dir = workspace / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    _write_last_state(debug_dir, state)
    if exception is not None:
        _write_error(debug_dir, exception)
    if tool_calls:
        _write_tool_calls_csv(debug_dir, tool_calls)
    _write_env(debug_dir, thread_id, extra_env)
    _write_workspace_tarball(debug_dir, workspace)

    # progress copy (read by humans first — keep it discoverable in the bundle)
    progress_src = workspace / "PROGRESS.md"
    if progress_src.exists():
        (debug_dir / "PROGRESS.md").write_text(progress_src.read_text(encoding="utf-8"))

    # Touch a marker that scripts can detect cheaply (find . -name _debug.ok)
    (debug_dir / "_debug.ok").write_text(str(time.time()))

    return debug_dir
