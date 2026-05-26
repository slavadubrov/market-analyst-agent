"""Per-thread workspace directories.

The article's production checklist calls for ``/workspaces/${THREAD_ID}`` per
session so two runs cannot corrupt each other's files (failure mode: "workspace
drift"). In development we mount ``./workspaces/`` into the worker container
and create a subdirectory per ``thread_id`` here.

In production the same code path should hand the workspace to a per-task
Daytona / Runloop sandbox — the bullet right under the compose section of the
article — but for the reference stack a plain directory is fine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_DEFAULT_ROOT = Path(os.getenv("WORKSPACE_ROOT", "./workspaces")).resolve()

# Conservative thread-id sanitizer. LangGraph generates UUIDs by default which
# pass this trivially; the regex is a defense-in-depth against a caller that
# passes user input directly as ``thread_id``.
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(thread_id: str) -> str:
    return _SAFE_ID.sub("_", thread_id) or "unknown"


def workspace_path_for(thread_id: str, root: Path | None = None) -> Path:
    """Return the workspace path for a thread; does NOT create the directory."""
    base = (root or _DEFAULT_ROOT).resolve()
    return base / _sanitize(thread_id)


def ensure_thread_workspace(thread_id: str, root: Path | None = None) -> Path:
    """Create the per-thread workspace (and its ``_debug`` subdir) if missing.

    Returns the workspace path. Safe to call repeatedly — uses ``exist_ok=True``.
    """
    path = workspace_path_for(thread_id, root)
    path.mkdir(parents=True, exist_ok=True)
    (path / "_debug").mkdir(parents=True, exist_ok=True)
    return path
