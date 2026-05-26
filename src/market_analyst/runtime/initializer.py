"""Initializer / progress-file pattern.

Per the article's "Feature amnesia across context windows" row: the
initializer writes ``claude-progress.txt`` / ``feature-list.json`` / ``init.sh``
into the workspace, and the coding agent reads them on every cold boot. In
our investment-research workflow we keep the same shape but tailor the
contents — a ``PROGRESS.md`` and a ``feature-list.json`` per thread, plus an
``InitializerContext`` the boot hook can return to the next session.

Recovery flow (matches §"The Healthy Run Lifecycle" step 1):

    workspace = ensure_thread_workspace(thread_id)
    init = InitializerContext.load(workspace)
    if init.is_resume():
        # rehydrate prior progress, append, continue
    else:
        # write the initial PROGRESS.md and continue
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROGRESS_FILENAME = "PROGRESS.md"
FEATURE_LIST_FILENAME = "feature-list.json"


@dataclass
class InitializerContext:
    """Snapshot of what the previous worker left behind in the workspace.

    Attributes:
        progress: Contents of ``PROGRESS.md``, or empty string on a fresh boot.
        feature_list: Parsed contents of ``feature-list.json``, or ``{}`` on a
            fresh boot. The structure is intentionally free-form so each
            workflow can put whatever it needs in there (planned steps,
            already-completed plan items, etc.).
        path: Absolute path to the workspace this context was loaded from.
    """

    progress: str = ""
    feature_list: dict = field(default_factory=dict)
    path: Path = field(default_factory=Path)

    def is_resume(self) -> bool:
        """True iff the workspace shows any sign of a previous run."""
        return bool(self.progress) or bool(self.feature_list)

    @classmethod
    def load(cls, workspace: Path) -> "InitializerContext":
        ctx = cls(path=workspace)
        progress_path = workspace / PROGRESS_FILENAME
        feature_path = workspace / FEATURE_LIST_FILENAME
        if progress_path.exists():
            ctx.progress = progress_path.read_text(encoding="utf-8")
        if feature_path.exists():
            try:
                ctx.feature_list = json.loads(feature_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Corrupted file from a crashed write — treat as missing.
                ctx.feature_list = {}
        return ctx


def write_progress(workspace: Path, *, line: str, prepend_timestamp: bool = True) -> Path:
    """Append a line to ``PROGRESS.md`` (creating the file if needed).

    Appends because the article emphasizes append-only logging — a crash mid-
    write should never wipe earlier progress.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    progress_path = workspace / PROGRESS_FILENAME
    prefix = ""
    if prepend_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        prefix = f"- [{ts}] "
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{prefix}{line}\n")
    return progress_path


def read_progress(workspace: Path) -> str:
    """Return ``PROGRESS.md`` contents, or empty string when the file is missing."""
    progress_path = workspace / PROGRESS_FILENAME
    if not progress_path.exists():
        return ""
    return progress_path.read_text(encoding="utf-8")


def write_feature_list(workspace: Path, features: dict) -> Path:
    """Write ``feature-list.json`` as a complete overwrite.

    Overwriting (not appending) is deliberate: the feature list is a desired-
    state document. Append-only semantics live in ``PROGRESS.md``.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    feature_path = workspace / FEATURE_LIST_FILENAME
    feature_path.write_text(json.dumps(features, indent=2), encoding="utf-8")
    return feature_path
