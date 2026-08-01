"""Idempotency-key store for tool calls with side effects.

Per the article: any tool with side effects needs an idempotency key derived
from ``(session_id, tool_call_id)``, stored *before* the side effect fires.
At-least-once delivery makes retries inevitable; without keys, duplicate
writes are inevitable too.

The store is a filesystem-backed key-value DB rooted at
``./.idempotency``. Each key becomes a file containing the JSON-serialized
result, so the second call returns the same payload it returned the first
time and the side effect never re-runs.

Why filesystem (not Postgres / Redis): the reference repo runs on
``docker compose up`` with no shared infra beyond what the agent already
needs. Production deployments should swap this for the same Postgres that
holds checkpoints (one transaction insert-or-fetch).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path(os.getenv("IDEMPOTENCY_ROOT", "./.idempotency")).resolve()


def _hash_key(thread_id: str, tool_call_id: str) -> str:
    """Hash (thread_id, tool_call_id) → 16-char hex for filesystem-safe keys."""
    digest = hashlib.sha256(f"{thread_id}::{tool_call_id}".encode()).hexdigest()
    return digest[:16]


@dataclass
class IdempotencyStore:
    """Filesystem-backed idempotency key store.

    Single-writer semantics within a process; concurrent writers across
    processes are safe at coarse granularity because each key is its own
    file (atomic create on POSIX via ``O_EXCL`` is below in
    :meth:`reserve_or_replay`).
    """

    root: Path = _DEFAULT_ROOT

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, thread_id: str, tool_call_id: str) -> Path:
        return self.root / f"{_hash_key(thread_id, tool_call_id)}.json"

    def has(self, thread_id: str, tool_call_id: str) -> bool:
        return self._path_for(thread_id, tool_call_id).exists()

    def fetch(self, thread_id: str, tool_call_id: str) -> Any | None:
        """Return the stored result, or ``None`` when the key is unknown."""
        path = self._path_for(thread_id, tool_call_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Half-written file from a crashed reservation; treat as missing
            # so the next call re-runs the side effect from scratch.
            return None

    def store(self, thread_id: str, tool_call_id: str, result: Any) -> None:
        """Persist the result of an idempotent operation.

        Writes to a temp file in the same directory then renames into place,
        so a partial write never leaves a corrupted JSON file behind.
        """
        path = self._path_for(thread_id, tool_call_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(path)

    def reserve_or_replay(self, thread_id: str, tool_call_id: str) -> tuple[bool, Any | None]:
        """Atomically: if seen, return ``(True, prior_result)``; else mark seen.

        The "reserve" semantics mean: by the time you call the underlying side
        effect, this function has already claimed the key. A duplicate call
        from another worker will see the marker and short-circuit.

        Returns:
            ``(seen, result)`` where ``seen`` is ``True`` when a prior result
            exists. The caller should return ``result`` directly on a replay
            and otherwise proceed with the side effect, calling
            :meth:`store` once the operation succeeds.
        """
        path = self._path_for(thread_id, tool_call_id)
        if path.exists():
            return True, self.fetch(thread_id, tool_call_id)
        # Reserve by creating an empty marker; .store() will overwrite later.
        # O_EXCL avoids two workers both passing this check on the same key.
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return True, self.fetch(thread_id, tool_call_id)
        return False, None


_store: IdempotencyStore | None = None


def get_idempotency_store() -> IdempotencyStore:
    """Return the process-singleton store. First call creates the directory."""
    global _store
    if _store is None:
        _store = IdempotencyStore()
    return _store
