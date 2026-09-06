"""Durable operation ledger: reserve before an effect, replay only completed work.

SQLite transactions coordinate local processes. Pending/unknown operations require
reconciliation; a timeout is never permission to repeat an external effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path(os.getenv("IDEMPOTENCY_ROOT", "./.idempotency")).resolve()


class OperationUncertain(RuntimeError):
    """An earlier attempt may have executed; inspect its outcome before retrying."""


@dataclass
class IdempotencyStore:
    """One local ledger shared by processes; use a shared DB for multiple hosts."""

    root: Path = _DEFAULT_ROOT

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS operations (key TEXT PRIMARY KEY, parameters TEXT NOT NULL, status TEXT NOT NULL, result TEXT)")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.root / "operations.sqlite3", timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _key(thread_id: str, operation_id: str) -> str:
        return json.dumps([thread_id, operation_id])

    def has(self, thread_id: str, tool_call_id: str) -> bool:
        with self._connection() as db:
            return db.execute("SELECT 1 FROM operations WHERE key=?", (self._key(thread_id, tool_call_id),)).fetchone() is not None

    def fetch(self, thread_id: str, tool_call_id: str) -> Any | None:
        with self._connection() as db:
            row = db.execute("SELECT status, result FROM operations WHERE key=?", (self._key(thread_id, tool_call_id),)).fetchone()
        if row is None:
            return None
        if row[0] != "completed":
            raise OperationUncertain(f"Operation {tool_call_id} is {row[0]}; reconciliation required")
        return json.loads(row[1])

    def store(self, thread_id: str, tool_call_id: str, result: Any) -> None:
        """Settle a reserved effect, or record a result confirmed by reconciliation."""
        with self._connection() as db:
            changed = db.execute(
                "UPDATE operations SET status='completed', result=? WHERE key=? AND status IN ('pending', 'unknown')",
                (json.dumps(result), self._key(thread_id, tool_call_id)),
            ).rowcount
            if not changed:
                raise ValueError("Operation must be reserved and not already completed")

    def mark_unknown(self, thread_id: str, tool_call_id: str) -> None:
        with self._connection() as db:
            db.execute("UPDATE operations SET status='unknown' WHERE key=? AND status='pending'", (self._key(thread_id, tool_call_id),))

    def reserve_or_replay(self, thread_id: str, tool_call_id: str, parameters: Any = None) -> tuple[bool, Any | None]:
        """Claim a business operation; changed arguments or uncertain outcomes stop."""
        key = self._key(thread_id, tool_call_id)
        payload = json.dumps(parameters, sort_keys=True, allow_nan=False)
        # Old empty file reservations are uncertain too. Never silently re-execute.
        legacy = self.root / (hashlib.sha256(f"{thread_id}::{tool_call_id}".encode()).hexdigest()[:16] + ".json")
        if legacy.exists():
            raise OperationUncertain("Legacy operation requires reconciliation before migration")
        with self._connection() as db:
            inserted = db.execute("INSERT OR IGNORE INTO operations VALUES (?, ?, 'pending', NULL)", (key, payload)).rowcount
            if inserted:
                return False, None
            row = db.execute("SELECT parameters, status, result FROM operations WHERE key=?", (key,)).fetchone()
            if row[0] != payload:
                raise ValueError("Operation ID is already bound to different parameters")
            if row[1] != "completed":
                raise OperationUncertain(f"Operation {tool_call_id} is {row[1]}; reconciliation required")
            return True, json.loads(row[2])


_store: IdempotencyStore | None = None


def get_idempotency_store() -> IdempotencyStore:
    global _store
    if _store is None:
        _store = IdempotencyStore(root=_DEFAULT_ROOT)
    return _store
