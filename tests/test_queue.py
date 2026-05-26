"""Tests for the Redis Streams queue (Part 5: queue + worker + checkpoint DB).

The Redis client is mocked end-to-end because we want unit tests, not an
integration test that needs ``docker compose up``. Integration coverage runs
when the Redis service is available (see the ``integration`` pytest marker
in pyproject).
"""

from __future__ import annotations

from typing import Any

import pytest

# --- Helpers ----------------------------------------------------------------


class FakeRedis:
    """Tiny Redis double that implements just the calls our queue uses."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, Any]]] = []
        self.next_id = 1
        self.group_created = False
        self.acks: list[str] = []

    # XADD
    def xadd(self, _stream: str, fields: dict[str, Any]) -> str:
        msg_id = f"{self.next_id}-0"
        self.next_id += 1
        self.entries.append((msg_id, fields))
        return msg_id

    # XGROUP CREATE
    def xgroup_create(self, *_args: Any, **_kw: Any) -> bool:
        if self.group_created:
            # Mirror real BUSYGROUP behavior so the queue.ensure_consumer_group
            # exception filter is exercised.
            import redis as redis_mod

            raise redis_mod.ResponseError("BUSYGROUP Consumer Group name already exists")
        self.group_created = True
        return True

    # XREADGROUP
    def xreadgroup(self, *, groupname: str, consumername: str, streams: dict, count: int, block: int):
        _ = (groupname, consumername, streams, count, block)
        if not self.entries:
            return []
        message_id, fields = self.entries.pop(0)
        return [("stream", [(message_id, fields)])]

    def xack(self, _stream: str, _group: str, message_id: str) -> int:
        self.acks.append(message_id)
        return 1

    def xinfo_groups(self, _stream: str):
        from market_analyst.runtime.queue import CONSUMER_GROUP

        return [{"name": CONSUMER_GROUP, "pending": 0, "lag": len(self.entries)}]


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()

    def _patch(_url: str | None = None):
        return fake

    monkeypatch.setattr("market_analyst.runtime.queue._redis_client", _patch)
    return fake


# --- push_run / pull_one round trip -----------------------------------------


def test_push_run_returns_thread_id(fake_redis):
    from market_analyst.runtime.queue import push_run

    thread_id = push_run(query="Analyze NVDA", user_id="u1")
    assert thread_id
    assert len(fake_redis.entries) == 1


def test_pull_one_returns_job(fake_redis):
    from market_analyst.runtime.queue import pull_one, push_run

    push_run(query="Analyze NVDA", user_id="u1", mode="flash", thread_id="thread-1")
    job = pull_one(consumer_name="worker-1", block_ms=100)
    assert job is not None
    assert job.thread_id == "thread-1"
    assert job.query == "Analyze NVDA"
    assert job.mode == "flash"


def test_pull_one_returns_none_when_empty(fake_redis):
    from market_analyst.runtime.queue import pull_one

    job = pull_one(consumer_name="worker-1", block_ms=10)
    assert job is None


def test_ack_records_message_id(fake_redis):
    from market_analyst.runtime.queue import ack, pull_one, push_run

    push_run(query="x", thread_id="thread-2")
    job = pull_one(consumer_name="w", block_ms=10)
    assert job is not None
    ack(job.message_id)
    assert fake_redis.acks == [job.message_id]


def test_ensure_consumer_group_handles_busygroup(fake_redis):
    """Second call must not raise — the group already exists."""
    from market_analyst.runtime.queue import ensure_consumer_group

    ensure_consumer_group()
    ensure_consumer_group()  # would raise without the filter


def test_queue_depth_reports_remaining_entries(fake_redis):
    from market_analyst.runtime.queue import push_run, queue_depth

    push_run(query="a", thread_id="t1")
    push_run(query="b", thread_id="t2")
    assert queue_depth() == 2
