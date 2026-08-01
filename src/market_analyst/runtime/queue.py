"""Redis Streams queue + worker (Part 5: queue + worker + checkpoint DB shape).

The article calls this shape out as "the default I recommend for most teams":

> The app accepts a request, creates a session row, pushes a job, and returns
> a run ID. The worker pulls the job, runs the harness, writes checkpoints,
> streams status, and stores artifacts as it goes. Postgres survives, workers
> are cattle, queue depth gives you backpressure.

This module is the queue half (push + pull); :mod:`market_analyst.runtime.worker`
is the consumer loop.

Implementation notes
--------------------

- Redis Streams (``XADD`` / ``XREADGROUP``) gives at-least-once delivery and
  a consumer-group abstraction without standing up RabbitMQ.
- The job payload is a JSON dict with ``thread_id``, ``user_id``, ``query``,
  ``mode``. It's the minimum the worker needs to reconstruct an invocation.
- Acknowledgement (``XACK``) happens only after the worker writes a checkpoint
  successfully. A crash between pull and ack causes the message to be
  re-delivered, which is fine because the checkpoint plus the idempotency
  store make resumes safe.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, cast

import redis

STREAM_KEY = os.getenv("RUN_QUEUE_STREAM", "market_analyst:runs")
CONSUMER_GROUP = os.getenv("RUN_QUEUE_GROUP", "workers")


def _redis_client(url: str | None = None) -> redis.Redis:
    """Build a Redis client. URL precedence: arg > REDIS_URL > default."""
    redis_url = url or os.getenv("REDIS_URL") or "redis://localhost:6379"
    return redis.Redis.from_url(
        redis_url,
        decode_responses=True,
    )


def ensure_consumer_group(client: redis.Redis | None = None) -> None:
    """Create the consumer group on the stream if it doesn't already exist."""
    r = client or _redis_client()
    try:
        r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:
        # BUSYGROUP means the group already exists. Anything else propagates.
        if "BUSYGROUP" not in str(exc):
            raise


@dataclass
class RunJob:
    """A unit of work pulled off the queue."""

    message_id: str
    thread_id: str
    user_id: str
    query: str
    mode: str  # "auto" | "deep" | "flash"


def push_run(
    query: str,
    *,
    user_id: str = "default",
    mode: str = "auto",
    thread_id: str | None = None,
    client: redis.Redis | None = None,
) -> str:
    """Enqueue a run; return the ``thread_id`` for client tracking.

    ``thread_id`` is generated here (UUID4) so the caller has it before the
    worker picks the job up. This matches the article's "the app … returns a
    run ID" flow.
    """
    r = client or _redis_client()
    thread_id = thread_id or str(uuid.uuid4())
    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "user_id": user_id,
        "query": query,
        "mode": mode,
    }
    r.xadd(STREAM_KEY, {"job": json.dumps(payload)})
    return thread_id


def pull_one(
    consumer_name: str,
    *,
    block_ms: int = 5000,
    client: redis.Redis | None = None,
) -> RunJob | None:
    """Block for one message; return ``None`` on timeout."""
    r = client or _redis_client()
    ensure_consumer_group(r)
    streams = cast(
        list[tuple[str, list[tuple[str, dict[str, str]]]]],
        r.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=consumer_name,
            streams={STREAM_KEY: ">"},
            count=1,
            block=block_ms,
        ),
    )
    if not streams:
        return None
    _stream, entries = streams[0]
    if not entries:
        return None
    message_id, fields = entries[0]
    payload = json.loads(fields["job"])
    return RunJob(message_id=message_id, **payload)


def ack(message_id: str, client: redis.Redis | None = None) -> None:
    """Mark a message as processed so it isn't redelivered."""
    r = client or _redis_client()
    r.xack(STREAM_KEY, CONSUMER_GROUP, message_id)


def queue_depth(client: redis.Redis | None = None) -> int:
    """Return the number of un-acked messages — useful for backpressure dashboards."""
    r = client or _redis_client()
    try:
        info = cast(list[dict[str, Any]], r.xinfo_groups(STREAM_KEY))
    except redis.ResponseError:
        return 0
    for group in info:
        if group["name"] == CONSUMER_GROUP:
            return int(group.get("pending", 0)) + int(group.get("lag", 0) or 0)
    return 0
