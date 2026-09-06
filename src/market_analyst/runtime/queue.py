"""At-least-once Redis delivery with pending reclamation and a dead-letter stream.

Reclamation does not establish ownership of a running graph. The worker also
holds a process lock for its run on the shared local workspace volume.
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
_claim_cursors: dict[str, str] = {}


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
    deliveries: int = 1


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
    reclaim_idle_ms: int = 60000,
) -> RunJob | None:
    """Block for one message; return ``None`` on timeout."""
    r = client or _redis_client()
    ensure_consumer_group(r)
    claimed = cast(
        Any, r.xautoclaim(STREAM_KEY, CONSUMER_GROUP, consumer_name, min_idle_time=reclaim_idle_ms, start_id=_claim_cursors.get(consumer_name, "0-0"), count=1)
    )
    _claim_cursors[consumer_name] = claimed[0]
    entries = claimed[1]
    if not entries:
        streams = cast(Any, r.xreadgroup(groupname=CONSUMER_GROUP, consumername=consumer_name, streams={STREAM_KEY: ">"}, count=1, block=block_ms))
        entries = streams[0][1] if streams else []
    if not entries:
        return None
    message_id, fields = entries[0]
    try:
        payload = json.loads(fields["job"])
        if payload.get("mode") not in {"auto", "deep", "flash"} or not all(
            isinstance(payload.get(k), str) and payload[k] for k in ("thread_id", "user_id", "query")
        ):
            raise ValueError("Invalid job payload")
        pending = cast(Any, r.xpending_range(STREAM_KEY, CONSUMER_GROUP, message_id, message_id, 1))
        return RunJob(message_id=message_id, **payload, deliveries=pending[0]["times_delivered"] if pending else 1)
    except (ValueError, TypeError, KeyError) as exc:
        dead_letter(message_id, fields, str(exc), client=r)
        return None


def dead_letter(message_id: str, payload: dict, error: str, *, client=None) -> None:
    """Record failure and acknowledge atomically; an unavailable Redis leaves pending."""
    r = client or _redis_client()
    with r.pipeline(transaction=True) as pipe:
        pipe.xadd(STREAM_KEY + ":dead", {"message_id": message_id, "payload": json.dumps(payload), "error": error})
        pipe.xack(STREAM_KEY, CONSUMER_GROUP, message_id)
        pipe.execute()


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
