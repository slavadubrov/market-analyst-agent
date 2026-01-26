"""Redis connection management for hot memory."""

import os

from langgraph.checkpoint.redis import RedisSaver
from redis import Redis


def get_connection_url() -> str:
    """Get Redis connection URL from environment variables."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def get_redis_saver() -> RedisSaver:
    """Create and configure a Redis checkpointer.

    Returns:
        Configured RedisSaver instance
    """
    url = get_connection_url()
    # We need to maintain a reference to the connection,
    # but RedisSaver handles its own connection internally when passed a connection object or url.
    # However, looking at docs, RedisSaver usually takes a sync or async connection.
    # Let's create a sync Redis client.

    # Note: langgraph-checkpoint-redis documentation usually suggests:
    # conn = Redis.from_url(...)
    # saver = RedisSaver(conn)

    conn = Redis.from_url(url)
    return RedisSaver(conn)
