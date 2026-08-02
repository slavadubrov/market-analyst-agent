"""Redis connection management for hot memory."""

import os

from langgraph.checkpoint.redis import RedisSaver


def get_connection_url() -> str:
    """Get Redis connection URL from environment variables."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def get_redis_saver() -> RedisSaver:
    """Create and configure a Redis checkpointer.

    Returns:
        Configured RedisSaver instance
    """
    checkpointer = RedisSaver(redis_url=get_connection_url())
    checkpointer.setup()
    return checkpointer
