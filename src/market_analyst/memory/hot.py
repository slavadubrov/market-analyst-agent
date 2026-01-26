"""Hot memory (short-term state) management using PostgreSQL or Redis.

This module enables:
1. Pause and resume mid-analysis
2. Crash recovery
3. Time-travel debugging
"""

import os
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver

from market_analyst.memory.postgres_store import get_postgres_saver
from market_analyst.memory.redis_store import get_redis_saver


def get_checkpointer() -> BaseCheckpointSaver:
    """Create and configure a checkpointer (Postgres or Redis).

    The checkpointer stores:
    - Full agent state at each step
    - Message history
    - Tool call results
    - Plan progress

    Returns:
        Configured BaseCheckpointSaver instance
    """
    provider = os.getenv("HOT_MEMORY_PROVIDER", "postgres").lower()

    if provider == "redis":
        return get_redis_saver()

    # Default to Postgres
    return get_postgres_saver()


@contextmanager
def checkpointer_context():
    """Context manager for checkpointer that ensures proper cleanup.

    Usage:
        with checkpointer_context() as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            graph.invoke(...)
    """
    checkpointer = get_checkpointer()
    try:
        yield checkpointer
    finally:
        pass  # Pool manages connection lifecycle, handled by postgres module


def get_thread_state(thread_id: str, checkpointer: BaseCheckpointSaver) -> dict | None:
    """Retrieve the latest state for a thread.

    Useful for debugging or resuming a conversation.

    Args:
        thread_id: The thread identifier
        checkpointer: BaseCheckpointSaver instance

    Returns:
        The latest state dict, or None if not found
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        return checkpointer.get(config)
    except Exception:
        return None


def list_thread_history(
    thread_id: str, checkpointer: BaseCheckpointSaver, limit: int = 10
) -> list:
    """List checkpoint history for a thread.

    Enables "time travel" by listing all saved states.

    Args:
        thread_id: The thread identifier
        checkpointer: BaseCheckpointSaver instance
        limit: Maximum number of checkpoints to return

    Returns:
        List of checkpoint metadata
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = []

    for checkpoint in checkpointer.list(config, limit=limit):
        history.append(
            {
                "checkpoint_id": checkpoint.config.get("checkpoint_id"),
                "thread_id": thread_id,
                "timestamp": checkpoint.metadata.get("created_at"),
                "step": checkpoint.metadata.get("step"),
            }
        )

    return history
