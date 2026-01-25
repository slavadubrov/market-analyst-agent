"""PostgreSQL checkpointer for state persistence.

This module provides checkpoint functionality using PostgresSaver,
enabling the agent to:
1. Pause and resume mid-analysis
2. Survive system crashes
3. Support time-travel debugging
"""

import os
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


def get_postgres_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "market_analyst")
    user = os.getenv("POSTGRES_USER", "analyst")
    password = os.getenv("POSTGRES_PASSWORD", "analyst_pass")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# Global connection pool - reused across calls
_connection_pool: ConnectionPool | None = None


def _close_pool():
    """Close the connection pool on exit."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.close()
        except Exception:
            pass
        _connection_pool = None


# Register cleanup handler
import atexit

atexit.register(_close_pool)


def _get_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        connection_string = get_postgres_connection_string()
        _connection_pool = ConnectionPool(
            connection_string,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True},  # Important for setup() and checkpointing
        )
    return _connection_pool


def get_checkpointer() -> PostgresSaver:
    """Create and configure a PostgreSQL checkpointer.

    The checkpointer stores:
    - Full agent state at each step
    - Message history
    - Tool call results
    - Plan progress

    This enables:
    - Pause/resume functionality
    - Crash recovery
    - State inspection for debugging

    Returns:
        Configured PostgresSaver instance
    """
    pool = _get_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return checkpointer


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
        pass  # Pool manages connection lifecycle


def get_thread_state(thread_id: str, checkpointer: PostgresSaver) -> dict | None:
    """Retrieve the latest state for a thread.

    Useful for debugging or resuming a conversation.

    Args:
        thread_id: The thread identifier
        checkpointer: PostgresSaver instance

    Returns:
        The latest state dict, or None if not found
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        return checkpointer.get(config)
    except Exception:
        return None


def list_thread_history(
    thread_id: str, checkpointer: PostgresSaver, limit: int = 10
) -> list:
    """List checkpoint history for a thread.

    Enables "time travel" by listing all saved states.

    Args:
        thread_id: The thread identifier
        checkpointer: PostgresSaver instance
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
