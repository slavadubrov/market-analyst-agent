"""PostgreSQL connection and checkpointer management."""

import atexit
import os

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


def get_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "market_analyst")
    user = os.getenv("POSTGRES_USER", "analyst")
    password = os.getenv("POSTGRES_PASSWORD", "analyst_pass")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# Global connection pool - reused across calls
_connection_pool: ConnectionPool | None = None


def close_pool():
    """Close the connection pool on exit."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.close()
        except Exception:
            pass
        _connection_pool = None


# Register cleanup handler
atexit.register(close_pool)


def get_connection_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        connection_string = get_connection_string()
        _connection_pool = ConnectionPool(
            connection_string,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True},  # Important for setup() and checkpointing
        )
    return _connection_pool


def get_postgres_saver() -> PostgresSaver:
    """Create and configure a Postgres checkpointer.

    Returns:
        Configured PostgresSaver instance
    """
    pool = get_connection_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer
