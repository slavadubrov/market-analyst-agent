"""PostgreSQL connection and checkpointer management."""

import atexit
import logging
import os
from typing import Any, cast
from urllib.parse import quote

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from market_analyst.memory.encryption import checkpoint_serializer

logger = logging.getLogger(__name__)


def get_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "market_analyst")
    user = os.getenv("POSTGRES_USER", "analyst")
    password = os.getenv("POSTGRES_PASSWORD", "analyst_pass")

    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{quote(db, safe='')}"


# Global connection pool - reused across calls
_connection_pool: ConnectionPool[Connection[dict[str, Any]]] | None = None


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


def get_connection_pool() -> ConnectionPool[Connection[dict[str, Any]]]:
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        connection_string = get_connection_string()
        _connection_pool = cast(
            ConnectionPool[Connection[dict[str, Any]]],
            ConnectionPool(
                connection_string,
                min_size=1,
                max_size=10,
                kwargs={"autocommit": True, "row_factory": dict_row},
            ),
        )
    return _connection_pool


def get_postgres_saver() -> PostgresSaver:
    """Create and configure a Postgres checkpointer.

    If ``CHECKPOINT_ENCRYPTION_KEY`` is set, the checkpointer wraps payloads
    with ``EncryptedSerializer`` per the article's production-hardening list.
    Otherwise the default serializer is used.

    Returns:
        Configured PostgresSaver instance
    """
    pool = get_connection_pool()
    serde = checkpoint_serializer()
    checkpointer = PostgresSaver(pool, serde=serde)
    checkpointer.setup()
    return checkpointer
