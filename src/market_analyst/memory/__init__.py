"""Memory module for Market Analyst Agent.

Restructured into:
- hot: Short-term state persistence (PostgreSQL)
- long: Long-term memory and profiles (Qdrant)
"""

from market_analyst.memory.hot import (
    checkpointer_context,
    get_checkpointer,
    get_thread_state,
    list_thread_history,
)
from market_analyst.memory.long import (
    LongTermMemory,
    get_long_term_memory,
    load_user_profile,
    save_user_profile,
)
from market_analyst.memory.postgres_store import get_connection_string

__all__ = [
    # Hot Memory
    "get_checkpointer",
    "checkpointer_context",
    "get_thread_state",
    "list_thread_history",
    "get_connection_string",
    # Long Memory
    "LongTermMemory",
    "get_long_term_memory",
    "load_user_profile",
    "save_user_profile",
]
