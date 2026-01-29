"""Utility functions for state access."""

from typing import Any, TypeVar

T = TypeVar("T")


def get_state_attr(state: Any, attr: str, default: T = None) -> T:
    """Safely extract attribute from state object or dict.

    Args:
        state: State object (can be a dataclass, Pydantic model, or dict).
        attr: Attribute name to extract.
        default: Default value if attribute is not found.

    Returns:
        The attribute value or default if not found.
    """
    if hasattr(state, attr):
        return getattr(state, attr)
    if isinstance(state, dict):
        return state.get(attr, default)
    return default
