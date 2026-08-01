"""Utility functions for state access."""

import re
from typing import Any

_TICKER_RE = re.compile(r"[A-Z][A-Z0-9]{0,4}(?:[.-][A-Z0-9]{1,4})?")


def normalize_ticker(value: str) -> str:
    """Normalize a common exchange ticker and reject unsafe query text."""
    ticker = value.strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")
    return ticker


def get_state_attr(state: Any, attr: str, default: Any = None) -> Any:
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
