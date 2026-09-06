"""Optional checkpoint encryption. A configured but unusable key fails closed.

Uses the SDK cipher implementation; no key rotation is implemented by this demo.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENCRYPTION_KEY_ENV = "CHECKPOINT_ENCRYPTION_KEY"


def get_encrypted_serializer(serde: Any = None) -> Any | None:
    """Return encryption when configured; never silently downgrade to plaintext."""
    key = os.getenv(ENCRYPTION_KEY_ENV)
    if not key:
        return None

    try:
        from langgraph.checkpoint.serde.encrypted import EncryptedSerializer  # type: ignore
    except ImportError as exc:
        raise ValueError("Checkpoint encryption is configured but its serializer is unavailable") from exc

    try:
        decoded_key = base64.urlsafe_b64decode(key)
        if len(decoded_key) != 32:
            raise ValueError(f"{ENCRYPTION_KEY_ENV} must decode to 32 bytes")
        if serde is not None:
            return EncryptedSerializer.from_pycryptodome_aes(serde=serde, key=decoded_key)
        return EncryptedSerializer.from_pycryptodome_aes(key=decoded_key)
    except (AttributeError, binascii.Error, ImportError, ValueError) as exc:
        raise ValueError("Cannot configure checkpoint encryption") from exc


def checkpoint_serializer():
    """Explicitly allow demo state types for PostgreSQL and in-memory checkpoints."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("market_analyst.schemas", name)
            for name in (
                "AgentState",
                "DraftReport",
                "ExecutionMode",
                "GuardianDecision",
                "GuardianResult",
                "PlanStep",
                "ResearchData",
                "ReWOOPlanStep",
                "TradeRequest",
                "TradeAction",
                "UserProfile",
            )
        ],
    )
    return get_encrypted_serializer(serde) or serde
