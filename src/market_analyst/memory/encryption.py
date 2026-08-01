"""EncryptedSerializer support for the Postgres checkpointer.

Per the article's "What Has to Change Before This Goes to Production" list:

> The Postgres checkpointer should run with ``EncryptedSerializer`` for any
> sensitive payloads, plus ``LANGGRAPH_STRICT_MSGPACK=true`` to defend
> against deserialization attacks if the DB is ever compromised.

LangGraph ships ``EncryptedSerializer`` under
``langgraph.checkpoint.serde.encrypted``. It encrypts the checkpoint payload
with a symmetric key (AES-256-GCM) before it hits Postgres. The key comes
from ``CHECKPOINT_ENCRYPTION_KEY`` -- a 32-byte URL-safe base64 string -- and
is rotated by re-encrypting on read.

This module is a thin "if the env var is set, return a serializer; otherwise
return ``None``" wrapper. ``postgres_store.get_postgres_saver`` calls it and
hands the serializer to ``PostgresSaver`` when present.

Why a wrapper instead of always-on encryption: in development the demo runs
on ``docker compose up`` with no secret manager. Requiring a key up front
would break the friction-free start the article (and the README) sells.
Production overrides the default by setting the env var.
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
    """Return an EncryptedSerializer if a key is configured, else ``None``.

    Soft-fails when the LangGraph version in use does not ship the encrypted
    serializer (older releases) — we log a warning and continue with the
    default unencrypted serializer rather than crashing the boot.
    """
    key = os.getenv(ENCRYPTION_KEY_ENV)
    if not key:
        return None

    try:
        from langgraph.checkpoint.serde.encrypted import EncryptedSerializer  # type: ignore
    except ImportError:
        logger.warning(
            "%s is set but langgraph.checkpoint.serde.encrypted is unavailable; running without checkpoint encryption.",
            ENCRYPTION_KEY_ENV,
        )
        return None

    try:
        decoded_key = base64.urlsafe_b64decode(key)
        if len(decoded_key) != 32:
            raise ValueError(f"{ENCRYPTION_KEY_ENV} must decode to 32 bytes")
        if serde is not None:
            return EncryptedSerializer.from_pycryptodome_aes(serde=serde, key=decoded_key)
        return EncryptedSerializer.from_pycryptodome_aes(key=decoded_key)
    except (AttributeError, binascii.Error, ImportError, ValueError) as exc:
        logger.warning("Failed to build EncryptedSerializer: %s", exc)
        return None
