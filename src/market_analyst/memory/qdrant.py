"""Qdrant client management."""

import os

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Constants
DEFAULT_COLLECTION_NAME = "user_profiles"
VECTOR_SIZE = 768  # Standard size for many embedding models


def get_client() -> QdrantClient:
    """Get Qdrant client configured from environment variables."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def ensure_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_size: int = VECTOR_SIZE,
):
    """Ensure a collection exists with the specified configuration."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
