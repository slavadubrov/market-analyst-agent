"""Qdrant-based store for long-term memory.

This module provides the interface to Qdrant Vector DB for:
1. Storing User Profiles (long-term preferences)
2. Storing Semantic Memories (knowledge)
3. Vector Search capabilities (RAG)
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from market_analyst.schemas import UserProfile


class QdrantStore:
    """Qdrant-based store for user profiles and semantic memory.

    Implements long-term storage, enabling semantic search.
    """

    COLLECTION_NAME = "user_profiles"
    VECTOR_SIZE = 768  # Standard size for many embedding models (e.g. HuggingFace)

    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client."""
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists."""
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )

    def _get_dummy_vector(self) -> List[float]:
        """Generate a dummy vector for when we don't have an embedding model yet.

        TODO: Integrate a real embedding model (e.g. fastembed or OpenAI/Anthropic).
        For now, we just want to enable storage and exact retrieval by user_id.
        """
        return [0.0] * self.VECTOR_SIZE

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve user profile by User ID (Exact Match).

        We store the user_id in the payload and use a Filter for retrieval.
        """
        # Search by payload filter (exact match on user_id)
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=1,
        )

        points, _ = results
        if points:
            payload = points[0].payload
            # Reconstruction of UserProfile from payload
            # We filter out internal keys if any
            profile_data = {k: v for k, v in payload.items() if k != "user_id"}
            return UserProfile(**profile_data)

        return UserProfile()  # Return default

    def save_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Save user profile to Qdrant.

        Upserts the point. We use a deterministic UUID based on user_id for the Point ID
        to ensure updates overwrite old data.
        """
        try:
            # Create a deterministic UUID from the user_id string
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id))

            payload = profile.model_dump()
            payload["user_id"] = user_id  # Add user_id to payload for filtering

            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=self._get_dummy_vector(),  # Placeholder for actual embedding
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as e:
            print(f"Error saving to Qdrant: {e}")
            return False

    def search_profiles(
        self, query_vector: List[float], limit: int = 5
    ) -> List[UserProfile]:
        """Search profiles by vector similarity.

        This enables finding "similar users" or accessing profiles based on semantic
        matching if we were embedding their bio/preferences.
        """
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
        )

        profiles = []
        for point in results:
            if point.payload:
                profile_data = {
                    k: v for k, v in point.payload.items() if k != "user_id"
                }
                profiles.append(UserProfile(**profile_data))

        return profiles


def get_qdrant_store() -> QdrantStore:
    """Factory function."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantStore(host=host, port=port)
