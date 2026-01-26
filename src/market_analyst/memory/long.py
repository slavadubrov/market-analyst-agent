"""Long-term memory management using Qdrant.

This module provides the interface for:
1. Storing User Profiles (long-term preferences)
2. Storing Semantic Memories (knowledge)
3. Vector Search capabilities (RAG)
"""

import uuid

from qdrant_client.http import models

from market_analyst.memory.qdrant import (
    DEFAULT_COLLECTION_NAME,
    VECTOR_SIZE,
    ensure_collection,
    get_client,
)
from market_analyst.schemas import UserProfile


class LongTermMemory:
    """Manager for long-term memory storage and retrieval."""

    def __init__(self):
        self.client = get_client()
        ensure_collection(self.client)
        self.vector_size = VECTOR_SIZE
        self.collection_name = DEFAULT_COLLECTION_NAME

    def _get_dummy_vector(self) -> list[float]:
        """Generate a dummy vector for when we don't have an embedding model yet.

        TODO: Integrate a real embedding model (e.g. fastembed or OpenAI/Anthropic).
        For now, we just want to enable storage and exact retrieval by user_id.
        """
        return [0.0] * self.vector_size

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve user profile by User ID (Exact Match)."""
        # Search by payload filter (exact match on user_id)
        results = self.client.scroll(
            collection_name=self.collection_name,
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
                collection_name=self.collection_name,
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
            print(f"Error saving to long-term memory: {e}")
            return False

    def update_profile(self, user_id: str, **updates) -> UserProfile:
        """Partially update a user profile."""
        profile = self.get_profile(user_id)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self.save_profile(user_id, profile)
        return profile

    def search_profiles(
        self, query_vector: list[float], limit: int = 5
    ) -> list[UserProfile]:
        """Search profiles by vector similarity."""
        results = self.client.search(
            collection_name=self.collection_name,
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


def get_long_term_memory() -> LongTermMemory:
    """Factory function."""
    return LongTermMemory()


# Convenience functions for backward compatibility or easy access
def load_user_profile(user_id: str) -> UserProfile:
    """Load a user's profile."""
    memory = get_long_term_memory()
    return memory.get_profile(user_id)


def save_user_profile(user_id: str, profile: UserProfile) -> bool:
    """Save a user's profile."""
    memory = get_long_term_memory()
    return memory.save_profile(user_id, profile)
