"""Long-term memory management using Qdrant.

This module provides the interface for:
1. Storing User Profiles (long-term preferences)
2. Storing Semantic Memories (knowledge)
3. Vector Search capabilities (RAG)
"""

import uuid
from datetime import datetime, timedelta, timezone

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

    def __init__(self, client=None):
        self.client = client if client is not None else get_client()
        ensure_collection(self.client)
        self.vector_size = VECTOR_SIZE
        self.collection_name = DEFAULT_COLLECTION_NAME

    def _get_dummy_vector(self) -> list[float]:
        """Generate a placeholder vector for storage without embeddings.

        Returns a zero vector of the configured size. This enables storage
        and exact-match retrieval by user_id without requiring an embedding model.

        TODO: Integrate a real embedding model (e.g. fastembed or OpenAI/Anthropic)
        for semantic search capabilities.

        Returns:
            Zero vector of length self.vector_size.
        """
        return [0.0] * self.vector_size

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve user profile by User ID (Exact Match)."""
        # Search by payload filter (exact match on user_id)
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._scope(user_id),
            limit=1,
        )

        points, _ = results
        if points:
            payload = points[0].payload or {}
            # Reconstruction of UserProfile from payload
            # We filter out internal keys if any
            profile_data = {k: v for k, v in payload.items() if k in UserProfile.model_fields}
            return UserProfile(**profile_data)

        return UserProfile()  # Return default

    def save_profile(self, user_id: str, profile: UserProfile, *, ttl_days: int = 365) -> bool:
        """Save user profile to Qdrant.

        Upserts the point. We use a deterministic UUID based on user_id for the Point ID
        to ensure updates overwrite old data.
        """
        if not user_id or ttl_days <= 0:
            raise ValueError("A principal and positive profile TTL are required")
        try:
            # Create a deterministic UUID from the user_id string
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id))

            payload = profile.model_dump()
            payload["user_id"] = user_id
            payload["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

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
            raise RuntimeError("Profile could not be saved") from e

    def update_profile(self, user_id: str, **updates) -> UserProfile:
        """Partially update a user profile with provided fields.

        Args:
            user_id: Unique user identifier.
            **updates: Key-value pairs of profile fields to update.

        Returns:
            Updated UserProfile instance.
        """
        profile = self.get_profile(user_id)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self.save_profile(user_id, profile)
        return profile

    def search_profiles(self, query_vector: list[float], limit: int = 5, *, user_id: str) -> list[UserProfile]:
        """Search profiles by vector similarity."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._scope(user_id),
            limit=limit,
        ).points

        profiles = []
        for point in results:
            if point.payload:
                profile_data = {k: v for k, v in point.payload.items() if k in UserProfile.model_fields}
                profiles.append(UserProfile(**profile_data))

        return profiles

    @staticmethod
    def _scope(user_id: str) -> models.Filter:
        if not user_id:
            raise ValueError("A principal is required")
        return models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="expires_at", range=models.DatetimeRange(gt=datetime.now(timezone.utc))),
            ]
        )

    def delete_profile(self, user_id: str) -> None:
        if not user_id:
            raise ValueError("A principal is required")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])),
            wait=True,
        )

    def purge_expired(self) -> None:
        """Physically remove expired records; reads exclude them immediately."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="expires_at", range=models.DatetimeRange(lte=datetime.now(timezone.utc)))])
            ),
            wait=True,
        )


def get_long_term_memory() -> LongTermMemory:
    """Factory function."""
    return LongTermMemory()


# Convenience functions for backward compatibility or easy access
def load_user_profile(user_id: str) -> UserProfile:
    """Load a user's profile."""
    try:
        return get_long_term_memory().get_profile(user_id)
    except Exception:
        return UserProfile()


def save_user_profile(user_id: str, profile: UserProfile) -> bool:
    """Save a user's profile."""
    memory = get_long_term_memory()
    return memory.save_profile(user_id, profile)
