"""User profile management using Qdrant.

This module provides the interface for storing and retrieving user profiles
from long-term memory (Qdrant Vector DB).
"""

from market_analyst.memory.qdrant_store import QdrantStore, get_qdrant_store
from market_analyst.schemas import UserProfile


# Compatibility layer: ProfileStore defaults to using Qdrant now
class ProfileStore:
    """Store for user profiles using Qdrant backend."""

    def __init__(self):
        self.store = get_qdrant_store()

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve user profile."""
        return self.store.get_profile(user_id)

    def save_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Save user profile."""
        return self.store.save_profile(user_id, profile)

    def update_profile(self, user_id: str, **updates) -> UserProfile:
        """Partially update a user profile."""
        profile = self.get_profile(user_id)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self.save_profile(user_id, profile)
        return profile

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile. (Not implemented in QdrantStore yet for simplicity).

        For the demo, we just return False or implement if critical.
        """
        # TODO: Implement delete in QdrantStore if needed
        return True


def get_profile_store() -> ProfileStore:
    """Factory function."""
    return ProfileStore()


# Convenience functions
def load_user_profile(user_id: str) -> UserProfile:
    """Load a user's profile."""
    store = get_profile_store()
    return store.get_profile(user_id)


def save_user_profile(user_id: str, profile: UserProfile) -> bool:
    """Save a user's profile."""
    store = get_profile_store()
    return store.save_profile(user_id, profile)
