"""Redis-based user profile store for cross-thread memory.

This provides long-term memory that persists across conversation threads,
allowing the agent to remember user preferences like risk tolerance.

This demonstrates the separation of:
- Thread-level memory (PostgreSQL checkpoints) - conversation state
- Cross-thread memory (Redis store) - user preferences
"""

import json
import os
from typing import Optional

import redis

from market_analyst.schemas import UserProfile


class ProfileStore:
    """Redis-based store for user profiles.

    Stores user preferences that persist across conversation threads:
    - Risk tolerance
    - Investment horizon
    - Preferred sectors
    - Custom notes
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Initialize Redis connection.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._prefix = "market_analyst:profile:"

    def _key(self, user_id: str) -> str:
        """Generate Redis key for a user."""
        return f"{self._prefix}{user_id}"

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve user profile from Redis.

        Returns default profile if not found.

        Args:
            user_id: User identifier

        Returns:
            UserProfile instance
        """
        try:
            data = self.client.get(self._key(user_id))
            if data:
                profile_dict = json.loads(data)
                return UserProfile(**profile_dict)
        except (redis.RedisError, json.JSONDecodeError):
            pass

        return UserProfile()  # Return default profile

    def save_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Save user profile to Redis.

        Args:
            user_id: User identifier
            profile: UserProfile to save

        Returns:
            True if saved successfully
        """
        try:
            data = profile.model_dump_json()
            self.client.set(self._key(user_id), data)
            return True
        except redis.RedisError:
            return False

    def update_profile(self, user_id: str, **updates) -> UserProfile:
        """Partially update a user profile.

        Args:
            user_id: User identifier
            **updates: Fields to update

        Returns:
            Updated UserProfile
        """
        profile = self.get_profile(user_id)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self.save_profile(user_id, profile)
        return profile

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile.

        Args:
            user_id: User identifier

        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete(self._key(user_id))
            return True
        except redis.RedisError:
            return False

    def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            return self.client.ping()
        except redis.RedisError:
            return False


def get_profile_store() -> ProfileStore:
    """Factory function to create ProfileStore with environment config.

    Returns:
        Configured ProfileStore instance
    """
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))

    return ProfileStore(host=host, port=port)


# Convenience functions for direct profile access
def load_user_profile(user_id: str) -> UserProfile:
    """Load a user's profile from Redis.

    Args:
        user_id: User identifier

    Returns:
        UserProfile (default if not found)
    """
    store = get_profile_store()
    return store.get_profile(user_id)


def save_user_profile(user_id: str, profile: UserProfile) -> bool:
    """Save a user's profile to Redis.

    Args:
        user_id: User identifier
        profile: Profile to save

    Returns:
        True if successful
    """
    store = get_profile_store()
    return store.save_profile(user_id, profile)
