import os

from market_analyst.memory.hot import get_checkpointer, list_thread_history
from market_analyst.memory.long import LongTermMemory
from market_analyst.memory.postgres_store import (
    get_connection_string,
    get_postgres_saver,
)
from market_analyst.memory.redis_store import get_connection_url, get_redis_saver
from market_analyst.schemas import UserProfile

# --- Test Hot Memory (hot.py) ---


def test_get_checkpointer_default(mocker):
    """Test defaults to Postgres when env var is not set."""
    mock_get_redis = mocker.patch("market_analyst.memory.hot.get_redis_saver")
    mock_get_postgres = mocker.patch("market_analyst.memory.hot.get_postgres_saver")

    mocker.patch.dict(os.environ, {}, clear=True)

    get_checkpointer()
    mock_get_postgres.assert_called_once()
    mock_get_redis.assert_not_called()


def test_get_checkpointer_redis(mocker):
    """Test uses Redis when env var is set."""
    mock_get_redis = mocker.patch("market_analyst.memory.hot.get_redis_saver")
    mock_get_postgres = mocker.patch("market_analyst.memory.hot.get_postgres_saver")

    mocker.patch.dict(os.environ, {"HOT_MEMORY_PROVIDER": "redis"})

    get_checkpointer()
    mock_get_redis.assert_called_once()
    mock_get_postgres.assert_not_called()


def test_list_thread_history(mocker):
    """Test listing thread history from checkpointer."""
    mock_checkpointer = mocker.MagicMock()
    # Mock return value of .list()
    mock_checkpoint = mocker.MagicMock()
    mock_checkpoint.config = {"checkpoint_id": "cp1"}
    mock_checkpoint.metadata = {"created_at": "2024-01-01", "step": 1}
    mock_checkpointer.list.return_value = [mock_checkpoint]

    history = list_thread_history("thread-1", mock_checkpointer)

    assert len(history) == 1
    assert history[0]["thread_id"] == "thread-1"
    assert history[0]["checkpoint_id"] == "cp1"


# --- Test Long Memory (long.py) ---


def test_long_term_memory_get_profile(mocker):
    """Test retrieving user profile."""
    _ = mocker.patch("market_analyst.memory.long.ensure_collection")
    mock_get_client = mocker.patch("market_analyst.memory.long.get_client")

    mock_client = mocker.MagicMock()
    mock_get_client.return_value = mock_client

    # Mock scroll result
    mock_point = mocker.MagicMock()
    mock_point.payload = {"user_id": "u1", "risk_tolerance": "moderate"}

    # client.scroll returns (points, offset)
    mock_client.scroll.return_value = ([mock_point], None)

    memory = LongTermMemory()
    profile = memory.get_profile("u1")

    assert isinstance(profile, UserProfile)
    assert profile.risk_tolerance == "moderate"


def test_long_term_memory_save_profile(mocker):
    """Test saving user profile."""
    _ = mocker.patch("market_analyst.memory.long.ensure_collection")
    mock_get_client = mocker.patch("market_analyst.memory.long.get_client")

    mock_client = mocker.MagicMock()
    mock_get_client.return_value = mock_client

    memory = LongTermMemory()
    profile = UserProfile(risk_tolerance="conservative")
    success = memory.save_profile("u1", profile)

    assert success is True
    mock_client.upsert.assert_called_once()

    # Verify call args
    call_args = mock_client.upsert.call_args
    assert call_args.kwargs["collection_name"] == memory.collection_name
    points = call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["risk_tolerance"] == "conservative"
    assert points[0].payload["user_id"] == "u1"


# --- Test Postgres Store (postgres_store.py) ---


def test_postgres_connection_string(mocker):
    """Test connection string generation from env vars."""
    mocker.patch.dict(
        os.environ,
        {
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_pass",
            "POSTGRES_HOST": "test_host",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "test_db",
        },
    )

    conn_str = get_connection_string()
    assert conn_str == "postgresql://test_user:test_pass@test_host:5432/test_db"


def test_get_postgres_saver(mocker):
    """Test creating PostgresSaver."""
    mock_pool = mocker.patch("market_analyst.memory.postgres_store.ConnectionPool")
    mock_saver = mocker.patch("market_analyst.memory.postgres_store.PostgresSaver")

    _ = get_postgres_saver()
    mock_pool.assert_called_once()
    mock_saver.assert_called_once()
    mock_saver.return_value.setup.assert_called_once()


# --- Test Redis Store (redis_store.py) ---


def test_redis_connection_url(mocker):
    """Test Redis URL from env vars."""
    mocker.patch.dict(os.environ, {"REDIS_URL": "redis://test:6379"})
    url = get_connection_url()
    assert url == "redis://test:6379"


def test_get_redis_saver(mocker):
    """Test creating RedisSaver."""
    mock_redis = mocker.patch("market_analyst.memory.redis_store.Redis")
    mock_saver = mocker.patch("market_analyst.memory.redis_store.RedisSaver")

    _ = get_redis_saver()
    mock_redis.from_url.assert_called_once()
    mock_saver.assert_called_once()
