"""Constants for the Market Analyst Agent."""

# Model Configuration
MODEL_ENV_VAR = "MARKET_ANALYST_MODEL"

# Model Mappings
MODEL_MAP = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

DEFAULT_MODEL_KEY = "sonnet"
DEFAULT_MODEL = MODEL_MAP[DEFAULT_MODEL_KEY]

# User defaults
DEFAULT_USER_ID = "default"
