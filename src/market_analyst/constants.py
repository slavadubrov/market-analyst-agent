"""Constants for the Market Analyst Agent."""

# Model Configuration
MODEL_ENV_VAR = "MARKET_ANALYST_MODEL"
MODEL_PROVIDER_ENV_VAR = "MARKET_ANALYST_PROVIDER"

# Model Mappings
MODEL_MAP = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

OPENAI_MODEL_MAP = {
    "sonnet": "gpt-5.6-sol",
    "haiku": "gpt-5.6-luna",
}

DEFAULT_MODEL_KEY = "sonnet"
DEFAULT_MODEL = MODEL_MAP[DEFAULT_MODEL_KEY]

# User defaults
DEFAULT_USER_ID = "default"
