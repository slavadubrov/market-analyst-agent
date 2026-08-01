"""Shared chat model selection."""

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from market_analyst.constants import (
    DEFAULT_MODEL,
    MODEL_ENV_VAR,
    MODEL_PROVIDER_ENV_VAR,
    OPENAI_MODEL_MAP,
)


def get_chat_model(temperature: float = 0) -> BaseChatModel:
    """Use OpenAI when configured, otherwise Anthropic."""
    provider = os.getenv(MODEL_PROVIDER_ENV_VAR, "auto").lower()
    if provider == "auto":
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"

    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    if provider == "openai":
        role = "haiku" if "haiku" in model_name else "sonnet"
        openai_model = model_name if model_name.startswith("gpt-") else OPENAI_MODEL_MAP[role]
        return ChatOpenAI(model=openai_model, timeout=None, use_responses_api=True)
    if provider == "anthropic":
        return ChatAnthropic(model_name=model_name, temperature=temperature, timeout=None, stop=None)
    raise ValueError(f"Unsupported {MODEL_PROVIDER_ENV_VAR}: {provider}")


def get_structured_model(schema: type[BaseModel], temperature: float = 0) -> Runnable[Any, Any]:
    """Create structured output compatible with either provider."""
    model = get_chat_model(temperature)
    if isinstance(model, ChatOpenAI):
        return model.with_structured_output(schema, method="function_calling")
    return model.with_structured_output(schema)
