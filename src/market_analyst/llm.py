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


def resolve_model(model_name: str | None = None) -> tuple[str, str]:
    """Resolve the active provider and provider-specific model name."""
    provider = os.getenv(MODEL_PROVIDER_ENV_VAR, "auto").lower()
    if provider == "auto":
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"

    model_name = model_name or os.getenv(MODEL_ENV_VAR) or DEFAULT_MODEL
    if provider == "openai":
        role = "haiku" if "haiku" in model_name else "sonnet"
        return provider, model_name if model_name.startswith("gpt-") else OPENAI_MODEL_MAP[role]
    if provider == "anthropic":
        return provider, model_name
    raise ValueError(f"Unsupported {MODEL_PROVIDER_ENV_VAR}: {provider}")


def get_chat_model(temperature: float = 0, model_name: str | None = None) -> BaseChatModel:
    """Use OpenAI when configured, otherwise Anthropic."""
    provider, resolved_model = resolve_model(model_name)
    if provider == "openai":
        return ChatOpenAI(model=resolved_model, timeout=None, use_responses_api=True)
    return ChatAnthropic(model_name=resolved_model, temperature=temperature, timeout=None, stop=None)


def get_structured_model(
    schema: type[BaseModel],
    temperature: float = 0,
    model_name: str | None = None,
) -> Runnable[Any, Any]:
    """Create structured output compatible with either provider."""
    model = get_chat_model(temperature, model_name)
    if isinstance(model, ChatOpenAI):
        return model.with_structured_output(schema, method="function_calling")
    return model.with_structured_output(schema)
