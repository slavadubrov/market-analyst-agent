"""Provider adapters. Run settings are immutable; credentials stay in the environment.

Use OpenAI or Anthropic directly, or point ``compatible`` at a LiteLLM gateway.
The gateway owns provider routing; this application preserves its model names.
"""

import os
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR, MODEL_MAP, MODEL_PROVIDER_ENV_VAR, OPENAI_MODEL_MAP


class ModelSettings(BaseModel):
    """Serializable settings saved with the run; API keys are never checkpointed."""

    model_config = ConfigDict(frozen=True)
    provider: Literal["openai", "anthropic", "compatible"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    timeout: float = Field(default=90, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    context_tokens: int = Field(default=16000, ge=1000)

    @classmethod
    def from_env(cls, model: str | None = None, provider: str | None = None):
        provider = provider or os.getenv(MODEL_PROVIDER_ENV_VAR, "auto").lower()
        if provider == "auto":
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
        if provider not in {"openai", "anthropic", "compatible"}:
            raise ValueError(f"Unsupported {MODEL_PROVIDER_ENV_VAR}: {provider}")
        name = model or os.getenv(MODEL_ENV_VAR)
        if not name:
            name = OPENAI_MODEL_MAP["haiku"] if provider == "openai" else DEFAULT_MODEL
        if provider == "openai":
            # Historical CLI aliases only; never rewrite an arbitrary deployment ID.
            aliases = {**OPENAI_MODEL_MAP, **{v: OPENAI_MODEL_MAP[k] for k, v in MODEL_MAP.items()}}
            name = aliases.get(name, name)
        elif provider == "anthropic":
            name = MODEL_MAP.get(name, name)
        base_url = os.getenv("MARKET_ANALYST_BASE_URL")
        if provider == "compatible" and not base_url:
            raise ValueError("compatible provider requires MARKET_ANALYST_BASE_URL")
        return cls.model_validate({"provider": provider, "model": name, "base_url": base_url})


def resolve_model(model_name: str | None = None, config: RunnableConfig | None = None) -> tuple[str, str]:
    settings = _settings(model_name, config)
    return settings.provider, settings.model


def _settings(model_name: str | None, config: RunnableConfig | None) -> ModelSettings:
    saved = (config or {}).get("configurable", {}).get("model_settings")
    settings = ModelSettings.model_validate(saved) if saved else ModelSettings.from_env(model_name)
    return settings.model_copy(update={"model": model_name}) if saved and model_name else settings


def get_chat_model(temperature: float = 0, model_name: str | None = None, *, config: RunnableConfig | None = None) -> BaseChatModel:
    settings = _settings(model_name, config)
    if settings.provider == "anthropic":
        return ChatAnthropic(
            model_name=settings.model, temperature=temperature, timeout=settings.timeout, max_tokens_to_sample=settings.max_tokens, max_retries=0, stop=None
        )
    options: dict[str, Any] = {
        "model": settings.model,
        "timeout": settings.timeout,
        "max_tokens": settings.max_tokens,
        "max_retries": 0,
        "use_responses_api": settings.provider == "openai",
    }
    if settings.provider == "openai":
        options["include"] = ["reasoning.encrypted_content"]
        options["store"] = False
    if settings.base_url:
        options["base_url"] = settings.base_url
    if settings.provider == "compatible":
        # A gateway key must be explicit: don't send an OpenAI key to another host.
        options["api_key"] = os.environ["MARKET_ANALYST_API_KEY"]
    return ChatOpenAI(**options)


def get_structured_model(
    schema: type[BaseModel],
    temperature: float = 0,
    model_name: str | None = None,
    *,
    config: RunnableConfig | None = None,
) -> Runnable[Any, Any]:
    model = get_chat_model(temperature, model_name, config=config)
    return model.with_structured_output(schema, method="function_calling") if isinstance(model, ChatOpenAI) else model.with_structured_output(schema)
