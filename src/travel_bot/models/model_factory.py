"""Creates LangChain chat models for each agent role (GPT or Gemini).

Agents receive an injected `RoleModel`; no agent instantiates a provider model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from travel_bot.config import AGENT_ROLES, Settings
from travel_bot.models.model_config import (
    SUPPORTED_PROVIDERS,
    ModelSpec,
    RoleModelConfig,
    check_credentials,
    load_role_configs,
)


@dataclass
class RoleModel:
    role: str
    model_id: str
    model: BaseChatModel
    fallback_id: str | None = None
    fallback: BaseChatModel | None = None


def build_chat_model(
    spec: ModelSpec, cfg: RoleModelConfig, settings: Settings
) -> BaseChatModel:
    kwargs: dict[str, Any] = {"timeout": cfg.timeout, "max_retries": cfg.max_retries}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    secret = getattr(settings, SUPPORTED_PROVIDERS[spec.provider])
    if secret is not None:
        kwargs["api_key"] = secret.get_secret_value()
    # The explicit provider prefix selects the integration (no implicit inference).
    return init_chat_model(spec.name, model_provider=spec.provider, **kwargs)


def build_role_models(
    settings: Settings, roles: tuple[str, ...] = AGENT_ROLES, *, require_credentials: bool = True
) -> dict[str, RoleModel]:
    configs = load_role_configs(settings, roles)
    if require_credentials:
        check_credentials(settings, configs)
    models: dict[str, RoleModel] = {}
    for role, cfg in configs.items():
        models[role] = RoleModel(
            role=role,
            model_id=cfg.primary.model_id,
            model=build_chat_model(cfg.primary, cfg, settings),
            fallback_id=cfg.fallback.model_id if cfg.fallback else None,
            fallback=build_chat_model(cfg.fallback, cfg, settings) if cfg.fallback else None,
        )
    return models
