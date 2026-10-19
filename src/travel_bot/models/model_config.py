"""Reads and validates role -> model settings."""

from __future__ import annotations

from dataclasses import dataclass

from travel_bot.config import AGENT_ROLES, Settings
from travel_bot.errors import ModelConfigurationError

# Provider prefix -> settings attribute holding its API key.
SUPPORTED_PROVIDERS = {
    "openai": "openai_api_key",
    "google_genai": "google_api_key",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str  # provider-qualified, e.g. "openai:gpt-5.5"

    @property
    def provider(self) -> str:
        return self.model_id.split(":", 1)[0]

    @property
    def name(self) -> str:
        return self.model_id.split(":", 1)[1]


@dataclass(frozen=True)
class RoleModelConfig:
    role: str
    primary: ModelSpec
    fallback: ModelSpec | None
    temperature: float | None
    timeout: float
    max_retries: int


def parse_model_id(model_id: str, *, setting: str) -> ModelSpec:
    if ":" not in model_id:
        raise ModelConfigurationError(
            f"{setting}={model_id!r} must be provider-qualified, e.g. 'openai:<model>' "
            "or 'google_genai:<model>'"
        )
    spec = ModelSpec(model_id.strip())
    if spec.provider not in SUPPORTED_PROVIDERS:
        raise ModelConfigurationError(
            f"{setting}: unsupported provider {spec.provider!r}; "
            f"expected one of {sorted(SUPPORTED_PROVIDERS)}"
        )
    if not spec.name:
        raise ModelConfigurationError(f"{setting}: model name is empty")
    return spec


def load_role_configs(settings: Settings, roles: tuple[str, ...] = AGENT_ROLES) -> dict[str, RoleModelConfig]:
    configs: dict[str, RoleModelConfig] = {}
    for role in roles:
        primary = parse_model_id(settings.model_id_for(role), setting=f"{role.upper()}_MODEL")
        fb_id = settings.fallback_model_id_for(role)
        fallback = parse_model_id(fb_id, setting=f"{role.upper()}_FALLBACK_MODEL") if fb_id else None
        configs[role] = RoleModelConfig(
            role=role,
            primary=primary,
            fallback=fallback,
            temperature=settings.temperature_for(role),
            timeout=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )
    return configs


def check_credentials(settings: Settings, configs: dict[str, RoleModelConfig]) -> None:
    missing: set[str] = set()
    for cfg in configs.values():
        for spec in filter(None, (cfg.primary, cfg.fallback)):
            key_attr = SUPPORTED_PROVIDERS[spec.provider]
            if getattr(settings, key_attr) is None:
                missing.add(key_attr.upper())
    if missing:
        raise ModelConfigurationError(
            "Missing API key(s) for configured models: " + ", ".join(sorted(missing))
            + ". Set them in .env (see .env.example) or switch the role's model."
        )
