"""Application settings, loaded from environment variables / `.env`.

Model selection and secrets live here (trusted configuration), never in prompts
or under end-user control.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AGENT_ROLES = (
    "supervisor",
    "small_talk",
    "travel_entity",
    "question_generator",
    "guardrail_classifier",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Persistence -------------------------------------------------------
    # Application data (SQLAlchemy async URL). Production: postgresql+asyncpg://...
    database_url: str = "sqlite+aiosqlite:///./travel_bot.db"
    # LangGraph checkpointer. Kept in its own SQLite file by default so the
    # checkpointer and the application tables never contend for one write lock.
    checkpoint_backend: str = "sqlite"  # "sqlite" | "memory"
    checkpoint_db_path: str = "./travel_bot_checkpoints.db"
    # Dev convenience: create tables on startup. Use Alembic migrations otherwise.
    db_auto_create: bool = True

    # --- Provider credentials (never logged; read from env / secret manager) --
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # --- Models (provider-qualified IDs, e.g. openai:... / google_genai:...) --
    supervisor_model: str = "openai:gpt-5.5"
    small_talk_model: str = "openai:gpt-5.5"
    travel_entity_model: str = "openai:gpt-5.5"
    question_generator_model: str = "openai:gpt-5.5"
    guardrail_classifier_model: str = "openai:gpt-5.5"

    # Optional fallbacks: only configure after the fallback passes the same evals.
    supervisor_fallback_model: str | None = None
    small_talk_fallback_model: str | None = None
    travel_entity_fallback_model: str | None = None
    question_generator_fallback_model: str | None = None
    guardrail_classifier_fallback_model: str | None = None

    # Per-role temperature. Left unset by default because some models only
    # accept their default temperature.
    supervisor_temperature: float | None = None
    small_talk_temperature: float | None = None
    travel_entity_temperature: float | None = None
    question_generator_temperature: float | None = None
    guardrail_classifier_temperature: float | None = None

    model_timeout_seconds: float = 30.0
    model_max_retries: int = 2
    structured_output_retries: int = 1

    # --- Conversation / intake policy ---------------------------------------
    reference_timezone: str = "UTC"
    supported_languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["en", "es", "fr", "de", "it", "pt"]
    )
    default_language: str = "en"
    history_window_messages: int = 8
    ask_recommended_fields: bool = True
    min_confidence_known: float = 0.6
    max_party_size: int = 20
    session_ttl_hours: int = 72

    # --- Guardrails / limits --------------------------------------------------
    max_message_chars: int = 2000
    rate_limit_per_minute: int = 30
    guardrail_llm_enabled: bool = False

    @field_validator("supported_languages", mode="before")
    @classmethod
    def _split_languages(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip().lower() for v in value.split(",") if v.strip()]
        return value

    def model_id_for(self, role: str) -> str:
        return getattr(self, f"{role}_model")

    def fallback_model_id_for(self, role: str) -> str | None:
        return getattr(self, f"{role}_fallback_model") or None

    def temperature_for(self, role: str) -> float | None:
        return getattr(self, f"{role}_temperature")


@lru_cache
def get_settings() -> Settings:
    # Also export .env to the process so LangChain/LangSmith see e.g. LANGSMITH_*.
    load_dotenv(override=False)
    return Settings()
