"""Composition root: wires settings, persistence, models, agents and the graph."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from travel_bot.agents.question_generator import QuestionGenerator
from travel_bot.agents.small_talk import SmallTalkAgent
from travel_bot.agents.supervisor import Supervisor
from travel_bot.agents.travel_entity import TravelEntityAgent
from travel_bot.config import Settings, get_settings
from travel_bot.database.connection import Database
from travel_bot.entities.catalog import load_catalog
from travel_bot.graph.checkpointer import open_checkpointer
from travel_bot.graph.workflow import GraphDeps, build_graph
from travel_bot.guardrails.input_guardrails import InputGuardrail
from travel_bot.logging_setup import configure_logging
from travel_bot.models.model_factory import RoleModel, build_role_models
from travel_bot.services.chat_service import ChatService, RateLimiter
from travel_bot.services.entity_service import EntityService


def build_deps(settings: Settings, db: Database, models: dict[str, RoleModel]) -> GraphDeps:
    catalog = load_catalog()
    retries = settings.structured_output_retries
    entity_service = EntityService(
        db, catalog, timezone=settings.reference_timezone,
        max_party_size=settings.max_party_size,
        min_confidence_known=settings.min_confidence_known,
    )
    return GraphDeps(
        db=db,
        catalog=catalog,
        entity_service=entity_service,
        guardrail=InputGuardrail(
            supported_languages=settings.supported_languages,
            default_language=settings.default_language,
            max_chars=settings.max_message_chars,
            classifier=models.get("guardrail_classifier") if settings.guardrail_llm_enabled else None,
        ),
        supervisor=Supervisor(models["supervisor"], retries=retries),
        small_talk=SmallTalkAgent(models["small_talk"], retries=retries),
        travel_entity=TravelEntityAgent(models["travel_entity"], catalog,
                                        history_window=settings.history_window_messages,
                                        retries=retries),
        question_generator=QuestionGenerator(models["question_generator"], catalog, retries=retries),
        supported_languages=settings.supported_languages,
        timezone=settings.reference_timezone,
        history_window=settings.history_window_messages,
        ask_recommended=settings.ask_recommended_fields,
    )


@asynccontextmanager
async def build_chat_service(
    settings: Settings | None = None,
    *,
    models: dict[str, RoleModel] | None = None,
) -> AsyncIterator[ChatService]:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    if models is None:
        roles = ("supervisor", "small_talk", "travel_entity", "question_generator")
        if settings.guardrail_llm_enabled:
            roles += ("guardrail_classifier",)
        models = build_role_models(settings, roles)

    db = Database(settings.database_url)
    if settings.db_auto_create:
        await db.create_all()
    try:
        async with open_checkpointer(settings.checkpoint_backend, settings.checkpoint_db_path) as cp:
            deps = build_deps(settings, db, models)
            yield ChatService(
                settings=settings, db=db, catalog=deps.catalog,
                entity_service=deps.entity_service,
                graph=build_graph(deps, checkpointer=cp),
                rate_limiter=RateLimiter(settings.rate_limit_per_minute),
            )
    finally:
        await db.dispose()
