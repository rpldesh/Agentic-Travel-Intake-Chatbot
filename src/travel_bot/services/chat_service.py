"""Turn handling: authenticate, validate, deduplicate, persist, run the graph."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from travel_bot.config import Settings
from travel_bot.database.connection import Database
from travel_bot.entities.catalog import EntityCatalog
from travel_bot.entities.rendering import render_state
from travel_bot.entities.requirements import compute_missing
from travel_bot.errors import RateLimitedError, RequestValidationError
from travel_bot.graph.progress import NODE_LABELS
from travel_bot.graph.state import TurnContext, turn_input
from travel_bot.graph.workflow import user_message
from travel_bot.guardrails.input_guardrails import clean_text, redact_sensitive
from travel_bot.repositories.message_repository import MessageRepository
from travel_bot.repositories.session_repository import SessionRepository
from travel_bot.schemas.api import (
    ChatResponse,
    CreateSessionResponse,
    EntitiesResponse,
    EntityView,
    MessageView,
)
from travel_bot.schemas.enums import ResponseType
from travel_bot.services.entity_service import EntityService

logger = logging.getLogger(__name__)

TECHNICAL_ERROR_REPLY = ("Sorry, something went wrong on my side. "
                         "Could you send that again?")


class RateLimiter:
    """In-process sliding window per session. Use a shared store (e.g. Redis) when scaling out."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            raise RateLimitedError(key)
        hits.append(now)


@dataclass(frozen=True)
class PreparedTurn:
    session_id: str
    message_id: str
    text: str  # redacted
    user_id: str | None = None


@dataclass
class ChatService:
    settings: Settings
    db: Database
    catalog: EntityCatalog
    entity_service: EntityService
    graph: object  # compiled LangGraph
    rate_limiter: RateLimiter

    async def create_session(
        self, *, user_id: str | None, preferred_language: str | None = None
    ) -> CreateSessionResponse:
        lang = preferred_language if preferred_language in self.settings.supported_languages \
            else self.settings.default_language
        async with self.db.session() as s:
            row, token = await SessionRepository(s).create(
                user_id=user_id, preferred_language=lang,
                schema_version=self.catalog.schema_version,
            )
        return CreateSessionResponse(session_id=row.session_id, session_token=token,
                                     status=row.status)

    async def authorize(self, session_id: str, token: str | None, user_id: str | None = None) -> None:
        async with self.db.session() as s:
            await SessionRepository(s).authorize(
                session_id, token, user_id=user_id, ttl_hours=self.settings.session_ttl_hours
            )

    async def handle_turn(
        self,
        *,
        session_id: str,
        session_token: str | None,
        message_id: str,
        content: str,
        user_id: str | None = None,
    ) -> ChatResponse:
        prepared = await self.prepare_turn(session_id=session_id, session_token=session_token,
                                           message_id=message_id, content=content, user_id=user_id)
        if isinstance(prepared, ChatResponse):
            return prepared
        final: ChatResponse | None = None
        async for event in self.run_turn(prepared):
            if event["type"] == "final":
                final = ChatResponse.model_validate(event["response"])
        assert final is not None
        return final

    async def prepare_turn(
        self,
        *,
        session_id: str,
        session_token: str | None,
        message_id: str,
        content: str,
        user_id: str | None = None,
    ) -> PreparedTurn | ChatResponse:
        """Validate, authorize, deduplicate and persist the user message.

        Raises before any streaming starts, so HTTP errors keep proper status codes.
        Returns the original reply for a duplicate delivery.
        """
        # 1. Request validation (hard limits before anything else).
        text = clean_text(content or "")
        if not text:
            raise RequestValidationError("empty_message", "Message is empty.")
        if len(text) > self.settings.max_message_chars:
            raise RequestValidationError(
                "message_too_long", f"Message exceeds {self.settings.max_message_chars} characters."
            )

        # 2. Session ownership. The model never chooses the session.
        await self.authorize(session_id, session_token, user_id)

        # 3. Idempotency: a retried delivery returns the original reply.
        async with self.db.session() as s:
            prior = await MessageRepository(s).get_reply_for(session_id, message_id)
        if prior is not None:
            return ChatResponse(
                message_id=message_id, reply=prior.content,
                response_type=prior.meta.get("response_type", ResponseType.ASK_QUESTION.value),
                target_entity_keys=prior.meta.get("target_entity_keys", []),
                session_status=await self._status(session_id), duplicate=True,
                trace_id=prior.meta.get("trace_id"),
            )

        self.rate_limiter.check(session_id)

        # 4. Persist the incoming message (sensitive data redacted, never stored raw).
        redacted, redactions = redact_sensitive(text)
        async with self.db.session() as s:
            await MessageRepository(s).add_user_message(
                session_id, message_id, redacted,
                meta={"redactions": redactions} if redactions else None,
            )
        return PreparedTurn(session_id=session_id, message_id=message_id,
                            text=redacted, user_id=user_id)

    async def run_turn(self, turn: PreparedTurn) -> AsyncIterator[dict]:
        """Run the graph, yielding progress events and finally {"type": "final", ...}.

        Progress events: step (agent/node start + done with ms), tool, model_call,
        decision. `t` is milliseconds since the turn started.
        """
        trace_id = uuid.uuid4().hex
        ctx = TurnContext(session_id=turn.session_id, message_id=turn.message_id,
                          trace_id=trace_id, user_id=turn.user_id)
        config = {"configurable": {"thread_id": turn.session_id}, "recursion_limit": 25}
        t0 = time.perf_counter()
        started: dict[str, float] = {}
        state: dict = {}

        def elapsed() -> int:
            return int((time.perf_counter() - t0) * 1000)

        yield {"type": "turn_started", "trace_id": trace_id, "t": 0}
        try:
            async for mode, data in self.graph.astream(
                turn_input(message=user_message(turn.text, turn.message_id), ctx=ctx, text=turn.text),
                config, context=ctx, stream_mode=["tasks", "custom", "values"],
            ):
                if mode == "values":
                    state = data
                elif mode == "tasks":
                    agent, label = NODE_LABELS.get(data["name"], (data["name"], ""))
                    base = {"type": "step", "node": data["name"], "agent": agent, "label": label}
                    if "result" in data:  # finished
                        ms = int((time.perf_counter() - started.pop(data["id"], t0)) * 1000)
                        status = "failed" if data.get("error") else "done"
                        yield {**base, "status": status, "ms": ms, "t": elapsed()}
                    else:
                        started[data["id"]] = time.perf_counter()
                        yield {**base, "status": "started", "t": elapsed()}
                elif mode == "custom":
                    yield {**data, "t": elapsed()}
        except Exception:
            # Nothing is claimed as saved; the same message_id can be retried safely.
            logger.exception("graph execution failed", extra={"session_id": turn.session_id,
                                                               "trace_id": trace_id})
            response = ChatResponse(message_id=turn.message_id, reply=TECHNICAL_ERROR_REPLY,
                                    response_type=ResponseType.CLARIFICATION.value,
                                    session_status=await self._status(turn.session_id),
                                    trace_id=trace_id)
            yield {"type": "final", "response": response.model_dump(mode="json"), "t": elapsed()}
            return

        response = state.get("assistant_response") or {}
        result = ChatResponse(
            message_id=turn.message_id,
            reply=response.get("text", ""),
            response_type=response.get("response_type", ResponseType.ASK_QUESTION.value),
            target_entity_keys=response.get("target_entity_keys", []),
            session_status=state.get("session_status") or await self._status(turn.session_id),
            trace_id=trace_id,
        )
        yield {"type": "final", "response": result.model_dump(mode="json"), "t": elapsed()}

    async def entities(self, session_id: str) -> EntitiesResponse:
        snapshot = await self.entity_service.get_snapshot(session_id)
        missing = compute_missing(self.catalog, snapshot)
        views = []
        for defn in self.catalog.entities:
            state = snapshot.get(defn.key)
            views.append(EntityView(
                entity_key=defn.key, label=defn.label, status=state.status.value,
                value=state.value, user_confirmed=state.user_confirmed,
            ))
        return EntitiesResponse(
            session_id=session_id, session_status=await self._status(session_id),
            version=snapshot.version, entities=views,
            missing_required=missing.missing_required, ambiguous=missing.ambiguous,
            required_complete=missing.required_complete,
        )

    async def entity_summary(self, session_id: str) -> list[str]:
        snapshot = await self.entity_service.get_snapshot(session_id)
        return [f"{d.label}: {render_state(d, snapshot.get(d.key))}"
                for d in self.catalog.entities if snapshot.status_of(d.key).value != "missing"]

    async def messages(self, session_id: str) -> list[MessageView]:
        async with self.db.session() as s:
            rows = await MessageRepository(s).list_messages(session_id)
        return [MessageView(message_id=r.message_id, role=r.role, content=r.content,
                            created_at=r.created_at) for r in rows]

    async def _status(self, session_id: str) -> str:
        async with self.db.session() as s:
            row = await SessionRepository(s).get(session_id)
        return row.status if row else "unknown"
