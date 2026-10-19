"""Graph state and trusted per-turn runtime context.

State values are plain JSON-compatible data (plus LangChain messages) so the
checkpointer can run with strict deserialization. `entity_snapshot` is only a
working copy: the database is the source of truth and is reloaded every turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


@dataclass(frozen=True)
class TurnContext:
    """Supplied by the application from the authenticated request; never by the model."""

    session_id: str
    message_id: str
    trace_id: str
    user_id: str | None = None
    tenant_id: str = "default"


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # Trusted identifiers (mirrors TurnContext for traceability).
    session_id: str
    user_id: str | None
    tenant_id: str
    current_message_id: str
    idempotency_key: str
    trace_id: str

    # Per-turn working data.
    user_input: str
    guardrail_result: dict[str, Any] | None
    detected_language: str
    route: dict[str, Any] | None
    entity_snapshot: dict[str, Any] | None
    entity_candidates: dict[str, Any] | None
    accepted_entity_updates: list[dict[str, Any]]
    rejected_entity_updates: list[dict[str, Any]]
    missing_required_entities: list[str]
    missing_recommended_entities: list[str]
    ambiguous_entities: list[str]
    required_complete: bool
    persistence_error: bool
    small_talk_text: str
    assistant_response: dict[str, Any] | None
    call_log: list[dict[str, Any]]
    agent_calls: int

    # Carried across turns.
    session_status: str
    last_question_target_keys: list[str]
    last_response_type: str | None
    question_history: list[dict[str, Any]]
    offered_keys: list[str]


def turn_input(
    *,
    message: AnyMessage,
    ctx: TurnContext,
    text: str,
) -> TravelState:
    """Initial values for a new turn: resets per-turn fields, keeps cross-turn ones."""
    return TravelState(
        messages=[message],
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        tenant_id=ctx.tenant_id,
        current_message_id=ctx.message_id,
        idempotency_key=ctx.message_id,
        trace_id=ctx.trace_id,
        user_input=text,
        guardrail_result=None,
        route=None,
        entity_candidates=None,
        accepted_entity_updates=[],
        rejected_entity_updates=[],
        persistence_error=False,
        small_talk_text="",
        assistant_response=None,
        call_log=[],
        agent_calls=0,
    )
