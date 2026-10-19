"""Entity contracts: what the extractor proposes and what the database holds."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from travel_bot.schemas.enums import ChangeActor, ChangeIntent, EntityStatus


# ---------------------------------------------------------------------------
# TravelEntityAgent output (LLM structured output).
#
# `value_json` is a JSON-encoded string rather than a free-form object: strict
# structured-output modes (OpenAI and Gemini) do not reliably accept open
# dictionaries, and the application validates/normalizes the value anyway.
# ---------------------------------------------------------------------------
class EntityCandidate(BaseModel):
    entity_key: str = Field(description="One of the catalog entity keys.")
    value_json: str | None = Field(
        default=None,
        description="The value encoded as a JSON string, using the format given "
        "for this entity. Null when status is declined/not_applicable or no value.",
    )
    status: EntityStatus = Field(
        description="known | flexible | ambiguous | not_applicable | declined"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = Field(
        description="Exact quote from the CURRENT user message supporting this update."
    )
    change_intent: ChangeIntent = ChangeIntent.SET


class AmbiguityNote(BaseModel):
    entity_key: str
    description: str = Field(description="What is unclear, in a few words.")


class EntityExtraction(BaseModel):
    updates: list[EntityCandidate] = Field(default_factory=list)
    ambiguities: list[AmbiguityNote] = Field(default_factory=list)
    user_explicitly_declined: list[str] = Field(default_factory=list)
    confirms_summary: bool = Field(
        default=False,
        description="True only if the assistant just showed a trip summary and the "
        "user clearly confirms it is correct, with no corrections.",
    )


# ---------------------------------------------------------------------------
# Canonical state (from the database).
# ---------------------------------------------------------------------------
class EntityState(BaseModel):
    entity_key: str
    value: Any | None = None
    status: EntityStatus = EntityStatus.MISSING
    confidence: float | None = None
    evidence_text: str | None = None
    source_message_id: str | None = None
    user_confirmed: bool = False
    updated_at: datetime | None = None


class EntitySnapshot(BaseModel):
    session_id: str
    version: int
    schema_version: str
    entities: dict[str, EntityState] = Field(default_factory=dict)

    def get(self, key: str) -> EntityState:
        return self.entities.get(key) or EntityState(entity_key=key)

    def status_of(self, key: str) -> EntityStatus:
        return self.get(key).status

    def value_of(self, key: str) -> Any | None:
        return self.get(key).value

    def known_items(self) -> dict[str, Any]:
        return {
            k: {"status": e.status.value, "value": e.value}
            for k, e in self.entities.items()
            if e.status != EntityStatus.MISSING
        }


class MissingReport(BaseModel):
    missing_required: list[str] = Field(default_factory=list)
    missing_recommended: list[str] = Field(default_factory=list)
    ambiguous: list[str] = Field(default_factory=list)
    declined: list[str] = Field(default_factory=list)
    required_complete: bool = False


# ---------------------------------------------------------------------------
# Result of applying a candidate batch.
# ---------------------------------------------------------------------------
class AcceptedUpdate(BaseModel):
    entity_key: str
    old_value: Any | None = None
    new_value: Any | None = None
    old_status: EntityStatus = EntityStatus.MISSING
    new_status: EntityStatus
    change_type: str
    actor: ChangeActor = ChangeActor.USER_INPUT
    confidence: float | None = None
    evidence_text: str | None = None


class RejectedUpdate(BaseModel):
    entity_key: str
    reason_code: str
    detail: str = ""


class EntityUpdateResult(BaseModel):
    accepted: list[AcceptedUpdate] = Field(default_factory=list)
    rejected: list[RejectedUpdate] = Field(default_factory=list)
    already_applied: bool = False
    conflict_retries: int = 0
    new_version: int | None = None
