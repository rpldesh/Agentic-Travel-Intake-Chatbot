from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    preferred_language: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    session_token: str = Field(
        description="Secret bearer token for this session. Send as X-Session-Token."
    )
    status: str


class ChatRequest(BaseModel):
    message_id: str = Field(
        min_length=1,
        max_length=128,
        description="Client-generated unique ID per turn; doubles as idempotency key.",
    )
    content: str


class ChatResponse(BaseModel):
    message_id: str
    reply: str
    response_type: str
    target_entity_keys: list[str] = Field(default_factory=list)
    session_status: str
    duplicate: bool = False
    trace_id: str | None = None


class EntityView(BaseModel):
    entity_key: str
    label: str
    status: str
    value: Any | None = None
    user_confirmed: bool = False


class EntitiesResponse(BaseModel):
    session_id: str
    session_status: str
    version: int
    entities: list[EntityView]
    missing_required: list[str]
    ambiguous: list[str]
    required_complete: bool


class MessageView(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
