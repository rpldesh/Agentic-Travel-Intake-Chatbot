"""Application tables.

Portable across SQLite (dev/test) and PostgreSQL (production): generic `JSON`,
UUID-compatible string keys, timezone-aware UTC timestamps. LangGraph's
checkpoint tables are owned by the checkpointer and are not modeled here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TravelSession(Base):
    __tablename__ = "travel_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    session_token_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="active")
    preferred_language: Mapped[str] = mapped_column(String(16), default="en")
    entity_schema_version: Mapped[str] = mapped_column(String(16), default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Optimistic concurrency for entity batches (bumped on every accepted batch).
    row_version: Mapped[int] = mapped_column(Integer, default=1)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint("session_id", "message_id", name="uq_message_per_session"),
        Index("ix_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(160))
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("travel_sessions.session_id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # For assistant messages: the user message this replies to (idempotency lookup).
    reply_to_message_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    guardrail_status: Mapped[str | None] = mapped_column(String(48), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SessionEntity(Base):
    __tablename__ = "session_entities"
    __table_args__ = (
        UniqueConstraint("session_id", "entity_key", name="uq_entity_per_session"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("travel_sessions.session_id", ondelete="CASCADE")
    )
    entity_key: Mapped[str] = mapped_column(String(64))
    value_json: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(24))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_message_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, default=1)


class EntityChangeHistory(Base):
    """Immutable audit trail. Stores concise evidence only, never model reasoning."""

    __tablename__ = "entity_change_history"
    __table_args__ = (
        # One change per entity per source message per actor: makes retries idempotent.
        UniqueConstraint(
            "session_id", "entity_key", "source_message_id", "actor",
            name="uq_history_idempotency",
        ),
        Index("ix_history_session_created", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("travel_sessions.session_id", ondelete="CASCADE")
    )
    entity_key: Mapped[str] = mapped_column(String(64))
    old_value_json: Mapped[Any] = mapped_column(JSON, nullable=True)
    new_value_json: Mapped[Any] = mapped_column(JSON, nullable=True)
    old_status: Mapped[str] = mapped_column(String(24))
    new_status: Mapped[str] = mapped_column(String(24))
    source_message_id: Mapped[str] = mapped_column(String(160))
    change_type: Mapped[str] = mapped_column(String(24))
    actor: Mapped[str] = mapped_column(String(32))
    evidence_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
