from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from travel_bot.database.models import ConversationMessage
from travel_bot.schemas.enums import MessageRole


def assistant_message_id(user_message_id: str) -> str:
    return f"{user_message_id}:assistant"


class MessageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, session_id: str, message_id: str) -> ConversationMessage | None:
        result = await self.db.execute(
            select(ConversationMessage).where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.message_id == message_id,
            )
        )
        return result.scalar_one_or_none()

    async def add_user_message(
        self,
        session_id: str,
        message_id: str,
        content: str,
        *,
        guardrail_status: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[ConversationMessage, bool]:
        """Insert a user message. Returns (row, created); idempotent on message_id."""
        existing = await self.get(session_id, message_id)
        if existing is not None:
            return existing, False
        row = ConversationMessage(
            session_id=session_id,
            message_id=message_id,
            role=MessageRole.USER.value,
            content=content,
            guardrail_status=guardrail_status,
            meta=meta or {},
        )
        self.db.add(row)
        await self.db.flush()
        return row, True

    async def get_reply_for(
        self, session_id: str, user_message_id: str
    ) -> ConversationMessage | None:
        result = await self.db.execute(
            select(ConversationMessage).where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.reply_to_message_id == user_message_id,
                ConversationMessage.role == MessageRole.ASSISTANT.value,
            )
        )
        return result.scalar_one_or_none()

    async def add_assistant_message(
        self,
        session_id: str,
        reply_to_message_id: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        existing = await self.get_reply_for(session_id, reply_to_message_id)
        if existing is not None:
            return existing
        row = ConversationMessage(
            session_id=session_id,
            message_id=assistant_message_id(reply_to_message_id),
            role=MessageRole.ASSISTANT.value,
            content=content,
            reply_to_message_id=reply_to_message_id,
            meta=meta or {},
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_messages(
        self, session_id: str, *, limit: int | None = None
    ) -> list[ConversationMessage]:
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = list((await self.db.execute(stmt)).scalars())
        rows.reverse()
        return rows
