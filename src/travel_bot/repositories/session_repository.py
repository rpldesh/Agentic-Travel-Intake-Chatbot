from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from travel_bot.database.models import TravelSession, utcnow
from travel_bot.errors import (
    SessionExpiredError,
    SessionNotFoundError,
    SessionUnauthorizedError,
)
from travel_bot.schemas.enums import SessionStatus


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes even for timezone-aware columns.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class SessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        user_id: str | None,
        tenant_id: str = "default",
        preferred_language: str = "en",
        schema_version: str = "v1",
    ) -> tuple[TravelSession, str]:
        """Create a session with a fresh, unguessable ID and bearer token."""
        token = secrets.token_urlsafe(32)
        row = TravelSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            session_token_hash=hash_token(token),
            status=SessionStatus.ACTIVE.value,
            preferred_language=preferred_language,
            entity_schema_version=schema_version,
            row_version=1,
        )
        self.db.add(row)
        await self.db.flush()
        return row, token

    async def get(self, session_id: str) -> TravelSession | None:
        return await self.db.get(TravelSession, session_id)

    async def authorize(
        self,
        session_id: str,
        token: str | None,
        *,
        user_id: str | None = None,
        ttl_hours: int | None = None,
    ) -> TravelSession:
        row = await self.get(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        if not token or not hmac.compare_digest(row.session_token_hash, hash_token(token)):
            raise SessionUnauthorizedError(session_id)
        if row.user_id is not None and user_id is not None and row.user_id != user_id:
            raise SessionUnauthorizedError(session_id)
        if row.status == SessionStatus.EXPIRED.value:
            raise SessionExpiredError(session_id)
        if ttl_hours is not None:
            idle = datetime.now(UTC) - _as_utc(row.last_activity_at)
            if idle > timedelta(hours=ttl_hours):
                row.status = SessionStatus.EXPIRED.value
                await self.db.flush()
                raise SessionExpiredError(session_id)
        return row

    async def get_version(self, session_id: str) -> int:
        result = await self.db.execute(
            select(TravelSession.row_version).where(TravelSession.session_id == session_id)
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise SessionNotFoundError(session_id)
        return version

    async def bump_version(self, session_id: str, expected_version: int) -> bool:
        """Compare-and-swap the session row version. False on conflict."""
        result = await self.db.execute(
            update(TravelSession)
            .where(
                TravelSession.session_id == session_id,
                TravelSession.row_version == expected_version,
            )
            .values(row_version=expected_version + 1, updated_at=utcnow())
        )
        return result.rowcount == 1

    async def set_status(self, session_id: str, status: SessionStatus) -> None:
        await self.db.execute(
            update(TravelSession)
            .where(TravelSession.session_id == session_id)
            .values(status=status.value, updated_at=utcnow())
        )

    async def touch(self, session_id: str, *, language: str | None = None) -> None:
        values: dict = {"last_activity_at": utcnow()}
        if language:
            values["preferred_language"] = language
        await self.db.execute(
            update(TravelSession).where(TravelSession.session_id == session_id).values(**values)
        )
