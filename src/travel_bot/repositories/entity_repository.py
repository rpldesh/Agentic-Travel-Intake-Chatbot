from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from travel_bot.database.models import EntityChangeHistory, SessionEntity, utcnow
from travel_bot.schemas.entities import EntitySnapshot, EntityState
from travel_bot.schemas.enums import EntityStatus


class EntityRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_rows(self, session_id: str) -> list[SessionEntity]:
        result = await self.db.execute(
            select(SessionEntity).where(SessionEntity.session_id == session_id)
        )
        return list(result.scalars())

    async def snapshot(
        self, session_id: str, *, version: int, schema_version: str
    ) -> EntitySnapshot:
        rows = await self.list_rows(session_id)
        return EntitySnapshot(
            session_id=session_id,
            version=version,
            schema_version=schema_version,
            entities={
                r.entity_key: EntityState(
                    entity_key=r.entity_key,
                    value=r.value_json,
                    status=EntityStatus(r.status),
                    confidence=r.confidence,
                    evidence_text=r.evidence_text,
                    source_message_id=r.source_message_id,
                    user_confirmed=r.user_confirmed,
                    updated_at=r.updated_at,
                )
                for r in rows
            },
        )

    async def upsert(
        self,
        session_id: str,
        entity_key: str,
        *,
        value: Any,
        status: EntityStatus,
        confidence: float | None,
        evidence_text: str | None,
        source_message_id: str,
    ) -> None:
        result = await self.db.execute(
            select(SessionEntity).where(
                SessionEntity.session_id == session_id,
                SessionEntity.entity_key == entity_key,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            self.db.add(
                SessionEntity(
                    session_id=session_id,
                    entity_key=entity_key,
                    value_json=value,
                    status=status.value,
                    confidence=confidence,
                    evidence_text=evidence_text,
                    source_message_id=source_message_id,
                    user_confirmed=False,
                    row_version=1,
                )
            )
        else:
            row.value_json = value
            row.status = status.value
            row.confidence = confidence
            row.evidence_text = evidence_text
            row.source_message_id = source_message_id
            row.user_confirmed = False
            row.confirmed_at = None
            row.row_version = row.row_version + 1
        await self.db.flush()

    async def mark_confirmed(self, session_id: str) -> list[str]:
        rows = await self.list_rows(session_id)
        now = utcnow()
        confirmed = []
        for row in rows:
            if row.status != EntityStatus.MISSING.value and not row.user_confirmed:
                row.user_confirmed = True
                row.confirmed_at = now
                confirmed.append(row.entity_key)
        await self.db.flush()
        return confirmed

    async def add_history(
        self,
        session_id: str,
        entity_key: str,
        *,
        old_value: Any,
        new_value: Any,
        old_status: EntityStatus,
        new_status: EntityStatus,
        source_message_id: str,
        change_type: str,
        actor: str,
        evidence_text: str | None,
    ) -> None:
        self.db.add(
            EntityChangeHistory(
                session_id=session_id,
                entity_key=entity_key,
                old_value_json=old_value,
                new_value_json=new_value,
                old_status=old_status.value,
                new_status=new_status.value,
                source_message_id=source_message_id,
                change_type=change_type,
                actor=actor,
                evidence_text=evidence_text,
            )
        )
        await self.db.flush()

    async def history_exists_for_message(
        self, session_id: str, source_message_id: str, *, actor: str | None = None
    ) -> bool:
        stmt = select(EntityChangeHistory.id).where(
            EntityChangeHistory.session_id == session_id,
            EntityChangeHistory.source_message_id == source_message_id,
        )
        if actor:
            stmt = stmt.where(EntityChangeHistory.actor == actor)
        return (await self.db.execute(stmt.limit(1))).first() is not None

    async def list_history(
        self, session_id: str, entity_key: str | None = None
    ) -> list[EntityChangeHistory]:
        stmt = select(EntityChangeHistory).where(EntityChangeHistory.session_id == session_id)
        if entity_key:
            stmt = stmt.where(EntityChangeHistory.entity_key == entity_key)
        stmt = stmt.order_by(EntityChangeHistory.id)
        return list((await self.db.execute(stmt)).scalars())
