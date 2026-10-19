"""Entity application service: the only path by which entity state changes.

Validates candidates, resolves them against the canonical snapshot, and persists
the accepted batch + audit history in one transaction, idempotently and with an
optimistic version check.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError, OperationalError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from travel_bot.database.connection import Database
from travel_bot.entities.catalog import EntityCatalog
from travel_bot.entities.normalizers import NormalizationContext
from travel_bot.entities.requirements import compute_missing
from travel_bot.entities.resolution import reconcile, resolve_candidates
from travel_bot.errors import ConcurrencyConflictError, PersistenceError
from travel_bot.repositories.entity_repository import EntityRepository
from travel_bot.repositories.session_repository import SessionRepository
from travel_bot.schemas.entities import (
    AcceptedUpdate,
    EntityCandidate,
    EntitySnapshot,
    EntityUpdateResult,
    MissingReport,
)
from travel_bot.schemas.enums import ChangeActor, EntityStatus, SessionStatus

logger = logging.getLogger(__name__)

_MAX_CONFLICT_RETRIES = 2


class EntityService:
    def __init__(
        self,
        db: Database,
        catalog: EntityCatalog,
        *,
        timezone: str = "UTC",
        max_party_size: int = 20,
        min_confidence_known: float = 0.6,
        write_attempts: int = 3,
    ) -> None:
        self.db = db
        self.catalog = catalog
        self.timezone = timezone
        self.max_party_size = max_party_size
        self.min_confidence_known = min_confidence_known
        self.write_attempts = write_attempts

    def reference_date(self) -> date:
        return datetime.now(ZoneInfo(self.timezone)).date()

    def normalization_context(self) -> NormalizationContext:
        return NormalizationContext(
            reference_date=self.reference_date(), max_party_size=self.max_party_size
        )

    # ------------------------------------------------------------------ reads
    async def get_snapshot(self, session_id: str) -> EntitySnapshot:
        async with self.db.session() as s:
            version = await SessionRepository(s).get_version(session_id)
            return await EntityRepository(s).snapshot(
                session_id, version=version, schema_version=self.catalog.schema_version
            )

    async def get_missing(self, session_id: str) -> MissingReport:
        return compute_missing(self.catalog, await self.get_snapshot(session_id))

    # ----------------------------------------------------------------- writes
    async def record_updates(
        self,
        session_id: str,
        candidates: list[EntityCandidate],
        *,
        source_message_id: str,
        current_message_text: str,
        expected_version: int | None = None,
    ) -> EntityUpdateResult:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.write_attempts),
                wait=wait_exponential(multiplier=0.2, max=2),
                retry=retry_if_exception_type(OperationalError),
                reraise=True,
            ):
                with attempt:
                    return await self._record_with_conflict_retry(
                        session_id, candidates, source_message_id,
                        current_message_text, expected_version,
                    )
        except OperationalError as exc:
            logger.error("entity write failed after retries", extra={"session_id": session_id})
            raise PersistenceError(str(exc)) from exc
        raise AssertionError("unreachable")

    async def _record_with_conflict_retry(
        self,
        session_id: str,
        candidates: list[EntityCandidate],
        source_message_id: str,
        current_message_text: str,
        expected_version: int | None,
    ) -> EntityUpdateResult:
        conflicts = 0
        while True:
            try:
                result = await self._record_once(
                    session_id, candidates, source_message_id,
                    current_message_text, expected_version,
                )
                result.conflict_retries = conflicts
                return result
            except (ConcurrencyConflictError, IntegrityError):
                conflicts += 1
                if conflicts > _MAX_CONFLICT_RETRIES:
                    raise PersistenceError("concurrency conflict persisted") from None
                logger.warning(
                    "entity version conflict; reloading canonical state",
                    extra={"session_id": session_id, "attempt": conflicts},
                )
                expected_version = None  # re-resolve against the fresh snapshot

    async def _record_once(
        self,
        session_id: str,
        candidates: list[EntityCandidate],
        source_message_id: str,
        current_message_text: str,
        expected_version: int | None,
    ) -> EntityUpdateResult:
        async with self.db.session() as s:
            sessions = SessionRepository(s)
            entities = EntityRepository(s)

            if await entities.history_exists_for_message(
                session_id, source_message_id, actor=ChangeActor.USER_INPUT.value
            ):
                return EntityUpdateResult(
                    already_applied=True, new_version=await sessions.get_version(session_id)
                )

            version = await sessions.get_version(session_id)
            if expected_version is not None and expected_version != version:
                raise ConcurrencyConflictError(f"expected {expected_version}, found {version}")

            snapshot = await entities.snapshot(
                session_id, version=version, schema_version=self.catalog.schema_version
            )
            planned, rejected = resolve_candidates(
                candidates, snapshot, self.catalog, self.normalization_context(),
                current_message_text=current_message_text,
                min_confidence_known=self.min_confidence_known,
            )
            final, cross_rejected = reconcile(snapshot, planned)
            rejected.extend(cross_rejected)

            if not final:
                return EntityUpdateResult(rejected=rejected, new_version=version)

            for ch in final:
                await entities.upsert(
                    session_id, ch.entity_key,
                    value=ch.new_value, status=ch.new_status,
                    confidence=ch.confidence, evidence_text=ch.evidence_text,
                    source_message_id=source_message_id,
                )
                await entities.add_history(
                    session_id, ch.entity_key,
                    old_value=ch.old_value, new_value=ch.new_value,
                    old_status=ch.old_status, new_status=ch.new_status,
                    source_message_id=source_message_id,
                    change_type=ch.change_type, actor=ch.actor.value,
                    evidence_text=ch.evidence_text,
                )

            if not await sessions.bump_version(session_id, version):
                raise ConcurrencyConflictError("row_version changed during write")

            # New information after completion reopens the intake for confirmation.
            session_row = await sessions.get(session_id)
            if session_row and session_row.status == SessionStatus.INTAKE_COMPLETE.value:
                await sessions.set_status(session_id, SessionStatus.ACTIVE)

            return EntityUpdateResult(
                accepted=[
                    AcceptedUpdate(
                        entity_key=ch.entity_key,
                        old_value=ch.old_value, new_value=ch.new_value,
                        old_status=ch.old_status, new_status=ch.new_status,
                        change_type=ch.change_type, actor=ch.actor,
                        confidence=ch.confidence, evidence_text=ch.evidence_text,
                    )
                    for ch in final
                ],
                rejected=rejected,
                new_version=version + 1,
            )

    async def confirm_intake(self, session_id: str, *, source_message_id: str) -> list[str]:
        """User confirmed the summary: mark entities confirmed and complete the intake."""
        async with self.db.session() as s:
            entities = EntityRepository(s)
            sessions = SessionRepository(s)
            snapshot = await entities.snapshot(
                session_id, version=await sessions.get_version(session_id),
                schema_version=self.catalog.schema_version,
            )
            if not compute_missing(self.catalog, snapshot).required_complete:
                return []
            confirmed = await entities.mark_confirmed(session_id)
            for key in confirmed:
                state = snapshot.get(key)
                await entities.add_history(
                    session_id, key,
                    old_value=state.value, new_value=state.value,
                    old_status=state.status, new_status=state.status,
                    source_message_id=source_message_id, change_type="confirm",
                    actor=ChangeActor.USER_CONFIRMATION.value, evidence_text=None,
                )
            await sessions.set_status(session_id, SessionStatus.INTAKE_COMPLETE)
            return confirmed or [k for k, e in snapshot.entities.items()
                                 if e.status != EntityStatus.MISSING]
