"""Narrow, typed entity tools.

The session scope is bound when the tools are built for a turn (from the
authenticated runtime context). None of the tool schemas expose a session_id,
so a model holding these tools cannot address another session. There is no
SQL access, only these three operations.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from travel_bot.graph.progress import tool_call
from travel_bot.schemas.entities import (
    EntityCandidate,
    EntitySnapshot,
    EntityUpdateResult,
    MissingReport,
)
from travel_bot.services.entity_service import EntityService


class RecordEntityUpdatesArgs(BaseModel):
    updates: list[EntityCandidate] = Field(description="Validated candidate updates.")
    expected_version: int | None = Field(
        default=None, description="Snapshot version the updates were based on."
    )


def _update_summary(result: EntityUpdateResult) -> str:
    if result.already_applied:
        return "already applied (duplicate delivery)"
    parts = [f"{len(result.accepted)} saved"]
    if result.rejected:
        parts.append(f"{len(result.rejected)} rejected")
    return ", ".join(parts)


@dataclass
class EntityTools:
    """Session-bound entity operations, usable directly or as LangChain tools."""

    service: EntityService
    session_id: str
    source_message_id: str
    current_message_text: str

    async def get_session_entities(self) -> EntitySnapshot:
        with tool_call("get_session_entities") as out:
            snapshot = await self.service.get_snapshot(self.session_id)
            out["summary"] = f"{len(snapshot.known_items())} saved details"
            return snapshot

    async def get_missing_entities(self) -> MissingReport:
        with tool_call("get_missing_entities") as out:
            report = await self.service.get_missing(self.session_id)
            out["summary"] = f"{len(report.missing_required)} required still missing"
            return report

    async def record_entity_updates(
        self, updates: list[EntityCandidate], expected_version: int | None = None
    ) -> EntityUpdateResult:
        with tool_call("record_entity_updates", candidates=len(updates)) as out:
            result = await self.service.record_updates(
                self.session_id,
                updates,
                source_message_id=self.source_message_id,
                current_message_text=self.current_message_text,
                expected_version=expected_version,
            )
            out["summary"] = _update_summary(result)
            out["accepted_keys"] = [a.entity_key for a in result.accepted]
            out["rejected"] = [f"{r.entity_key}:{r.reason_code}" for r in result.rejected]
            return result

    def as_langchain_tools(self) -> list[BaseTool]:
        async def _get_entities() -> dict:
            return (await self.get_session_entities()).model_dump(mode="json")

        async def _get_missing() -> dict:
            return (await self.get_missing_entities()).model_dump(mode="json")

        async def _record(
            updates: list[EntityCandidate], expected_version: int | None = None
        ) -> dict:
            result = await self.record_entity_updates(updates, expected_version)
            return result.model_dump(mode="json")

        return [
            StructuredTool.from_function(
                coroutine=_get_entities,
                name="get_session_entities",
                description="Return the canonical trip entity snapshot for this session.",
            ),
            StructuredTool.from_function(
                coroutine=_get_missing,
                name="get_missing_entities",
                description="Return required/recommended missing and ambiguous entity keys.",
            ),
            StructuredTool.from_function(
                coroutine=_record,
                name="record_entity_updates",
                description="Validate and persist a batch of entity updates for this session.",
                args_schema=RecordEntityUpdatesArgs,
            ),
        ]
