"""Versioned, configuration-driven entity catalog."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, model_validator

from travel_bot.schemas.enums import EntityStatus, EntityTier

_CATALOG_DIR = Path(__file__).parent


class EntityDefinition(BaseModel):
    key: str
    label: str
    type: str
    tier: EntityTier
    required_rule: str
    ask_rule: str
    priority: int
    dependencies: list[str]
    ask_with: list[str]
    normalizer: str
    validation_rule: str
    allowed_statuses: list[EntityStatus]
    value_format: str
    question_template: str
    schema_version: str = ""


class EntityCatalog(BaseModel):
    schema_version: str
    entities: list[EntityDefinition]

    @model_validator(mode="after")
    def _check(self) -> "EntityCatalog":
        keys = [e.key for e in self.entities]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate entity keys in catalog")
        for e in self.entities:
            e.schema_version = self.schema_version
            for ref in [*e.dependencies, *e.ask_with]:
                if ref not in keys:
                    raise ValueError(f"{e.key} references unknown entity {ref}")
        self.entities.sort(key=lambda e: e.priority)
        return self

    @property
    def keys(self) -> list[str]:
        return [e.key for e in self.entities]

    def get(self, key: str) -> EntityDefinition | None:
        for e in self.entities:
            if e.key == key:
                return e
        return None

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


@lru_cache
def load_catalog(version: str = "v1") -> EntityCatalog:
    path = _CATALOG_DIR / f"catalog_{version}.json"
    return EntityCatalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
