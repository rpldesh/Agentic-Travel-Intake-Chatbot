import os

# Strict checkpoint deserialization, as recommended for production use.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

import uuid
from datetime import date

import pytest
import pytest_asyncio

from travel_bot.database.connection import Database
from travel_bot.entities.catalog import load_catalog
from travel_bot.entities.normalizers import NormalizationContext
from travel_bot.repositories.session_repository import SessionRepository
from travel_bot.schemas.entities import EntityCandidate
from travel_bot.services.entity_service import EntityService

REF_DATE = date(2026, 10, 18)


@pytest.fixture
def catalog():
    return load_catalog()


@pytest.fixture
def ctx():
    return NormalizationContext(reference_date=REF_DATE)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    await database.create_all()
    yield database
    await database.dispose()


@pytest_asyncio.fixture
async def entity_service(db, catalog, monkeypatch):
    service = EntityService(db, catalog)
    monkeypatch.setattr(service, "reference_date", lambda: REF_DATE)
    return service


@pytest_asyncio.fixture
async def session_id(db):
    async with db.session() as s:
        row, _token = await SessionRepository(s).create(user_id="user-1")
        return row.session_id


def cand(key, value_json=None, status="known", evidence="", confidence=0.95, intent="set"):
    return EntityCandidate(
        entity_key=key,
        value_json=value_json,
        status=status,
        confidence=confidence,
        evidence_text=evidence,
        change_intent=intent,
    )


def new_id() -> str:
    return f"msg_{uuid.uuid4().hex[:10]}"
