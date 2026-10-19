import pytest
from sqlalchemy.exc import OperationalError

from travel_bot.errors import (
    PersistenceError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionUnauthorizedError,
)
from travel_bot.repositories.entity_repository import EntityRepository
from travel_bot.repositories.session_repository import SessionRepository
from travel_bot.schemas.enums import EntityStatus, SessionStatus
from travel_bot.services.entity_service import EntityService
from travel_bot.tools.entity_tools import EntityTools

from tests.conftest import cand, new_id


async def test_record_and_reload(entity_service, session_id):
    msg = "Flying from Dallas to Rome with my wife"
    result = await entity_service.record_updates(
        session_id,
        [cand("destination", '{"city": "Rome"}', evidence="to Rome"),
         cand("origin", '{"city": "Dallas"}', evidence="from Dallas")],
        source_message_id=new_id(), current_message_text=msg,
    )
    assert {a.entity_key for a in result.accepted} == {"destination", "origin"}
    snap = await entity_service.get_snapshot(session_id)
    assert snap.value_of("destination") == {"city": "Rome"}
    assert snap.version == 2


async def test_correction_keeps_audit_history(entity_service, session_id, db):
    await entity_service.record_updates(
        session_id, [cand("origin", '{"city": "Dallas"}', evidence="Dallas")],
        source_message_id="m1", current_message_text="from Dallas",
    )
    await entity_service.record_updates(
        session_id, [cand("origin", '{"city": "Austin"}', evidence="Austin", intent="correct")],
        source_message_id="m2", current_message_text="Actually Austin",
    )
    snap = await entity_service.get_snapshot(session_id)
    assert snap.value_of("origin") == {"city": "Austin"}
    async with db.session() as s:
        history = await EntityRepository(s).list_history(session_id, "origin")
    assert [(h.old_value_json, h.new_value_json, h.change_type) for h in history] == [
        (None, {"city": "Dallas"}, "create"),
        ({"city": "Dallas"}, {"city": "Austin"}, "correct"),
    ]


async def test_duplicate_delivery_is_idempotent(entity_service, session_id, db):
    args = dict(source_message_id="dup-1", current_message_text="to Rome")
    first = await entity_service.record_updates(
        session_id, [cand("destination", '{"city": "Rome"}', evidence="to Rome")], **args
    )
    second = await entity_service.record_updates(
        session_id, [cand("destination", '{"city": "Rome"}', evidence="to Rome")], **args
    )
    assert first.accepted and not second.accepted and second.already_applied
    async with db.session() as s:
        assert len(await EntityRepository(s).list_history(session_id)) == 1


async def test_stale_expected_version_reloads_and_resolves(entity_service, session_id):
    await entity_service.record_updates(
        session_id, [cand("origin", '{"city": "Dallas"}', evidence="Dallas")],
        source_message_id="m1", current_message_text="Dallas",
    )
    # A writer that read version 1 must not blindly overwrite: it re-resolves.
    result = await entity_service.record_updates(
        session_id, [cand("destination", '{"city": "Rome"}', evidence="Rome")],
        source_message_id="m2", current_message_text="Rome", expected_version=1,
    )
    assert result.conflict_retries == 1
    snap = await entity_service.get_snapshot(session_id)
    assert snap.value_of("origin") == {"city": "Dallas"}
    assert snap.value_of("destination") == {"city": "Rome"}


async def test_transient_db_failure_raises_persistence_error(entity_service, session_id, monkeypatch):
    async def boom(*_a, **_k):
        raise OperationalError("stmt", {}, Exception("database is locked"))

    monkeypatch.setattr(entity_service, "_record_with_conflict_retry", boom)
    with pytest.raises(PersistenceError):
        await entity_service.record_updates(
            session_id, [cand("destination", '{"city": "Rome"}', evidence="Rome")],
            source_message_id="m1", current_message_text="Rome",
        )
    assert (await entity_service.get_snapshot(session_id)).status_of("destination") == EntityStatus.MISSING


async def test_sessions_are_isolated(entity_service, db):
    async with db.session() as s:
        a, _ = await SessionRepository(s).create(user_id="a")
        b, _ = await SessionRepository(s).create(user_id="b")
    await entity_service.record_updates(
        a.session_id, [cand("destination", '{"city": "Rome"}', evidence="Rome")],
        source_message_id="m1", current_message_text="Rome",
    )
    await entity_service.record_updates(
        b.session_id, [cand("destination", '{"city": "Tokyo"}', evidence="Tokyo")],
        source_message_id="m1", current_message_text="Tokyo",
    )
    assert (await entity_service.get_snapshot(a.session_id)).value_of("destination") == {"city": "Rome"}
    assert (await entity_service.get_snapshot(b.session_id)).value_of("destination") == {"city": "Tokyo"}


async def test_session_authorization(db):
    async with db.session() as s:
        repo = SessionRepository(s)
        row, token = await repo.create(user_id="owner")
        assert (await repo.authorize(row.session_id, token, user_id="owner")).session_id == row.session_id
        with pytest.raises(SessionUnauthorizedError):
            await repo.authorize(row.session_id, "wrong-token")
        with pytest.raises(SessionUnauthorizedError):
            await repo.authorize(row.session_id, token, user_id="someone-else")
        with pytest.raises(SessionNotFoundError):
            await repo.authorize("nope", token)
        await repo.set_status(row.session_id, SessionStatus.EXPIRED)
        await s.refresh(row)
        with pytest.raises(SessionExpiredError):
            await repo.authorize(row.session_id, token)


async def test_confirm_intake_requires_complete_required_fields(entity_service, session_id, db):
    assert await entity_service.confirm_intake(session_id, source_message_id="c0") == []
    msg = "Austin to Rome 2027-05-03 one way 2 adults"
    await entity_service.record_updates(session_id, [
        cand("destination", '{"city": "Rome"}', evidence="Rome"),
        cand("origin", '{"city": "Austin"}', evidence="Austin"),
        cand("departure_date", '{"date": "2027-05-03"}', evidence="2027-05-03"),
        cand("trip_type", '{"type": "one_way"}', evidence="one way"),
        cand("travelers.adults", '{"count": 2}', evidence="2 adults"),
    ], source_message_id="m1", current_message_text=msg)
    confirmed = await entity_service.confirm_intake(session_id, source_message_id="c1")
    assert "destination" in confirmed
    async with db.session() as s:
        assert (await SessionRepository(s).get(session_id)).status == "intake_complete"
    # a later change reopens the intake
    await entity_service.record_updates(
        session_id, [cand("travelers.adults", '{"count": 3}', evidence="3 of us", intent="correct")],
        source_message_id="m2", current_message_text="actually 3 of us",
    )
    async with db.session() as s:
        assert (await SessionRepository(s).get(session_id)).status == "active"


async def test_tools_are_session_bound(entity_service, session_id):
    tools = EntityTools(entity_service, session_id, "m1", "to Rome").as_langchain_tools()
    for tool in tools:
        assert "session_id" not in tool.args
    record = next(t for t in tools if t.name == "record_entity_updates")
    out = await record.ainvoke({"updates": [cand("destination", '{"city": "Rome"}', evidence="to Rome").model_dump()]})
    assert out["accepted"][0]["entity_key"] == "destination"
    missing = next(t for t in tools if t.name == "get_missing_entities")
    assert "destination" not in (await missing.ainvoke({}))["missing_required"]


def test_reference_date_uses_configured_timezone(db, catalog):
    # Not mocked: Windows needs the tzdata package for IANA zones.
    from datetime import date

    for tz in ("UTC", "Asia/Colombo", "America/Los_Angeles"):
        assert isinstance(EntityService(db, catalog, timezone=tz).reference_date(), date)
