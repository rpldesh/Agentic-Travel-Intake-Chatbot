from travel_bot.entities.requirements import compute_missing
from travel_bot.entities.resolution import (
    apply_to_snapshot,
    evidence_in_message,
    reconcile,
    resolve_candidates,
)
from travel_bot.schemas.entities import EntitySnapshot, EntityState
from travel_bot.schemas.enums import EntityStatus

from tests.conftest import cand


def snap(**entities):
    return EntitySnapshot(
        session_id="s", version=1, schema_version="v1",
        entities={
            k.replace("__", "."): EntityState(entity_key=k.replace("__", "."), value=v[0], status=v[1])
            for k, v in entities.items()
        },
    )


def resolve(cands, snapshot, catalog, ctx, msg):
    planned, rejected = resolve_candidates(cands, snapshot, catalog, ctx, current_message_text=msg)
    final, cross = reconcile(snapshot, planned)
    return final, rejected + cross


def test_evidence_matching():
    assert evidence_in_message("go to Rome", "I want to go to Rome!")
    assert evidence_in_message("two adults", "We are two adults and a kid")
    assert not evidence_in_message("Paris", "I want to go to Rome")
    assert not evidence_in_message("", "anything")


def test_multi_entity_extraction(catalog, ctx):
    msg = "Flying from Dallas to Rome on 2027-05-03, two adults"
    final, rejected = resolve([
        cand("destination", '{"city": "Rome"}', evidence="to Rome"),
        cand("origin", '{"city": "Dallas"}', evidence="from Dallas"),
        cand("departure_date", '{"date": "2027-05-03"}', evidence="on 2027-05-03"),
        cand("travelers.adults", '{"count": 2}', evidence="two adults"),
    ], snap(), catalog, ctx, msg)
    assert not rejected
    assert {c.entity_key for c in final} == {"destination", "origin", "departure_date", "travelers.adults"}
    assert all(c.change_type == "create" for c in final)


def test_explicit_correction_updates_value(catalog, ctx):
    s = snap(origin=({"city": "Dallas"}, EntityStatus.KNOWN))
    final, _ = resolve([cand("origin", '{"city": "Austin"}', evidence="Austin, not Dallas",
                             intent="correct")], s, catalog, ctx, "Sorry, Austin, not Dallas")
    assert final[0].change_type == "correct"
    assert final[0].old_value == {"city": "Dallas"}
    assert final[0].new_value == {"city": "Austin"}


def test_unclear_overwrite_becomes_ambiguous(catalog, ctx):
    s = snap(origin=({"city": "Dallas"}, EntityStatus.KNOWN))
    final, _ = resolve([cand("origin", '{"city": "Houston"}', evidence="Houston")],
                       s, catalog, ctx, "Houston")
    assert final[0].new_status == EntityStatus.AMBIGUOUS
    assert final[0].new_value == {"current": {"city": "Dallas"}, "proposed": {"city": "Houston"}}


def test_stale_value_from_history_is_rejected(catalog, ctx):
    s = snap(origin=({"city": "Austin"}, EntityStatus.KNOWN))
    final, rejected = resolve([cand("origin", '{"city": "Dallas"}', evidence="from Dallas",
                                    intent="correct")], s, catalog, ctx, "Two adults please")
    assert not final
    assert rejected[0].reason_code == "evidence_not_in_current_message"


def test_flexible_and_declined(catalog, ctx):
    final, _ = resolve([
        cand("destination", None, status="flexible", evidence="don't know where yet"),
        cand("budget", None, status="declined", evidence="rather not say"),
    ], snap(), catalog, ctx, "I don't know where yet and I'd rather not say the budget")
    by_key = {c.entity_key: c for c in final}
    assert by_key["destination"].new_status == EntityStatus.FLEXIBLE
    assert by_key["budget"].new_status == EntityStatus.DECLINED


def test_low_confidence_known_downgraded(catalog, ctx):
    final, _ = resolve([cand("travelers.adults", '{"count": 4}', evidence="my family",
                             confidence=0.3)], snap(), catalog, ctx, "traveling with my family")
    assert final[0].new_status == EntityStatus.AMBIGUOUS


def test_invalid_values_and_statuses_rejected(catalog, ctx):
    final, rejected = resolve([
        cand("departure_date", '{"date": "2020-01-01"}', evidence="2020-01-01"),
        cand("travelers.adults", '{"count": 0}', evidence="0"),
        cand("trip_type", None, status="flexible", evidence="whatever"),
        cand("hotel_stars", '{"x": 1}', evidence="whatever"),
    ], snap(), catalog, ctx, "2020-01-01 0 whatever")
    codes = {r.entity_key: r.reason_code for r in rejected}
    assert codes == {
        "departure_date": "past_date",
        "travelers.adults": "invalid_adult_count",
        "trip_type": "status_not_allowed",
        "hotel_stars": "unknown_entity",
    }
    assert not final


def test_list_values_merge(catalog, ctx):
    s = snap(interests=({"items": ["museums"]}, EntityStatus.KNOWN))
    final, _ = resolve([cand("interests", '{"items": ["food"]}', evidence="love food")],
                       s, catalog, ctx, "we also love food")
    assert final[0].change_type == "merge"
    assert final[0].new_value == {"items": ["museums", "food"]}


def test_no_op_when_value_unchanged(catalog, ctx):
    s = snap(destination=({"city": "Rome"}, EntityStatus.KNOWN))
    final, rejected = resolve([cand("destination", '{"city": "Rome"}', evidence="Rome")],
                              s, catalog, ctx, "Rome")
    assert not final and not rejected


def test_derivations_round_trip_duration(catalog, ctx):
    msg = "leaving 2027-05-03 and back 2027-05-10"
    final, _ = resolve([
        cand("departure_date", '{"date": "2027-05-03"}', evidence="leaving 2027-05-03"),
        cand("return_date", '{"date": "2027-05-10"}', evidence="back 2027-05-10"),
    ], snap(), catalog, ctx, msg)
    by_key = {c.entity_key: c for c in final}
    assert by_key["trip_type"].new_value == {"type": "round_trip"}
    assert by_key["trip_type"].actor == "system_normalization"
    assert by_key["trip_duration"].new_value == {"nights": 7}


def test_one_way_marks_return_not_applicable(catalog, ctx):
    final, _ = resolve([cand("trip_type", '{"type": "one_way"}', evidence="one way")],
                       snap(), catalog, ctx, "one way please")
    by_key = {c.entity_key: c for c in final}
    assert by_key["return_date"].new_status == EntityStatus.NOT_APPLICABLE


def test_return_before_departure_rejected(catalog, ctx):
    s = snap(departure_date=({"date": "2027-05-10"}, EntityStatus.KNOWN))
    final, rejected = resolve([cand("return_date", '{"date": "2027-05-01"}', evidence="2027-05-01")],
                              s, catalog, ctx, "back on 2027-05-01")
    assert not final
    assert rejected[0].reason_code == "return_before_departure"


def test_month_range_sets_date_flexibility(catalog, ctx):
    final, _ = resolve([cand("departure_date", '{"earliest": "2027-05-01", "latest": "2027-05-31"}',
                             status="flexible", evidence="sometime in May")],
                       snap(), catalog, ctx, "sometime in May")
    by_key = {c.entity_key: c for c in final}
    assert by_key["date_flexibility"].new_value == {"mode": "flexible"}


def test_missing_report_priorities(catalog):
    report = compute_missing(catalog, snap())
    assert report.missing_required == ["destination", "origin", "departure_date", "trip_type", "travelers.adults"]
    assert not report.required_complete

    s = snap(
        destination=({"city": "Rome"}, EntityStatus.KNOWN),
        origin=({"city": "Austin"}, EntityStatus.KNOWN),
        departure_date=({"date": "2027-05-03"}, EntityStatus.KNOWN),
        trip_type=({"type": "round_trip"}, EntityStatus.KNOWN),
        travelers__adults=({"count": 2}, EntityStatus.KNOWN),
    )
    report = compute_missing(catalog, s)
    assert report.missing_required == ["return_date"]  # round trip needs an end

    s2 = apply_to_snapshot(s, [])
    s2.entities["trip_duration"] = EntityState(entity_key="trip_duration", value={"nights": 7},
                                               status=EntityStatus.KNOWN)
    report = compute_missing(catalog, s2)
    assert report.required_complete
    # optional fields never block completion; recommended ones are offered once
    assert report.missing_recommended == ["travelers.children", "budget", "date_flexibility"]


def test_flexible_destination_promotes_interests(catalog):
    s = snap(destination=(None, EntityStatus.FLEXIBLE))
    report = compute_missing(catalog, s)
    assert report.missing_recommended[0] == "interests"
