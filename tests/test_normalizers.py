import pytest

from travel_bot.entities.normalizers import (
    NormalizationError,
    decode_value,
    normalize_budget,
    normalize_children,
    normalize_count,
    normalize_date,
    normalize_duration,
    normalize_list,
    normalize_location,
    normalize_trip_type,
)


def test_location_from_string_and_dict(ctx):
    assert normalize_location("  Rome ", ctx) == {"city": "Rome"}
    assert normalize_location({"city": "Austin", "airport_code": "aus"}, ctx) == {
        "city": "Austin", "airport_code": "AUS"
    }
    with pytest.raises(NormalizationError):
        normalize_location({}, ctx)
    with pytest.raises(NormalizationError) as err:
        normalize_location({"airport_code": "AUST"}, ctx)
    assert err.value.reason_code == "invalid_airport_code"


def test_dates(ctx):
    assert normalize_date({"date": "2027-05-03"}, ctx) == {"date": "2027-05-03"}
    assert normalize_date("2027-05-03", ctx) == {"date": "2027-05-03"}
    assert normalize_date({"earliest": "2027-05-01", "latest": "2027-05-31"}, ctx) == {
        "earliest": "2027-05-01", "latest": "2027-05-31"
    }
    with pytest.raises(NormalizationError) as err:
        normalize_date({"date": "2025-01-01"}, ctx)
    assert err.value.reason_code == "past_date"
    with pytest.raises(NormalizationError) as err:
        normalize_date({"date": "next tuesday"}, ctx)
    assert err.value.reason_code == "invalid_date"
    with pytest.raises(NormalizationError) as err:
        normalize_date({"earliest": "2027-06-01", "latest": "2027-05-01"}, ctx)
    assert err.value.reason_code == "invalid_range"


def test_partially_past_range_is_clamped(ctx):
    assert normalize_date({"earliest": "2026-10-01", "latest": "2026-10-30"}, ctx) == {
        "earliest": "2026-10-18", "latest": "2026-10-30"
    }


def test_trip_type_and_counts(ctx):
    assert normalize_trip_type("Round Trip", ctx) == {"type": "round_trip"}
    assert normalize_trip_type({"type": "one-way"}, ctx) == {"type": "one_way"}
    with pytest.raises(NormalizationError):
        normalize_trip_type("cruise", ctx)
    assert normalize_count({"count": "2"}, ctx) == {"count": 2}
    with pytest.raises(NormalizationError):
        normalize_count({"count": 2.5}, ctx)
    with pytest.raises(NormalizationError) as err:
        normalize_count({"count": 99}, ctx)
    assert err.value.reason_code == "party_too_large"
    assert normalize_children({"count": 2, "ages": [4, 9]}, ctx) == {"count": 2, "ages": [4, 9]}
    with pytest.raises(NormalizationError):
        normalize_children({"count": 1, "ages": [25]}, ctx)


def test_duration_budget_lists(ctx):
    assert normalize_duration({"nights": 7}, ctx) == {"nights": 7}
    assert normalize_duration({"days": 5}, ctx) == {"days": 5, "nights": 4}
    assert normalize_budget({"amount": "3,000", "currency": "$", "basis": "per person"}, ctx) == {
        "currency": "USD", "amount": 3000.0, "basis": "per_person"
    }
    with pytest.raises(NormalizationError) as err:
        normalize_budget({"amount": 100, "currency": "XYZ"}, ctx)
    assert err.value.reason_code == "invalid_currency"
    assert normalize_list({"items": ["Food", "food", " beaches "]}, ctx) == {
        "items": ["Food", "beaches"]
    }


def test_decode_value_tolerates_bare_strings():
    assert decode_value('{"city": "Rome"}') == {"city": "Rome"}
    assert decode_value("round_trip") == "round_trip"
    assert decode_value(None) is None
