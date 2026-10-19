"""Deterministic value normalization + validation, keyed by catalog `normalizer`.

Each normalizer takes the decoded candidate value and a context and returns the
canonical value (a JSON-serializable dict) or raises `NormalizationError`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from dateutil import parser as date_parser


class NormalizationError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class NormalizationContext:
    reference_date: date
    max_party_size: int = 20
    max_days_ahead: int = 3 * 365


Normalizer = Callable[[Any, NormalizationContext], dict[str, Any]]


def decode_value(value_json: str | None) -> Any:
    if value_json is None or value_json == "":
        return None
    try:
        return json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        # Tolerate a bare string value, e.g. "round_trip" or "Rome".
        return value_json


def _clean_str(value: Any, max_len: int = 200) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text[:max_len] or None


def _require_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalizationError("invalid_format", f"expected an object, got {type(value).__name__}")
    return value


def _to_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise NormalizationError("invalid_number", field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError("invalid_number", field) from exc
    if not number.is_integer():
        raise NormalizationError("invalid_number", f"{field} must be a whole number")
    return int(number)


def _to_number(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise NormalizationError("invalid_number", field)
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError("invalid_number", field) from exc


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def normalize_location(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"city": value}
    value = _require_dict(value)
    out = {
        "city": _clean_str(value.get("city"), 100),
        "region": _clean_str(value.get("region"), 100),
        "country": _clean_str(value.get("country"), 100),
        "airport_code": None,
        "description": _clean_str(value.get("description"), 200),
    }
    code = _clean_str(value.get("airport_code"), 8)
    if code:
        code = code.upper()
        if not (len(code) == 3 and code.isalpha()):
            raise NormalizationError("invalid_airport_code", code)
        out["airport_code"] = code
    out = {k: v for k, v in out.items() if v is not None}
    if not out:
        raise NormalizationError("empty_value", "location needs at least one part")
    return out


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def _parse_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise NormalizationError("invalid_date", field)
    try:
        # Require an explicit year-month-day; reject fuzzy strings.
        parsed = date_parser.isoparse(value.strip())
    except (ValueError, OverflowError) as exc:
        raise NormalizationError("invalid_date", f"{field}={value!r}") from exc
    return parsed.date() if hasattr(parsed, "date") else parsed


def normalize_date(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"date": value}
    value = _require_dict(value)
    horizon = ctx.reference_date + timedelta(days=ctx.max_days_ahead)

    if value.get("date"):
        d = _parse_iso_date(value["date"], "date")
        if d < ctx.reference_date:
            raise NormalizationError("past_date", d.isoformat())
        if d > horizon:
            raise NormalizationError("date_too_far", d.isoformat())
        return {"date": d.isoformat()}

    earliest = value.get("earliest")
    latest = value.get("latest")
    if not earliest and not latest:
        raise NormalizationError("empty_value", "date needs date or earliest/latest")
    e = _parse_iso_date(earliest, "earliest") if earliest else None
    l = _parse_iso_date(latest, "latest") if latest else None
    if e and l and e > l:
        raise NormalizationError("invalid_range", "earliest is after latest")
    if l and l < ctx.reference_date:
        raise NormalizationError("past_date", l.isoformat())
    if e and e > horizon:
        raise NormalizationError("date_too_far", e.isoformat())
    if e and e < ctx.reference_date:
        e = ctx.reference_date  # partially-past range: clamp to today
    out: dict[str, Any] = {}
    if e:
        out["earliest"] = e.isoformat()
    if l:
        out["latest"] = l.isoformat()
    if e and l and e == l:
        return {"date": e.isoformat()}
    return out


def date_bounds(value: dict[str, Any] | None) -> tuple[date | None, date | None]:
    """(earliest, latest) of a normalized date value."""
    if not value:
        return None, None
    if value.get("date"):
        d = date.fromisoformat(value["date"])
        return d, d
    e = date.fromisoformat(value["earliest"]) if value.get("earliest") else None
    l = date.fromisoformat(value["latest"]) if value.get("latest") else None
    return e, l


# ---------------------------------------------------------------------------
# Enumerations / counts
# ---------------------------------------------------------------------------
_TRIP_TYPES = {
    "one_way": "one_way", "oneway": "one_way", "one-way": "one_way", "one way": "one_way",
    "round_trip": "round_trip", "roundtrip": "round_trip", "round-trip": "round_trip",
    "round trip": "round_trip", "return": "round_trip",
    "multi_city": "multi_city", "multi-city": "multi_city", "multicity": "multi_city",
    "multi city": "multi_city",
    "other": "other",
}


def normalize_trip_type(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    raw = value.get("type") if isinstance(value, dict) else value
    key = (_clean_str(raw, 30) or "").lower()
    if key not in _TRIP_TYPES:
        raise NormalizationError("invalid_enum", f"trip_type={raw!r}")
    return {"type": _TRIP_TYPES[key]}


def normalize_count(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    raw = value.get("count") if isinstance(value, dict) else value
    count = _to_int(raw, "count")
    if count < 0:
        raise NormalizationError("negative_count", str(count))
    if count > ctx.max_party_size:
        raise NormalizationError("party_too_large", str(count))
    return {"count": count}


def normalize_children(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {"count": value}
    out = normalize_count(value, ctx)
    ages_raw = value.get("ages") or []
    if not isinstance(ages_raw, list):
        raise NormalizationError("invalid_format", "ages must be a list")
    ages = [_to_int(a, "age") for a in ages_raw]
    if any(a < 0 or a > 17 for a in ages):
        raise NormalizationError("invalid_child_age", str(ages))
    if ages and len(ages) > out["count"]:
        if out["count"] == 0:
            out["count"] = len(ages)
        else:
            raise NormalizationError("ages_count_mismatch", f"{len(ages)} ages for {out['count']}")
    if ages:
        out["ages"] = ages
    return out


def normalize_duration(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {"nights": value}
    nights = value.get("nights")
    days = value.get("days")
    if nights is None and days is None:
        raise NormalizationError("empty_value", "duration needs nights or days")
    out: dict[str, Any] = {}
    if nights is not None:
        out["nights"] = _to_int(nights, "nights")
    if days is not None:
        out["days"] = _to_int(days, "days")
        out.setdefault("nights", max(out["days"] - 1, 0))
    if out["nights"] < 0 or out["nights"] > 365:
        raise NormalizationError("invalid_duration", str(out["nights"]))
    return out


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
_CURRENCY_ALIASES = {
    "$": "USD", "us$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "¥": "JPY", "jpy": "JPY", "yen": "JPY",
    "₹": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "lkr": "LKR", "rs": "LKR", "aud": "AUD", "a$": "AUD", "cad": "CAD", "c$": "CAD",
    "chf": "CHF", "cny": "CNY", "sgd": "SGD", "aed": "AED", "nzd": "NZD", "mxn": "MXN",
    "brl": "BRL", "zar": "ZAR", "sek": "SEK", "nok": "NOK", "dkk": "DKK", "thb": "THB",
}
_KNOWN_CURRENCIES = set(_CURRENCY_ALIASES.values()) | {
    "HKD", "KRW", "MYR", "IDR", "PHP", "VND", "PLN", "CZK", "HUF", "TRY", "ILS", "SAR",
    "QAR", "EGP", "KES", "NGN", "ARS", "CLP", "COP", "PEN", "PKR", "BDT", "NPR", "TWD",
}


def normalize_currency(raw: Any) -> str:
    text = (_clean_str(raw, 10) or "").strip()
    if not text:
        raise NormalizationError("missing_currency", "")
    alias = _CURRENCY_ALIASES.get(text.lower())
    if alias:
        return alias
    if text.upper() in _KNOWN_CURRENCIES:
        return text.upper()
    raise NormalizationError("invalid_currency", text)


def normalize_budget(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    value = _require_dict(value)
    amount = _to_number(value.get("amount"), "amount")
    low = _to_number(value.get("min"), "min")
    high = _to_number(value.get("max"), "max")
    if amount is None and low is None and high is None:
        raise NormalizationError("empty_value", "budget needs an amount or range")
    for n in (amount, low, high):
        if n is not None and n <= 0:
            raise NormalizationError("invalid_budget", str(n))
    if low is not None and high is not None and low > high:
        raise NormalizationError("invalid_range", "min above max")
    out: dict[str, Any] = {"currency": normalize_currency(value.get("currency"))}
    if amount is not None:
        out["amount"] = amount
    if low is not None:
        out["min"] = low
    if high is not None:
        out["max"] = high
    basis = (_clean_str(value.get("basis"), 20) or "").lower().replace("-", "_").replace(" ", "_")
    if basis in {"total", "per_person"}:
        out["basis"] = basis
    elif basis:
        raise NormalizationError("invalid_enum", f"basis={basis}")
    return out


def normalize_date_flexibility(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"mode": value}
    value = _require_dict(value)
    mode = (_clean_str(value.get("mode"), 30) or "").lower()
    if mode not in {"fixed", "plus_minus_days", "flexible"}:
        raise NormalizationError("invalid_enum", f"mode={mode}")
    out: dict[str, Any] = {"mode": mode}
    if value.get("days") is not None:
        days = _to_int(value["days"], "days")
        if days < 0 or days > 60:
            raise NormalizationError("invalid_number", f"days={days}")
        out["days"] = days
    if mode == "plus_minus_days" and "days" not in out:
        raise NormalizationError("missing_days", "plus_minus_days needs days")
    return out


# ---------------------------------------------------------------------------
# Free-form-ish
# ---------------------------------------------------------------------------
def normalize_list(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("items")
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    if not isinstance(items, list):
        raise NormalizationError("invalid_format", "items must be a list")
    seen: dict[str, str] = {}
    for item in items:
        text = _clean_str(item, 80)
        if text and text.lower() not in seen:
            seen[text.lower()] = text
    if not seen:
        raise NormalizationError("empty_value", "list is empty")
    return {"items": list(seen.values())[:20]}


def normalize_accommodation(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"notes": value}
    value = _require_dict(value)
    out: dict[str, Any] = {}
    for field in ("type", "location", "notes"):
        text = _clean_str(value.get(field), 200)
        if text:
            out[field] = text
    if value.get("rooms") is not None:
        rooms = _to_int(value["rooms"], "rooms")
        if rooms < 1 or rooms > 20:
            raise NormalizationError("invalid_number", f"rooms={rooms}")
        out["rooms"] = rooms
    rating = _to_number(value.get("min_rating"), "min_rating")
    if rating is not None:
        if not 0 < rating <= 5:
            raise NormalizationError("invalid_number", f"min_rating={rating}")
        out["min_rating"] = rating
    if not out:
        raise NormalizationError("empty_value", "accommodation preferences empty")
    return out


def normalize_text(value: Any, ctx: NormalizationContext) -> dict[str, Any]:
    raw = value.get("text") if isinstance(value, dict) else value
    text = _clean_str(raw, 500)
    if not text:
        raise NormalizationError("empty_value", "text is empty")
    return {"text": text}


NORMALIZERS: dict[str, Normalizer] = {
    "location": normalize_location,
    "date": normalize_date,
    "trip_type": normalize_trip_type,
    "count": normalize_count,
    "children": normalize_children,
    "duration": normalize_duration,
    "budget": normalize_budget,
    "date_flexibility": normalize_date_flexibility,
    "list": normalize_list,
    "accommodation": normalize_accommodation,
    "text": normalize_text,
}
