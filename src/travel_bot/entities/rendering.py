"""Human-readable rendering of canonical entity values (for summaries/prompts)."""

from __future__ import annotations

from datetime import date
from typing import Any

from travel_bot.entities.catalog import EntityCatalog, EntityDefinition
from travel_bot.schemas.entities import EntitySnapshot, EntityState
from travel_bot.schemas.enums import EntityStatus


def _fmt_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def render_value(defn: EntityDefinition, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict) and "proposed" in value:
        cur = render_value(defn, value.get("current"))
        new = render_value(defn, value.get("proposed"))
        return f"{cur} or {new}? (to confirm)" if cur else f"{new}? (to confirm)"
    t = defn.type
    if t == "location":
        parts = [value.get("city"), value.get("region"), value.get("country")]
        text = ", ".join(p for p in parts if p)
        if value.get("airport_code"):
            text = f"{text} ({value['airport_code']})" if text else value["airport_code"]
        return text or value.get("description", "")
    if t == "date":
        if value.get("date"):
            return _fmt_date(value["date"])
        e, l = value.get("earliest"), value.get("latest")
        if e and l:
            return f"between {_fmt_date(e)} and {_fmt_date(l)}"
        return f"from {_fmt_date(e)}" if e else f"by {_fmt_date(l)}"
    if t == "enum":
        return str(value.get("type", "")).replace("_", " ").capitalize()
    if t == "count":
        return str(value.get("count"))
    if t == "children":
        count = value.get("count", 0)
        if not count:
            return "None"
        ages = value.get("ages")
        return f"{count} (ages {', '.join(map(str, ages))})" if ages else str(count)
    if t == "duration":
        nights = value.get("nights")
        return f"{nights} night{'s' if nights != 1 else ''}" if nights is not None else f"{value.get('days')} days"
    if t == "budget":
        cur = value.get("currency", "")
        if value.get("amount") is not None:
            amount = f"{cur} {value['amount']:,.0f}"
        else:
            lo, hi = value.get("min"), value.get("max")
            amount = (f"{cur} {lo:,.0f}–{hi:,.0f}" if lo is not None and hi is not None
                      else f"{cur} up to {hi:,.0f}" if hi is not None else f"{cur} from {lo:,.0f}")
        basis = value.get("basis")
        return f"{amount} {basis.replace('_', ' ')}" if basis else amount
    if t == "date_flexibility":
        mode = value.get("mode")
        if mode == "plus_minus_days":
            return f"± {value.get('days')} days"
        return str(mode).capitalize()
    if t == "list":
        return ", ".join(value.get("items", []))
    if t == "accommodation":
        bits = [value.get("type"), value.get("location"), value.get("notes")]
        if value.get("rooms"):
            bits.append(f"{value['rooms']} room(s)")
        if value.get("min_rating"):
            bits.append(f"{value['min_rating']}+ stars")
        return ", ".join(str(b) for b in bits if b)
    if t == "text":
        return value.get("text", "")
    return str(value)


def render_state(defn: EntityDefinition, state: EntityState) -> str:
    rendered = render_value(defn, state.value)
    if state.status == EntityStatus.FLEXIBLE:
        if defn.key == "destination" and rendered:
            return f"Open ({rendered})"
        return f"{rendered} (flexible)" if rendered else "Flexible / not decided yet"
    if state.status == EntityStatus.DECLINED:
        return "Prefer not to say"
    if state.status == EntityStatus.NOT_APPLICABLE:
        return "Not applicable"
    return rendered


def summary_lines(catalog: EntityCatalog, snapshot: EntitySnapshot) -> list[str]:
    lines = []
    for defn in catalog.entities:
        state = snapshot.get(defn.key)
        if state.status == EntityStatus.MISSING:
            continue
        if state.status == EntityStatus.NOT_APPLICABLE and defn.key == "return_date":
            continue
        if defn.key == "date_flexibility" and state.status == EntityStatus.KNOWN \
                and (state.value or {}).get("mode") == "flexible" \
                and snapshot.status_of("departure_date") == EntityStatus.FLEXIBLE:
            continue  # implied by the date range; avoid noise
        text = render_state(defn, state)
        if text:
            lines.append(f"- {defn.label}: {text}")
    return lines


def render_accepted(catalog: EntityCatalog, accepted: list[dict]) -> list[str]:
    """User-provided changes from this turn, rendered for an acknowledgement."""
    out = []
    for upd in accepted:
        if upd.get("actor") != "user_input":
            continue
        defn = catalog.get(upd["entity_key"])
        if defn is None:
            continue
        state = EntityState(entity_key=defn.key, value=upd.get("new_value"),
                            status=EntityStatus(upd["new_status"]))
        if state.status == EntityStatus.AMBIGUOUS:
            continue
        out.append(f"{defn.label}: {render_state(defn, state) or '(cleared)'}")
    return out
