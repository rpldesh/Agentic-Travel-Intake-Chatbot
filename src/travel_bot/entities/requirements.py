"""Which entities are required / worth asking, given the canonical snapshot.

Rules are referenced by name from the catalog so the catalog stays declarative.
"""

from __future__ import annotations

from collections.abc import Callable

from travel_bot.entities.catalog import EntityCatalog, EntityDefinition
from travel_bot.schemas.entities import EntitySnapshot, MissingReport
from travel_bot.schemas.enums import EntityStatus, EntityTier

# Statuses that satisfy a required field. `declined` also counts as resolved so
# the bot never nags a user who chose not to answer; the summary shows it.
RESOLVED_STATUSES = {
    EntityStatus.KNOWN,
    EntityStatus.FLEXIBLE,
    EntityStatus.NOT_APPLICABLE,
    EntityStatus.DECLINED,
}

Rule = Callable[[EntitySnapshot], bool]


def _resolved(snapshot: EntitySnapshot, key: str) -> bool:
    return snapshot.status_of(key) in RESOLVED_STATUSES


def _trip_type(snapshot: EntitySnapshot) -> str | None:
    if snapshot.status_of("trip_type") != EntityStatus.KNOWN:
        return None
    value = snapshot.value_of("trip_type") or {}
    return value.get("type")


def _round_trip_needs_end(snapshot: EntitySnapshot) -> bool:
    return _trip_type(snapshot) == "round_trip" and not _resolved(snapshot, "trip_duration")


def _departure_exact(snapshot: EntitySnapshot) -> bool:
    if snapshot.status_of("departure_date") != EntityStatus.KNOWN:
        return False
    return bool((snapshot.value_of("departure_date") or {}).get("date"))


def _destination_flexible(snapshot: EntitySnapshot) -> bool:
    return snapshot.status_of("destination") == EntityStatus.FLEXIBLE


RULES: dict[str, Rule] = {
    "always": lambda _s: True,
    "never": lambda _s: False,
    "round_trip_needs_end": _round_trip_needs_end,
    "departure_exact": _departure_exact,
    "destination_flexible": _destination_flexible,
}


def _rule(name: str) -> Rule:
    try:
        return RULES[name]
    except KeyError as exc:  # catalog misconfiguration should fail loudly
        raise ValueError(f"unknown catalog rule: {name}") from exc


def is_required(defn: EntityDefinition, snapshot: EntitySnapshot) -> bool:
    if defn.tier in (EntityTier.REQUIRED, EntityTier.CONDITIONAL):
        return _rule(defn.required_rule)(snapshot)
    return False


def is_askable(defn: EntityDefinition, snapshot: EntitySnapshot) -> bool:
    """Non-required fields worth asking about once (recommended, or promoted optional)."""
    if is_required(defn, snapshot):
        return False
    if defn.tier == EntityTier.RECOMMENDED or defn.tier == EntityTier.OPTIONAL:
        return _rule(defn.ask_rule)(snapshot)
    return False


def compute_missing(catalog: EntityCatalog, snapshot: EntitySnapshot) -> MissingReport:
    report = MissingReport()
    for defn in catalog.entities:  # already priority-sorted
        status = snapshot.status_of(defn.key)
        if status == EntityStatus.AMBIGUOUS:
            report.ambiguous.append(defn.key)
            continue
        if status == EntityStatus.DECLINED:
            report.declined.append(defn.key)
        if _resolved(snapshot, defn.key):
            continue
        if is_required(defn, snapshot):
            report.missing_required.append(defn.key)
        elif is_askable(defn, snapshot):
            report.missing_recommended.append(defn.key)
    required_ambiguous = [
        k for k in report.ambiguous
        if (d := catalog.get(k)) and d.tier in (EntityTier.REQUIRED, EntityTier.CONDITIONAL)
    ]
    report.required_complete = not report.missing_required and not required_ambiguous
    return report
