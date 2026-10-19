"""Pure merge/update logic: candidate updates + canonical snapshot -> planned changes.

No I/O here, so every rule (correction, ambiguity, flexibility, stale writes,
cross-field consistency) is unit-testable without a database or a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from travel_bot.entities.catalog import EntityCatalog
from travel_bot.entities.normalizers import (
    NORMALIZERS,
    NormalizationContext,
    NormalizationError,
    date_bounds,
    decode_value,
)
from travel_bot.schemas.entities import (
    EntityCandidate,
    EntitySnapshot,
    EntityState,
    RejectedUpdate,
)
from travel_bot.schemas.enums import ChangeActor, ChangeIntent, EntityStatus

DERIVED_EVIDENCE = "[derived]"


@dataclass
class PlannedChange:
    entity_key: str
    old_value: Any
    new_value: Any
    old_status: EntityStatus
    new_status: EntityStatus
    change_type: str  # create | update | correct | merge | conflict | clear | derive
    actor: ChangeActor
    confidence: float | None
    evidence_text: str | None


# ---------------------------------------------------------------------------
# Evidence check: the quote must come from the CURRENT message. This is what
# stops an older value found in history from silently overwriting a newer one.
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def evidence_in_message(evidence: str, message: str) -> bool:
    ev_tokens = _tokens(evidence or "")
    if not ev_tokens:
        return False
    msg_tokens = _tokens(message or "")
    if " ".join(ev_tokens) in " ".join(msg_tokens):
        return True
    msg_set = set(msg_tokens)
    overlap = sum(1 for t in ev_tokens if t in msg_set) / len(ev_tokens)
    return overlap >= 0.6


def _same(a: Any, b: Any) -> bool:
    return a == b


def _validate_extra(rule: str, value: dict[str, Any]) -> None:
    if rule == "party_size" and value.get("count", 0) < 1:
        raise NormalizationError("invalid_adult_count", "at least one adult is required")


def _normalize(
    catalog: EntityCatalog, key: str, value_json: str | None, ctx: NormalizationContext
) -> Any:
    defn = catalog.get(key)
    assert defn is not None
    raw = decode_value(value_json)
    if raw is None:
        return None
    value = NORMALIZERS[defn.normalizer](raw, ctx)
    _validate_extra(defn.validation_rule, value)
    return value


def resolve_candidates(
    candidates: list[EntityCandidate],
    snapshot: EntitySnapshot,
    catalog: EntityCatalog,
    ctx: NormalizationContext,
    *,
    current_message_text: str,
    min_confidence_known: float = 0.6,
) -> tuple[list[PlannedChange], list[RejectedUpdate]]:
    planned: list[PlannedChange] = []
    rejected: list[RejectedUpdate] = []

    # One candidate per key: keep the most confident.
    best: dict[str, EntityCandidate] = {}
    for cand in candidates:
        prev = best.get(cand.entity_key)
        if prev is None or cand.confidence > prev.confidence:
            if prev is not None:
                rejected.append(RejectedUpdate(entity_key=prev.entity_key, reason_code="duplicate_candidate"))
            best[cand.entity_key] = cand
        else:
            rejected.append(RejectedUpdate(entity_key=cand.entity_key, reason_code="duplicate_candidate"))

    for key, cand in best.items():
        defn = catalog.get(key)
        if defn is None:
            rejected.append(RejectedUpdate(entity_key=key, reason_code="unknown_entity"))
            continue
        if not evidence_in_message(cand.evidence_text, current_message_text):
            rejected.append(RejectedUpdate(
                entity_key=key, reason_code="evidence_not_in_current_message",
                detail=cand.evidence_text[:120],
            ))
            continue

        existing = snapshot.get(key)
        status = cand.status

        if cand.change_intent == ChangeIntent.CLEAR:
            if existing.status != EntityStatus.MISSING:
                planned.append(_change(existing, None, EntityStatus.MISSING, "clear", cand))
            continue

        if status == EntityStatus.MISSING or status not in defn.allowed_statuses:
            rejected.append(RejectedUpdate(
                entity_key=key, reason_code="status_not_allowed", detail=status.value
            ))
            continue

        value: Any = None
        if status in (EntityStatus.KNOWN, EntityStatus.FLEXIBLE, EntityStatus.AMBIGUOUS):
            try:
                value = _normalize(catalog, key, cand.value_json, ctx)
            except NormalizationError as err:
                if status == EntityStatus.AMBIGUOUS:
                    value = None  # ambiguity is recorded even without a usable value
                else:
                    rejected.append(RejectedUpdate(
                        entity_key=key, reason_code=err.reason_code, detail=err.detail
                    ))
                    continue
            if status == EntityStatus.KNOWN and value is None:
                rejected.append(RejectedUpdate(entity_key=key, reason_code="missing_value"))
                continue

        if status == EntityStatus.KNOWN and cand.confidence < min_confidence_known:
            status = EntityStatus.AMBIGUOUS  # weak clue: keep as a proposal, ask to confirm

        # --- merge with existing canonical state --------------------------
        if _same(existing.value, value) and existing.status == status:
            continue  # no-op: nothing new

        if defn.type == "list" and status == EntityStatus.KNOWN and value is not None \
                and existing.status == EntityStatus.KNOWN \
                and cand.change_intent in (ChangeIntent.ADD, ChangeIntent.SET):
            merged = {"items": _merge_items(existing.value, value)}
            if merged != existing.value:
                planned.append(_change(existing, merged, status, "merge", cand))
            continue

        if existing.status == EntityStatus.KNOWN and status == EntityStatus.KNOWN \
                and cand.change_intent != ChangeIntent.CORRECT:
            # Different value without explicit correction wording: don't guess.
            conflict = {"current": existing.value, "proposed": value}
            planned.append(_change(existing, conflict, EntityStatus.AMBIGUOUS, "conflict", cand))
            continue

        if existing.status == EntityStatus.AMBIGUOUS and isinstance(existing.value, dict) \
                and "current" in existing.value and status == EntityStatus.AMBIGUOUS:
            continue  # still unresolved; keep the original conflict record

        change_type = (
            "create" if existing.status == EntityStatus.MISSING
            else "correct" if cand.change_intent == ChangeIntent.CORRECT
            else "update"
        )
        planned.append(_change(existing, value, status, change_type, cand))

    return planned, rejected


def _merge_items(old: Any, new: dict[str, Any]) -> list[str]:
    items = list((old or {}).get("items", []))
    lower = {i.lower() for i in items}
    for item in new.get("items", []):
        if item.lower() not in lower:
            items.append(item)
            lower.add(item.lower())
    return items[:20]


def _change(
    existing: EntityState,
    value: Any,
    status: EntityStatus,
    change_type: str,
    cand: EntityCandidate,
) -> PlannedChange:
    return PlannedChange(
        entity_key=existing.entity_key,
        old_value=existing.value,
        new_value=value,
        old_status=existing.status,
        new_status=status,
        change_type=change_type,
        actor=ChangeActor.USER_INPUT,
        confidence=cand.confidence,
        evidence_text=cand.evidence_text[:500],
    )


# ---------------------------------------------------------------------------
# Cross-field consistency and deterministic derivations.
# ---------------------------------------------------------------------------
def apply_to_snapshot(snapshot: EntitySnapshot, changes: list[PlannedChange]) -> EntitySnapshot:
    working = snapshot.model_copy(deep=True)
    for ch in changes:
        working.entities[ch.entity_key] = EntityState(
            entity_key=ch.entity_key,
            value=ch.new_value,
            status=ch.new_status,
            confidence=ch.confidence,
            evidence_text=ch.evidence_text,
        )
    return working


def _derive(
    working: EntitySnapshot, key: str, value: Any, status: EntityStatus
) -> PlannedChange | None:
    existing = working.get(key)
    if existing.value == value and existing.status == status:
        return None
    return PlannedChange(
        entity_key=key,
        old_value=existing.value,
        new_value=value,
        old_status=existing.status,
        new_status=status,
        change_type="derive",
        actor=ChangeActor.SYSTEM_NORMALIZATION,
        confidence=1.0,
        evidence_text=DERIVED_EVIDENCE,
    )


def reconcile(
    snapshot: EntitySnapshot, changes: list[PlannedChange]
) -> tuple[list[PlannedChange], list[RejectedUpdate]]:
    """Validate cross-field consistency and add derived changes.

    Returns the final change list (user changes that survived + derivations).
    """
    rejected: list[RejectedUpdate] = []
    changed_keys = {c.entity_key for c in changes}

    # 1. Return must not precede departure. Reject whichever side just changed.
    working = apply_to_snapshot(snapshot, changes)
    dep_lo, _ = date_bounds(_date_value(working, "departure_date"))
    _, ret_hi = date_bounds(_date_value(working, "return_date"))
    if dep_lo and ret_hi and ret_hi < dep_lo:
        bad = "return_date" if "return_date" in changed_keys else "departure_date"
        changes = [c for c in changes if c.entity_key != bad]
        changed_keys.discard(bad)
        rejected.append(RejectedUpdate(
            entity_key=bad, reason_code="return_before_departure",
            detail=f"departure {dep_lo}, return {ret_hi}",
        ))
        working = apply_to_snapshot(snapshot, changes)

    derived: list[PlannedChange] = []

    def add(key: str, value: Any, status: EntityStatus) -> None:
        if key in changed_keys:
            return  # never override what the user just said
        ch = _derive(working, key, value, status)
        if ch:
            derived.append(ch)
            working.entities[key] = EntityState(entity_key=key, value=value, status=status,
                                                evidence_text=DERIVED_EVIDENCE)

    trip_type = (working.value_of("trip_type") or {}).get("type") \
        if working.status_of("trip_type") == EntityStatus.KNOWN else None

    # 2. One-way trips have no return date; round trips need one again.
    if trip_type == "one_way" and working.status_of("return_date") != EntityStatus.NOT_APPLICABLE:
        add("return_date", None, EntityStatus.NOT_APPLICABLE)
    if trip_type == "round_trip" and working.status_of("return_date") == EntityStatus.NOT_APPLICABLE:
        add("return_date", None, EntityStatus.MISSING)

    # 3. A stated return date implies a round trip.
    if trip_type is None and working.status_of("trip_type") == EntityStatus.MISSING \
            and working.status_of("return_date") in (EntityStatus.KNOWN, EntityStatus.FLEXIBLE):
        add("trip_type", {"type": "round_trip"}, EntityStatus.KNOWN)

    # 4. Exact dates determine the duration.
    dep = _date_value(working, "departure_date")
    ret = _date_value(working, "return_date")
    if dep and ret and dep.get("date") and ret.get("date") \
            and {"departure_date", "return_date"} & ({c.entity_key for c in changes}):
        nights = (date.fromisoformat(ret["date"]) - date.fromisoformat(dep["date"])).days
        duration = working.get("trip_duration")
        user_duration = duration.status == EntityStatus.KNOWN \
            and duration.evidence_text != DERIVED_EVIDENCE
        if not user_duration:
            add("trip_duration", {"nights": nights}, EntityStatus.KNOWN)
        elif (duration.value or {}).get("nights") != nights:
            ch = _derive(working, "trip_duration",
                         {"current": duration.value, "proposed": {"nights": nights}},
                         EntityStatus.AMBIGUOUS)
            if ch and "trip_duration" not in changed_keys:
                derived.append(ch)

    # 5. A date range means the dates are flexible.
    if dep and not dep.get("date") and working.status_of("date_flexibility") == EntityStatus.MISSING:
        add("date_flexibility", {"mode": "flexible"}, EntityStatus.KNOWN)

    return changes + derived, rejected


def _date_value(snapshot: EntitySnapshot, key: str) -> dict[str, Any] | None:
    state = snapshot.get(key)
    if state.status in (EntityStatus.KNOWN, EntityStatus.FLEXIBLE) and isinstance(state.value, dict) \
            and ("date" in state.value or "earliest" in state.value or "latest" in state.value):
        return state.value
    return None
