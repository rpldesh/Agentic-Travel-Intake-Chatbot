"""Deterministic question policy: WHAT to ask next. The LLM only phrases it.

Order of precedence:
  1. Problems with values the user gave this turn (e.g. a past date).
  2. Ambiguous entities (required first).
  3. Missing required entities (by priority, combined with `ask_with` partners).
  4. Recommended entities not yet offered (each offered once).
  5. Summary confirmation, then completion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from travel_bot.entities.catalog import EntityCatalog
from travel_bot.entities.requirements import RESOLVED_STATUSES
from travel_bot.schemas.entities import EntitySnapshot, MissingReport
from travel_bot.schemas.enums import EntityTier, ResponseType

MAX_TARGETS = 3

REJECTION_HINTS = {
    "past_date": "that date seems to be in the past",
    "date_too_far": "that date is very far in the future",
    "invalid_date": "I couldn't understand that date",
    "invalid_range": "the date range looked reversed",
    "return_before_departure": "the return date came before the departure date",
    "party_too_large": "that's a very large group",
    "invalid_adult_count": "at least one adult needs to be traveling",
    "invalid_currency": "I didn't recognize the currency",
    "invalid_child_age": "a child's age didn't look right",
}


@dataclass
class QuestionPlan:
    response_type: ResponseType
    target_keys: list[str] = field(default_factory=list)
    issue: str | None = None          # explanation for a clarification
    times_asked: int = 0              # consecutive prior asks of these targets
    offer_skip: bool = False          # invite "not sure yet"
    newly_offered_keys: list[str] = field(default_factory=list)


def fingerprint(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _consecutive_asks(history: list[dict], targets: list[str]) -> int:
    count = 0
    for item in reversed(history):
        if item.get("target_keys") == targets:
            count += 1
        else:
            break
    return count


def plan_next(
    catalog: EntityCatalog,
    snapshot: EntitySnapshot,
    missing: MissingReport,
    *,
    rejected: list[dict],
    question_history: list[dict],
    offered_keys: list[str],
    session_complete: bool,
    ask_recommended: bool = True,
) -> QuestionPlan:
    # 1. Values we just refused: explain and re-ask (one at a time).
    for rej in rejected:
        key = rej.get("entity_key")
        hint = REJECTION_HINTS.get(rej.get("reason_code", ""))
        if key in catalog and hint and snapshot.status_of(key) not in RESOLVED_STATUSES:
            return QuestionPlan(ResponseType.CLARIFICATION, [key], issue=hint,
                                times_asked=_consecutive_asks(question_history, [key]))

    # 2. Ambiguities before anything optional.
    if missing.ambiguous:
        required_first = sorted(
            missing.ambiguous,
            key=lambda k: (catalog.get(k).tier not in (EntityTier.REQUIRED, EntityTier.CONDITIONAL),
                           catalog.get(k).priority),
        )
        key = required_first[0]
        n = _consecutive_asks(question_history, [key])
        return QuestionPlan(ResponseType.CLARIFICATION, [key], times_asked=n, offer_skip=n >= 2)

    # 3. Missing required. A promoted preference question that outranks the next
    #    required field (e.g. interests when the destination is open) goes first.
    if missing.missing_required:
        first = missing.missing_required[0]
        for key in missing.missing_recommended:
            if catalog.get(key).priority < catalog.get(first).priority and key not in offered_keys:
                return QuestionPlan(ResponseType.ASK_QUESTION, [key], newly_offered_keys=[key])
        targets = [first]
        for partner in catalog.get(first).ask_with:
            if len(targets) >= MAX_TARGETS:
                break
            unresolved = snapshot.status_of(partner) not in RESOLVED_STATUSES
            if unresolved and (partner in missing.missing_required
                               or partner in missing.missing_recommended
                               or catalog.get(partner).tier in (EntityTier.CONDITIONAL,
                                                                EntityTier.OPTIONAL)):
                targets.append(partner)
        n = _consecutive_asks(question_history, targets)
        newly = [t for t in targets if t not in missing.missing_required and t not in offered_keys]
        return QuestionPlan(ResponseType.ASK_QUESTION, targets, times_asked=n,
                            offer_skip=n >= 2, newly_offered_keys=newly)

    # 4. Recommended, offered once each.
    if ask_recommended and not session_complete:
        for key in missing.missing_recommended:
            if key not in offered_keys:
                return QuestionPlan(ResponseType.ASK_QUESTION, [key], newly_offered_keys=[key])

    # 5. Summary / completion.
    if session_complete:
        return QuestionPlan(ResponseType.INTAKE_COMPLETE)
    return QuestionPlan(ResponseType.CONFIRM_SUMMARY)
