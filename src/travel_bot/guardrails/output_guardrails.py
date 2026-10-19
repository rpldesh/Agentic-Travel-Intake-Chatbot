"""Post-agent output validation: schema, leakage, unsupported claims, false saves."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from travel_bot.guardrails.input_guardrails import redact_sensitive
from travel_bot.guardrails.patterns import INTERNAL_LEAK, SECRET_PATTERNS, UNSUPPORTED_CLAIMS

MAX_REPLY_CHARS = 2500
_SAVED_CLAIM = re.compile(r"\b(saved|stored|recorded|noted down|updated)\b", re.IGNORECASE)


@dataclass
class OutputCheck:
    ok: bool
    text: str
    reasons: list[str] = field(default_factory=list)


def validate_output(text: str, *, persistence_failed: bool = False) -> OutputCheck:
    reasons: list[str] = []
    text = (text or "").strip()
    if not text:
        return OutputCheck(False, text, ["empty_reply"])
    if len(text) > MAX_REPLY_CHARS:
        reasons.append("reply_too_long")
    if any(p.search(text) for p in SECRET_PATTERNS):
        reasons.append("secret_leak")
    if INTERNAL_LEAK.search(text):
        reasons.append("internal_leak")
    if UNSUPPORTED_CLAIMS.search(text):
        reasons.append("unsupported_claim")
    if persistence_failed and _SAVED_CLAIM.search(text):
        reasons.append("false_save_claim")
    redacted, redactions = redact_sensitive(text)
    if redactions:
        text = redacted
    return OutputCheck(not reasons, text, reasons)
