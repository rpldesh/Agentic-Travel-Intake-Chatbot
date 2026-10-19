"""Deterministic patterns for guardrails. Kept conservative to limit false positives."""

import re

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|the|any)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|polic(y|ies)|directions?)\b",
    r"\b(reveal|show|print|repeat|output|display|tell me|give me|leak)\b.{0,40}\b(system|developer|hidden|initial|original)\b.{0,15}\b(prompt|message|instructions?)\b",
    r"\byour (system|developer|hidden) (prompt|instructions?)\b",
    r"\b(you are now|from now on you are|act as|pretend (to be|you are)|roleplay as)\b.{0,60}\b(dan|jailbr\w*|unrestricted|unfiltered|developer mode|without (rules|restrictions|filters))\b",
    r"\b(developer|god|admin|debug|sudo) mode\b",
    r"\bjailbr(eak|oken)\b",
    r"\bnew (system )?instructions?\s*:",
    r"\b(drop|truncate|delete from|alter)\s+table\b",
    r"\b(select|update)\b.{0,40}\bfrom\b.{0,40}\b(session|conversation|entity|user)s?\b",
    r"\b(other|another|all|every)\s+(users?|sessions?|customers?)'?s?\b.{0,40}\b(data|history|messages?|details|conversations?|entities)\b",
    r"\b(switch|change|use)\b.{0,20}\bsession[_ ]?id\b",
    r"\b(call|invoke|run|use)\b.{0,20}\b(tool|function)\b.{0,30}\b(record_entity_updates|get_session_entities|sql|shell|database)\b",
]

UNSAFE_PATTERNS = [
    r"\b(smuggl\w*|traffick\w*)\b.{0,40}\b(drugs?|people|humans?|weapons?|guns?|cocaine|heroin)\b",
    r"\b(bomb|explosives?|weapon|gun)\b.{0,40}\b(on|onto|through|past)\b.{0,20}\b(plane|flight|airport security|security)\b",
    r"\b(fake|forged|counterfeit)\b.{0,20}\b(passport|visa|id|documents?)\b",
    r"\b(evade|avoid|sneak past)\b.{0,20}\b(customs|border control|immigration)\b",
]

# Sensitive data that intake never needs: redact before storing / prompting.
CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
CVV = re.compile(r"\b(cvv|cvc|security code)\b\D{0,10}\d{3,4}\b", re.IGNORECASE)
PASSPORT = re.compile(r"\bpassport\b(?:\s*(?:number|no\.?|#))?\s*(?:is|:)?\s*([A-Z0-9]{6,9})\b", re.IGNORECASE)
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Output checks.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
]
INTERNAL_LEAK = re.compile(
    r"(TRAVEL_ENTITY|SMALL_TALK|reason_code|value_json|entity_key|target_entity_keys|"
    r"PROMPT_VERSION|system prompt|my instructions|developer message|\{common\})",
    re.IGNORECASE,
)
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(I(?:'ve| have)? (?:booked|reserved|purchased|paid)|"
    r"(?:your )?(?:booking|reservation) (?:is|has been) (?:confirmed|made|complete)|"
    r"I (?:found|searched|checked|looked up) (?:\w+ ){0,2}(?:flights?|hotels?|fares?|prices?|availability|deals?)|"
    r"confirmation (?:number|code) (?:is|:)|ticket number)\b",
    re.IGNORECASE,
)


def compile_all(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


INJECTION_RES = compile_all(INJECTION_PATTERNS)
UNSAFE_RES = compile_all(UNSAFE_PATTERNS)


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0
