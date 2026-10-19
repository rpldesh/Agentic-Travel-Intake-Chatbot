from enum import StrEnum


class EntityStatus(StrEnum):
    MISSING = "missing"
    KNOWN = "known"
    AMBIGUOUS = "ambiguous"
    FLEXIBLE = "flexible"
    NOT_APPLICABLE = "not_applicable"
    DECLINED = "declined"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    INTAKE_COMPLETE = "intake_complete"
    ABANDONED = "abandoned"
    EXPIRED = "expired"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChangeIntent(StrEnum):
    SET = "set"
    CORRECT = "correct"
    ADD = "add"
    CLEAR = "clear"


class ChangeActor(StrEnum):
    USER_INPUT = "user_input"
    SYSTEM_NORMALIZATION = "system_normalization"
    USER_CONFIRMATION = "user_confirmation"


class EntityTier(StrEnum):
    REQUIRED = "required"
    CONDITIONAL = "conditional"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class Route(StrEnum):
    SMALL_TALK = "SMALL_TALK"
    TRAVEL_ENTITY = "TRAVEL_ENTITY"
    CLARIFY = "CLARIFY"


class GuardrailOutcome(StrEnum):
    PASS = "PASS"
    PASS_WITH_UNTRUSTED_INSTRUCTIONS = "PASS_WITH_UNTRUSTED_INSTRUCTIONS"
    CLARIFY_LANGUAGE = "CLARIFY_LANGUAGE"
    BLOCK_SAFE_REPLY = "BLOCK_SAFE_REPLY"


class ResponseType(StrEnum):
    ASK_QUESTION = "ask_question"
    CONFIRM_SUMMARY = "confirm_summary"
    INTAKE_COMPLETE = "intake_complete"
    SAFE_REDIRECT = "safe_redirect"
    CLARIFICATION = "clarification"


class CompletionStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETE = "complete"
