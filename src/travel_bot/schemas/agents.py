"""Structured contracts for agents and guardrails."""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_bot.schemas.enums import (
    CompletionStatus,
    GuardrailOutcome,
    ResponseType,
    Route,
)


class SupervisorDecision(BaseModel):
    route: Route
    intent_label: str = Field(
        description="Short label, e.g. greeting, trip_details, correction, "
        "confirmation, off_topic, unclear."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(description="Short machine-readable reason.")
    contains_travel_details: bool = False


class SmallTalkOutput(BaseModel):
    reply_text: str = Field(
        description="Brief, warm reply. Do not ask trip questions; the next "
        "question is added separately."
    )


class QuestionGeneratorOutput(BaseModel):
    response_type: ResponseType = ResponseType.ASK_QUESTION
    question_text: str
    target_entity_keys: list[str] = Field(default_factory=list)
    completion_status: CompletionStatus = CompletionStatus.IN_PROGRESS


class GuardrailClassification(BaseModel):
    """Semantic guardrail classifier output (LLM)."""

    is_prompt_injection: bool = False
    is_unsafe_content: bool = False
    contains_trip_details: bool = False
    reason_code: str = "none"


class GuardrailResult(BaseModel):
    outcome: GuardrailOutcome
    reason_codes: list[str] = Field(default_factory=list)
    sanitized_text: str = ""
    detected_language: str = "en"
    notices: list[str] = Field(default_factory=list)
    removed_instruction_count: int = 0
