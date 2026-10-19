"""Supervisor: a structured-output router over an allow-listed set of routes.

The graph's conditional edges perform the actual routing, so the model cannot
invoke tools or invent agents. It never writes entity state.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from travel_bot.models.model_factory import RoleModel
from travel_bot.models.structured import CallLog, invoke_structured
from travel_bot.prompts import COMMON_RULES
from travel_bot.prompts import supervisor as prompt
from travel_bot.schemas.agents import SupervisorDecision
from travel_bot.schemas.enums import Route

MIN_CONFIDENCE = 0.35


@dataclass
class SupervisorInput:
    message: str
    session_status: str
    known_keys: list[str]
    missing_keys: list[str]
    last_assistant: str | None
    guardrail_outcome: str


class Supervisor:
    def __init__(self, model: RoleModel, *, retries: int = 1) -> None:
        self.model = model
        self.retries = retries

    async def route(self, inp: SupervisorInput, *, call_log: CallLog | None = None) -> SupervisorDecision:
        messages = [
            SystemMessage(prompt.SYSTEM.format(common=COMMON_RULES)),
            HumanMessage(prompt.USER.format(
                session_status=inp.session_status,
                known_keys=", ".join(inp.known_keys) or "(nothing)",
                missing_keys=", ".join(inp.missing_keys) or "(nothing)",
                last_assistant=(inp.last_assistant or "(none - first turn)")[:400],
                guardrail=inp.guardrail_outcome,
                message=inp.message,
            )),
        ]
        decision = await invoke_structured(
            self.model, SupervisorDecision, messages,
            retries=self.retries, prompt_version=prompt.PROMPT_VERSION, call_log=call_log,
        )
        if decision is None:
            # Safe default: extraction is harmless on small talk (yields no updates),
            # and the question generator still produces a useful next question.
            return SupervisorDecision(route=Route.TRAVEL_ENTITY, intent_label="unknown",
                                      confidence=0.0, reason_code="router_unavailable",
                                      contains_travel_details=True)
        return self._apply_policy(decision)

    @staticmethod
    def _apply_policy(decision: SupervisorDecision) -> SupervisorDecision:
        if decision.route == Route.SMALL_TALK and decision.contains_travel_details:
            # Mixed message: extraction happens after the small-talk preface.
            return decision
        if decision.route != Route.CLARIFY and decision.confidence < MIN_CONFIDENCE \
                and not decision.contains_travel_details:
            return decision.model_copy(update={"route": Route.CLARIFY,
                                               "reason_code": "low_confidence"})
        return decision
