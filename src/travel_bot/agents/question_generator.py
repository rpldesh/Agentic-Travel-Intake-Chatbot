"""Question Generator: turns a deterministic QuestionPlan into a user-facing message.

No write tools. Uses the refreshed canonical snapshot supplied by the graph.
Summaries are rendered deterministically so they can only contain saved facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from travel_bot.entities.catalog import EntityCatalog
from travel_bot.entities.rendering import summary_lines
from travel_bot.models.model_factory import RoleModel
from travel_bot.models.structured import CallLog, invoke_structured
from travel_bot.prompts import COMMON_RULES
from travel_bot.prompts import question_generator as prompt
from travel_bot.schemas.agents import QuestionGeneratorOutput
from travel_bot.schemas.entities import EntitySnapshot
from travel_bot.schemas.enums import CompletionStatus, ResponseType
from travel_bot.services.question_planner import QuestionPlan, fingerprint

MAX_QUESTION_CHARS = 500


@dataclass
class QuestionInput:
    plan: QuestionPlan
    snapshot: EntitySnapshot
    language: str
    user_message: str
    captured: list[str] = field(default_factory=list)
    previous_question: str | None = None
    recent_questions: list[str] = field(default_factory=list)
    prefix: str = ""


def strip_prefix(text: str, prefix: str) -> str:
    """Drop a repeat of the small-talk preface that the model echoed despite instructions."""
    if not prefix or not text:
        return text
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):].lstrip(" ,.!-—")
    # Also drop any leading sentences that each appear in the prefix (e.g. a repeated greeting).
    sentences = re.split(r"(?<=[.!?])\s+", text)
    prefix_fp = fingerprint(prefix)
    while len(sentences) > 1 and fingerprint(sentences[0]) and fingerprint(sentences[0]) in prefix_fp:
        sentences.pop(0)
    return " ".join(sentences)


class QuestionGenerator:
    def __init__(self, model: RoleModel, catalog: EntityCatalog, *, retries: int = 1) -> None:
        self.model = model
        self.catalog = catalog
        self.retries = retries

    async def generate(self, inp: QuestionInput, *, call_log: CallLog | None = None) -> QuestionGeneratorOutput:
        plan = inp.plan
        if plan.response_type == ResponseType.CONFIRM_SUMMARY:
            return await self._summary(inp, call_log, complete=False)
        if plan.response_type == ResponseType.INTAKE_COMPLETE:
            return await self._summary(inp, call_log, complete=True)

        result = await invoke_structured(
            self.model, QuestionGeneratorOutput, self._messages(inp),
            retries=self.retries, prompt_version=prompt.PROMPT_VERSION, call_log=call_log,
        )
        text = result.question_text.strip() if result else ""
        text = strip_prefix(text, inp.prefix)
        if not self._acceptable(text, inp):
            text = self.fallback_text(plan)
        return QuestionGeneratorOutput(
            response_type=plan.response_type,
            question_text=text,
            target_entity_keys=plan.target_keys,  # enforced: the plan decides targets
            completion_status=CompletionStatus.IN_PROGRESS,
        )

    def _messages(self, inp: QuestionInput):
        plan = inp.plan
        known = [ln[2:] for ln in summary_lines(self.catalog, inp.snapshot)]
        targets = "; ".join(
            f"{k} ({self.catalog.get(k).label}; e.g. \"{self.catalog.get(k).question_template}\")"
            for k in plan.target_keys
        )
        system = prompt.SYSTEM.format(common=COMMON_RULES, language=inp.language)
        user = prompt.USER.format(
            response_type=plan.response_type.value,
            targets=targets,
            issue=plan.issue or "(none)",
            previous=inp.previous_question if plan.times_asked else "(none)",
            offer_skip="yes" if plan.offer_skip else "no",
            captured="; ".join(inp.captured) or "(nothing)",
            known="; ".join(known) or "(nothing yet)",
            prefix=inp.prefix or "(none)",
            message=inp.user_message[:600],
        )
        return [SystemMessage(system), HumanMessage(user)]

    def _acceptable(self, text: str, inp: QuestionInput) -> bool:
        if not text or len(text) > MAX_QUESTION_CHARS or "?" not in text and "？" not in text:
            return False
        recent = {fingerprint(q) for q in [*inp.recent_questions, inp.previous_question or ""] if q}
        if fingerprint(text) in recent:
            return False  # verbatim repeat of a recent question
        return True

    def fallback_text(self, plan: QuestionPlan) -> str:
        """Deterministic phrasing used when the model is unavailable or unusable."""
        if not plan.target_keys:
            return "Could you tell me a bit more about your trip?"
        first = self.catalog.get(plan.target_keys[0])
        text = first.question_template
        if plan.issue:
            text = f"It looks like {plan.issue}. {text}"
        elif plan.response_type == ResponseType.CLARIFICATION:
            text = f"Just to make sure I have it right: {text[0].lower()}{text[1:]}"
        elif plan.times_asked:
            text = f"Sorry to ask again: {text[0].lower()}{text[1:]}"
        if plan.offer_skip:
            text += " It's fine to say you're not sure yet."
        return text

    async def _summary(self, inp: QuestionInput, call_log: CallLog | None, *, complete: bool) -> QuestionGeneratorOutput:
        lines = summary_lines(self.catalog, inp.snapshot)
        body = "\n".join(lines)
        if complete:
            text = ("Thanks! Your trip details are saved:\n" + body +
                    "\n\nIf anything changes, just tell me and I'll update it.")
            rtype, status = ResponseType.INTAKE_COMPLETE, CompletionStatus.COMPLETE
        else:
            text = ("Here's what I have so far:\n" + body +
                    "\n\nDoes everything look right? Let me know if you'd like to change anything.")
            rtype, status = ResponseType.CONFIRM_SUMMARY, CompletionStatus.AWAITING_CONFIRMATION
        if inp.language != "en":
            translated = await invoke_structured(
                self.model, QuestionGeneratorOutput,
                [SystemMessage(prompt.TRANSLATE_SYSTEM.format(language=inp.language,
                                                              response_type=rtype.value)),
                 HumanMessage(text)],
                retries=self.retries, prompt_version=prompt.PROMPT_VERSION, call_log=call_log,
            )
            if translated and translated.question_text.strip() \
                    and len(translated.question_text) < len(text) * 3:
                text = translated.question_text.strip()
        return QuestionGeneratorOutput(response_type=rtype, question_text=text,
                                       target_entity_keys=[], completion_status=status)
