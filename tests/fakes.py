"""Deterministic stand-ins for chat models, so graph tests need no API keys."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableLambda

from travel_bot.models.model_factory import RoleModel
from travel_bot.schemas.agents import (
    QuestionGeneratorOutput,
    SmallTalkOutput,
    SupervisorDecision,
)
from travel_bot.schemas.entities import EntityCandidate, EntityExtraction

Responder = Callable[[type, list], Any]


class FakeStructuredModel:
    """Implements just `with_structured_output(...).ainvoke(messages)`."""

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.calls: list[tuple[type, list]] = []

    def with_structured_output(self, schema, **_kwargs):
        async def run(messages):
            self.calls.append((schema, messages))
            out = self.responder(schema, messages)
            if isinstance(out, Exception):
                raise out
            return out

        return RunnableLambda(run)


_CURRENT = re.compile(r"<<<\n(.*?)\n>>>", re.DOTALL)
_TARGETS = re.compile(r"Target fields: (.*)")


def current_message(messages) -> str:
    m = _CURRENT.search(messages[-1].content)
    return m.group(1).strip() if m else messages[-1].content


def U(key, value_json=None, status="known", evidence=None, intent="set", confidence=0.95):
    """Shorthand for an extraction update; evidence defaults to the value's text."""
    return EntityCandidate(entity_key=key, value_json=value_json, status=status,
                           confidence=confidence, evidence_text=evidence or "",
                           change_intent=intent)


class ScriptedBrain:
    """Maps the current user message to a supervisor route + extraction.

    script: {message_text: (route, intent_label, [updates], extras)} where extras may
    contain contains_travel_details / confirms_summary.
    """

    def __init__(self, script: dict[str, tuple]) -> None:
        self.script = script
        self.fail: set[type] = set()
        self.seen: list[tuple[str, str]] = []

    def respond(self, schema, messages):
        if schema in self.fail:
            return TimeoutError("simulated model timeout")
        if schema is SupervisorDecision:
            text = current_message(messages)
            route, intent, _updates, extras = self._lookup(text)
            self.seen.append(("supervisor", text))
            return SupervisorDecision(
                route=route, intent_label=intent, confidence=0.9, reason_code="scripted",
                contains_travel_details=extras.get("contains_travel_details", route == "TRAVEL_ENTITY"),
            )
        if schema is EntityExtraction:
            text = current_message(messages)
            self.seen.append(("extract", text))
            _route, _intent, updates, extras = self._lookup(text)
            fixed = [u.model_copy(update={"evidence_text": u.evidence_text or text}) for u in updates]
            return EntityExtraction(updates=fixed,
                                    confirms_summary=extras.get("confirms_summary", False))
        if schema is SmallTalkOutput:
            return SmallTalkOutput(reply_text="Hello! Happy to help plan your trip.")
        if schema is QuestionGeneratorOutput:
            prompt = messages[-1].content
            m = _TARGETS.search(prompt)
            targets = m.group(1) if m else ""
            keys = re.findall(r"([\w.]+) \(", targets)
            return QuestionGeneratorOutput(question_text=f"Q[{','.join(keys)}]?", target_entity_keys=keys)
        raise AssertionError(f"unexpected schema {schema}")

    def _lookup(self, text):
        entry = self.script.get(text)
        if entry is None:
            return ("SMALL_TALK", "off_topic", [], {})
        route, intent, updates, *rest = entry
        return route, intent, updates, (rest[0] if rest else {})


def fake_models(brain: ScriptedBrain) -> dict[str, RoleModel]:
    roles = ("supervisor", "small_talk", "travel_entity", "question_generator", "guardrail_classifier")
    return {r: RoleModel(role=r, model_id=f"fake:{r}", model=FakeStructuredModel(brain.respond))
            for r in roles}
