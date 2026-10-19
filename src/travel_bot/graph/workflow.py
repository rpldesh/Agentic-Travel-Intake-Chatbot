"""The explicit LangGraph workflow for one user turn.

    input_guardrail ─┬─(block / unsupported language)─> safe_reply ───────────────┐
                     └─> load_context ─> supervisor ─┬─> extract_entities         │
                                                     └─> small_talk ─┬─(clarify)──┤
                                                                     ├─> extract  │
                                                                     └─> question │
    extract_entities ─> apply_entity_updates ─> refresh_entities ─> question      │
    question_generator ─> output_validation <─────────────────────────────────────┘
    output_validation ─> persist_response ─> END

Every route is allow-listed by conditional edges; the graph is acyclic, so a
turn makes a bounded number of model calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from travel_bot.agents.question_generator import QuestionGenerator, QuestionInput
from travel_bot.agents.small_talk import SmallTalkAgent, mode_for
from travel_bot.agents.supervisor import Supervisor, SupervisorInput
from travel_bot.agents.travel_entity import ExtractionInput, TravelEntityAgent
from travel_bot.database.connection import Database
from travel_bot.entities.catalog import EntityCatalog
from travel_bot.entities.rendering import render_accepted
from travel_bot.entities.requirements import compute_missing
from travel_bot.errors import PersistenceError
from travel_bot.graph.state import TravelState, TurnContext
from travel_bot.guardrails.input_guardrails import InputGuardrail, safe_reply_for
from travel_bot.guardrails.output_guardrails import validate_output
from travel_bot.models.structured import CallLog
from travel_bot.repositories.message_repository import MessageRepository
from travel_bot.repositories.session_repository import SessionRepository
from travel_bot.schemas.agents import GuardrailResult
from travel_bot.schemas.entities import EntityExtraction, EntitySnapshot
from travel_bot.schemas.enums import (
    GuardrailOutcome,
    ResponseType,
    Route,
    SessionStatus,
)
from travel_bot.services.entity_service import EntityService
from travel_bot.services.question_planner import plan_next
from travel_bot.tools.entity_tools import EntityTools

logger = logging.getLogger(__name__)

MAX_AGENT_CALLS_PER_TURN = 6
MAX_STATE_MESSAGES = 40
QUESTION_HISTORY_LEN = 6
EXTRACTION_FAILED_PREFIX = "Sorry, I didn't quite catch all of that."
PERSISTENCE_ERROR_TEXT = ("Sorry, I had trouble saving that just now, so it didn't go through. "
                          "Could you repeat your last details?")


@dataclass
class GraphDeps:
    db: Database
    catalog: EntityCatalog
    entity_service: EntityService
    guardrail: InputGuardrail
    supervisor: Supervisor
    small_talk: SmallTalkAgent
    travel_entity: TravelEntityAgent
    question_generator: QuestionGenerator
    supported_languages: list[str]
    timezone: str = "UTC"
    history_window: int = 8
    ask_recommended: bool = True


def _last_assistant_text(state: TravelState) -> str | None:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return None


def _prior_messages(state: TravelState) -> list:
    """History excluding the current user message (the last one)."""
    return list(state.get("messages", []))[:-1]


def _text(state: TravelState) -> str:
    g = state.get("guardrail_result") or {}
    return g.get("sanitized_text") or state.get("user_input", "")


def _merge_calls(state: TravelState, log: CallLog) -> dict[str, Any]:
    return {
        "call_log": [*state.get("call_log", []), *log.as_dicts()],
        "agent_calls": state.get("agent_calls", 0) + len(log.calls),
    }


def _budget_left(state: TravelState) -> bool:
    return state.get("agent_calls", 0) < MAX_AGENT_CALLS_PER_TURN


def build_graph(deps: GraphDeps, checkpointer: Checkpointer | None = None):
    catalog = deps.catalog

    # ------------------------------------------------------------- guardrail
    async def input_guardrail(state: TravelState, runtime: Runtime[TurnContext]) -> dict:
        log = CallLog()
        async with deps.db.session() as s:
            session = await SessionRepository(s).get(runtime.context.session_id)
            session_language = session.preferred_language if session else None
        result = await deps.guardrail.check(
            state.get("user_input", ""), session_language=session_language, call_log=log
        )
        return {
            "guardrail_result": result.model_dump(mode="json"),
            "detected_language": result.detected_language,
            **_merge_calls(state, log),
        }

    def after_guardrail(state: TravelState) -> str:
        outcome = (state.get("guardrail_result") or {}).get("outcome")
        if outcome in (GuardrailOutcome.BLOCK_SAFE_REPLY, GuardrailOutcome.CLARIFY_LANGUAGE):
            return "safe_reply"
        return "load_context"

    async def safe_reply(state: TravelState) -> dict:
        result = GuardrailResult.model_validate(state["guardrail_result"])
        return {"assistant_response": {
            "response_type": ResponseType.SAFE_REDIRECT.value,
            "text": safe_reply_for(result, deps.supported_languages),
            "target_entity_keys": [],
        }}

    # --------------------------------------------------------------- context
    async def _load(session_id: str) -> tuple[EntitySnapshot, str]:
        snapshot = await deps.entity_service.get_snapshot(session_id)
        async with deps.db.session() as s:
            session = await SessionRepository(s).get(session_id)
        return snapshot, session.status if session else SessionStatus.ACTIVE.value

    def _snapshot_fields(snapshot: EntitySnapshot, status: str) -> dict:
        missing = compute_missing(catalog, snapshot)
        return {
            "entity_snapshot": snapshot.model_dump(mode="json"),
            "missing_required_entities": missing.missing_required,
            "missing_recommended_entities": missing.missing_recommended,
            "ambiguous_entities": missing.ambiguous,
            "required_complete": missing.required_complete,
            "session_status": status,
        }

    async def load_context(state: TravelState, runtime: Runtime[TurnContext]) -> dict:
        snapshot, status = await _load(runtime.context.session_id)
        cached = state.get("entity_snapshot")
        if cached and cached.get("version") not in (None, snapshot.version):
            # Checkpoint and DB diverged (e.g. an out-of-band write). The DB wins.
            logger.info("checkpoint snapshot refreshed from database",
                        extra={"session_id": runtime.context.session_id,
                               "checkpoint_version": cached.get("version"),
                               "db_version": snapshot.version})
        return _snapshot_fields(snapshot, status)

    # ------------------------------------------------------------ supervisor
    async def supervisor(state: TravelState) -> dict:
        log = CallLog()
        decision = await deps.supervisor.route(SupervisorInput(
            message=_text(state),
            session_status=state.get("session_status", "active"),
            known_keys=list((state.get("entity_snapshot") or {}).get("entities", {}).keys()),
            missing_keys=state.get("missing_required_entities", []),
            last_assistant=_last_assistant_text({"messages": _prior_messages(state)}),
            guardrail_outcome=(state.get("guardrail_result") or {}).get("outcome", "PASS"),
        ), call_log=log)
        return {"route": decision.model_dump(mode="json"), **_merge_calls(state, log)}

    def after_supervisor(state: TravelState) -> str:
        route = (state.get("route") or {}).get("route")
        return "extract_entities" if route == Route.TRAVEL_ENTITY else "small_talk"

    # ------------------------------------------------------------ small talk
    async def small_talk(state: TravelState) -> dict:
        route = state.get("route") or {}
        mode = mode_for(
            route.get("intent_label", ""),
            is_clarify=route.get("route") == Route.CLARIFY,
            session_complete=state.get("session_status") == SessionStatus.INTAKE_COMPLETE,
        )
        log = CallLog()
        text = await deps.small_talk.reply(
            message=_text(state), mode=mode, language=state.get("detected_language", "en"),
            last_assistant=_last_assistant_text({"messages": _prior_messages(state)}),
            call_log=log,
        )
        update: dict = {"small_talk_text": text, **_merge_calls(state, log)}
        if mode in ("clarify", "complete"):
            update["assistant_response"] = {
                "response_type": (ResponseType.CLARIFICATION if mode == "clarify"
                                  else ResponseType.INTAKE_COMPLETE).value,
                "text": text,
                "target_entity_keys": [],
            }
        return update

    def after_small_talk(state: TravelState) -> str:
        if state.get("assistant_response"):
            return "output_validation"
        if (state.get("route") or {}).get("contains_travel_details") and _budget_left(state):
            return "extract_entities"
        return "question_generator"

    # ------------------------------------------------------ entity extraction
    async def extract_entities(state: TravelState, runtime: Runtime[TurnContext]) -> dict:
        snapshot = EntitySnapshot.model_validate(state["entity_snapshot"])
        log = CallLog()
        guard = state.get("guardrail_result") or {}
        extraction = await deps.travel_entity.extract(ExtractionInput(
            message=_text(state),
            history=_prior_messages(state),
            snapshot=snapshot,
            reference_date=deps.entity_service.reference_date(),
            timezone=deps.timezone,
            last_target_keys=state.get("last_question_target_keys", []),
            last_response_type=state.get("last_response_type"),
            untrusted_instructions_present=guard.get("outcome")
            == GuardrailOutcome.PASS_WITH_UNTRUSTED_INSTRUCTIONS,
        ), call_log=log)
        return {
            "entity_candidates": extraction.model_dump(mode="json") if extraction else None,
            **_merge_calls(state, log),
        }

    async def apply_entity_updates(state: TravelState, runtime: Runtime[TurnContext]) -> dict:
        raw = state.get("entity_candidates")
        if not raw:
            return {"accepted_entity_updates": [], "rejected_entity_updates": []}
        extraction = EntityExtraction.model_validate(raw)
        tools = EntityTools(
            deps.entity_service, runtime.context.session_id,
            runtime.context.message_id, _text(state),
        )
        try:
            result = await tools.record_entity_updates(
                extraction.updates,
                expected_version=(state.get("entity_snapshot") or {}).get("version"),
            )
            if extraction.confirms_summary and not result.accepted \
                    and state.get("last_response_type") == ResponseType.CONFIRM_SUMMARY:
                await deps.entity_service.confirm_intake(
                    runtime.context.session_id, source_message_id=runtime.context.message_id
                )
        except PersistenceError:
            logger.error("entity persistence failed", extra={"session_id": runtime.context.session_id})
            return {"persistence_error": True, "accepted_entity_updates": [],
                    "rejected_entity_updates": []}
        return {
            "accepted_entity_updates": [a.model_dump(mode="json") for a in result.accepted],
            "rejected_entity_updates": [r.model_dump(mode="json") for r in result.rejected],
        }

    async def refresh_entities(state: TravelState, runtime: Runtime[TurnContext]) -> dict:
        snapshot, status = await _load(runtime.context.session_id)
        return _snapshot_fields(snapshot, status)

    # ------------------------------------------------------ question generator
    async def question_generator(state: TravelState) -> dict:
        if state.get("persistence_error"):
            return {"assistant_response": {
                "response_type": ResponseType.CLARIFICATION.value,
                "text": PERSISTENCE_ERROR_TEXT,
                "target_entity_keys": state.get("last_question_target_keys", []),
            }}

        snapshot = EntitySnapshot.model_validate(state["entity_snapshot"])
        missing = compute_missing(catalog, snapshot)
        complete = state.get("session_status") == SessionStatus.INTAKE_COMPLETE
        history = state.get("question_history", [])
        plan = plan_next(
            catalog, snapshot, missing,
            rejected=state.get("rejected_entity_updates", []),
            question_history=history,
            offered_keys=state.get("offered_keys", []),
            session_complete=complete,
            ask_recommended=deps.ask_recommended,
        )
        prefix = state.get("small_talk_text", "")
        if (state.get("route") or {}).get("route") == Route.TRAVEL_ENTITY                 and state.get("entity_candidates") is None and not prefix:
            prefix = EXTRACTION_FAILED_PREFIX  # model failed: don't pretend we understood
        just_completed = complete and state.get("last_response_type") != ResponseType.INTAKE_COMPLETE
        if complete and not just_completed and not state.get("accepted_entity_updates"):
            # Nothing new after completion: don't re-send the full summary.
            text = prefix or "Your trip details are saved. Just tell me if anything changes."
            return {"assistant_response": {"response_type": ResponseType.INTAKE_COMPLETE.value,
                                           "text": text, "target_entity_keys": []}}

        log = CallLog()
        if _budget_left(state):
            out = await deps.question_generator.generate(QuestionInput(
                plan=plan, snapshot=snapshot,
                language=state.get("detected_language", "en"),
                user_message=_text(state),
                captured=render_accepted(catalog, state.get("accepted_entity_updates", [])),
                previous_question=history[-1]["text"] if history else None,
                recent_questions=[h["text"] for h in history],
                prefix=prefix,
            ), call_log=log)
            text, rtype = out.question_text, out.response_type.value
        else:
            text, rtype = deps.question_generator.fallback_text(plan), plan.response_type.value

        if prefix:
            text = f"{prefix} {text}" if "\n" not in text else f"{prefix}\n\n{text}"
        return {
            "assistant_response": {"response_type": rtype, "text": text,
                                   "target_entity_keys": plan.target_keys},
            "offered_keys": [*state.get("offered_keys", []),
                             *[k for k in plan.newly_offered_keys
                               if k not in state.get("offered_keys", [])]],
            **_merge_calls(state, log),
        }

    # ------------------------------------------------------- output validation
    async def output_validation(state: TravelState) -> dict:
        response = dict(state.get("assistant_response") or {})
        guard = state.get("guardrail_result") or {}
        check = validate_output(response.get("text", ""),
                                persistence_failed=bool(state.get("persistence_error")))
        if not check.ok:
            logger.warning("output validation failed", extra={"reasons": check.reasons})
            response["text"] = _safe_fallback(state, deps)
            response["output_violations"] = check.reasons
        else:
            response["text"] = check.text
        notices = guard.get("notices") or []
        if notices:
            response["text"] = " ".join(notices) + "\n\n" + response["text"]
        return {"assistant_response": response}

    # -------------------------------------------------------------- persist
    async def persist_response(state: TravelState, runtime: Runtime[TurnContext]) -> dict:
        response = state.get("assistant_response") or {}
        text = response.get("text", "")
        rtype = response.get("response_type", ResponseType.ASK_QUESTION.value)
        targets = response.get("target_entity_keys", [])
        ctx = runtime.context
        trace = {
            "trace_id": ctx.trace_id,
            "response_type": rtype,
            "target_entity_keys": targets,
            "route": state.get("route"),
            "guardrail": {k: (state.get("guardrail_result") or {}).get(k)
                          for k in ("outcome", "reason_codes", "detected_language")},
            "accepted_keys": [a["entity_key"] for a in state.get("accepted_entity_updates", [])],
            "rejected": [{"entity_key": r["entity_key"], "reason_code": r["reason_code"]}
                         for r in state.get("rejected_entity_updates", [])],
            "persistence_error": bool(state.get("persistence_error")),
            "output_violations": response.get("output_violations", []),
            "model_calls": state.get("call_log", []),
        }
        async with deps.db.session() as s:
            await MessageRepository(s).add_assistant_message(
                ctx.session_id, ctx.message_id, text, meta=trace
            )
            await SessionRepository(s).touch(ctx.session_id,
                                              language=state.get("detected_language"))
            session = await SessionRepository(s).get(ctx.session_id)
            status = session.status if session else state.get("session_status")

        history = state.get("question_history", [])
        if rtype in (ResponseType.ASK_QUESTION, ResponseType.CLARIFICATION) and targets:
            history = [*history, {"target_keys": targets, "text": text}][-QUESTION_HISTORY_LEN:]

        messages = state.get("messages", [])
        removals = [RemoveMessage(id=m.id) for m in messages[: max(0, len(messages) + 1 - MAX_STATE_MESSAGES)]
                    if m.id]
        return {
            "messages": [*removals, AIMessage(content=text, id=f"{ctx.message_id}:assistant")],
            "last_question_target_keys": targets,
            "last_response_type": rtype,
            "question_history": history,
            "session_status": status,
        }

    # ---------------------------------------------------------------- wiring
    g = StateGraph(TravelState, context_schema=TurnContext)
    g.add_node("input_guardrail", input_guardrail)
    g.add_node("safe_reply", safe_reply)
    g.add_node("load_context", load_context)
    g.add_node("supervisor", supervisor)
    g.add_node("small_talk", small_talk)
    g.add_node("extract_entities", extract_entities)
    g.add_node("apply_entity_updates", apply_entity_updates)
    g.add_node("refresh_entities", refresh_entities)
    g.add_node("question_generator", question_generator)
    g.add_node("output_validation", output_validation)
    g.add_node("persist_response", persist_response)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", after_guardrail, ["safe_reply", "load_context"])
    g.add_edge("safe_reply", "output_validation")
    g.add_edge("load_context", "supervisor")
    g.add_conditional_edges("supervisor", after_supervisor, ["extract_entities", "small_talk"])
    g.add_conditional_edges("small_talk", after_small_talk,
                            ["output_validation", "extract_entities", "question_generator"])
    g.add_edge("extract_entities", "apply_entity_updates")
    g.add_edge("apply_entity_updates", "refresh_entities")
    g.add_edge("refresh_entities", "question_generator")
    g.add_edge("question_generator", "output_validation")
    g.add_edge("output_validation", "persist_response")
    g.add_edge("persist_response", END)
    return g.compile(checkpointer=checkpointer)


def _safe_fallback(state: TravelState, deps: GraphDeps) -> str:
    missing = state.get("missing_required_entities") or []
    if missing:
        return deps.catalog.get(missing[0]).question_template
    return "Could you tell me a bit more about your trip?"


def user_message(text: str, message_id: str) -> HumanMessage:
    return HumanMessage(content=text, id=message_id)
