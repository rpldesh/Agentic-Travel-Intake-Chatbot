"""Structured-output invocation with bounded retries, fallback and telemetry.

Output is re-validated in application code even when the provider enforces a
schema. Callers get `None` on failure and must use a safe deterministic fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from travel_bot.graph.progress import emit
from travel_bot.models.model_factory import RoleModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class CallRecord:
    role: str
    model_id: str
    ok: bool
    latency_ms: int
    attempts: int
    used_fallback: bool = False
    error: str | None = None
    prompt_version: str | None = None


@dataclass
class CallLog:
    """Per-turn record of model calls; ends up in the assistant message trace."""

    calls: list[CallRecord] = field(default_factory=list)

    def as_dicts(self) -> list[dict]:
        return [c.__dict__ for c in self.calls]


async def invoke_structured(
    role_model: RoleModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    retries: int = 1,
    prompt_version: str | None = None,
    call_log: CallLog | None = None,
) -> T | None:
    candidates = [(role_model.model_id, role_model.model, False)]
    if role_model.fallback is not None:
        candidates.append((role_model.fallback_id or "fallback", role_model.fallback, True))

    for model_id, model, is_fallback in candidates:
        runnable = model.with_structured_output(schema)
        emit({"type": "model_call", "status": "started", "role": role_model.role,
              "model_id": model_id, "fallback": is_fallback})
        attempts = 0
        start = time.perf_counter()
        last_error: str | None = None
        while attempts <= retries:
            attempts += 1
            try:
                raw = await runnable.ainvoke(messages)
                result = raw if isinstance(raw, schema) else schema.model_validate(raw)
                _record(call_log, role_model.role, model_id, True, start, attempts,
                        is_fallback, None, prompt_version)
                if is_fallback:
                    logger.warning("model fallback used", extra={"role": role_model.role,
                                                                 "model_id": model_id})
                return result
            except ValidationError as exc:
                last_error = f"invalid_output: {exc.error_count()} errors"
            except Exception as exc:  # provider/network/timeout errors
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        _record(call_log, role_model.role, model_id, False, start, attempts,
                is_fallback, last_error, prompt_version)
        logger.warning("structured call failed", extra={"role": role_model.role,
                                                        "model_id": model_id,
                                                        "error": last_error})
    return None


def _record(log, role, model_id, ok, start, attempts, fallback, error, prompt_version) -> None:
    latency_ms = int((time.perf_counter() - start) * 1000)
    emit({"type": "model_call", "status": "done" if ok else "failed", "role": role,
          "model_id": model_id, "ms": latency_ms, "attempts": attempts, "fallback": fallback})
    if log is None:
        return
    log.calls.append(CallRecord(
        role=role, model_id=model_id, ok=ok, latency_ms=latency_ms,
        attempts=attempts, used_fallback=fallback, error=error,
        prompt_version=prompt_version,
    ))
