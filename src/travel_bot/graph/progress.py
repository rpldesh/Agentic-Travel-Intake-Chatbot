"""Live progress events for the UI (which agent is working, tool and model calls).

Nodes emit through LangGraph's custom stream. Outside a streaming graph run
(e.g. unit tests calling an agent directly) `emit` is a no-op. Events carry
only names, codes, counts and timings, never message content or values.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from langgraph.config import get_stream_writer

# node -> (agent / component name, what it is doing)
NODE_LABELS: dict[str, tuple[str, str]] = {
    "input_guardrail": ("Input Guardrail", "Checking your message"),
    "safe_reply": ("Input Guardrail", "Preparing a safe reply"),
    "load_context": ("Context Loader", "Loading your trip details"),
    "supervisor": ("Supervisor", "Understanding your request"),
    "small_talk": ("Small Talk Agent", "Writing a reply"),
    "extract_entities": ("TravelEntityAgent", "Extracting trip details"),
    "apply_entity_updates": ("Entity Service", "Validating and saving details"),
    "refresh_entities": ("Context Loader", "Reloading saved trip details"),
    "question_generator": ("Question Generator", "Choosing the next question"),
    "output_validation": ("Output Guardrail", "Checking the reply"),
    "persist_response": ("Persistence", "Saving the conversation"),
}


def emit(event: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):
        return  # not inside a graph run (e.g. a LangChain tool invoked directly)
    writer(event)


@contextmanager
def tool_call(name: str, **fields: Any):
    """Emit tool start/end events. The body may set `result["summary"]`."""
    start = time.perf_counter()
    emit({"type": "tool", "name": name, "status": "started", **fields})
    result: dict[str, Any] = {}
    try:
        yield result
    except Exception as exc:
        emit({"type": "tool", "name": name, "status": "failed", "error": type(exc).__name__,
              "ms": int((time.perf_counter() - start) * 1000), **fields})
        raise
    emit({"type": "tool", "name": name, "status": "done",
          "ms": int((time.perf_counter() - start) * 1000), **fields, **result})
