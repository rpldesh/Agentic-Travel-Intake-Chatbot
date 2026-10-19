"""Small helpers shared by agents."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def format_history(messages: list[BaseMessage], limit: int) -> str:
    lines = []
    for m in messages[-limit:]:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {text[:600]}")
    return "\n".join(lines) or "(no earlier turns)"


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
