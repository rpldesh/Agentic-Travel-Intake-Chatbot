"""Small Talk Agent: greetings, social chat, graceful redirects. No tools."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from travel_bot.models.model_factory import RoleModel
from travel_bot.models.structured import CallLog, invoke_structured
from travel_bot.prompts import COMMON_RULES
from travel_bot.prompts import small_talk as prompt
from travel_bot.schemas.agents import SmallTalkOutput

FALLBACKS = {
    "greeting": "Hi! I'd be happy to help plan your trip.",
    "off_topic": "I'm best at helping plan trips.",
    "thanks": "You're welcome!",
    "clarify": "Sorry, I didn't quite follow. Could you tell me a bit more about what you have in mind?",
    "complete": "Your trip details are saved. Just tell me if anything changes.",
}

MAX_CHARS = 400


def mode_for(intent_label: str, *, is_clarify: bool, session_complete: bool) -> str:
    if is_clarify:
        return "clarify"
    if session_complete:
        return "complete"
    if intent_label in ("greeting", "thanks"):
        return intent_label
    return "off_topic"


class SmallTalkAgent:
    def __init__(self, model: RoleModel, *, retries: int = 1) -> None:
        self.model = model
        self.retries = retries

    async def reply(
        self,
        *,
        message: str,
        mode: str,
        language: str,
        last_assistant: str | None,
        call_log: CallLog | None = None,
    ) -> str:
        messages = [
            SystemMessage(prompt.SYSTEM.format(common=COMMON_RULES, language=language, mode=mode)),
            HumanMessage(prompt.USER.format(
                last_assistant=(last_assistant or "(none - first turn)")[:400], message=message,
            )),
        ]
        out = await invoke_structured(
            self.model, SmallTalkOutput, messages,
            retries=self.retries, prompt_version=prompt.PROMPT_VERSION, call_log=call_log,
        )
        text = out.reply_text.strip() if out else ""
        if not text or len(text) > MAX_CHARS:
            return FALLBACKS[mode]
        return text
