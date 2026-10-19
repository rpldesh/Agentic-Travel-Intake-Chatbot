"""Terminal chat for local development: `uv run travel-bot`."""

from __future__ import annotations

import argparse
import asyncio
import uuid

from travel_bot.bootstrap import build_chat_service
from travel_bot.errors import ModelConfigurationError, TravelBotError


async def _chat(show_entities: bool) -> None:
    async with build_chat_service() as svc:
        session = await svc.create_session(user_id="cli-user")
        print("Travel intake chat. Type /entities to see saved details, /quit to exit.\n")
        while True:
            try:
                text = await asyncio.to_thread(input, "you> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip() in ("/quit", "/exit"):
                break
            if text.strip() == "/entities":
                for line in await svc.entity_summary(session.session_id) or ["(nothing yet)"]:
                    print("  " + line)
                continue
            try:
                r = await svc.handle_turn(session_id=session.session_id,
                                          session_token=session.session_token,
                                          message_id=uuid.uuid4().hex, content=text)
            except TravelBotError as exc:
                print(f"[error] {exc}")
                continue
            print(f"bot> {r.reply}\n")
            if show_entities:
                print(f"  [{r.response_type}; targets={r.target_entity_keys}; status={r.session_status}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the travel intake bot.")
    parser.add_argument("--debug", action="store_true", help="show response metadata")
    args = parser.parse_args()
    try:
        asyncio.run(_chat(args.debug))
    except ModelConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None


if __name__ == "__main__":  # pragma: no cover
    main()
