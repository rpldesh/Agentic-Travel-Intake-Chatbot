"""Terminal chat for local development: `uv run travel-bot [--debug]`."""

from __future__ import annotations

import argparse
import asyncio
import uuid

from travel_bot.bootstrap import build_chat_service
from travel_bot.errors import ModelConfigurationError, TravelBotError
from travel_bot.schemas.api import ChatResponse


async def _chat(verbose: bool) -> None:
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
                r = await _run(svc, session, text, verbose=verbose)
            except TravelBotError as exc:
                print(f"[error] {exc}")
                continue
            print(f"bot> {r.reply}\n")
            if verbose:
                print(f"  [{r.response_type}; targets={r.target_entity_keys}; status={r.session_status}]\n")


async def _run(svc, session, text: str, *, verbose: bool) -> ChatResponse:
    """Run one turn, printing agent/tool/model progress live when verbose."""
    prepared = await svc.prepare_turn(session_id=session.session_id,
                                      session_token=session.session_token,
                                      message_id=uuid.uuid4().hex, content=text)
    if isinstance(prepared, ChatResponse):
        return prepared
    final = None
    async for ev in svc.run_turn(prepared):
        kind = ev["type"]
        if kind == "final":
            final = ChatResponse.model_validate(ev["response"])
            if verbose:
                print(f"  total {ev['t']} ms")
        elif not verbose:
            continue
        elif kind == "step" and ev["status"] != "started":
            print(f"  {_mark(ev)} {ev['agent']}: {ev['label']} ({ev['ms']} ms)")
        elif kind == "model_call" and ev["status"] != "started":
            print(f"      {_mark(ev)} model {ev['model_id']} [{ev['role']}] ({ev['ms']} ms)")
        elif kind == "tool" and ev["status"] != "started":
            print(f"      {_mark(ev)} tool {ev['name']}: {ev.get('summary', '')} ({ev['ms']} ms)")
        elif kind == "decision":
            print(f"      - {ev['detail']}")
    assert final is not None
    return final


def _mark(ev: dict) -> str:
    return "x" if ev.get("status") == "failed" else "+"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the travel intake bot.")
    parser.add_argument("--debug", action="store_true",
                        help="show live agent, tool and model activity with timings")
    args = parser.parse_args()
    try:
        asyncio.run(_chat(args.debug))
    except ModelConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None


if __name__ == "__main__":  # pragma: no cover
    main()
