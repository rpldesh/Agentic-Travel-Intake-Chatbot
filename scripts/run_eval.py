"""Run the golden conversation set against the models configured in .env.

    uv run python scripts/run_eval.py [--only id1,id2] [--out report.json]

Uses a throwaway SQLite database per run. Reports entity precision/recall per
key, first-target accuracy, repeated-question rate, latency and model errors.
Run it for every role/model/prompt change before deploying (GPT and Gemini).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

from travel_bot.bootstrap import build_chat_service
from travel_bot.config import get_settings
from travel_bot.repositories.message_repository import MessageRepository
from travel_bot.services.question_planner import fingerprint

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden_conversations.json"


def _matches(expected: dict, state) -> bool:
    if "status" in expected and state.status.value != expected["status"]:
        return False
    if "status_in" in expected and state.status.value not in expected["status_in"]:
        return False
    for k, v in expected.get("value_subset", {}).items():
        actual = (state.value or {}).get(k) if isinstance(state.value, dict) else None
        if isinstance(v, str) and isinstance(actual, str):
            if v.lower() != actual.lower():
                return False
        elif actual != v:
            return False
    return True


async def run(only: set[str] | None, out: Path | None) -> int:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    ref = date.fromisoformat(golden["reference_date"])
    tmp = Path(tempfile.mkdtemp(prefix="travel_eval_"))
    settings = get_settings().model_copy(update={
        "database_url": f"sqlite+aiosqlite:///{(tmp / 'eval.db').as_posix()}",
        "checkpoint_db_path": str(tmp / "eval_ckpt.db"),
        "rate_limit_per_minute": 10_000,
    })

    tp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    results, repeats, questions, latencies, model_errors = [], 0, 0, [], 0

    async with build_chat_service(settings) as svc:
        svc.entity_service.reference_date = lambda: ref
        for conv in golden["conversations"]:
            if only and conv["id"] not in only:
                continue
            s = await svc.create_session(user_id="eval")
            replies, prints = [], []
            for turn in conv["turns"]:
                t0 = time.perf_counter()
                r = await svc.handle_turn(session_id=s.session_id, session_token=s.session_token,
                                          message_id=uuid.uuid4().hex, content=turn)
                latencies.append(time.perf_counter() - t0)
                replies.append(r)
                fpnt = fingerprint(r.reply)
                questions += 1
                repeats += fpnt in prints
                prints.append(fpnt)

            snap = await svc.entity_service.get_snapshot(s.session_id)
            failures = []
            for key, exp in conv["expect_entities"].items():
                if _matches(exp, snap.get(key)):
                    tp[key] += 1
                else:
                    fn[key] += 1
                    failures.append(f"{key}: expected {exp}, got "
                                    f"{snap.get(key).status.value} {snap.get(key).value}")
            for key, st in snap.entities.items():  # unexpected values (precision)
                if key not in conv["expect_entities"] and st.status.value == "known" \
                        and st.evidence_text != "[derived]" and conv["expect_entities"]:
                    fp[key] += 1
            if "expect_first_target" in conv and replies[0].target_entity_keys != conv["expect_first_target"]:
                failures.append(f"first target {replies[0].target_entity_keys} != {conv['expect_first_target']}")
            async with svc.db.session() as db:
                msgs = await MessageRepository(db).list_messages(s.session_id)
            for m in msgs:
                for call in (m.meta or {}).get("model_calls", []):
                    model_errors += not call["ok"]
            results.append({"id": conv["id"], "ok": not failures, "failures": failures,
                            "replies": [r.reply for r in replies]})
            print(("PASS " if not failures else "FAIL ") + conv["id"])
            for f in failures:
                print("     - " + f)

    keys = sorted(set(tp) | set(fn) | set(fp))
    per_key = {k: {"precision": round(tp[k] / (tp[k] + fp[k]), 2) if tp[k] + fp[k] else None,
                   "recall": round(tp[k] / (tp[k] + fn[k]), 2) if tp[k] + fn[k] else None}
               for k in keys}
    report = {
        "models": {r: settings.model_id_for(r) for r in
                   ("supervisor", "small_talk", "travel_entity", "question_generator")},
        "passed": sum(r["ok"] for r in results), "total": len(results),
        "repeated_question_rate": round(repeats / questions, 3) if questions else 0,
        "avg_turn_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "model_call_failures": model_errors,
        "per_entity": per_key, "conversations": results,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "conversations"}, indent=2))
    if out:
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sys.exit(asyncio.run(run({x for x in a.only.split(",") if x} or None, a.out)))
