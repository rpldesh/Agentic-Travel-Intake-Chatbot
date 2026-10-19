import httpx
from asgi_lifespan import LifespanManager

from travel_bot.api.app import create_app

from tests.conftest import REF_DATE
from tests.fakes import U, ScriptedBrain, fake_models
from tests.test_conversations import settings_for


async def test_http_flow(tmp_path):
    brain = ScriptedBrain({
        "Hi": ("SMALL_TALK", "greeting", [], {"contains_travel_details": False}),
        "Rome": ("TRAVEL_ENTITY", "answer", [U("destination", '{"city": "Rome"}')]),
    })
    app = create_app(settings_for(tmp_path), fake_models(brain))
    async with LifespanManager(app):
        app.state.chat.entity_service.reference_date = lambda: REF_DATE
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            session = (await client.post("/sessions", json={})).json()
            sid, h = session["session_id"], {"X-Session-Token": session["session_token"]}

            r = await client.post(f"/sessions/{sid}/messages", headers=h,
                                  json={"message_id": "m1", "content": "Hi"})
            assert r.status_code == 200 and r.json()["target_entity_keys"] == ["destination"]
            r = await client.post(f"/sessions/{sid}/messages", headers=h,
                                  json={"message_id": "m2", "content": "Rome"})
            assert r.json()["target_entity_keys"] == ["origin"]

            ents = (await client.get(f"/sessions/{sid}/entities", headers=h)).json()
            dest = next(e for e in ents["entities"] if e["entity_key"] == "destination")
            assert dest["value"] == {"city": "Rome"} and "destination" not in ents["missing_required"]
            assert len((await client.get(f"/sessions/{sid}/messages", headers=h)).json()) == 4

            # authorization and validation
            assert (await client.get(f"/sessions/{sid}/entities")).status_code == 403
            assert (await client.get(f"/sessions/{sid}/entities",
                                     headers={"X-Session-Token": "nope"})).status_code == 403
            assert (await client.get("/sessions/missing/entities", headers=h)).status_code == 404
            r = await client.post(f"/sessions/{sid}/messages", headers=h,
                                  json={"message_id": "m3", "content": "  "})
            assert r.status_code == 422 and r.json()["reason_code"] == "empty_message"
            assert (await client.get("/")).status_code == 200


async def test_stream_reports_agents_tools_and_timings(tmp_path):
    import json

    brain = ScriptedBrain({"Rome": ("TRAVEL_ENTITY", "answer", [U("destination", '{"city": "Rome"}')])})
    app = create_app(settings_for(tmp_path), fake_models(brain))
    async with LifespanManager(app):
        app.state.chat.entity_service.reference_date = lambda: REF_DATE
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session = (await client.post("/sessions", json={})).json()
            sid, h = session["session_id"], {"X-Session-Token": session["session_token"]}
            r = await client.post(f"/sessions/{sid}/messages/stream", headers=h,
                                  json={"message_id": "m1", "content": "Rome"})
            assert r.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]

            # auth errors are raised before streaming starts
            bad = await client.post(f"/sessions/{sid}/messages/stream",
                                    headers={"X-Session-Token": "nope"},
                                    json={"message_id": "m2", "content": "Rome"})
            assert bad.status_code == 403

            # duplicate delivery streams only the original final reply
            dup = await client.post(f"/sessions/{sid}/messages/stream", headers=h,
                                    json={"message_id": "m1", "content": "Rome"})
            dup_events = [json.loads(l[6:]) for l in dup.text.splitlines() if l.startswith("data: ")]

    assert events[0]["type"] == "turn_started" and events[-1]["type"] == "final"
    done_agents = [e["agent"] for e in events if e["type"] == "step" and e["status"] == "done"]
    assert done_agents == ["Input Guardrail", "Context Loader", "Supervisor", "TravelEntityAgent",
                           "Entity Service", "Context Loader", "Question Generator",
                           "Output Guardrail", "Persistence"]
    assert all(isinstance(e["ms"], int) for e in events if e.get("status") == "done")
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)  # monotonic timeline
    record = next(e for e in events if e["type"] == "tool" and e["name"] == "record_entity_updates"
                  and e["status"] == "done")
    assert record["summary"] == "1 saved" and record["accepted_keys"] == ["destination"]
    roles = [e["role"] for e in events if e["type"] == "model_call" and e["status"] == "done"]
    assert roles == ["supervisor", "travel_entity", "question_generator"]
    assert any(e["type"] == "decision" and e["detail"].startswith("Route: TRAVEL_ENTITY") for e in events)
    assert events[-1]["response"]["target_entity_keys"] == ["origin"]
    # progress events never carry the user's message text or entity values
    assert "Rome" not in json.dumps([e for e in events if e["type"] != "final"])
    assert [e["type"] for e in dup_events] == ["final"] and dup_events[0]["response"]["duplicate"]
