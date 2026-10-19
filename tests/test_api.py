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
