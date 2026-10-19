import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from travel_bot.agents.question_generator import QuestionGenerator, QuestionInput
from travel_bot.config import AGENT_ROLES, Settings
from travel_bot.entities.requirements import compute_missing
from travel_bot.errors import ModelConfigurationError
from travel_bot.models.model_factory import RoleModel, build_role_models
from travel_bot.models.structured import CallLog, invoke_structured
from travel_bot.schemas.agents import QuestionGeneratorOutput, SupervisorDecision
from travel_bot.schemas.entities import EntitySnapshot, EntityState
from travel_bot.schemas.enums import EntityStatus, ResponseType
from travel_bot.services.question_planner import plan_next

from tests.fakes import FakeStructuredModel


def snap(**kv):
    return EntitySnapshot(session_id="s", version=1, schema_version="v1", entities={
        k.replace("__", "."): EntityState(entity_key=k.replace("__", "."), value=v, status=s)
        for k, (v, s) in kv.items()})


def plan(catalog, snapshot, **kw):
    args = dict(rejected=[], question_history=[], offered_keys=[], session_complete=False)
    args.update(kw)
    return plan_next(catalog, snapshot, compute_missing(catalog, snapshot), **args)


K = EntityStatus.KNOWN


def test_planner_order_and_combinations(catalog):
    assert plan(catalog, snap()).target_keys == ["destination"]
    s = snap(destination=({"city": "Rome"}, K), origin=({"city": "Austin"}, K),
             departure_date=({"date": "2027-05-03"}, K))
    p = plan(catalog, s)
    assert p.target_keys == ["trip_type", "return_date"]  # natural pair
    s.entities["trip_type"] = EntityState(entity_key="trip_type", value={"type": "one_way"}, status=K)
    s.entities["return_date"] = EntityState(entity_key="return_date", status=EntityStatus.NOT_APPLICABLE)
    assert plan(catalog, s).target_keys == ["travelers.adults", "travelers.children"]


def test_planner_prioritises_rejections_then_ambiguity(catalog):
    s = snap(origin=({"current": {"city": "Dallas"}, "proposed": {"city": "Houston"}}, EntityStatus.AMBIGUOUS))
    p = plan(catalog, s, rejected=[{"entity_key": "departure_date", "reason_code": "past_date"}])
    assert p.response_type == ResponseType.CLARIFICATION and p.target_keys == ["departure_date"]
    assert "past" in p.issue
    p = plan(catalog, s)
    assert p.response_type == ResponseType.CLARIFICATION and p.target_keys == ["origin"]


def test_planner_repeat_tracking_and_completion(catalog):
    history = [{"target_keys": ["destination"], "text": "Where to?"}] * 2
    p = plan(catalog, snap(), question_history=history)
    assert p.times_asked == 2 and p.offer_skip

    done = snap(destination=({"city": "Rome"}, K), origin=({"city": "Austin"}, K),
                departure_date=({"date": "2027-05-03"}, K), trip_type=({"type": "one_way"}, K),
                return_date=(None, EntityStatus.NOT_APPLICABLE), travelers__adults=({"count": 1}, K))
    offered = ["travelers.children", "budget", "date_flexibility"]
    assert plan(catalog, done, offered_keys=offered).response_type == ResponseType.CONFIRM_SUMMARY
    assert plan(catalog, done, offered_keys=offered, session_complete=True).response_type == \
        ResponseType.INTAKE_COMPLETE
    assert plan(catalog, done).target_keys == ["travelers.children"]


async def test_question_generator_enforces_plan_targets_and_falls_back(catalog):
    outputs = iter([
        QuestionGeneratorOutput(question_text="And how about your budget?", target_entity_keys=["budget"]),
        QuestionGeneratorOutput(question_text="no question mark here", target_entity_keys=[]),
    ])
    gen = QuestionGenerator(RoleModel("question_generator", "fake:q",
                                      FakeStructuredModel(lambda s, m: next(outputs))), catalog)
    p = plan(catalog, snap())
    out = await gen.generate(QuestionInput(plan=p, snapshot=snap(), language="en", user_message="hi"))
    assert out.target_entity_keys == ["destination"]  # model's keys ignored
    out = await gen.generate(QuestionInput(plan=p, snapshot=snap(), language="en", user_message="hi"))
    assert out.question_text == "Where are you thinking of going?"


async def test_summary_only_contains_saved_facts(catalog):
    gen = QuestionGenerator(RoleModel("q", "fake", FakeStructuredModel(lambda s, m: None)), catalog)
    s = snap(destination=({"city": "Rome", "country": "Italy"}, K), budget=(None, EntityStatus.DECLINED))
    p = plan(catalog, s, offered_keys=["travelers.children", "budget"])
    p.response_type = ResponseType.CONFIRM_SUMMARY
    out = await gen.generate(QuestionInput(plan=p, snapshot=s, language="en", user_message="x"))
    assert "- Destination: Rome, Italy" in out.question_text
    assert "- Budget: Prefer not to say" in out.question_text
    assert "Origin" not in out.question_text


# ---- model factory / provider abstraction ---------------------------------
def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_every_role_can_use_gpt_or_gemini():
    gpt = _settings(openai_api_key=SecretStr("sk-test"),
                    **{f"{r}_model": "openai:gpt-5.5" for r in AGENT_ROLES})
    gem = _settings(google_api_key=SecretStr("g-test"),
                    **{f"{r}_model": "google_genai:gemini-3.1-pro-preview" for r in AGENT_ROLES})
    for role, m in build_role_models(gpt).items():
        assert isinstance(m.model, ChatOpenAI) and m.model_id == "openai:gpt-5.5"
    for role, m in build_role_models(gem).items():
        assert isinstance(m.model, ChatGoogleGenerativeAI)


def test_mixed_providers_and_fallback():
    s = _settings(openai_api_key=SecretStr("sk"), google_api_key=SecretStr("g"),
                  supervisor_model="google_genai:gemini-3.1-pro-preview",
                  supervisor_fallback_model="openai:gpt-5.5",
                  travel_entity_model="google_genai:gemini-3.1-pro-preview",
                  travel_entity_temperature=0.0)
    models = build_role_models(s)
    assert isinstance(models["supervisor"].model, ChatGoogleGenerativeAI)
    assert isinstance(models["supervisor"].fallback, ChatOpenAI)
    assert models["travel_entity"].model.temperature == 0.0


@pytest.mark.parametrize("kw,msg", [
    ({"supervisor_model": "gpt-5.5"}, "provider-qualified"),
    ({"supervisor_model": "anthropic:x"}, "unsupported provider"),
    ({}, "OPENAI_API_KEY"),
])
def test_model_config_errors(kw, msg, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelConfigurationError, match=msg):
        build_role_models(_settings(**kw))


async def test_structured_invoke_retries_then_fallback():
    calls = {"n": 0}

    def flaky(schema, messages):
        calls["n"] += 1
        return TimeoutError("slow")

    good = SupervisorDecision(route="SMALL_TALK", intent_label="greeting", confidence=1, reason_code="ok")
    rm = RoleModel("supervisor", "fake:primary", FakeStructuredModel(flaky),
                   fallback_id="fake:fallback", fallback=FakeStructuredModel(lambda s, m: good))
    log = CallLog()
    out = await invoke_structured(rm, SupervisorDecision, [], retries=1, call_log=log)
    assert out == good and calls["n"] == 2  # one retry on primary
    assert [c.ok for c in log.calls] == [False, True] and log.calls[1].used_fallback

    bad = RoleModel("supervisor", "fake", FakeStructuredModel(lambda s, m: {"route": "BOOK_FLIGHT"}))
    assert await invoke_structured(bad, SupervisorDecision, [], retries=0) is None  # off-allow-list


def test_echoed_small_talk_prefix_is_removed():
    from travel_bot.agents.question_generator import strip_prefix

    prefix = "Hi there! I’d be happy to help plan your trip."
    assert strip_prefix(prefix + " Where are you thinking of going?", prefix) == "Where are you thinking of going?"
    assert strip_prefix("Hi there! Where are you thinking of going?", prefix) == "Where are you thinking of going?"
    assert strip_prefix("Where to?", "") == "Where to?"
    assert strip_prefix("Where to?", prefix) == "Where to?"
