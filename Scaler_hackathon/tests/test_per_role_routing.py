"""Each role must use its own client, model and call settings, and the trace must say so."""
import json

import pytest

from server import baseline
from server.energy_grid_environment import EnergyGridEnvironment
from server.model_config import ROLES, build_team, load_team_config
from server.trace import TraceRecorder
from tests.conftest import FakeClient

CANARY = "sk-canary-987654"


@pytest.fixture(autouse=True)
def one_step(monkeypatch):
    monkeypatch.setattr(baseline.time, "sleep", lambda *_: None)
    monkeypatch.setattr(baseline, "MAX_EVAL_STEPS", 1)


def make_team(extra_env=None):
    env = {"SCALER_PROVIDER": "nebius", "NEBIUS_API_KEY": CANARY,
           "SCALER_PLANNING_MODEL": "org/planner", "SCALER_DISPATCH_MODEL": "org/dispatcher",
           "SCALER_MARKET_MODEL": "org/marketer",
           "SCALER_PLANNING_MAX_TOKENS": "111", "SCALER_PLANNING_PLAN_MAX_TOKENS": "999", "SCALER_PLANNING_TEMPERATURE": "0.1",
           "SCALER_DISPATCH_MAX_TOKENS": "222", "SCALER_DISPATCH_TEMPERATURE": "0.5",
           "SCALER_MARKET_MAX_TOKENS": "333", "SCALER_MARKET_TEMPERATURE": "0.9", "SCALER_MARKET_TIMEOUT_S": "15",
           "SCALER_MARKET_BASE_URL": "https://market.example/v1", "SCALER_MARKET_API_KEY_ENV": "MARKET_KEY",
           "MARKET_KEY": "other-secret",
           "SCALER_PLANNING_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}',
           **(extra_env or {})}
    cfg = load_team_config(env=env)
    clients = {}

    def factory(base_url, key, timeout):
        clients[base_url] = FakeClient(base_url=base_url)
        return clients[base_url]

    return cfg, build_team(cfg, client_factory=factory, env=env), clients


def test_each_role_hits_its_own_client_with_its_own_settings():
    cfg, team, clients = make_team()
    nebius, market_host = clients["https://api.tokenfactory.nebius.com/v1/"], clients["https://market.example/v1"]
    baseline.run_task(EnergyGridEnvironment(), None, "ignored", "medium", verbose=False, team=team)

    assert {c["role"] for c in market_host.calls} == {"market"}
    assert {c["role"] for c in nebius.calls} == {"planning", "dispatch"}

    def kw(client, role):
        return [c for c in client.calls if c["role"] == role][0]

    d, p, m = kw(nebius, "dispatch"), kw(nebius, "planning"), kw(market_host, "market")
    assert (d["model"], d["kwargs"]["max_tokens"], d["kwargs"]["temperature"]) == ("org/dispatcher", 222, 0.5)
    assert (p["model"], p["kwargs"]["max_tokens"], p["kwargs"]["temperature"]) == ("org/planner", 111, 0.1)
    assert (m["model"], m["kwargs"]["max_tokens"], m["kwargs"]["temperature"]) == ("org/marketer", 333, 0.9)
    assert m["kwargs"]["timeout"] == 15 and d["kwargs"]["timeout"] == 60.0


def test_extra_body_is_sent_only_to_the_role_that_configures_it():
    _, team, clients = make_team()
    nebius = clients["https://api.tokenfactory.nebius.com/v1/"]
    baseline.run_task(EnergyGridEnvironment(), None, "x", "medium", verbose=False, team=team)
    planning = [c for c in nebius.calls if c["role"] == "planning"][0]["kwargs"]
    dispatch = [c for c in nebius.calls if c["role"] == "dispatch"][0]["kwargs"]
    assert planning["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "extra_body" not in dispatch


def test_hard_task_plan_call_uses_plan_budget_then_step_budget():
    _, team, clients = make_team()
    nebius = clients["https://api.tokenfactory.nebius.com/v1/"]
    baseline.run_task(EnergyGridEnvironment(), None, "x", "hard", verbose=False, team=team)
    planning_calls = [c for c in nebius.calls if c["role"] == "planning"]
    assert planning_calls[0]["kwargs"]["max_tokens"] == 999        # one-shot strategic plan
    assert all(c["kwargs"]["max_tokens"] == 111 for c in planning_calls[1:])


def test_legacy_runtime_sends_no_extra_request_fields():
    client = FakeClient()
    baseline.run_task(EnergyGridEnvironment(), client, "m", "medium", verbose=False)
    for c in client.calls:
        assert "extra_body" not in c["kwargs"] and "timeout" not in c["kwargs"]
        assert c["kwargs"]["temperature"] == 0.2 and c["kwargs"]["max_tokens"] in (256, 600)


def test_trace_manifest_records_per_role_identity_and_never_a_key():
    _, team, _ = make_team()
    trace = TraceRecorder()
    result = baseline.run_task(EnergyGridEnvironment(), None, "x", "medium", verbose=False, trace=trace, team=team)
    start = trace.events[0]
    assert start["heterogeneous"] is True
    assert start["models"] == {"planning": "org/planner", "dispatch": "org/dispatcher", "market": "org/marketer"}
    assert start["roles"]["market"]["provider_host"] == "market.example"
    assert start["roles"]["planning"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert start["roles"]["market"]["api_key_env"] == "MARKET_KEY"
    dumped = json.dumps(trace.events) + json.dumps(result, default=str)
    assert CANARY not in dumped and "other-secret" not in dumped
    calls = [e for e in trace.events if e["type"] == "agent_call"]
    assert {e["agent"]: e["model"] for e in calls} == {"planning": "org/planner", "dispatch": "org/dispatcher", "market": "org/marketer"}


def test_homogeneous_team_is_reported_as_homogeneous():
    _, team, _ = make_team({"SCALER_PLANNING_MODEL": "org/same", "SCALER_DISPATCH_MODEL": "org/same", "SCALER_MARKET_MODEL": "org/same"})
    trace = TraceRecorder()
    baseline.run_task(EnergyGridEnvironment(), None, "x", "medium", verbose=False, trace=trace, team=team)
    assert trace.events[0]["heterogeneous"] is False
