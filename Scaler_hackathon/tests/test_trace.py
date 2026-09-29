"""The trace must let us inspect one handoff end to end:
sender output -> receiver input -> resulting decision -> simulator response."""
import json

import pytest

from server import baseline
from server.energy_grid_environment import EnergyGridEnvironment
from server.trace import TraceRecorder, handoff_chain, load_trace
from tests.conftest import DEFAULT_REPLIES, FakeClient


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(baseline.time, "sleep", lambda *_: None)
    monkeypatch.setattr(baseline, "MAX_EVAL_STEPS", 2)


def traced_run(client=None, task="medium", path=None):
    client = client or FakeClient()
    trace = TraceRecorder(path)
    result = baseline.run_task(EnergyGridEnvironment(), client, "model-x", task, verbose=False, trace=trace)
    return client, trace, result


def calls(trace, **match):
    return [e for e in trace.events if e["type"] == "agent_call" and all(e.get(k) == v for k, v in match.items())]


def test_event_sequence_and_manifest():
    _, trace, result = traced_run()
    types = [e["type"] for e in trace.events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("env_step") == result["steps_completed"] == 2
    assert [e["seq"] for e in trace.events] == list(range(1, len(trace.events) + 1))
    m = trace.events[0]
    assert m["task_id"] == "medium" and m["protocol"] == "p1_free_text"
    assert m["models"]["dispatch"] == m["models"]["market"] == "model-x"
    assert isinstance(m["seed"], int) and m["git_sha"]
    assert "api_key" not in json.dumps(trace.events).lower()


def test_heterogeneity_is_derived_from_actual_model_ids(monkeypatch):
    monkeypatch.setattr(baseline, "PLANNING_MODEL", "model-x")
    _, trace, _ = traced_run()
    assert trace.events[0]["heterogeneous"] is False
    monkeypatch.setattr(baseline, "PLANNING_MODEL", "other-model")
    _, trace, _ = traced_run()
    assert trace.events[0]["heterogeneous"] is True


def test_every_llm_call_is_traced():
    client, trace, result = traced_run()
    assert len(calls(trace)) == len(client.calls) == result["llm_stats"]["calls"]


def test_exact_prompt_and_raw_response_recorded():
    client, trace, _ = traced_run()
    d1 = calls(trace, agent="dispatch", round="proposal", step=0)[0]
    assert d1["raw_response"] == DEFAULT_REPLIES["dispatch"]
    sent = [c for c in client.calls if c["role"] == "dispatch"][0]
    assert d1["user_prompt"] == sent["user"] and d1["system_prompt"] == sent["system"]
    assert d1["model"] == "model-x" and d1["prompt_tokens"] == 11 and d1["completion_tokens"] == 7


def test_revision_inputs_resolve_to_round1_calls_and_carry_their_content():
    _, trace, _ = traced_run()
    rev = calls(trace, agent="dispatch", round="revision", step=0)[0]
    by_id = {e["call_id"]: e for e in trace.events if e["type"] == "agent_call"}
    assert len(rev["inputs_from"]) == 3
    for cid in rev["inputs_from"]:
        assert by_id[cid]["step"] == 0 and by_id[cid]["round"] == "proposal"
    # the receiver was actually shown the sender's proposal
    sender = by_id[[c for c in rev["inputs_from"] if c.endswith(":dispatch")][0]]
    assert "NEGOTIATION" in rev["user_prompt"]
    assert f"coal_delta={sender['applied_action']['coal_delta']}" in rev["user_prompt"]


def test_env_step_records_submitted_actions_and_simulator_response():
    _, trace, _ = traced_run()
    chain = handoff_chain(trace.events, step=0)
    env_step = chain["env_step"]
    rev = [c for c in chain["calls"] if c["round"] == "revision" and c["agent"] == "dispatch"][0]
    assert env_step["submitted"]["dispatch"]["coal_delta"] == rev["applied_action"]["coal_delta"]
    assert isinstance(env_step["reward"], float)
    assert "frequency_hz" in env_step["observation"]
    assert len(chain["calls"]) == 5  # planning, dispatch, market, dispatch rev, market rev


def test_failed_calls_and_control_overrides_appear_in_trace():
    client = FakeClient(replies={"dispatch": RuntimeError("provider 503")})
    _, trace, _ = traced_run(client)
    bad = calls(trace, agent="dispatch", step=0, round="proposal")[0]
    assert bad["ok"] is False and bad["parse_status"] == "call_failed"
    assert "provider 503" in bad["call_error"] and bad["raw_response"] is None
    assert trace.events[-1]["valid_run"] is False

    client = FakeClient(replies={"dispatch": 'Action: {"coal_delta": 1, "plant_action": "build_solar"}'})
    _, trace, _ = traced_run(client)
    d = calls(trace, agent="dispatch", step=0, round="proposal")[0]
    assert d["control_changes"]["plant_action"] == ["build_solar", "none"]


def test_hard_task_plan_call_is_linked_to_agents_that_received_it():
    _, trace, _ = traced_run(task="hard")
    plan = calls(trace, round="plan")[0]
    d = calls(trace, agent="dispatch", round="proposal", step=0)[0]
    assert d["plan_call_id"] == plan["call_id"]
    assert plan["step"] is None


def test_jsonl_file_roundtrip(tmp_path):
    path = tmp_path / "run.jsonl"
    _, trace, _ = traced_run(path=path)
    assert load_trace(path) == json.loads(json.dumps(trace.events, default=str))


def test_unwritable_trace_path_does_not_break_the_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    _, trace, result = traced_run(path=blocker / "sub" / "run.jsonl")  # parent is a file -> mkdir fails
    assert result["steps_completed"] == 2
    assert len(trace.events) > 0  # still recorded in memory
