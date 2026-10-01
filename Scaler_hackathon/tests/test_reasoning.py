"""Reasoning models (e.g. Nemotron on Token Factory) may return empty `content` with the
text in `reasoning_content`, or wrap thinking in <think> tags. These must be handled
explicitly: parsed when an answer exists, flagged (never silently idle) when it does not."""
import pytest

from server import baseline
from server.energy_grid_environment import EnergyGridEnvironment
from server.llm_adapter import split_reasoning
from server.trace import TraceRecorder
from tests.conftest import FakeClient

ACTION = '{"coal_delta": 7, "battery_mode": "idle"}'


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(baseline.time, "sleep", lambda *_: None)
    monkeypatch.setattr(baseline, "MAX_EVAL_STEPS", 1)


@pytest.mark.parametrize("text, expected", [
    (None, ("", None, False)),
    ("", ("", None, False)),
    ("plain answer", ("plain answer", None, False)),
    ("<think>pondering</think>answer", ("answer", "pondering", False)),
    ("pondering</think>answer", ("answer", "pondering", False)),          # opening tag omitted
    ("<think>never finished", ("", "never finished", True)),
    ("pre<think>never finished", ("pre", "never finished", True)),
])
def test_split_reasoning(text, expected):
    assert split_reasoning(text) == expected


def invoke(reply):
    env = EnergyGridEnvironment()
    obs = env.reset("medium")
    client = FakeClient(replies={"dispatch": reply})
    stats = baseline._new_stats()
    trace = TraceRecorder()
    res = baseline._invoke_agent(client, "m", "dispatch", "You are the Dispatch Agent.", "state", obs, stats,
                                 verbose=False, trace=trace, run_id="r", step=0)
    return res, stats, trace


def test_think_tags_are_stripped_before_parsing_but_kept_in_trace():
    res, stats, trace = invoke(f"<think>maybe more coal?</think>Action: {ACTION}")
    assert res.status == "ok" and res.action.coal_delta == 7
    assert res.reasoning == "maybe more coal?"
    ev = trace.events[0]
    assert ev["raw_response"].startswith("<think>") and ev["reasoning"] == "maybe more coal?"
    assert stats["parse_failures"] == 0


def test_reasoning_content_field_is_captured_when_answer_present():
    res, _, trace = invoke({"content": f"Action: {ACTION}", "reasoning_content": "step by step"})
    assert res.status == "ok" and res.action.coal_delta == 7 and trace.events[0]["reasoning"] == "step by step"


def test_budget_exhausted_by_reasoning_is_flagged_not_silent():
    res, stats, trace = invoke({"content": None, "reasoning_content": "thinking... thinking...", "finish_reason": "length"})
    assert res.status == "reasoning_truncated" and res.ok is False
    assert res.action.coal_delta == 0                      # safe default, but flagged
    assert stats["parse_failures"] == 1
    ev = trace.events[0]
    assert ev["parse_status"] == "reasoning_truncated" and ev["finish_reason"] == "length" and ev["ok"] is False


def test_unclosed_think_tag_is_truncated_reasoning():
    res, stats, _ = invoke({"content": "<think>still going", "finish_reason": "length"})
    assert res.status == "reasoning_truncated" and stats["parse_failures"] == 1


def test_reasoning_without_answer_and_normal_stop_is_reasoning_only():
    res, _, _ = invoke({"content": None, "reasoning_content": "the answer is in here", "finish_reason": "stop"})
    assert res.status == "reasoning_only" and res.ok is False


def test_plain_empty_stays_empty():
    res, _, _ = invoke({"content": None, "finish_reason": "stop"})
    assert res.status == "empty"


def test_json_inside_reasoning_is_not_used_as_the_action():
    # A draft action in the thinking text must not be executed as the decision.
    res, _, _ = invoke({"content": None, "reasoning_content": f"draft: {ACTION}", "finish_reason": "length"})
    assert res.status == "reasoning_truncated" and res.action.coal_delta == 0


def test_hard_task_plan_truncated_by_reasoning_continues_without_plan():
    client = FakeClient(replies={"planning": {"content": None, "reasoning_content": "hmm", "finish_reason": "length"}})
    trace = TraceRecorder()
    res = baseline.run_task(EnergyGridEnvironment(), client, "m", "hard", verbose=False, trace=trace)
    plan_ev = [e for e in trace.events if e.get("round") == "plan"][0]
    assert plan_ev["parse_status"] == "reasoning_truncated" and plan_ev["reasoning"] == "hmm"
    assert res["valid_run"] is False and res["llm_stats"]["parse_failures"] >= 1
    d = [e for e in trace.events if e["type"] == "agent_call" and e["agent"] == "dispatch"][0]
    assert d["plan_call_id"] is None                       # no plan was produced, so none was injected
