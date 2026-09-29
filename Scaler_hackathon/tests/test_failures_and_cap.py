"""Failed model calls / unparseable output must be visible, never silent successes."""
import pytest

from models import EnergyGridAction
from server import baseline
from server.energy_grid_environment import EnergyGridEnvironment
from tests.conftest import FakeClient


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(baseline.time, "sleep", lambda *_: None)


def run(client, task="medium", cap=2, monkeypatch=None):
    baseline.MAX_EVAL_STEPS = cap
    try:
        return baseline.run_task(EnergyGridEnvironment(), client, "test-model", task, verbose=False)
    finally:
        baseline.MAX_EVAL_STEPS = 20


def test_clean_run_is_valid_and_counts_tokens():
    client = FakeClient(usage=(11, 7))
    res = run(client, cap=2)
    st = res["llm_stats"]
    assert res["valid_run"] is True
    assert st["call_failures"] == 0 and st["parse_failures"] == 0
    assert st["calls"] == len(client.calls) > 0
    assert st["prompt_tokens"] == 11 * st["calls"]
    assert st["completion_tokens"] == 7 * st["calls"]


def test_call_failure_does_not_crash_and_is_flagged():
    client = FakeClient(replies={"dispatch": RuntimeError("provider 503")})
    res = run(client, cap=2)  # previously: exception aborted the whole episode
    assert res["steps_completed"] == 2
    assert res["valid_run"] is False
    assert res["llm_stats"]["call_failures"] >= 2


def test_empty_content_counts_as_parse_failure():
    # e.g. reasoning models that put everything in a separate field and return null content
    client = FakeClient(replies={"market": None})
    res = run(client, cap=2)
    assert res["valid_run"] is False
    assert res["llm_stats"]["parse_failures"] >= 2
    assert res["llm_stats"]["call_failures"] == 0


def test_unparseable_text_counts_as_parse_failure():
    client = FakeClient(replies={"dispatch": "I think we should increase coal output."})
    res = run(client, cap=1)
    assert res["llm_stats"]["parse_failures"] >= 1
    assert res["valid_run"] is False


def test_checked_parser_statuses():
    assert baseline._parse_action_checked("")[1] == "empty"
    assert baseline._parse_action_checked(None)[1] == "empty"
    assert baseline._parse_action_checked("no json")[1] == "unparseable"
    action, status = baseline._parse_action_checked('Action: {"coal_delta": 12}')
    assert status == "ok" and action.coal_delta == 12
    # backward-compatible wrapper still degrades to the safe default
    assert baseline._parse_action("").coal_delta == 0


def test_legacy_wrapper_still_raises_after_retries():
    client = FakeClient(replies={"dispatch": RuntimeError("down")})
    with pytest.raises(RuntimeError):
        baseline._call_llm_with_retry(client, "m", "You are the Dispatch Agent.", [{"role": "user", "content": "x"}])


def test_step_cap_reports_truncation():
    res = run(FakeClient(), cap=2)
    assert res["truncated"] is True
    assert res["steps_completed"] == 2 and res["eval_step_cap"] == 2


def test_cap_zero_runs_full_episode():
    res = run(FakeClient(), task="easy", cap=0)
    assert res["truncated"] is False
    assert res["eval_step_cap"] == res["total_steps"]


def test_control_layer_overrides_are_recorded():
    env = EnergyGridEnvironment()
    obs = env.reset("medium")
    client = FakeClient(replies={"dispatch": 'Action: {"coal_delta": 1, "plant_action": "build_solar"}'})
    stats = baseline._new_stats()
    res = baseline._invoke_agent(client, "m", "dispatch", "You are the Dispatch Agent.", "state", obs, stats, verbose=False)
    assert res.proposed["plant_action"] == "build_solar"
    assert res.action.plant_action == "none"
    assert res.control_changes["plant_action"] == ["build_solar", "none"]
    assert stats["control_overrides"] == 1
