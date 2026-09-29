"""Regression: round-2 (revision) prompts must reflect canonical simulator state.

Bug (pre-fix): run_task built round-2 prompts from the observation returned by
env.step_market(), which is filtered for the *market* role and back-filled with
model defaults. Dispatch therefore revised its action from wrong battery,
reserve and blackout-risk values.
"""
import re

import pytest

from server import baseline
from server.energy_grid_environment import EnergyGridEnvironment
from tests.conftest import FakeClient

STATE_LINES = re.compile(r"^- (Freq:.*|Battery:.*)$", re.MULTILINE)


def state_lines(text: str):
    return STATE_LINES.findall(text)


@pytest.fixture(autouse=True)
def one_step(monkeypatch):
    monkeypatch.setattr(baseline, "MAX_EVAL_STEPS", 1)


def run_one_step(task_id="medium"):
    client = FakeClient()
    env = EnergyGridEnvironment()
    baseline.run_task(env, client, "test-model", task_id, verbose=False)
    return client


def by_role(client, role):
    return [c for c in client.calls if c["role"] == role]


@pytest.mark.parametrize("role", ["dispatch", "market"])
def test_round2_state_matches_round1(role):
    client = run_one_step("medium")
    r1, r2 = by_role(client, role)[:2]
    # The simulator does not advance between rounds, so the state lines
    # (frequency, reserve, battery, risk) must be identical.
    assert state_lines(r1["user"]) == state_lines(r2["user"])
    assert state_lines(r2["user"]), "state lines missing from prompt"


@pytest.mark.parametrize("role", ["dispatch", "market"])
def test_round2_contains_negotiation_history(role):
    client = run_one_step("medium")
    r1, r2 = by_role(client, role)[:2]
    assert "NEGOTIATION" not in r1["user"]
    assert "NEGOTIATION" in r2["user"]


def test_round2_prompt_shows_real_reserve_not_default():
    env = EnergyGridEnvironment()
    canonical = env.reset("medium")
    client = FakeClient()
    env2 = EnergyGridEnvironment()
    baseline.run_task(env2, client, "test-model", "medium", verbose=False)
    r2 = by_role(client, "dispatch")[1]
    required = f"{canonical.spinning_reserve_mw:.0f}/{canonical.spinning_reserve_required_mw:.0f}MW"
    assert f"Reserve: {required}" in r2["user"]
