import pytest
from server.energy_grid_environment import EnergyGridEnvironment
from models import EnergyGridAction


@pytest.fixture
def env():
    return EnergyGridEnvironment()


def test_reset_and_step(env):
    obs = env.reset(task_id="easy")
    assert obs.demand_mw > 0
    act = EnergyGridAction(coal_delta=0.0, battery_mode="idle")
    new_obs = env.step(act)
    assert isinstance(new_obs, type(obs))


def test_grader_runs(env):
    obs = env.reset(task_id="easy")
    for _ in range(5):
        obs = env.step(EnergyGridAction(coal_delta=0.0, battery_mode="idle"))
    grade = env.grade_current_episode()
    assert grade is not None
    assert 0.0 <= grade["total_score"] <= 1.0


def test_all_tasks_reset(env):
    for task_id in ["easy", "medium", "hard"]:
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id
        assert obs.demand_mw > 0