"""
Deterministic grader for the Energy Grid Management Environment.

Each task has a grader that produces a score in [0.0, 1.0] based on
an EpisodeLog collected during the episode. Graders are:
    - Deterministic: same log always produces the same score
    - Reproducible: no randomness, no external calls
    - Transparent: each component score is returned separately

Grader components per task:

    Easy:
        reliability      (0.60) — % steps with demand fully met
        cost_efficiency  (0.40) — normalised operational cost

    Medium:
        reliability      (0.60) — % steps with demand fully met
        cost_efficiency  (0.30) — normalised operational cost
        battery_health   (0.10) — final battery level / capacity

    Hard:
        reliability      (0.40) — % steps with demand fully met
        cost_efficiency  (0.20) — normalised operational cost
        emissions        (0.10) — normalised CO2 reduction vs coal-only baseline
        reservoir_mgmt   (0.10) — hydro reservoir management quality
        battery_health   (0.10) — final battery level / capacity
        capital_eff      (0.10) — did plant investment improve reliability?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tasks import TASKS


# ---------------------------------------------------------------------------
# Episode log — collected step by step during an episode
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    """Snapshot of key metrics at a single simulation step."""
    step: int
    demand_mw: float
    total_supply_mw: float
    unmet_demand_mw: float
    grid_frequency: float
    coal_output_mw: float
    solar_output_mw: float
    wind_output_mw: float
    hydro_output_mw: float
    nuclear_output_mw: float
    battery_level_mwh: float    
    battery_capacity_mwh: float
    reservoir_level_mwh: float
    reservoir_capacity_mwh: float
    cumulative_cost: float
    cumulative_emissions_tons: float
    feedin_credits_mwh: float
    coal_price: float
    load_shedding_mw: float
    active_events: List[str]
    capital_budget_remaining: float
    plants_built: List[str]
    blackout: bool
    reward: float


@dataclass
class EpisodeLog:
    """Complete record of an episode for grading."""
    task_id: str
    total_steps: int
    steps_logged: List[StepLog] = field(default_factory=list)

    # Final state snapshots (populated at episode end)
    final_battery_level_mwh: float = 0.0
    final_battery_capacity_mwh: float = 200.0
    final_reservoir_level_mwh: float = 0.0
    final_reservoir_capacity_mwh: float = 1000.0
    final_capital_budget: float = 0.0
    initial_capital_budget: float = 0.0
    total_cumulative_cost: float = 0.0
    total_cumulative_emissions: float = 0.0
    plants_built_during_episode: List[str] = field(default_factory=list)
    blackout_occurred: bool = False
    early_termination_step: Optional[int] = None

    def log_step(self, log: StepLog) -> None:
        self.steps_logged.append(log)
        if log.blackout:
            self.blackout_occurred = True
            self.early_termination_step = log.step

    def finalise(
        self,
        battery_level: float,
        battery_capacity: float,
        reservoir_level: float,
        reservoir_capacity: float,
        capital_remaining: float,
        total_cost: float,
        total_emissions: float,
        plants_built: List[str],
    ) -> None:
        self.final_battery_level_mwh = battery_level
        self.final_battery_capacity_mwh = battery_capacity
        self.final_reservoir_level_mwh = reservoir_level
        self.final_reservoir_capacity_mwh = reservoir_capacity
        self.final_capital_budget = capital_remaining
        self.total_cumulative_cost = total_cost
        self.total_cumulative_emissions = total_emissions
        self.plants_built_during_episode = plants_built


# ---------------------------------------------------------------------------
# Individual scoring components
# ---------------------------------------------------------------------------

def score_reliability(log: EpisodeLog) -> float:
    """
    Fraction of steps where demand was fully met (unmet < 1 MW tolerance).

    Returns 0.0 if a blackout occurred (catastrophic failure).
    Partial credit proportional to steps survived before blackout.
    """
    if log.blackout_occurred:
        steps_before_blackout = log.early_termination_step or 0
        if steps_before_blackout == 0:
            return 0.0
        # Penalty proportional to survival rather than cliff effect
        # Surviving half the episode: 0.25, surviving most: close to 1.0
        survival_fraction = steps_before_blackout / max(1, log.total_steps)
        return 0.5 * survival_fraction  # max 0.5 for blackout

    good_steps = sum(1 for s in log.steps_logged if s.unmet_demand_mw < 0.1)
    return good_steps / max(1, log.total_steps)


def score_cost_efficiency(log: EpisodeLog, task_id: str) -> float:
    """
    Normalised operational cost efficiency.

    Score = 1.0 − (actual_cost / max_expected_cost), clamped to [-0.2, 1.0].
    A perfectly optimised agent scores near 1.0; a wasteful one scores near 0.
    Overspending beyond max_expected_cost produces negative scores.
    """
    max_cost = TASKS[task_id]["max_expected_cost"]
    normalised = log.total_cumulative_cost / max(0.01, max_cost)
    return max(-0.2, min(1.0, 1.0 - normalised))


def score_frequency(log: EpisodeLog) -> float:
    """
    Fraction of steps where grid frequency was within ±0.2 Hz of 50.0 Hz.
    """
    if not log.steps_logged:
        return 0.0
    stable_steps = sum(
        1 for s in log.steps_logged
        if abs(s.grid_frequency - 50.0) <= 0.2
    )
    return stable_steps / len(log.steps_logged)


def score_battery_health(log: EpisodeLog) -> float:
    """
    Final battery state of charge as fraction of remaining capacity.

    Rewards agents that don't drain the battery to zero — preserves
    flexibility for future steps.
    """
    if log.final_battery_capacity_mwh <= 0:
        return 0.0
    return max(0.0, min(1.0, log.final_battery_level_mwh / log.final_battery_capacity_mwh))


def score_reservoir_management(log: EpisodeLog) -> float:
    """
    Hydro reservoir management quality.

    Penalises:
        - Draining the reservoir below 10% (desperate usage)
        - Allowing reservoir above 95% for multiple steps (spillage waste)

    Rewards keeping reservoir in the 20–80% operating band.
    """
    if not log.steps_logged:
        return 0.5  # neutral if no hydro data

    scores = []
    for s in log.steps_logged:
        cap = s.reservoir_capacity_mwh
        if cap <= 0:
            scores.append(0.5)
            continue

        pct = s.reservoir_level_mwh / cap

        if pct < 0.10:
            scores.append(0.0)   # critically low
        elif pct < 0.20:
            scores.append(0.3)
        elif pct <= 0.80:
            scores.append(1.0)   # optimal operating band
        elif pct <= 0.95:
            scores.append(0.7)   # slightly high but okay
        else:
            scores.append(0.3)   # near-spillage

    return sum(scores) / len(scores)


def score_emissions(log: EpisodeLog) -> float:
    """
    Emissions reduction score relative to a coal-only baseline.

    Baseline: all demand served by coal at 0.9 t CO2/MWh.
    Score = 1 − (actual_emissions / baseline_emissions).
    """
    if not log.steps_logged:
        return 0.0

    total_demand_mwh = sum(s.demand_mw for s in log.steps_logged)
    baseline_emissions = total_demand_mwh * 0.9   # all-coal baseline

    if baseline_emissions <= 0:
        return 1.0

    reduction_ratio = max(
        0.0,
        1.0 - (log.total_cumulative_emissions / baseline_emissions),
    )
    return min(1.0, reduction_ratio)


def score_capital_efficiency(log: EpisodeLog) -> float:
    """
    Did the agent's plant investment decisions actually improve outcomes?

    Proxy metric: reliability improvement per capital spent.
        - If no capital was spent: penalized (agent avoided required infrastructure)
        - If capital spent and reliability ≥ 0.7: good (scales with reliability)
        - If capital spent and reliability < 0.5: poor investment

    Also rewards finishing with some budget remaining (financial prudence).
    """
    capital_spent = log.initial_capital_budget - log.final_capital_budget

    if capital_spent <= 0:
        # Agent didn't build anything — penalized, especially on hard task
        # where infrastructure is critical
        return 0.0

    reliability = score_reliability(log)
    budget_remaining_pct = log.final_capital_budget / max(1.0, log.initial_capital_budget)

    # ROI proxy: reliability achieved per 1000 capital spent
    roi = reliability / max(1.0, capital_spent / 1000.0)
    roi_score = min(1.0, roi)

    # Blend ROI with budget conservation
    return 0.7 * roi_score + 0.3 * budget_remaining_pct


# ---------------------------------------------------------------------------
# Master grader
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """Full grading result with component breakdown."""
    task_id: str
    total_score: float                          # final 0.0–1.0 score
    component_scores: Dict[str, float]          # individual components
    component_weights: Dict[str, float]         # weights used
    weighted_components: Dict[str, float]       # weight × score per component
    blackout_occurred: bool
    early_termination_step: Optional[int]
    steps_completed: int
    total_steps: int
    metadata: Dict[str, Any]


def grade_episode(log: EpisodeLog) -> GradeResult:
    """
    Grade a completed episode and return a GradeResult.

    This is the primary public interface of the grader module.
    Called by the /grader endpoint and the baseline script.
    """
    task_id = log.task_id
    task = TASKS[task_id]
    weights = task["grader_weights"]

    # Compute all component scores
    reliability = score_reliability(log)
    cost_eff = score_cost_efficiency(log, task_id)
    frequency = score_frequency(log)
    battery = score_battery_health(log)
    reservoir = score_reservoir_management(log)
    emissions = score_emissions(log)
    capital_eff = score_capital_efficiency(log)

    # frequency_stability is computed for all tasks but only weighted in medium (via internal use)
    # — it appears in component_scores for transparency but does not affect easy or hard totals.
    component_scores: Dict[str, float] = {
        "reliability": reliability,
        "cost_efficiency": cost_eff,
        "frequency_stability": frequency,
        "battery_health": battery,
        "reservoir_management": reservoir,
        "emissions": emissions,
        "capital_efficiency": capital_eff,
    }

    # Apply task-specific weights
    # Note: frequency_stability is always computed but not always weighted
    # (easy/medium use it internally; hard uses emissions + capital_eff instead)
    if task_id == "easy":
        weighted = {
            "reliability": weights["reliability"] * reliability,
            "cost_efficiency": weights["cost_efficiency"] * cost_eff,
        }
    elif task_id == "medium":
        weighted = {
            "reliability": weights["reliability"] * reliability,
            "cost_efficiency": weights["cost_efficiency"] * cost_eff,
            "battery_health": weights["battery_health"] * battery,
        }
    else:  # hard
        weighted = {
            "reliability": weights["reliability"] * reliability,
            "cost_efficiency": weights["cost_efficiency"] * cost_eff,
            "emissions": weights["emissions"] * emissions,
            "reservoir_management": weights["reservoir_management"] * reservoir,
            "battery_health": weights["battery_health"] * battery,
            "capital_efficiency": weights["capital_efficiency"] * capital_eff,
        }

    total_score = sum(weighted.values())
    total_score = max(-0.05, min(1.0, total_score))

    # Explicitly add frequency_stability: 0.0 to weights for easy and hard tasks
    # to reflect actual usage (computed but not weighted)
    if task_id in ("easy", "hard"):
        weights = dict(weights)  # make a copy
        weights["frequency_stability"] = 0.0

    return GradeResult(
        task_id=task_id,
        total_score=round(total_score, 4),
        component_scores=component_scores,
        component_weights=weights,
        weighted_components=weighted,
        blackout_occurred=log.blackout_occurred,
        early_termination_step=log.early_termination_step,
        steps_completed=len(log.steps_logged),
        total_steps=log.total_steps,
        metadata={
            "total_cumulative_cost": round(log.total_cumulative_cost, 3),
            "total_cumulative_emissions_tons": round(log.total_cumulative_emissions, 2),
            "plants_built": log.plants_built_during_episode,
            "final_battery_pct": round(
                100 * log.final_battery_level_mwh / max(1, log.final_battery_capacity_mwh), 1
            ),
            "final_reservoir_pct": round(
                100 * log.final_reservoir_level_mwh / max(1, log.final_reservoir_capacity_mwh), 1
            ),
            "final_capital_remaining": round(log.final_capital_budget, 1),
        },
    )


def grade_result_to_dict(result: GradeResult) -> Dict[str, Any]:
    """Serialise a GradeResult to a JSON-safe dict for API responses."""
    return {
        "task_id": result.task_id,
        "total_score": result.total_score,
        "component_scores": {k: round(v, 4) for k, v in result.component_scores.items()},
        "component_weights": result.component_weights,
        "weighted_components": {k: round(v, 4) for k, v in result.weighted_components.items()},
        "blackout_occurred": result.blackout_occurred,
        "early_termination_step": result.early_termination_step,
        "steps_completed": result.steps_completed,
        "total_steps": result.total_steps,
        "metadata": result.metadata,
    }