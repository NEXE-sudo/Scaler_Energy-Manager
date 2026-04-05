"""
Task definitions for the Energy Grid Management Environment.

Each task defines a concrete objective with fixed parameters:
    - Episode length (steps)
    - Season (affects base demand)
    - Random seed (ensures deterministic episodes)
    - Available plants
    - Starting conditions
    - Grader weights
    - Human-readable description

Tasks are ordered easy → medium → hard with genuine difficulty progression.
The hard task is designed to challenge frontier LLMs with cascading failures,
long-horizon plant investment decisions, and competing objectives.
"""

from __future__ import annotations

from typing import Any, Dict
from .simulator import PLANT_BUILD_SPECS

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # EASY — Baseline Dispatch
    # ------------------------------------------------------------------
    "easy": {
        "id": "easy",
        "name": "Baseline Dispatch",
        "description": (
            "Operate a simple grid with a single coal plant and battery "
            "storage over one simulated day (24 steps). "
            "No renewable sources. No stochastic events. "
            "The agent must learn the daily demand curve and dispatch "
            "coal output + battery charge/discharge to meet demand at "
            "minimum cost while maintaining grid frequency. "
            "A well-calibrated agent should score 0.70–0.85."
        ),
        "difficulty": "easy",
        "total_steps": 24,
        "season": "spring",
        "seed": 42,
        # Plants available from start
        "plants_available": ["coal", "battery"],
        # Plants the agent can build (none in easy)
        "buildable_plants": [],
        # Starting battery level (MWh)
        "battery_start_mwh": 100.0,
        # Capital budget (unused in easy)
        "capital_budget": 0.0,
        # Coal starting output
        "coal_start_mw": 400.0,
        # Grader weights — must sum to 1.0
        "grader_weights": {
            "reliability": 0.60,   # % steps demand fully met
            "cost_efficiency": 0.40,  # normalised operational cost
            "reservoir_management": 0.00,
            "battery_health": 0.00,
            "capital_efficiency": 0.00,
            "emissions": 0.00,
        },
        # Normalisation reference for cost (worst-case cost for this task)
        "max_expected_cost": 15.0,
        # Expected LLM score range (for README / documentation)
        "expected_llm_score_range": (0.70, 0.85),
        "action_schema": {
            "coal_delta": "float, -100 to +100 MW",
            "battery_mode": "string: charge | discharge | idle",
            "hydro_delta": "float (ignored, no hydro)",
            "nuclear_delta": "float (ignored, no nuclear)",
            "plant_action": "string (ignored, no building in easy)",
            "emergency_coal_boost": "bool",
            "demand_response_mw": "float, 0 to 150 MW",
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — Renewable Integration
    # ------------------------------------------------------------------
    "medium": {
        "id": "medium",
        "name": "Renewable Integration",
        "description": (
            "Manage a grid with coal, solar, wind, and battery storage "
            "over two simulated days (48 steps) during summer. "
            "Solar and wind introduce variability — cloud cover and calm "
            "periods can cut renewable output without warning. "
            "Heatwaves and cold snaps cause demand surges. "
            "Rainfall events replenish the (not yet built) hydro reservoir. "
            "The agent must balance cost optimisation against reliability "
            "while coping with stochastic weather. "
            "A well-calibrated agent should score 0.50–0.70."
        ),
        "difficulty": "medium",
        "total_steps": 48,
        "season": "summer",    # higher base demand
        "seed": 137,
        "plants_available": ["coal", "solar", "wind", "battery"],
        "buildable_plants": [],
        "battery_start_mwh": 80.0,
        "capital_budget": 0.0,
        "coal_start_mw": 400.0,
        "grader_weights": {
            "reliability": 0.60,
            "cost_efficiency": 0.30,
            "battery_health": 0.10,
            "reservoir_management": 0.00,      # no hydro in medium task
            "capital_efficiency": 0.00,
            "emissions": 0.00,
        },
        "max_expected_cost": 35.0,
        "expected_llm_score_range": (0.50, 0.70),
        "action_schema": {
            "coal_delta": "float, -100 to +100 MW",
            "battery_mode": "string: charge | discharge | idle",
            "hydro_delta": "float (ignored, no hydro)",
            "nuclear_delta": "float (ignored, no nuclear)",
            "plant_action": "string (ignored, no building in medium)",
            "emergency_coal_boost": "bool",
            "demand_response_mw": "float, 0 to 150 MW",
        },
    },

    # ------------------------------------------------------------------
    # HARD — Full Grid Management
    # ------------------------------------------------------------------
    "hard": {
        "id": "hard",
        "name": "Full Grid Management",
        "description": (
            "Operate a full national grid over three simulated days (72 steps) "
            "during winter — the highest-demand season. "
            "All generation sources are available to build (solar, wind, hydro, "
            "nuclear) using a limited capital budget of 2000 units. "
            "The agent faces cascading challenges: a guaranteed coal outage on "
            "day 2, possible nuclear SCRAM, coal price spikes, drought reducing "
            "hydro reservoir inflow, and transmission faults. "
            "Nuclear takes 15 steps to build and provides cheap baseload — but "
            "building it too late is useless. Wind (6 steps) is faster but "
            "variable. Hydro (10 steps) provides dispatchable renewable power "
            "but requires reservoir management. "
            "The agent must balance four competing objectives: reliability, "
            "cost, emissions, and long-term grid stability. "
            "Demand response is available at capital cost. "
            "A frontier LLM should score 0.30–0.50."
        ),
        "difficulty": "hard",
        "total_steps": 72,
        "season": "winter",    # highest base demand
        "seed": 271,
        "plants_available": ["coal", "solar", "wind", "battery"],
        "buildable_plants": ["solar", "wind", "hydro", "nuclear"],
        "battery_start_mwh": 60.0,     # starts at 30% — stressed
        "capital_budget": 2000.0,
        "coal_start_mw": 350.0,
        "grader_weights": {
            "reliability": 0.40,
            "cost_efficiency": 0.20,
            "reservoir_management": 0.10,
            "battery_health": 0.10,
            "capital_efficiency": 0.10,   # did investment pay off?
            "emissions": 0.10,
        },
        "max_expected_cost": 80.0,
        "expected_llm_score_range": (0.30, 0.50),
        "action_schema": {
            "coal_delta": "float, -100 to +100 MW",
            "battery_mode": "string: charge | discharge | idle",
            "hydro_delta": "float, -80 to +80 MW (if hydro built)",
            "nuclear_delta": "float, -10 to +10 MW (if nuclear built)",
            "plant_action": (
                "string: none | build_solar | build_wind | "
                "build_hydro | build_nuclear | close_coal"
            ),
            "emergency_coal_boost": "bool",
            "demand_response_mw": "float, 0 to 150 MW (costs 0.5 capital/MW)",
        },
    },
}

# Ordered list for consistent iteration
TASK_ORDER: list[str] = ["easy", "medium", "hard"]


def get_task(task_id: str) -> Dict[str, Any]:
    """
    Retrieve task config by ID.

    Raises:
        ValueError: if task_id is not recognised.
    """
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task '{task_id}'. Valid tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def get_all_tasks() -> Dict[str, Dict[str, Any]]:
    """Return all task configs ordered easy → medium → hard."""
    return {k: TASKS[k] for k in TASK_ORDER}


def get_tasks_summary() -> list[Dict[str, Any]]:
    """
    Return a lightweight summary list suitable for the /tasks endpoint.

    Each entry contains id, name, difficulty, total_steps, description,
    and action_schema — everything a client needs to understand the task
    without the internal grader weights.
    """
    summary = []
    for task_id in TASK_ORDER:
        task = TASKS[task_id]
        summary.append(
            {
                "id": task["id"],
                "name": task["name"],
                "difficulty": task["difficulty"],
                "total_steps": task["total_steps"],
                "season": task["season"],
                "description": task["description"],
                "plants_available_from_start": task["plants_available"],
                "buildable_plants": task["buildable_plants"],
                "capital_budget": task["capital_budget"],
                "expected_llm_score_range": task["expected_llm_score_range"],
                "action_schema": task["action_schema"],
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Plant build reference (exposed via /tasks for agent reference)
# ---------------------------------------------------------------------------

# Plant build descriptive notes (references; specs are in simulator.py PLANT_BUILD_SPECS)
PLANT_BUILD_NOTES: Dict[str, Dict[str, str]] = {
    "solar": {
        "fuel_cost": "0",
        "inertia_contribution": "none (inverter-based)",
        "notes": "Output 0 MW at night. Reduced by cloud/storm events.",
    },
    "wind": {
        "fuel_cost": "0",
        "inertia_contribution": "none (inverter-based)",
        "notes": (
            "24/7 availability but stochastic. Output follows cubic wind "
            "power curve. Calm periods can persist 4–6 steps."
        ),
    },
    "hydro": {
        "fuel_cost": "0",
        "inertia_contribution": "high (synchronous machine)",
        "notes": (
            "Dispatchable renewable. Reservoir depletes with generation. "
            "Rainfall refills reservoir. Drought reduces inflow severely."
        ),
    },
    "nuclear": {
        "fuel_cost": "0.05",
        "inertia_contribution": "very high (synchronous machine)",
        "notes": (
            "Cheapest per MWh once running. Minimum stable output 300 MW "
            "— cannot be turned off. SCRAM event drops output to 0 for "
            "8 steps. Cannot be decommissioned."
        ),
    },
}

# Generate PLANT_BUILD_REFERENCE from PLANT_BUILD_SPECS (authoritative source)
# and augment with descriptive notes from PLANT_BUILD_NOTES
PLANT_BUILD_REFERENCE: Dict[str, Dict[str, Any]] = {
    "solar": {
        "action": "build_solar",
        "cost_units": PLANT_BUILD_SPECS["build_solar"]["cost"],
        "build_steps": PLANT_BUILD_SPECS["build_solar"]["build_steps"],
        "capacity_mw": PLANT_BUILD_SPECS["build_solar"]["capacity_mw"],
        **PLANT_BUILD_NOTES["solar"],
    },
    "wind": {
        "action": "build_wind",
        "cost_units": PLANT_BUILD_SPECS["build_wind"]["cost"],
        "build_steps": PLANT_BUILD_SPECS["build_wind"]["build_steps"],
        "capacity_mw": PLANT_BUILD_SPECS["build_wind"]["capacity_mw"],
        **PLANT_BUILD_NOTES["wind"],
    },
    "hydro": {
        "action": "build_hydro",
        "cost_units": PLANT_BUILD_SPECS["build_hydro"]["cost"],
        "build_steps": PLANT_BUILD_SPECS["build_hydro"]["build_steps"],
        "capacity_mw": PLANT_BUILD_SPECS["build_hydro"]["capacity_mw"],
        **PLANT_BUILD_NOTES["hydro"],
    },
    "nuclear": {
        "action": "build_nuclear",
        "cost_units": PLANT_BUILD_SPECS["build_nuclear"]["cost"],
        "build_steps": PLANT_BUILD_SPECS["build_nuclear"]["build_steps"],
        "capacity_mw": PLANT_BUILD_SPECS["build_nuclear"]["capacity_mw"],
        **PLANT_BUILD_NOTES["nuclear"],
    },
}