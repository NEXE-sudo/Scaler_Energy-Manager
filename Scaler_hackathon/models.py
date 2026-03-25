# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Energy Grid Management Environment.

Defines typed Pydantic models for Actions and Observations used in the
OpenEnv-compliant energy grid simulation. Models cover all generation
sources (coal, solar, wind, hydro, nuclear), battery storage, grid
frequency, hydro reservoir, plant construction queue, and economics.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator
from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class EnergyGridAction(Action):
    """
    Action submitted by the agent each simulation step.

    The agent controls dispatchable generation (coal, hydro, nuclear),
    battery storage mode, optional plant construction, and emergency /
    demand-response tools.  Solar and wind output are weather-driven and
    cannot be directly controlled.

    All MW deltas are clamped inside the simulator to physical limits, so
    the agent never needs to worry about exceeding ramp-rate constraints —
    but exceeding them consistently will cost reward.
    """

    # ------------------------------------------------------------------
    # Dispatchable generation adjustments (MW change this step)
    # ------------------------------------------------------------------
    coal_delta: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description=(
            "Change in coal plant output this step (MW). "
            "Range: -100 to +100 MW. "
            "Clamped to min-stable (200 MW) and max (600 MW) in simulator. "
            "Plant must be online; ignored during startup sequence."
        ),
    )

    hydro_delta: float = Field(
        default=0.0,
        ge=-80.0,
        le=80.0,
        description=(
            "Change in hydro turbine output this step (MW). "
            "Range: -80 to +80 MW. "
            "Clamped to 0–200 MW. Depletes reservoir when positive. "
            "Only available if hydro plant is built."
        ),
    )

    nuclear_delta: float = Field(
        default=0.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Change in nuclear reactor output this step (MW). "
            "Range: -10 to +10 MW (extremely slow ramp). "
            "Clamped to min-stable (300 MW) and max (500 MW). "
            "Cannot be adjusted during a SCRAM event. "
            "Only available if nuclear plant is built."
        ),
    )

    # ------------------------------------------------------------------
    # Battery storage
    # ------------------------------------------------------------------
    battery_mode: str = Field(
        default="idle",
        description=(
            "Battery operation mode this step. "
            "Options: 'charge' (draw up to 50 MW from grid to store), "
            "'discharge' (inject up to 50 MW into grid), "
            "'idle' (no battery action). "
            "Cannot charge and discharge in the same step."
        ),
    )

    # ------------------------------------------------------------------
    # Plant construction / decommissioning (Hard task)
    # ------------------------------------------------------------------
    plant_action: str = Field(
        default="none",
        description=(
            "Long-term investment action. "
            "Options: 'none', 'build_solar', 'build_wind', "
            "'build_hydro', 'build_nuclear', 'close_coal'. "
            "Building deducts capital immediately; capacity comes online "
            "after build-time steps. 'close_coal' is permanent and "
            "incurs a decommissioning cost. Only available in Hard task."
        ),
    )

    # ------------------------------------------------------------------
    # Emergency tools
    # ------------------------------------------------------------------
    emergency_coal_boost: bool = Field(
        default=False,
        description=(
            "Override coal ramp limits: instantly add +200 MW above normal "
            "maximum. Damages plant — reduces max output by 50 MW for the "
            "next 5 steps. Use only to prevent imminent blackout."
        ),
    )

    demand_response_mw: float = Field(
        default=0.0,
        ge=0.0,
        le=150.0,
        description=(
            "Request industrial demand reduction this step (MW). "
            "Range: 0–150 MW. Reduces effective demand immediately but "
            "costs capital (0.5 units per MW). Available in all tasks."
        ),
    )

    @field_validator("battery_mode")
    @classmethod
    def validate_battery_mode(cls, v: str) -> str:
        allowed = {"charge", "discharge", "idle"}
        if v not in allowed:
            raise ValueError(f"battery_mode must be one of {allowed}, got '{v}'")
        return v

    @field_validator("plant_action")
    @classmethod
    def validate_plant_action(cls, v: str) -> str:
        allowed = {
            "none",
            "build_solar",
            "build_wind",
            "build_hydro",
            "build_nuclear",
            "close_coal",
        }
        if v not in allowed:
            raise ValueError(f"plant_action must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class EnergyGridObservation(Observation):
    """
    Full grid state returned to the agent after each step / reset.

    Covers demand, all generation sources, battery, hydro reservoir,
    frequency dynamics, active events, plant construction queue,
    economics, and episode metadata.
    """

    # ------------------------------------------------------------------
    # Demand & time
    # ------------------------------------------------------------------
    demand_mw: float = Field(
        default=0.0,
        description="Current total grid demand (MW).",
    )
    time_of_day: int = Field(
        default=0,
        ge=0,
        le=23,
        description="Hour of the simulated day (0–23).",
    )
    day: int = Field(
        default=1,
        ge=1,
        description="Current episode day number (1-indexed).",
    )
    step: int = Field(
        default=0,
        ge=0,
        description="Total simulation steps elapsed this episode.",
    )
    season: str = Field(
        default="spring",
        description=(
            "Current season affecting base demand: "
            "'spring', 'summer', 'autumn', 'winter'."
        ),
    )

    # ------------------------------------------------------------------
    # Coal plant
    # ------------------------------------------------------------------
    coal_output_mw: float = Field(
        default=0.0,
        description="Current coal plant output (MW).",
    )
    coal_online: bool = Field(
        default=True,
        description="Whether the coal plant is currently online.",
    )
    coal_startup_steps_remaining: int = Field(
        default=0,
        ge=0,
        description=(
            "Steps remaining before coal plant reaches minimum stable output "
            "after a restart. 0 = online and available."
        ),
    )
    coal_max_mw: float = Field(
        default=600.0,
        description=(
            "Current coal plant maximum output (MW). "
            "Reduced temporarily after emergency boost usage."
        ),
    )
    coal_price: float = Field(
        default=1.0,
        description="Current coal fuel price multiplier (1.0 = baseline).",
    )

    # ------------------------------------------------------------------
    # Solar plant
    # ------------------------------------------------------------------
    solar_output_mw: float = Field(
        default=0.0,
        description="Current solar farm output (MW). Zero at night.",
    )
    solar_available: bool = Field(
        default=True,
        description="Whether solar panels are installed.",
    )
    solar_weather: str = Field(
        default="clear",
        description=(
            "Current solar weather condition: "
            "'clear' (1.0×), 'partial' (0.6×), 'cloudy' (0.3×), 'storm' (0.0×)."
        ),
    )

    # ------------------------------------------------------------------
    # Wind plant
    # ------------------------------------------------------------------
    wind_output_mw: float = Field(
        default=0.0,
        description="Current wind farm output (MW).",
    )
    wind_available: bool = Field(
        default=False,
        description="Whether wind turbines are installed.",
    )
    wind_speed_ms: float = Field(
        default=0.0,
        description=(
            "Current wind speed (m/s). Useful for anticipating next-step output. "
            "Cut-in: 3 m/s, rated: 12 m/s, cut-out: 25 m/s."
        ),
    )

    # ------------------------------------------------------------------
    # Hydro plant
    # ------------------------------------------------------------------
    hydro_output_mw: float = Field(
        default=0.0,
        description="Current hydro turbine output (MW).",
    )
    hydro_available: bool = Field(
        default=False,
        description="Whether the hydro plant is installed.",
    )
    reservoir_level_mwh: float = Field(
        default=600.0,
        description="Current hydro reservoir water level (MWh equivalent).",
    )
    reservoir_capacity_mwh: float = Field(
        default=1000.0,
        description="Total hydro reservoir capacity (MWh equivalent).",
    )
    natural_inflow_mwh: float = Field(
        default=15.0,
        description=(
            "Current natural river inflow into reservoir per step (MWh). "
            "Reduced during drought events."
        ),
    )

    # ------------------------------------------------------------------
    # Nuclear plant
    # ------------------------------------------------------------------
    nuclear_output_mw: float = Field(
        default=0.0,
        description="Current nuclear reactor output (MW).",
    )
    nuclear_available: bool = Field(
        default=False,
        description="Whether the nuclear plant is installed.",
    )
    nuclear_online: bool = Field(
        default=False,
        description=(
            "Whether reactor is online. False during SCRAM and restart sequence."
        ),
    )
    nuclear_trip_steps_remaining: int = Field(
        default=0,
        ge=0,
        description=(
            "Steps remaining in nuclear restart sequence after a SCRAM. "
            "0 = online and available."
        ),
    )

    # ------------------------------------------------------------------
    # Battery storage
    # ------------------------------------------------------------------
    battery_level_mwh: float = Field(
        default=100.0,
        description="Current battery state of charge (MWh).",
    )
    battery_capacity_mwh: float = Field(
        default=200.0,
        description=(
            "Effective battery capacity (MWh). "
            "Degrades slightly with each charge/discharge cycle."
        ),
    )

    # ------------------------------------------------------------------
    # Grid health & frequency
    # ------------------------------------------------------------------
    unmet_demand_mw: float = Field(
        default=0.0,
        description="Demand not currently being served (MW). Target: 0.",
    )
    overproduction_mw: float = Field(
        default=0.0,
        description="Generation exceeding demand (MW). Small amounts are acceptable.",
    )
    grid_frequency: float = Field(
        default=50.0,
        description=(
            "Grid frequency (Hz). Target: 50.0 Hz. "
            "Load shedding starts at 49.0 Hz. "
            "Blackout at 47.5 Hz (under) or 51.5 Hz (over)."
        ),
    )
    rate_of_change_hz_per_step: float = Field(
        default=0.0,
        description=(
            "Rate of change of frequency (Hz/step). "
            "High absolute values indicate instability. "
            "Protection trips if |RoCoF| > 1.0 Hz/step."
        ),
    )
    system_inertia_seconds: float = Field(
        default=4.0,
        description=(
            "Total system inertia constant (seconds). "
            "Higher = more stable grid. "
            "Decreases as coal/hydro/nuclear are replaced with solar/wind."
        ),
    )
    primary_response_active: bool = Field(
        default=False,
        description=(
            "Whether automatic governor response is compensating for "
            "a frequency deviation. Agent must act within 3 steps."
        ),
    )
    load_shedding_mw: float = Field(
        default=0.0,
        description=(
            "Involuntary load currently being shed to protect grid (MW). "
            "Non-zero values indicate a grid emergency."
        ),
    )
    blackout_risk: str = Field(
        default="none",
        description=(
            "Qualitative blackout risk level: "
            "'none', 'low', 'medium', 'high', 'critical'. "
            "Critical = blackout imminent, agent must act immediately."
        ),
    )
    spinning_reserve_mw: float = Field(
        default=0.0,
        description=(
            "Available spinning reserve (MW headroom on online synchronous plants). "
            "Should be >= 20% of demand to meet grid code."
        ),
    )
    spinning_reserve_required_mw: float = Field(
        default=0.0,
        description="Required spinning reserve = 20% of current demand (MW).",
    )
    transmission_capacity_mw: float = Field(
        default=1200.0,
        description=(
            "Current transmission capacity (MW). "
            "Reduced during grid_fault events."
        ),
    )

    # ------------------------------------------------------------------
    # Active events
    # ------------------------------------------------------------------
    active_events: List[str] = Field(
        default_factory=list,
        description=(
            "List of currently active stochastic events. "
            "Possible values: 'heatwave', 'cold_snap', 'cloud', 'heavy_cloud', "
            "'storm', 'calm', 'rainfall', 'drought', 'coal_outage', "
            "'nuclear_trip', 'price_spike', 'grid_fault'."
        ),
    )

    # ------------------------------------------------------------------
    # Plant construction queue (Hard task)
    # ------------------------------------------------------------------
    plants_under_construction: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Plants currently being built. Each entry: "
            "{'type': str, 'steps_remaining': int, 'capacity_mw': float}."
        ),
    )

    # ------------------------------------------------------------------
    # Economics & emissions
    # ------------------------------------------------------------------
    capital_budget: float = Field(
        default=0.0,
        description=(
            "Remaining capital budget for plant investment (units). "
            "Only relevant in Hard task (starts at 2000)."
        ),
    )
    cumulative_cost: float = Field(
        default=0.0,
        description="Total operational cost accumulated this episode.",
    )
    cumulative_emissions_tons: float = Field(
        default=0.0,
        description=(
            "Total CO₂ emissions accumulated this episode (tons). "
            "Coal emits 0.9 t/MWh. Penalised in Hard task grader."
        ),
    )
    step_reward: float = Field(
        default=0.0,
        description="Reward received for the previous step.",
    )

    # ------------------------------------------------------------------
    # Episode metadata (inherited: done, reward from Observation base)
    # ------------------------------------------------------------------
    episode_ended_early: bool = Field(
        default=False,
        description=(
            "True if the episode ended before all steps due to a "
            "catastrophic blackout event."
        ),
    )
    task_id: str = Field(
        default="easy",
        description="Active task identifier: 'easy', 'medium', or 'hard'.",
    )