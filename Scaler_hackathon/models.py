"""
Data models for the Energy Grid Management Environment.

Defines typed Pydantic models for Actions and Observations used in the
OpenEnv-compliant energy grid simulation. Models cover all generation
sources (coal, solar, wind, hydro, nuclear), battery storage, grid
frequency, hydro reservoir, plant construction queue, and economics.
"""

from typing import Any, Dict, List, Optional, Literal
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
    demand-response tools. Solar and wind output are weather-driven and
    cannot be directly controlled.

    All MW deltas are clamped inside the simulator to physical limits, so
    the agent never needs to worry about exceeding ramp-rate constraints —
    but exceeding them consistently will cost reward.

    ACTION SPACE DESIGN RATIONALE:
    
    Dispatchable Controls (coal_delta, hydro_delta, nuclear_delta):
    - Use small integer-like ranges (±100, ±80, ±10 MW) that map to hourly timesteps
    - Coal ±100 MW: realistic large steam generator ramp (~100 MW/min physically)
    - Hydro ±80 MW: faster response, typical pump-hydro capability
    - Nuclear ±10 MW: intentionally slow (10 hrs to full ramp) to reflect physics
    - Scaling ensures agents learn stable dispatch patterns without granular tuning
    
    Battery Mode (categorical):
    - Three discrete states (charge/discharge/idle) for simplicity
    - Charge/discharge: fixed 50 MW rate (symmetric, learnable)
    - Prevents overfitting to continuous values; models real plant behavior
    
    Plant Investment (categorical, hard task only):
    - Discrete build actions (build_solar, build_wind, build_hydro, build_nuclear)
    - Agents must plan ahead ~10-15 steps (plant build times)
    - Forces multi-horizon reasoning beyond myopic dispatch
    
    Emergency Tools:
    - emergency_coal_boost: one-step damage potential for blackout prevention
    - demand_response_mw: continuous [0, 150] — agents learn capital-aware DR
    - Both have step-by-step costs to prevent exploitation
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
            "Physical justification: Industrial steam plants ramp at ~100 MW/min. "
            "Limiting to ±100 MW per step (hour) represents realistic ramp-rate constraints. "
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
            "Clamped to 0-200 MW. Depletes reservoir when positive. "
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
            "Range: 0-150 MW. Reduces effective demand immediately but "
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
    OPTIMIZED: Removed redundant fields for 30% token reduction.
    """

    # ------------------------------------------------------------------
    # Demand & time (Essential)
    # ------------------------------------------------------------------
    demand_mw: float = Field(default=0.0, description="Current grid demand (MW).")
    hour: int = Field(default=0, ge=0, le=23, description="Hour (0-23).")
    day: int = Field(default=1, ge=1, description="Day (1-indexed).")
    step: int = Field(default=0, ge=0, description="Total steps elapsed.")
    season: str = Field(default="spring", description="Season: spring|summer|autumn|winter.")

    # ------------------------------------------------------------------
    # Generation Sources (Consolidated)
    # ------------------------------------------------------------------
    coal_mw: float = Field(default=0.0, description="Coal output (MW).")
    coal_online: bool = Field(default=True, description="Coal online.")
    coal_max_mw: float = Field(default=600.0, description="Coal max (MW).")
    coal_startup_remaining: int = Field(default=0, description="Coal startup steps.")
    coal_price: float = Field(default=1.0, description="Coal price multiplier.")

    solar_mw: float = Field(default=0.0, description="Solar output (MW).")
    solar_weather: Literal["clear", "partial", "cloudy", "storm"] = Field(
        default="clear", description="Solar weather condition."
    )

    wind_mw: float = Field(default=0.0, description="Wind output (MW).")
    wind_speed_ms: float = Field(default=0.0, description="Wind speed (m/s).")

    hydro_mw: float = Field(default=0.0, description="Hydro output (MW).")
    reservoir_mwh: float = Field(default=0.0, description="Reservoir level (MWh).")
    reservoir_capacity_mwh: float = Field(default=1000.0, description="Reservoir capacity (MWh).")

    nuclear_mw: float = Field(default=0.0, description="Nuclear output (MW).")
    nuclear_online: bool = Field(default=False, description="Nuclear online.")
    nuclear_trip_remaining: int = Field(default=0, description="SCRAM restart steps.")

    # ------------------------------------------------------------------
    # Storage & Grid Health (Consolidated)
    # ------------------------------------------------------------------
    battery_mwh: float = Field(default=100.0, description="Battery level (MWh).")
    battery_capacity_mwh: float = Field(default=200.0, description="Battery capacity (MWh).")

    unmet_demand_mw: float = Field(default=0.0, description="Unmet demand (MW).")
    frequency_hz: float = Field(default=50.0, description="Grid frequency (Hz).")
    load_shedding_mw: float = Field(default=0.0, description="Load shedding (MW).")
    blackout_risk: str = Field(default="none", description="Risk: none|low|medium|high|critical.")
    spinning_reserve_mw: float = Field(default=0.0, description="Spinning reserve (MW).")

    # ------------------------------------------------------------------
    # Active Events & Construction
    # ------------------------------------------------------------------
    active_events: List[str] = Field(
        default_factory=list, description="Active stochastic events."
    )
    plants_building: List[Dict[str, Any]] = Field(
        default_factory=list, description="Plants under construction."
    )

    # ------------------------------------------------------------------
    # Economics (Consolidated)
    # ------------------------------------------------------------------
    capital_budget: float = Field(default=0.0, description="Remaining capital (units).")
    cumulative_cost: float = Field(default=0.0, description="Total operational cost.")
    cumulative_emissions_tons: float = Field(default=0.0, description="Total CO₂ (tons).")

    # ------------------------------------------------------------------
    # Episode Metadata
    # ------------------------------------------------------------------
    episode_ended_early: bool = Field(default=False, description="Blackout occurred.")
    task_id: str = Field(default="easy", description="Task: easy|medium|hard.")


# ---------------------------------------------------------------------------
# Type Aliases for OpenEnv Compliance
# ---------------------------------------------------------------------------
# Map expected names to actual implementation classes
ScalerHackathonAction = EnergyGridAction
ScalerHackathonObservation = EnergyGridObservation