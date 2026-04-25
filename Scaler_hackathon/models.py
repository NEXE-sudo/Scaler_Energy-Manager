"""
Data models for the Energy Grid Management Environment.

Multi-agent architecture — three specialized action types:

    PlanningAgentAction  — long-horizon capital decisions (infrequent)
    DispatchAgentAction  — real-time generation control (every step)
    MarketAgentAction    — economic optimization (every step)
    EnergyGridAction     — backward-compatible unified action (legacy / single-agent)

All agents read the same EnergyGridObservation each step.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import Field, field_validator, ConfigDict
from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Agent 1 — Planning Agent Action
# Capital investment and decommissioning decisions.
# Fires infrequently — only when a build decision is needed.
# Rewarded on: long-run cost, emissions reduction, capital efficiency.
# ---------------------------------------------------------------------------

class PlanningAgentAction(Action):
    """
    Long-horizon capital investment decisions.

    The planning agent decides WHAT to build and WHEN to decommission.
    It operates on a multi-step horizon (15–72 steps) and is rewarded
    for long-run cost, emissions reduction, and capital efficiency.

    Typical firing pattern: once at episode start, then reactively
    when a major event changes the capacity outlook (coal outage,
    nuclear SCRAM, price spike).

    Actions here are irreversible — build costs are deducted immediately,
    decommissioning is permanent. The planning agent must reason about
    build times (nuclear=15 steps, hydro=10, wind/solar=6–8) and
    anticipate future demand.
    """

    plant_action: str = Field(
        default="none",
        description=(
            "Capital investment decision. "
            "Options: 'none' | 'build_solar' | 'build_wind' | "
            "'build_hydro' | 'build_nuclear' | 'close_coal'. "
            "Cost deducted immediately. Capacity comes online after build steps. "
            "Only available in Hard task. 'close_coal' is permanent."
        ),
    )

    target_step: Optional[int] = Field(
        default=None,
        description=(
            "Optional hint: which future step this build is targeting. "
            "Used for logging and coordination with dispatch agent. "
            "Does not affect simulator physics."
        ),
    )

    rationale: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text rationale for the planning decision. "
            "Useful for chain-of-thought logging and post-hoc analysis. "
            "Max 200 chars. Not used by simulator."
        ),
        max_length=200,
    )

    @field_validator("plant_action")
    @classmethod
    def validate_plant_action(cls, v: str) -> str:
        allowed = {
            "none", "build_solar", "build_wind",
            "build_hydro", "build_nuclear", "close_coal",
        }
        if v not in allowed:
            raise ValueError(f"plant_action must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Agent 2 — Dispatch Agent Action
# Real-time generation control. Fires every step.
# Rewarded on: frequency stability, unmet demand, spinning reserve.
# ---------------------------------------------------------------------------

class DispatchAgentAction(Action):
    """
    Real-time dispatchable generation control.

    The dispatch agent controls all online generation sources and
    battery storage on a per-step basis. It sees the full observation
    and must respond to frequency deviations, demand surges, and
    weather-driven renewable variability within a single step.

    Rewarded on: frequency stability (target 50.0 Hz ± 0.1),
    unmet demand (target 0 MW), spinning reserve adequacy (≥20% demand).

    The dispatch agent does NOT make capital decisions — it works with
    whatever capacity the planning agent has built. If nuclear isn't
    online yet, the dispatch agent must compensate with coal + battery.
    """

    coal_delta: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description=(
            "Change in coal plant output this step (MW). "
            "Range: -100 to +100 MW. Clamped to min-stable (200 MW) "
            "and max (600 MW). Ignored during startup sequence."
        ),
    )

    hydro_delta: float = Field(
        default=0.0,
        ge=-80.0,
        le=80.0,
        description=(
            "Change in hydro turbine output this step (MW). "
            "Range: -80 to +80 MW. Depletes reservoir when positive. "
            "Only effective if hydro plant is built and available."
        ),
    )

    nuclear_delta: float = Field(
        default=0.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Change in nuclear output this step (MW). "
            "Range: -10 to +10 MW. Extremely slow ramp. "
            "Clamped to min-stable (300 MW). Zero during SCRAM."
        ),
    )

    battery_mode: str = Field(
        default="idle",
        description=(
            "Battery operation mode. "
            "'charge' | 'discharge' | 'idle'. "
            "Charge/discharge rate fixed at 50 MW. "
            "Cannot do both in the same step."
        ),
    )

    emergency_coal_boost: bool = Field(
        default=False,
        description=(
            "Override coal ramp limits: instantly add +200 MW above normal max. "
            "Damages plant — reduces max output by 50 MW for 5 steps. "
            "Use only to prevent imminent blackout. "
            "Blocked after 5 uses per episode."
        ),
    )
    
    reroute_transmission: bool = False

    @field_validator("battery_mode")
    @classmethod
    def validate_battery_mode(cls, v: str) -> str:
        allowed = {"charge", "discharge", "idle"}
        if v not in allowed:
            raise ValueError(f"battery_mode must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Agent 3 — Market Agent Action
# Economic optimization and demand-side management. Fires every step.
# Rewarded on: cost efficiency, grid trading profit, DR utilization.
# ---------------------------------------------------------------------------

class MarketAgentAction(Action):
    """
    Economic optimization and demand-side management.

    The market agent manages the economic layer of grid operation:
    voluntary demand reduction (industrial load shedding), coal price
    negotiation, and grid import/export trading with neighboring grids.

    Rewarded on: cumulative operational cost (minimize), demand
    response utilization efficiency (maximize DR value per MW),
    grid trading profit (maximize export revenue - import cost).

    Operates in tension with the dispatch agent: the dispatch agent
    wants maximum generation headroom, the market agent wants to
    minimize fuel spend. This tension mirrors real grid operations
    where the system operator (dispatch) and trading desk (market)
    have different incentives.
    """

    scheduled_dr_mw: float = 0.0
    scheduled_dr_start: int = 0
    scheduled_dr_duration: int = 0
    
    demand_response_mw: float = Field(
        default=0.0,
        ge=0.0,
        le=150.0,
        description=(
            "Request voluntary industrial demand reduction (MW). "
            "Range: 0–150 MW. Reduces effective demand immediately. "
            "Costs 0.5 capital units per MW in Hard task. "
            "Strong penalty in easy/medium to prevent DR spam."
        ),
    )

    grid_export_mw: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Export excess generation to neighboring grid (MW). "
            "Range: 0–100 MW. Earns feed-in credits. "
            "Only effective when total supply > demand + export capacity. "
            "New field — not in legacy single-agent action."
        ),
    )

    grid_import_mw: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Import power from neighboring grid (MW). "
            "Range: 0–100 MW. Costs coal_price * 1.2 per MWh. "
            "Useful during coal outage or nuclear SCRAM. "
            "New field — not in legacy single-agent action."
        ),
    )

    coal_price_bid: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=3.0,
        description=(
            "Optional bid to negotiate coal fuel price for next 3 steps. "
            "Range: 0.5–3.0 multiplier. Accepted probabilistically "
            "based on market conditions. None = accept spot price. "
            "New field — not in legacy single-agent action."
        ),
    )

    @field_validator("scheduled_dr_mw")
    @classmethod
    def validate_scheduled_dr(cls, v):
        return max(0.0, min(150.0, v))

    @field_validator("scheduled_dr_duration")
    @classmethod
    def validate_duration(cls, v):
        return max(0, min(24, v))


# ---------------------------------------------------------------------------
# Legacy unified action (backward compatibility)
# Single-agent mode — combines all three agent actions into one.
# Used by /step endpoint, baseline.py, and existing training scripts.
# ---------------------------------------------------------------------------

class EnergyGridAction(Action):
    """
    Unified action for single-agent mode (backward compatible).

    Combines PlanningAgentAction + DispatchAgentAction + MarketAgentAction
    into one flat structure. The simulator accepts this directly via the
    original /step endpoint.

    For multi-agent mode, use the three specialized action classes
    and the /step/planning, /step/dispatch, /step/market endpoints.
    """

    # Dispatch fields
    coal_delta: float = Field(default=0.0, ge=-100.0, le=100.0,
        description="Change in coal output (MW). Range: -100 to +100.")
    hydro_delta: float = Field(default=0.0, ge=-80.0, le=80.0,
        description="Change in hydro output (MW). Range: -80 to +80.")
    nuclear_delta: float = Field(default=0.0, ge=-10.0, le=10.0,
        description="Change in nuclear output (MW). Range: -10 to +10.")
    battery_mode: str = Field(default="idle",
        description="Battery mode: 'charge' | 'discharge' | 'idle'.")
    emergency_coal_boost: bool = Field(default=False,
        description="Override ramp limits. Damages plant for 5 steps.")

    # Planning field
    plant_action: str = Field(default="none",
        description=(
            "Capital investment: 'none' | 'build_solar' | 'build_wind' | "
            "'build_hydro' | 'build_nuclear' | 'close_coal'."
        ))

    # Market field
    demand_response_mw: float = Field(default=0.0, ge=0.0, le=150.0,
        description="Voluntary demand reduction (MW). Range: 0–150.")

    # Phase 3 additions
    reroute_transmission: bool = False
    scheduled_dr_mw: float = 0.0
    scheduled_dr_start: int = 0
    scheduled_dr_duration: int = 0

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
            "none", "build_solar", "build_wind",
            "build_hydro", "build_nuclear", "close_coal",
        }
        if v not in allowed:
            raise ValueError(f"plant_action must be one of {allowed}, got '{v}'")
        return v

    # ── Conversion helpers ─────────────────────────────────────────────────

    def to_dispatch(self) -> DispatchAgentAction:
        """Extract dispatch fields as a DispatchAgentAction."""
        return DispatchAgentAction(
            coal_delta=self.coal_delta,
            hydro_delta=self.hydro_delta,
            nuclear_delta=self.nuclear_delta,
            battery_mode=self.battery_mode,
            emergency_coal_boost=self.emergency_coal_boost,
        )

    def to_planning(self) -> PlanningAgentAction:
        """Extract planning fields as a PlanningAgentAction."""
        return PlanningAgentAction(plant_action=self.plant_action)

    def to_market(self) -> MarketAgentAction:
        """Extract market fields as a MarketAgentAction."""
        return MarketAgentAction(
        demand_response_mw=self.demand_response_mw,
        scheduled_dr_mw=self.scheduled_dr_mw,
        scheduled_dr_start=self.scheduled_dr_start,
        scheduled_dr_duration=self.scheduled_dr_duration,
        )
        
    @classmethod
    def from_agents(
        cls,
        dispatch: DispatchAgentAction,
        planning: PlanningAgentAction,
        market: MarketAgentAction,
    ) -> "EnergyGridAction":
        """Merge three agent actions back into a unified action."""
        return cls(
            coal_delta=dispatch.coal_delta,
            hydro_delta=dispatch.hydro_delta,
            nuclear_delta=dispatch.nuclear_delta,
            battery_mode=dispatch.battery_mode,
            emergency_coal_boost=dispatch.emergency_coal_boost,
            plant_action=planning.plant_action,
            demand_response_mw=market.demand_response_mw,
        )


# ---------------------------------------------------------------------------
# Observation (unchanged — all agents read the same state)
# ---------------------------------------------------------------------------

class EnergyGridObservation(Observation):
    """
    Full grid state returned to ALL agents after each step / reset.

    All three agents receive the identical observation. Each agent
    is expected to attend to the fields relevant to its role:
      - Planning agent: capital_budget, plants_building, cumulative_emissions
      - Dispatch agent: frequency_hz, unmet_demand_mw, blackout_risk, coal_mw
      - Market agent: cumulative_cost, coal_price, active_events
    """

    model_config = ConfigDict(extra='ignore')

    # Demand & time
    demand_mw: float = Field(default=0.0)
    hour: int = Field(default=0, ge=0, le=23)
    day: int = Field(default=1, ge=1)
    step: int = Field(default=0, ge=0)
    season: str = Field(default="spring")
    industrial_demand_mw: float = 0.0
    datacenter_demand_mw: float = 0.0
    scheduled_dr_mw: float = 0.0

    # Generation
    coal_mw: float = Field(default=0.0)
    coal_online: bool = Field(default=True)
    coal_max_mw: float = Field(default=600.0)
    coal_startup_remaining: int = Field(default=0)
    coal_price: float = Field(default=1.0)

    solar_mw: float = Field(default=0.0)
    solar_weather: Literal["clear", "partial", "cloudy", "storm"] = Field(default="clear")

    wind_mw: float = Field(default=0.0)
    wind_speed_ms: float = Field(default=0.0)

    hydro_mw: float = Field(default=0.0)
    reservoir_mwh: float = Field(default=0.0)
    reservoir_capacity_mwh: float = Field(default=1000.0)

    nuclear_mw: float = Field(default=0.0)
    nuclear_online: bool = Field(default=False)
    nuclear_trip_remaining: int = Field(default=0)

    battery_mwh: float = Field(default=100.0)
    battery_capacity_mwh: float = Field(default=200.0)

    # Grid health
    unmet_demand_mw: float = Field(default=0.0)
    frequency_hz: float = Field(default=50.0)
    load_shedding_mw: float = Field(default=0.0)
    blackout_risk: str = Field(default="none")
    spinning_reserve_mw: float = Field(default=0.0)
    spinning_reserve_required_mw: float = Field(default=0.0,
        description="Required spinning reserve (MW) based on N-1 contingency criterion. Dynamic.")

    duck_curve_stress: float = 0.0
    voltage_stability_index: float = 100.0
    spot_price: float = 1.0
    anomaly_score: float = 0.0

    # Events & construction
    active_events: List[str] = Field(default_factory=list)
    plants_building: List[Dict[str, Any]] = Field(default_factory=list)
    steps_until_projected_shortfall: int = 999
    steps_until_shortfall: int = 999
    fdi_active: bool = False

    # Economics
    capital_budget: float = Field(default=0.0)
    cumulative_cost: float = Field(default=0.0)
    cumulative_emissions_tons: float = Field(default=0.0)

    # Market fields (new — populated by market step)
    grid_export_mw: float = Field(default=0.0,
        description="MW currently being exported to neighboring grid.")
    grid_import_mw: float = Field(default=0.0,
        description="MW currently being imported from neighboring grid.")
    trading_credits: float = Field(default=0.0,
        description="Cumulative grid trading profit/loss.")

    # Episode metadata
    episode_ended_early: bool = Field(default=False)
    task_id: str = Field(default="easy")

    # Unified reward (step reward)
    reward: float = Field(default=0.0,
        description="Step reward signal for single-agent backward compatibility.")

    # Agent-specific reward breakdown (new)
    dispatch_reward: float = Field(default=0.0,
        description="Reward component attributable to dispatch decisions.")
    planning_reward: float = Field(default=0.0,
        description="Reward component attributable to planning decisions.")
    market_reward: float = Field(default=0.0,
        description="Reward component attributable to market decisions.")

    # Phase 1: New fields for feature parity
    # -- Coal health (Feature 3)
    coal_health_pct: float = Field(default=100.0, ge=0.0, le=100.0,
        description="Coal plant health (0-100%). Degrades with usage and boost damage.")
    
    # -- Duck curve stress (Feature 5)
    duck_curve_stress_mw_per_step: float = Field(default=0.0,
        description="Rate of change of net load (demand - renewables), MW/step. "
                    "Positive = increasing net load (ramp up). Negative = decreasing net load (ramp down).")
    
    # -- Spot price / LMP (Feature 8)
    spot_price: float = Field(default=1.0,
        description="Real-time electricity spot price (£/MWh equivalent). "
                    "Rises during scarcity; reflects coal + carbon cost + demand premium.")
    carbon_price_per_ton: float = Field(default=45.0,
        description="Carbon price (£/ton CO2). Visible in Hard task; affects spot price.")
    
    # -- Frequency rate of change (Foundation for grid stability)
    rate_of_change_hz_per_step: float = Field(default=0.0,
        description="Rate of change of frequency (Hz/step). RoCoF signal.")
    
    # -- Voltage stability index (Feature 10)
    voltage_stability_index: float = Field(default=100.0, ge=0.0, le=100.0,
        description="Voltage stability proxy (0-100). Ratio of synchronous to total generation. "
                    "100 = all synchronous (stable). 0 = all inverter-based (risky).")


# ---------------------------------------------------------------------------
# Filtered Observation Classes for Multi-Agent Architecture
# ---------------------------------------------------------------------------

class PlanningAgentObservation(EnergyGridObservation):
    """
    Observation filtered for Planning Agent.
    
    Sees long-run capital, construction state, cumulative metrics.
    Hides real-time dispatch details (frequency, spinning reserve, etc.).
    """
    pass  # Inherits all fields; filtering applied at environment level


class DispatchAgentObservation(EnergyGridObservation):
    """
    Observation filtered for Dispatch Agent.
    
    Sees real-time frequency, reserve, generation outputs, demand.
    Hides capital budget, economic details beyond current cost.
    """
    pass  # Inherits all fields; filtering applied at environment level


class MarketAgentObservation(EnergyGridObservation):
    """
    Observation filtered for Market Agent.
    
    Sees prices, costs, trading state, demand.
    Hides detailed plant internals, capital budget details.
    """
    pass  # Inherits all fields; filtering applied at environment level


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ScalerHackathonAction = EnergyGridAction
ScalerHackathonObservation = EnergyGridObservation