"""
Energy Grid Physics Simulator.

This module contains all physical simulation logic for the energy grid
environment. It is intentionally decoupled from the OpenEnv interface —
no FastAPI, no Pydantic models imported here. Pure simulation physics.

Systems modelled:
    - Demand: daily curve + seasonal multiplier + stochastic noise
    - Coal: ramp limits, min-stable generation, startup/shutdown,
            emergency boost, thermal damage, CO2 emissions
    - Solar: sine-curve irradiance, weather multiplier, panel degradation
    - Wind: autocorrelated stochastic wind speed, realistic power curve
    - Hydro: reservoir, natural inflow, spillage, drought/rainfall
    - Nuclear: baseload, slow ramp, SCRAM event, restart sequence
    - Battery: round-trip efficiency, C-rate limits, cycle degradation
    - Frequency: inertia model, RoCoF, governor response, load shedding,
                 cascade/blackout protection thresholds
    - Events: 12 event types, seeded deterministic scheduling
    - Plant construction: capital budget, build queue, capacity online
    - Economics: coal price fluctuation, cumulative cost tracking
    - Emissions: CO2 per MWh coal
    - Demand response: voluntary industrial load shed
    - Spinning reserve: headroom tracking and penalty signal
    - Transmission: capacity constraint, grid_fault reduction
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Coal
COAL_MIN_MW: float = 200.0          # minimum stable generation
COAL_MAX_MW: float = 600.0          # nameplate capacity
COAL_RAMP_MW: float = 100.0         # max ramp per step
COAL_STARTUP_STEPS: int = 3         # steps to reach min-stable after restart
COAL_EMISSION_FACTOR: float = 0.9   # tons CO2 per MWh
COAL_EMERGENCY_BOOST_INCREMENT_MW: float = 100.0   # per-step ramp during boost
COAL_EMERGENCY_BOOST_CEILING_MW: float = 150.0     # absolute MW above normal max_mw
COAL_BOOST_DAMAGE_MW: float = 50.0       # max reduction after boost
COAL_BOOST_DAMAGE_STEPS: int = 5         # steps damage lasts
COAL_RESTART_COST: float = 0.5           # startup fuel penalty

# Solar
SOLAR_MAX_MW: float = 300.0
SOLAR_DEGRADATION_PER_DAY: float = 0.001   # 0.1% per simulated day

# Wind
WIND_MAX_MW: float = 250.0
WIND_CUT_IN_MS: float = 3.0
WIND_RATED_MS: float = 12.0
WIND_CUT_OUT_MS: float = 25.0
WIND_MEAN_SPEED: float = 8.0
WIND_STD_SPEED: float = 3.0
WIND_AUTOCORR: float = 0.85         # autocorrelation coefficient

# Hydro
HYDRO_MAX_MW: float = 200.0
HYDRO_RAMP_MW: float = 80.0
HYDRO_RESERVOIR_CAP_MWH: float = 1000.0
HYDRO_SPILLAGE_THRESHOLD: float = 950.0
HYDRO_CRITICAL_LOW: float = 50.0
HYDRO_INFLOW_MEAN: float = 15.0     # MWh per step
HYDRO_INFLOW_STD: float = 3.0
HYDRO_INERTIA_PER_100MW: float = 3.0  # seconds

# Nuclear
NUCLEAR_MIN_MW: float = 300.0       # minimum stable (cannot go below)
NUCLEAR_MAX_MW: float = 500.0
NUCLEAR_RAMP_MW: float = 10.0       # very slow
NUCLEAR_STARTUP_STEPS: int = 12      # post-SCRAM restart
NUCLEAR_FUEL_COST: float = 0.05     # units per MWh
NUCLEAR_INERTIA_PER_100MW: float = 5.0  # seconds

# Battery
BATTERY_MAX_MWH: float = 200.0
BATTERY_CHARGE_RATE_MW: float = 50.0
BATTERY_DISCHARGE_RATE_MW: float = 50.0
BATTERY_EFFICIENCY: float = 0.92    # round-trip efficiency (per direction: sqrt)
BATTERY_STEP_EFFICIENCY: float = math.sqrt(0.92)  # ~0.959 per direction
BATTERY_DEGRADATION_PER_CYCLE: float = 0.0005  # 0.05% capacity per full cycle

# Coal inertia
COAL_INERTIA_PER_100MW: float = 4.0  # seconds

# Frequency
FREQ_NOMINAL: float = 50.0
FREQ_PRIMARY_RESPONSE_BAND: float = 0.5   # Hz — governor activates
FREQ_LOAD_SHED_1: float = 49.0            # Hz — first load shed
FREQ_LOAD_SHED_2: float = 48.5            # Hz — second load shed
FREQ_BLACKOUT_LOW: float = 47.5           # Hz — full blackout
FREQ_BLACKOUT_HIGH: float = 51.5          # Hz — over-frequency blackout
FREQ_ROCOF_TRIP: float = 1.0              # Hz/step — protection trip
SYSTEM_BASE_MVA: float = 1000.0           # Fixed rated capacity for swing equation
LOAD_SHED_1_MW: float = 100.0
LOAD_SHED_2_MW: float = 200.0

# Transmission
TRANSMISSION_NOMINAL_MW: float = 1200.0
TRANSMISSION_FAULT_REDUCTION: float = 0.20  # 20% capacity loss during fault

# Demand response
DR_COST_PER_MW: float = 0.5   # capital units per MW shed

# Seasonal demand multipliers
SEASON_MULTIPLIERS: Dict[str, float] = {
    "spring": 1.00,
    "summer": 1.20,
    "autumn": 1.05,
    "winter": 1.30,
}

# Base hourly demand curve (MW) — representative UK-style daily profile
BASE_DEMAND_CURVE: List[float] = [
    420, 400, 385, 375, 370, 380,   # 00:00-05:00  night trough
    480, 600, 700, 780, 820, 850,   # 06:00-11:00  morning ramp
    870, 880, 860, 840, 820, 800,   # 12:00-17:00  daytime plateau
    830, 870, 860, 780, 680, 560,   # 18:00-23:00  evening peak → night
]

# ---------------------------------------------------------------------------
# Plant build specs (used by Hard task)
# ---------------------------------------------------------------------------

PLANT_BUILD_SPECS: Dict[str, Dict[str, Any]] = {
    "build_solar": {
        "type": "solar",
        "cost": 500,
        "build_steps": 8,
        "capacity_mw": SOLAR_MAX_MW,
    },
    "build_wind": {
        "type": "wind",
        "cost": 400,
        "build_steps": 6,
        "capacity_mw": WIND_MAX_MW,
    },
    "build_hydro": {
        "type": "hydro",
        "cost": 600,
        "build_steps": 10,
        "capacity_mw": HYDRO_MAX_MW,
    },
    "build_nuclear": {
        "type": "nuclear",
        "cost": 1000,
        "build_steps": 15,
        "capacity_mw": NUCLEAR_MAX_MW,
    },
}

COAL_CLOSE_COST: float = 300.0   # decommissioning cost


# ---------------------------------------------------------------------------
# Dataclasses for internal simulator state
# ---------------------------------------------------------------------------

@dataclass
class CoalState:
    output_mw: float = 400.0
    online: bool = True
    startup_steps_remaining: int = 0
    max_mw: float = COAL_MAX_MW
    boost_damage_steps: int = 0       # steps of reduced max remaining
    available: bool = True            # False after close_coal action
    health_pct: float = 100.0          # Health metric (0-100%). Feature 3
    cumulative_runtime_hours: float = 0.0  # Track total usage for health degradation


@dataclass
class SolarState:
    output_mw: float = 0.0
    available: bool = False
    degradation_factor: float = 1.0   # 1.0 = new panels
    clearness_index: float = 0.8      # Continuous stochastic variation. Feature 13


@dataclass
class WindState:
    output_mw: float = 0.0
    available: bool = False
    wind_speed_ms: float = WIND_MEAN_SPEED
    weather: str = "normal"           # internal, not exposed directly


@dataclass
class HydroState:
    output_mw: float = 0.0
    available: bool = False
    reservoir_mwh: float = 600.0
    natural_inflow_mwh: float = HYDRO_INFLOW_MEAN
    drought_steps: int = 0


@dataclass
class NuclearState:
    output_mw: float = 0.0
    available: bool = False
    online: bool = False
    trip_steps_remaining: int = 0


@dataclass
class BatteryState:
    level_mwh: float = 100.0
    capacity_mwh: float = BATTERY_MAX_MWH
    total_cycles: float = 0.0         # cumulative full-cycle equivalent


@dataclass
class FrequencyState:
    frequency: float = FREQ_NOMINAL
    rocof: float = 0.0                # Hz per step
    primary_response_active: bool = False
    primary_response_steps: int = 0   # steps governor has been active
    load_shedding_mw: float = 0.0


@dataclass
class ConstructionEntry:
    plant_type: str = ""
    steps_remaining: int = 0
    capacity_mw: float = 0.0


@dataclass
class GridSimState:
    """Complete mutable state of the grid simulator."""
    # Plants
    coal: CoalState = field(default_factory=CoalState)
    solar: SolarState = field(default_factory=SolarState)
    wind: WindState = field(default_factory=WindState)
    hydro: HydroState = field(default_factory=HydroState)
    nuclear: NuclearState = field(default_factory=NuclearState)
    battery: BatteryState = field(default_factory=BatteryState)
    frequency: FrequencyState = field(default_factory=FrequencyState)
    scheduled_dr_queue: List[Tuple[int, float]] = field(default_factory=list)
    last_reroute: bool = False

    # Time
    step: int = 0
    day: int = 1
    season: str = "spring"

    # Grid
    demand_mw: float = 500.0
    baseline_demand_mw: float = 0.0
    industrial_demand_mw: float = 0.0
    datacenter_demand_mw: float = 0.0
    transmission_capacity_mw: float = TRANSMISSION_NOMINAL_MW
    solar_weather: str = "clear"
    prev_coal_delta: Optional[float] = None
    coal_flip_streak: int = 0
    prev_net_load_mw: float = 0.0     # For duck curve stress. Feature 5

    # Events
    active_events: List[str] = field(default_factory=list)

    # Construction queue
    construction_queue: List[ConstructionEntry] = field(default_factory=list)

    # Economics
    capital_budget: float = 0.0
    coal_price: float = 1.0
    carbon_price: float = 45.0   # £/ton CO2, realistic UK ETS price
    cumulative_cost: float = 0.0
    cumulative_emissions: float = 0.0
    cumulative_feedin_credits: float = 0.0

    # Episode tracking
    episode_ended: bool = False
    blackout_this_step: bool = False
    steps_demand_met: int = 0
    total_steps: int = 24

    # Random state (seeded per task for determinism)
    rng: random.Random = field(default_factory=random.Random)

    # ---- Demand response tracking ----
    total_demand_response: float = 0.0

    # ---- Emergency usage tracking ----
    boost_used_count: int = 0

    # ---- Battery mode tracking ----
    prev_battery_mode: str = "idle"


# ---------------------------------------------------------------------------
# Phase 1 Computation Functions (Features 2, 8, 10)
# ---------------------------------------------------------------------------

def compute_duck_curve_stress(state: GridSimState) -> float:
    """
    Duck curve stress = change in net load between steps.

    net_load = demand - renewable generation (solar + wind)

    Positive → ramp up required (evening peak)
    Negative → ramp down (solar surge)
    """

    renewable_gen = 0.0

    if state.solar.available:
        renewable_gen += state.solar.output_mw

    if state.wind.available:
        renewable_gen += state.wind.output_mw

    net_load = state.demand_mw - renewable_gen

    # Compute delta vs previous step
    stress = net_load - state.prev_net_load_mw

    # Update state for next step
    state.prev_net_load_mw = net_load

    return stress

def compute_required_spinning_reserve(state: 'GridSimState') -> float:
    """
    Dynamic N-1 spinning reserve requirement (Feature 2).
    
    Reserve scales with largest single infeed (contingency criterion).
    - If nuclear available and online: 80% of nuclear capacity
    - Else if coal available and online: 80% of coal capacity (or damaged max)
    - Else if hydro available: 50% of hydro output
    - Minimum floor: 15% of current demand
    """
    # Check nuclear first (most inertia)
    if state.nuclear.available and state.nuclear.online:
        contingency_mw = NUCLEAR_MAX_MW * 0.8
    # Check coal (largest conventional plant)
    elif state.coal.available and state.coal.online:
        contingency_mw = state.coal.max_mw * 0.8  # respects boost damage
    # Check hydro (fallback)
    elif state.hydro.available and state.hydro.output_mw > 0:
        contingency_mw = state.hydro.output_mw * 0.5
    else:
        contingency_mw = 0.0
    
    # Floor: never drop below 15% of demand (ensures minimum responsiveness)
    min_reserve = state.demand_mw * 0.15
    return max(min_reserve, contingency_mw)

def compute_spot_price(state, coal_price, carbon_price, task_id, actual_reserve, required_reserve) -> float:
    """
    Spot price driven purely by reserve scarcity (clean + consistent).

    Uses N-1 reserve ratio instead of naive supply/demand.
    """

    # Edge case: no demand
    if state.demand_mw < 1.0:
        return coal_price

    # ── Reserve-based scarcity ───────────────────────────────
    reserve_ratio = actual_reserve / max(required_reserve, 1.0)

    # ── Pricing curve ────────────────────────────────────────
    if reserve_ratio < 0.5:
        scarcity_mult = 3.0
    elif reserve_ratio < 1.0:
        scarcity_mult = 2.0 + (1.0 - reserve_ratio)
    elif reserve_ratio < 1.5:
        scarcity_mult = 1.2
    else:
        scarcity_mult = 1.0

    spot = coal_price * scarcity_mult

    # Carbon pricing (hard mode)
    if task_id == "hard" and carbon_price > 0:
        spot *= (1.0 + carbon_price / 100.0)

    return min(6.0, spot)

def compute_voltage_stability_index(
    coal_state: CoalState,
    hydro_state: HydroState,
    nuclear_state: NuclearState,
    solar_output: float,
    wind_output: float,
) -> float:
    """
    Voltage stability proxy (Feature 10).
    
    Ratio of synchronous to total generation.
    100 = all synchronous (stable). 0 = all inverter-based (risky).
    Real grids need >50% synchronous during high renewables.
    """
    sync_gen = 0.0
    if coal_state.online and coal_state.available:
        sync_gen += coal_state.output_mw
    if hydro_state.available and hydro_state.output_mw > 0:
        sync_gen += hydro_state.output_mw
    if nuclear_state.available and nuclear_state.online:
        sync_gen += nuclear_state.output_mw
    
    inverter_gen = solar_output + wind_output
    total_gen = sync_gen + inverter_gen
    
    if total_gen < 1.0:
        return 100.0  # edge case: no generation
    
    ratio = sync_gen / total_gen
    return ratio * 100.0

# ---------------------------------------------------------------------------
# Demand simulation
# ---------------------------------------------------------------------------

def compute_demand(
    hour: int,
    season: str,
    active_events: List[str],
    rng: random.Random,
    noise_std: float = 20.0,
) -> Tuple[float, float, float]:
    """
    Compute grid demand for the given hour.

    Base curve × seasonal multiplier + stochastic noise + event modifiers.
    """
    base = BASE_DEMAND_CURVE[hour]
    seasonal = SEASON_MULTIPLIERS.get(season, 1.0)

    demand = base * seasonal

    # Event modifiers
    for event in active_events:
        if event == "heatwave":
            demand *= 1.25
        elif event == "cold_snap":
            demand *= 1.20

     # Noise
    demand += rng.gauss(0, 20.0)

    demand = max(200.0, demand)

    baseline = demand * 0.6
    industrial = demand * 0.3
    datacenter = demand * 0.1

    return baseline, industrial, datacenter

def compute_anomaly_score(state, event_schedule, current_step, rng):
    """
    Noisy pre-warning signal for upcoming events (Feature 12).

    Rises 3–5 steps before an event.
    Non-deterministic due to noise.
    """

    score = 0.0

    for step, events in event_schedule.items():
        delta = step - current_step

        # Only look ahead up to 5 steps
        if 0 < delta <= 5:
            # closer event → higher signal
            score += (6 - delta) * 0.15

    # Add noise so agent cannot perfectly predict
    noise = rng.gauss(0, 0.05)

    return max(0.0, min(0.8, score + noise))

def compute_shortfall_projection(state: GridSimState) -> int:
    """
    Estimate steps until supply deficit under worst-case conditions.
    """

    supply = (
        state.coal.output_mw +
        state.solar.output_mw +
        state.wind.output_mw +
        state.hydro.output_mw +
        state.nuclear.output_mw
    )

    demand = state.demand_mw

    if supply >= demand:
        return 999

    deficit_ratio = (demand - supply) / max(demand, 1.0)

    if deficit_ratio > 0.3:
        return 1
    elif deficit_ratio > 0.2:
        return 2
    elif deficit_ratio > 0.1:
        return 3
    else:
        return 5

# ---------------------------------------------------------------------------
# Solar irradiance
# ---------------------------------------------------------------------------

def step_solar_clearness(solar_state, rng):
    """
    AR(1)-style stochastic solar variability (Feature 13)
    """

    alpha = 0.92
    noise = rng.gauss(0, 0.08)

    solar_state.clearness_index = (
        alpha * solar_state.clearness_index +
        (1 - alpha) * 0.8 +
        noise
    )

    solar_state.clearness_index = max(0.3, min(1.0, solar_state.clearness_index))

def compute_solar_output(
    hour: int,
    solar_state: SolarState,
    solar_weather: str,
) -> float:
    """
    Realistic solar output using sine irradiance curve.

    Panels only produce between 06:00 and 18:00. Weather and
    long-term degradation reduce output.
    """
    if not solar_state.available:
        return 0.0

    if hour < 6 or hour >= 18:
        return 0.0

    # Irradiance: sine curve peaking at noon
    irradiance = math.sin(math.pi * (hour - 6) / 12.0)

    weather_multipliers = {
        "clear": 1.0,
        "partial": 0.6,
        "cloudy": 0.3,
        "storm": 0.0,
    }
    weather_mult = weather_multipliers.get(solar_weather, 1.0)

    # Continuous clearness index noise superimposed on weather (Feature 13)
    output = SOLAR_MAX_MW * irradiance * weather_mult * solar_state.clearness_index * solar_state.degradation_factor
    return max(0.0, output)


# ---------------------------------------------------------------------------
# Wind power curve
# ---------------------------------------------------------------------------

def compute_wind_output(wind_state: WindState) -> float:
    """
    Realistic wind turbine power curve.

    Output scales as wind_speed³ between cut-in and rated speed.
    Zero below cut-in (calm) and above cut-out (storm protection).
    """
    if not wind_state.available:
        return 0.0

    v = wind_state.wind_speed_ms

    if v < WIND_CUT_IN_MS or v > WIND_CUT_OUT_MS:
        return 0.0

    if v >= WIND_RATED_MS:
        return WIND_MAX_MW

    # Cubic scaling between cut-in and rated speed
    fraction = (v - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS)
    output = WIND_MAX_MW * (fraction ** 3)
    return max(0.0, min(WIND_MAX_MW, output))


def step_wind_speed(wind_state: WindState, rng: random.Random) -> None:
    """
    Update wind speed using AR(1) autocorrelated process.

    Ensures wind is persistent (calm periods last multiple steps)
    rather than independent white noise.
    """
    innovation = rng.gauss(0, WIND_STD_SPEED * math.sqrt(1 - WIND_AUTOCORR**2))
    new_speed = (
        WIND_AUTOCORR * wind_state.wind_speed_ms
        + (1 - WIND_AUTOCORR) * WIND_MEAN_SPEED
        + innovation
    )
    wind_state.wind_speed_ms = max(0.0, min(35.0, new_speed))


# ---------------------------------------------------------------------------
# Hydro reservoir
# ---------------------------------------------------------------------------

def step_hydro(
    hydro_state: HydroState,
    requested_output_mw: float,
    active_events: List[str],
    rng: random.Random,
) -> Tuple[float, bool]:
    """
    Update hydro reservoir and compute actual output.

    Depletes reservoir when generating. Receives natural inflow each step.
    Spillage occurs if reservoir is near-full (waste + penalty signal).
    
    Returns: (output_mw, spillage_occurred)
    """
    if not hydro_state.available:
        return 0.0, False

    # Natural inflow (reduced by drought)
    if "drought" in active_events:
        inflow = rng.gauss(2.0, 0.5)
        hydro_state.drought_steps += 1
    elif "rainfall" in active_events:
        inflow = rng.gauss(HYDRO_INFLOW_MEAN + 50.0, 5.0)
    else:
        inflow = rng.gauss(HYDRO_INFLOW_MEAN, HYDRO_INFLOW_STD)

    inflow = max(0.0, inflow)
    hydro_state.natural_inflow_mwh = inflow
    hydro_state.reservoir_mwh = min(
        HYDRO_RESERVOIR_CAP_MWH,
        hydro_state.reservoir_mwh + inflow,
    )

    # Clamp requested output to what reservoir can support
    max_from_reservoir = min(
        requested_output_mw,
        hydro_state.reservoir_mwh,   # can't generate more than stored
        HYDRO_MAX_MW,
    )
    actual_output = max(0.0, max_from_reservoir)

    # Deplete reservoir (accounting for 87% efficiency)
    hydro_state.reservoir_mwh = max(0.0, hydro_state.reservoir_mwh - (actual_output / 0.87))

    # Automatic spillage if near capacity
    spillage_occurred = False
    if hydro_state.reservoir_mwh > HYDRO_SPILLAGE_THRESHOLD:
        spilled = hydro_state.reservoir_mwh - HYDRO_SPILLAGE_THRESHOLD
        hydro_state.reservoir_mwh = HYDRO_SPILLAGE_THRESHOLD
        spillage_occurred = True

    return actual_output, spillage_occurred


# ---------------------------------------------------------------------------
# Coal plant dynamics
# ---------------------------------------------------------------------------

def step_coal(
    coal_state: CoalState,
    delta_mw: float,
    emergency_boost: bool,
    state: Optional[GridSimState] = None,
) -> float:
    """
    Update coal plant output respecting ramp limits, startup sequence,
    minimum stable generation, and emergency boost mechanics.
    """
    if not coal_state.available:
        return 0.0

    # Recover from boost damage
    if coal_state.boost_damage_steps > 0:
        coal_state.boost_damage_steps -= 1
        if coal_state.boost_damage_steps == 0:
            coal_state.max_mw = COAL_MAX_MW   # restore after damage period

    # Update health based on usage and damage (Feature 3)
    if coal_state.online and coal_state.output_mw > COAL_MIN_MW:
        # Degrade by ~0.5% per step at full load; scales with output level
        usage_factor = coal_state.output_mw / COAL_MAX_MW  # 0 to 1
        health_degradation = usage_factor * 0.005  # 0.5% per step at 100% load
        coal_state.health_pct = max(0.0, coal_state.health_pct - health_degradation)
        coal_state.cumulative_runtime_hours += 1.0 / 60.0
    
    # Boost damage also reduces health
    if coal_state.boost_damage_steps > 0:
        health_penalty = 0.02 * (COAL_BOOST_DAMAGE_STEPS - coal_state.boost_damage_steps + 1)
        coal_state.health_pct = max(0.0, coal_state.health_pct - health_penalty)

    # Handle startup sequence
    if not coal_state.online:
        if coal_state.startup_steps_remaining > 0:
            coal_state.startup_steps_remaining -= 1
        if coal_state.startup_steps_remaining == 0:
            coal_state.online = True
            coal_state.output_mw = COAL_MIN_MW
        return coal_state.output_mw if coal_state.online else 0.0

    # Emergency boost (overrides normal ramp limits)
    if emergency_boost:
        # Boost target is always based on BASE capacity (600 + 150 = 750), not damaged capacity
        boost_target = coal_state.max_mw + COAL_EMERGENCY_BOOST_CEILING_MW
        # Apply damage to current max
        coal_state.max_mw = max(
            COAL_MIN_MW,
            coal_state.max_mw - COAL_BOOST_DAMAGE_MW,
        )
        # Increase output toward boost target (capped at 750)
        coal_state.output_mw = min(boost_target, coal_state.output_mw + COAL_EMERGENCY_BOOST_INCREMENT_MW)
        coal_state.boost_damage_steps = COAL_BOOST_DAMAGE_STEPS
    else:
        # Normal ramp
        clamped_delta = max(-COAL_RAMP_MW, min(COAL_RAMP_MW, delta_mw))
        new_output = coal_state.output_mw + clamped_delta

        # Enforce min-stable and max
        if new_output < COAL_MIN_MW:
            # Shutting down below min-stable — start shutdown and deduct restart cost
            coal_state.online = False
            coal_state.output_mw = 0.0
            coal_state.startup_steps_remaining = COAL_STARTUP_STEPS
            if state is not None:
                state.cumulative_cost += COAL_RESTART_COST
            return 0.0

        coal_state.output_mw = min(new_output, coal_state.max_mw)

    # CRITICAL FIX: Absolute ceiling of 750 MW (600 base + 150 emergency boost max)
    coal_state.output_mw = min(750.0, coal_state.output_mw)
    
    return coal_state.output_mw


# ---------------------------------------------------------------------------
# Nuclear plant dynamics
# ---------------------------------------------------------------------------

def step_nuclear(
    nuclear_state: NuclearState,
    delta_mw: float,
    scram_triggered: bool,
) -> float:
    """
    Update nuclear reactor output.

    Extremely slow ramp. SCRAM drops output to zero immediately.
    Restart takes NUCLEAR_STARTUP_STEPS steps to reach minimum stable.
    
    Note: Nuclear is effectively baseload once online; the ramp exists only
    for minor load-following. If online and delta would move output below
    NUCLEAR_MIN_MW, the delta is clamped to zero.
    """
    if not nuclear_state.available:
        return 0.0

    if scram_triggered and nuclear_state.online:
        nuclear_state.online = False
        nuclear_state.output_mw = 0.0
        nuclear_state.trip_steps_remaining = NUCLEAR_STARTUP_STEPS + 1
        return 0.0

    if not nuclear_state.online:
        if nuclear_state.trip_steps_remaining > 0:
            nuclear_state.trip_steps_remaining -= 1
        if nuclear_state.trip_steps_remaining == 0:
            nuclear_state.online = True
            nuclear_state.output_mw = NUCLEAR_MIN_MW
        return nuclear_state.output_mw

    # Normal ramp (very slow) — guard against dropping below min-stable
    clamped_delta = max(-NUCLEAR_RAMP_MW, min(NUCLEAR_RAMP_MW, delta_mw))
    if nuclear_state.output_mw + clamped_delta < NUCLEAR_MIN_MW:
        # Clamp delta to zero if it would violate minimum stable
        clamped_delta = 0.0
    new_output = nuclear_state.output_mw + clamped_delta
    nuclear_state.output_mw = max(NUCLEAR_MIN_MW, min(NUCLEAR_MAX_MW, new_output))

    return nuclear_state.output_mw


# ---------------------------------------------------------------------------
# Battery storage
# ---------------------------------------------------------------------------

def step_battery(
    battery_state: BatteryState,
    mode: str,
    demand_shortfall_mw: float,
) -> Tuple[float, float]:
    """
    Update battery state and return (discharge_mw, charge_mw).

    Enforces C-rate limits and round-trip efficiency losses.
    Tracks cycle count for degradation.

    Returns:
        (energy_injected_mw, energy_drawn_mw) — one will always be 0.
    """
    discharged = 0.0
    charged = 0.0

    if mode == "discharge":
        # How much can we discharge?
        available = min(
            battery_state.level_mwh,
            BATTERY_DISCHARGE_RATE_MW,
        )
        actual = min(available, max(0.0, demand_shortfall_mw))
        # Apply efficiency loss (energy out < energy stored)
        discharged = actual * BATTERY_STEP_EFFICIENCY
        battery_state.level_mwh -= actual
        battery_state.total_cycles += actual / battery_state.capacity_mwh

    elif mode == "charge":
        # How much headroom is available?
        headroom = battery_state.capacity_mwh - battery_state.level_mwh
        actual = min(headroom, BATTERY_CHARGE_RATE_MW)
        # Apply efficiency loss (stored < drawn from grid)
        energy_stored = actual * BATTERY_STEP_EFFICIENCY
        charged = actual
        battery_state.level_mwh += energy_stored
        battery_state.total_cycles += actual / battery_state.capacity_mwh

    # Degrade capacity based on accumulated cycles
    battery_state.capacity_mwh = max(
        50.0,   # floor: battery never totally unusable
        BATTERY_MAX_MWH * (1 - BATTERY_DEGRADATION_PER_CYCLE * battery_state.total_cycles),
    )

    battery_state.level_mwh = max(
        0.0,
        min(battery_state.capacity_mwh, battery_state.level_mwh),
    )

    return discharged, charged


# ---------------------------------------------------------------------------
# System inertia calculation
# ---------------------------------------------------------------------------

def compute_system_inertia(
    coal_state: CoalState,
    hydro_state: HydroState,
    nuclear_state: NuclearState,
) -> float:
    """
    Compute total system inertia constant (seconds).

    Only synchronous machines (coal, hydro, nuclear) contribute inertia.
    Solar, wind, and battery are inverter-based — zero inertia.
    """
    inertia = 0.0

    if coal_state.online and coal_state.available:
        inertia += (coal_state.output_mw / 100.0) * COAL_INERTIA_PER_100MW

    if hydro_state.available and hydro_state.output_mw > 0:
        inertia += (hydro_state.output_mw / 100.0) * HYDRO_INERTIA_PER_100MW

    if nuclear_state.available and nuclear_state.online:
        inertia += (nuclear_state.output_mw / 100.0) * NUCLEAR_INERTIA_PER_100MW

    return max(0.5, inertia)   # floor: some residual inertia always exists


# ---------------------------------------------------------------------------
# Frequency dynamics
# ---------------------------------------------------------------------------

def step_frequency(
    freq_state: FrequencyState,
    power_imbalance_mw: float,
    system_inertia: float,
    demand_mw: float,
) -> Tuple[bool, float]:
    """
    Update grid frequency using simplified swing equation.

    RoCoF = power_imbalance / (2 × system_inertia × system_size_proxy)

    Returns:
        (blackout_triggered, load_shed_mw_this_step)
    """
    # Swing equation: df/dt = ΔP / (2H × S_base)
    # Fixed S_base = 1000 MVA (industry standard)
    rocof = power_imbalance_mw / (2.0 * system_inertia * SYSTEM_BASE_MVA / 100.0)

    # Clamp RoCoF to physically plausible range
    rocof = max(-3.0, min(3.0, rocof))
    freq_state.rocof = rocof

    # Update frequency
    freq_state.frequency += rocof

    # Clamp to physically meaningful range
    freq_state.frequency = max(45.0, min(55.0, freq_state.frequency))

    load_shed = 0.0

    # Primary governor response
    if abs(freq_state.frequency - FREQ_NOMINAL) > FREQ_PRIMARY_RESPONSE_BAND:
        freq_state.primary_response_active = True
        freq_state.primary_response_steps += 1
        # Governors provide partial correction
        governor_correction = (FREQ_NOMINAL - freq_state.frequency) * 0.3
        freq_state.frequency += governor_correction
    else:
        freq_state.primary_response_active = False
        freq_state.primary_response_steps = 0
        # Natural damping toward nominal
        freq_state.frequency += (FREQ_NOMINAL - freq_state.frequency) * 0.1

    # Protection thresholds — load shedding
    if freq_state.frequency < FREQ_LOAD_SHED_2:
        load_shed = LOAD_SHED_2_MW
        freq_state.load_shedding_mw = load_shed
    elif freq_state.frequency < FREQ_LOAD_SHED_1:
        load_shed = LOAD_SHED_1_MW
        freq_state.load_shedding_mw = load_shed
    else:
        freq_state.load_shedding_mw = 0.0

    # Blackout check
    blackout = (
        freq_state.frequency < FREQ_BLACKOUT_LOW
        or freq_state.frequency > FREQ_BLACKOUT_HIGH
        or abs(rocof) > FREQ_ROCOF_TRIP
    )

    return blackout, load_shed


def classify_blackout_risk(freq_state: FrequencyState) -> str:
    """Return qualitative blackout risk level."""
    f = freq_state.frequency
    rocof = abs(freq_state.rocof)

    if f < 48.0 or f > 51.2 or rocof > 0.8:
        return "critical"
    if f < 48.5 or f > 51.0 or rocof > 0.5:
        return "high"
    if f < 49.0 or f > 50.8 or rocof > 0.3:
        return "medium"
    if f < 49.5 or f > 50.5 or rocof > 0.1:
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# Event engine
# ---------------------------------------------------------------------------

def schedule_events(
    task_id: str,
    total_steps: int,
    rng: random.Random,
) -> Dict[int, List[str]]:
    """
    Pre-schedule all stochastic events for the episode.

    Returns a dict mapping step → list of events that START on that step.
    Events have a duration; the simulator tracks when they end.
    Seeded via task RNG for full determinism.
    """
    schedule: Dict[int, List[str]] = {}

    def add_event(start: int, event: str) -> None:
        schedule.setdefault(start, [])
        if event not in schedule[start]:
            schedule[start].append(event)

    if task_id == "easy":
        # No stochastic events — clean baseline
        return schedule

    if task_id in ("medium", "hard"):
        # Weather events for solar
        for _ in range(rng.randint(2, 4)):
            step = rng.randint(6, total_steps - 6)
            add_event(step, rng.choice(["cloud", "heavy_cloud"]))

        # Heatwave or cold snap (demand surge)
        if rng.random() < 0.6:
            step = rng.randint(0, total_steps - 8)
            add_event(step, rng.choice(["heatwave", "cold_snap"]))

    if task_id == "hard":
        # Rainfall (positive event — refills reservoir) — only in hard task where hydro is available
        if rng.random() < 0.5:
            add_event(rng.randint(0, total_steps - 1), "rainfall")
            
    if task_id == "hard" and rng.random() < 0.3:
        add_event(rng.randint(5, total_steps - 5), "fdi_attack")

    if task_id == "hard":
        # Coal outage: base range 20-35 with per-episode variance to prevent memorization
        # Even with seeded RNG, add runtime variance window (±random offset)
        coal_base = rng.randint(20, 35)
        coal_variance = rng.randint(-3, 3)  # ±3 step variance per episode
        coal_step = max(1, min(total_steps - 5, coal_base + coal_variance))
        add_event(coal_step, "coal_outage")

        # Nuclear trip: randomized within 30-50 range with variance
        if rng.random() < 0.4:
            nuclear_base = rng.randint(30, 50)
            nuclear_variance = rng.randint(-3, 3)  # ±3 step variance
            nuclear_step = max(1, min(total_steps - 2, nuclear_base + nuclear_variance))
            add_event(nuclear_step, "nuclear_trip")

        # Coal price spike
        for _ in range(rng.randint(1, 2)):
            add_event(rng.randint(10, total_steps - 10), "price_spike")

        # Drought
        if rng.random() < 0.5:
            add_event(rng.randint(5, 20), "drought")

    return schedule


# Event durations in steps
EVENT_DURATIONS: Dict[str, int] = {
    "heatwave": 6,
    "cold_snap": 5,
    "cloud": 3,
    "heavy_cloud": 4,
    "storm": 2,
    "calm": 4,
    "rainfall": 2,
    "drought": 8,
    "coal_outage": 3,
    "nuclear_trip": 1,    # trigger only; restart handled by nuclear state
    "price_spike": 5,
    "grid_fault": 3,
    "fdi_attack": 4,
}


def apply_event_start(event: str, state: GridSimState) -> None:
    """Apply immediate effects when an event starts."""
    if event == "coal_outage":
        state.coal.max_mw = min(state.coal.max_mw, 300.0)
        if state.coal.output_mw > 300.0:
            state.coal.output_mw = 300.0

    elif event == "nuclear_trip":
        if state.nuclear.available and state.nuclear.online:
            state.nuclear.online = False
            state.nuclear.output_mw = 0.0
            state.nuclear.trip_steps_remaining = NUCLEAR_STARTUP_STEPS + 1

    elif event == "price_spike":
        state.coal_price = min(2.5, state.coal_price * 2.0)

    elif event == "grid_fault":
        state.transmission_capacity_mw = (
            TRANSMISSION_NOMINAL_MW * (1.0 - TRANSMISSION_FAULT_REDUCTION)
        )

    elif event == "storm":
        state.solar.degradation_factor = max(
            0.8, state.solar.degradation_factor - 0.02
        )


def apply_event_end(event: str, state: GridSimState) -> None:
    """Restore state when an event expires."""
    if event == "coal_outage":
        # Only restore coal max if no boost damage is currently active
        # Otherwise the damage window gets cut short
        if state.coal.boost_damage_steps == 0:
            state.coal.max_mw = COAL_MAX_MW
    elif event == "price_spike":
        # Preserve the random walk trend (don't snap back to 1.0)
        # Just cap at max to prevent unbounded growth
        state.coal_price = min(2.5, state.coal_price)
    elif event == "grid_fault":
        state.transmission_capacity_mw = TRANSMISSION_NOMINAL_MW


# ---------------------------------------------------------------------------
# Solar weather from active events
# ---------------------------------------------------------------------------

def derive_solar_weather(active_events: List[str]) -> str:
    if "storm" in active_events:
        return "storm"
    if "heavy_cloud" in active_events:
        return "cloudy"
    if "cloud" in active_events:
        return "partial"
    return "clear"


# ---------------------------------------------------------------------------
# Plant construction
# ---------------------------------------------------------------------------

def process_plant_action(
    action_str: str,
    state: GridSimState,
    task_id: str,
) -> Optional[str]:
    """
    Attempt to start a plant construction project or decommission coal.

    Returns an error message string if the action is invalid, else None.
    """
    if action_str == "none":
        return None

    if task_id != "hard":
        return "Plant actions only available in Hard task"

    if action_str == "close_coal":
        if not state.coal.available:
            return "Coal plant already closed"
        if state.capital_budget < COAL_CLOSE_COST:
            return "Insufficient capital for coal decommissioning"
        state.coal.available = False
        state.coal.online = False
        state.coal.output_mw = 0.0
        state.capital_budget -= COAL_CLOSE_COST
        state.capital_budget += 150.0  # 50% salvage value recovery (~150 of 300 cost)
        return None

    spec = PLANT_BUILD_SPECS.get(action_str)
    if spec is None:
        return f"Unknown plant action: {action_str}"

    # Check capital
    if state.capital_budget < spec["cost"]:
        return f"Insufficient capital: need {spec['cost']}, have {state.capital_budget:.0f}"

    # Check not already building same type
    for entry in state.construction_queue:
        if entry.plant_type == spec["type"]:
            return f"Already building a {spec['type']} plant"

    # Check not already available
    plant_state = getattr(state, spec["type"], None)
    if plant_state is not None and plant_state.available:
        return f"{spec['type']} plant already exists"

    # Deduct capital and queue
    state.capital_budget -= spec["cost"]
    state.construction_queue.append(
        ConstructionEntry(
            plant_type=spec["type"],
            steps_remaining=spec["build_steps"],
            capacity_mw=spec["capacity_mw"],
        )
    )
    return None


def advance_construction(state: GridSimState) -> List[str]:
    """
    Advance construction timers. Return list of newly completed plant types.
    """
    completed = []
    still_building = []

    for entry in state.construction_queue:
        entry.steps_remaining -= 1
        if entry.steps_remaining <= 0:
            # Bring plant online
            plant_state = getattr(state, entry.plant_type, None)
            if plant_state is not None:
                plant_state.available = True
                if entry.plant_type == "nuclear":
                    plant_state.online = True
                    plant_state.output_mw = NUCLEAR_MIN_MW
            completed.append(entry.plant_type)
        else:
            still_building.append(entry)

    state.construction_queue = still_building
    return completed


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    state: GridSimState,
    supply_mw: float,
    demand_mw: float,
    battery_discharged_mw: float,
    battery_charged_mw: float,
    load_shed_mw: float,
    blackout: bool,
    spillage_occurred: bool,
    task_id: str,
    feedin_mw: float = 0.0,
    demand_response_mw: float = 0.0,
    emergency_coal_boost: bool = False,
    duck_curve_stress: float = 0.0,
    actual_reserve: float = 0.0,
    required_reserve: float = 0.0,
    voltage_stability_index: float = 100.0,
    scheduled_dr: float = 0.0,          # ✅ NEW
    spot_price: float = 1.0
) -> float:
    """
    Compute step reward.

    Reward structure:
        - Heavy penalty for unmet demand (reliability is primary objective)
        - Small waste penalty for overproduction
        - Frequency stability bonus/penalty
        - Coal operational cost (fuel price sensitive)
        - Nuclear near-free running cost
        - Hydro spillage penalty (wasted water)
        - Hydro critical-low warning
        - Battery degradation penalty
        - Spinning reserve shortfall penalty
        - Emissions penalty (Hard task only)
        - Catastrophic blackout penalty
    """
    if blackout:
        return -500.0   # catastrophic failure

    unmet = max(0.0, demand_mw - supply_mw)
    over = max(0.0, supply_mw - demand_mw)

    freq_error = abs(state.frequency.frequency - FREQ_NOMINAL)

    # Base reward components
    reward = 0.0

    # ---- Reliability ----
    reward -= 0.25 * unmet          # prioritise reliability
    reward -= 0.002 * over          # reduce oversupply penalty
    reward -= 0.40 * load_shed_mw   # penalty for load shedding (prevent exploit)
    # ---- Demand response costs ----
    reward -= 0.01 * demand_response_mw
    reward -= 0.004 * scheduled_dr

    # ---- Rerouting penalty ----
    if state.last_reroute:
        reward -= 2.0

    # ---- Frequency stability ----
    reward -= 0.25 * freq_error      # reduce frequency dominance
    if freq_error < 0.1:
        reward += 0.25   # bonus for very stable frequency

    if demand_response_mw > 0 and task_id != "hard":
        reward -= 0.20 * demand_response_mw  # strong penalty to prevent DR spam

    # Penalty applies only when crossing the 500 MW threshold (not every step after)
    # Check: did we just cross 500 MW this step?
    if state.total_demand_response > 500 and (state.total_demand_response - demand_response_mw) <= 500:
        reward -= 5.0

    if unmet < 3 and over < 10:
        reward += 0.2

    # ---- Emergency boost penalty ----
    # Apply penalty only on the step boost is actually used, not permanently
    if emergency_coal_boost:
        reward -= 3.0

    # ---- Generation costs ----
    coal_mwh = state.coal.output_mw * (1.0 / 1.0)   # 1 step = 1 MWh equivalent
    reward -= 0.004 * coal_mwh * state.coal_price

    # Dynamic carbon price signal (£/ton CO2, realistic ETS pricing)
    reward -= (coal_mwh * COAL_EMISSION_FACTOR * state.carbon_price * 0.00012)

    if state.coal_flip_streak == 0 and unmet < 10:
        reward += 0.05

    if state.nuclear.available and state.nuclear.online:
        nuclear_mwh = state.nuclear.output_mw
        reward -= 0.0001 * nuclear_mwh * NUCLEAR_FUEL_COST

    # ---- RENEWABLE ENERGY BONUS (NEW: incentivize clean generation) ----
    # Provide positive rewards for using wind, solar, hydro
    wind_output = state.wind.output_mw if state.wind.available else 0.0
    solar_output = state.solar.output_mw if state.solar.available else 0.0
    hydro_output = state.hydro.output_mw if state.hydro.available else 0.0
    
    renewable_bonus = 0.02 * wind_output + 0.025 * solar_output + 0.015 * hydro_output
    renewable_bonus = min(0.30, renewable_bonus)  # Cap bonus to prevent stacking exploit
    reward += renewable_bonus
    
    # ---- Prosumer feed-in credit ----
    if feedin_mw > 0:
        reserve_ratio = actual_reserve / max(required_reserve, 1.0)

        if reserve_ratio < 0.8:
            feedin_rate = 0.08
        elif reserve_ratio < 1.2:
            feedin_rate = 0.05
        else:
            feedin_rate = 0.02

        feedin_value = feedin_mw * feedin_rate * spot_price
        reward += feedin_value

    # ---- Hydro management ----
    if spillage_occurred:
        reward -= 0.05
    if state.hydro.available and state.hydro.reservoir_mwh < HYDRO_CRITICAL_LOW:
        reward -= 0.1

    # ---- Battery wear ----
    # Use discharged amount only (already accounts for efficiency losses)
    # charging_inefficiency means charge_mw > actual stored, so use discharged as consistent unit
    cycle_delta = battery_discharged_mw / max(1.0, state.battery.capacity_mwh)
    reward -= 0.005 * cycle_delta

    # ---- Spinning reserve shortfall ----
    reserve_shortfall = max(
        0.0,
        required_reserve - actual_reserve,
    )

    reward -= 0.05 * reserve_shortfall / max(1.0, demand_mw)

    # ---- Duck curve ramp stress penalty (Feature 5) ----
    reward -= 0.004 * abs(duck_curve_stress)

    # ---- Voltage stability penalty (NEW - Phase 1 completion) ----
    voltage_index = voltage_stability_index

    # Soft penalty below safe threshold (~50%)
    if voltage_index < 50:
        reward -= 0.4 * (50 - voltage_index)
    elif voltage_index < 70:
        reward -= 0.1 * (70 - voltage_index)

    # ---- Oscillation penalty ----
    # Penalise repeatedly hitting max or min ramp — signals instability
    if state.coal_flip_streak >= 2:
        reward -= 1.0 * state.coal_flip_streak  # doubled penalty to strongly discourage oscillations (exploit fix)

    return reward


def _compute_spinning_reserve(state: GridSimState) -> float:
    """Compute available spinning reserve from online synchronous machines."""
    reserve = 0.0
    HYDRO_RESERVE_MIN_MWH = 10.0  # Minimum water to unlock reserve capacity

    if state.coal.online and state.coal.available:
        reserve += state.coal.max_mw - state.coal.output_mw

    # Only count hydro reserve if enough water is available
    if state.hydro.available and state.hydro.reservoir_mwh > HYDRO_RESERVE_MIN_MWH:
        reserve += HYDRO_MAX_MW - state.hydro.output_mw

    if state.nuclear.available and state.nuclear.online:
        reserve += NUCLEAR_MAX_MW - state.nuclear.output_mw

    return max(0.0, reserve)


# ---------------------------------------------------------------------------
# Coal price fluctuation (Hard task)
# ---------------------------------------------------------------------------

def update_coal_price(state: GridSimState, rng: random.Random) -> None:
    """
    Random-walk coal price within [0.8, 2.5].
    Only applied in Hard task; no-op otherwise.
    Also updates carbon price via random walk (£/ton CO2).
    """
    if "price_spike" in state.active_events:
        return   # already set by event

    delta = rng.gauss(0, 0.05)
    state.coal_price = max(0.8, min(2.5, state.coal_price + delta))
    
    # Carbon price random walk (realistic market volatility)
    carbon_delta = rng.gauss(0, 2.0)   # £2/ton std dev per step
    state.carbon_price = max(20.0, min(100.0, state.carbon_price + carbon_delta))


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------

def simulator_step(
    state: GridSimState,
    coal_delta: float,
    hydro_delta: float,
    nuclear_delta: float,
    battery_mode: str,
    emergency_coal_boost: bool,
    demand_response_mw: float,
    plant_action: str,
    event_schedule: Dict[int, List[str]],
    event_end_schedule: Dict[int, List[str]],
    task_id: str,
    scheduled_dr_mw: float = 0.0,
    scheduled_dr_start: int = 0,
    scheduled_dr_duration: int = 0,
    reroute_transmission: bool = False,
) -> Dict[str, Any]:
    """
    Advance the simulation by one step.

    Order of operations:
        1. End expired events
        2. Start new events
        3. Process plant construction
        4. Process plant action (build/close)
        5. Update demand
        6. Step wind speed
        7. Compute solar/wind output
        8. Step coal
        9. Step nuclear
        10. Step hydro
        11. Apply demand response
        12. Step battery
        13. Compute net supply/imbalance
        14. Clamp to transmission capacity
        15. Step frequency
        16. Compute reward
        17. Update economics/emissions
        18. Advance time

    Returns a dict of all values needed to build the observation.
    """
        
    PRIORITY = {
        "grid_fault": 3,
        "nuclear_trip": 3,
        "coal_outage": 2,
        "fdi_attack": 2,
    }

    state.active_events.sort(key=lambda e: PRIORITY.get(e, 1), reverse=True)

    state.active_events = state.active_events[:3]

    # 1. End expired events
    for event in event_end_schedule.get(state.step, []):
        if event in state.active_events:
            state.active_events.remove(event)
            apply_event_end(event, state)

    # 2. Start new events
    for event in event_schedule.get(state.step, []):
        if event not in state.active_events:
            state.active_events.append(event)
            apply_event_start(event, state)

    # 3. Advance construction queue
    advance_construction(state)

    # 4. Process plant action
    process_plant_action(plant_action, state, task_id)

    # 5. Demand
    hour = state.step % 24
    b, i, d = compute_demand(
        hour,
        state.season,
        state.active_events,
        state.rng
    )

    state.baseline_demand_mw = b
    state.industrial_demand_mw = i
    state.datacenter_demand_mw = d

    state.demand_mw = b + i + d
    
    # ---- Demand Response (FIXED + WORKING) ----

    # Immediate DR comes from function input
    instant_dr = max(0.0, demand_response_mw)

    # Scheduled DR inputs from action
    scheduled_dr_mw_input = scheduled_dr_mw
    scheduled_start_input = scheduled_dr_start
    scheduled_duration_input = scheduled_dr_duration

    # Enqueue scheduled DR (only once per trigger)
    if scheduled_dr_mw_input > 0 and getattr(state, "_dr_scheduled_flag", False) is False:
        for t in range(scheduled_duration_input):
            state.scheduled_dr_queue.append(
                (state.step + scheduled_start_input + t, scheduled_dr_mw_input)
            )
        state._dr_scheduled_flag = True

    # Reset flag when no new DR requested
    if scheduled_dr_mw_input == 0:
        state._dr_scheduled_flag = False

    # Active scheduled DR this step
    scheduled_dr = sum(mw for step, mw in state.scheduled_dr_queue if step == state.step)

    # Clean expired entries
    state.scheduled_dr_queue = [
        (s, mw) for s, mw in state.scheduled_dr_queue if s > state.step
    ]

    # Total DR applied (cap)
    total_dr = min(instant_dr + scheduled_dr, 150.0)

    # Track cumulative DR (IMPORTANT for reward)
    state.total_demand_response += total_dr

    # Effective demand
    effective_demand = max(0.0, state.demand_mw - total_dr)

    # 6. Wind speed update
    step_wind_speed(state.wind, state.rng)

    # 7. Passive generation (weather-driven, no agent control)
    state.solar_weather = derive_solar_weather(state.active_events)
    
    # Update solar clearness index for continuous noise (Feature 13)
    step_solar_clearness(state.solar, state.rng)
    
    solar_out = compute_solar_output(hour, state.solar, state.solar_weather)
    state.solar.output_mw = solar_out
    wind_out = compute_wind_output(state.wind)
    state.wind.output_mw = wind_out

    # 7b. Prosumer feed-in (rooftop solar returns excess to grid)
    feedin_mw = 0.0
    if state.solar.available and state.solar_weather in ("clear", "partial"):
        feedin_mw = solar_out * 0.05
        state.cumulative_feedin_credits += feedin_mw

    # 8. Coal

    # ---- Hard cap on emergency boost usage ----
    if state.boost_used_count >= 5:
        emergency_coal_boost = False
        logging.warning("Emergency coal boost blocked: boost_used_count >= 5")

    # ---- Clamp delta FIRST (this is the actual applied control) ----
    effective_coal_delta = max(-COAL_RAMP_MW, min(COAL_RAMP_MW, coal_delta))

    # Apply coal dynamics using EFFECTIVE delta
    coal_out = step_coal(state.coal, effective_coal_delta, emergency_coal_boost, state)
    state.coal.output_mw = coal_out

    # ---- Oscillation tracking (use EFFECTIVE delta, not raw) ----
    THRESHOLD = 10.0

    if state.prev_coal_delta is not None:
        if abs(effective_coal_delta) > THRESHOLD and abs(state.prev_coal_delta) > THRESHOLD:
            if state.prev_coal_delta * effective_coal_delta < 0:
                state.coal_flip_streak += 1
            else:
                # Decay slowly when not oscillating (only decay every 3 stable steps)
                state.coal_flip_streak = max(0, state.coal_flip_streak - (1 if state.coal_flip_streak % 3 == 0 else 0))
        else:
            # Outside threshold - decay slowly (1 per 3 steps) to keep memory of oscillations
            state.coal_flip_streak = max(0, state.coal_flip_streak - (1 if state.coal_flip_streak % 3 == 0 else 0))

    state.prev_coal_delta = effective_coal_delta

    if emergency_coal_boost:
        state.boost_used_count += 1

    # 9. Nuclear
    # Use active_events (after apply_event_start) to avoid double-triggering
    scram_nuclear = "nuclear_trip" in state.active_events
    nuclear_out = step_nuclear(state.nuclear, nuclear_delta, scram_nuclear)
    state.nuclear.output_mw = nuclear_out

    # 10. Hydro (reservoir-aware)
    target_hydro = max(
        0.0,
        min(
            HYDRO_MAX_MW,
            state.hydro.output_mw + hydro_delta,
        ),
    )
    prev_reservoir = state.hydro.reservoir_mwh
    hydro_out, spillage_occurred = step_hydro(state.hydro, target_hydro, state.active_events, state.rng)
    state.hydro.output_mw = hydro_out

    # 12. Battery
    passive_supply = solar_out + wind_out + coal_out + hydro_out + nuclear_out
    shortfall = max(0.0, effective_demand - passive_supply)
    battery_discharged, battery_charged = step_battery(
        state.battery, battery_mode, shortfall
    )

    # Track battery mode for oscillation detection
    if battery_mode != state.prev_battery_mode and battery_mode != "idle" and state.prev_battery_mode != "idle":
        # Direct switch between charge and discharge — apply cycle penalty
        state.battery.total_cycles += 0.02
    state.prev_battery_mode = battery_mode

    # 13. Net supply and imbalance
    total_supply = passive_supply + battery_discharged - battery_charged + feedin_mw
    power_imbalance = total_supply - effective_demand

    # ---- Transmission capacity (with rerouting) ----
    state.last_reroute = False

    if "grid_fault" in state.active_events:
        if reroute_transmission:
            state.transmission_capacity_mw = 1080
            state.last_reroute = True
        else:
            state.transmission_capacity_mw = 960
    else:
        state.transmission_capacity_mw = 1200

    # 14. Transmission capacity constraint
    if total_supply > state.transmission_capacity_mw:
        # Curtail overproduction at transmission limit
        total_supply = state.transmission_capacity_mw
        power_imbalance = total_supply - effective_demand

    anomaly_score = min(compute_anomaly_score(
        state,
        event_schedule,
        state.step,
        state.rng
    ), 0.8)

    # 15. Frequency dynamics
    system_inertia = compute_system_inertia(state.coal, state.hydro, state.nuclear)
    blackout, load_shed_mw = step_frequency(
        state.frequency, power_imbalance, system_inertia, effective_demand
    )
    state.blackout_this_step = blackout

    if blackout:
        state.episode_ended = True

    # 16. Track demand-met
    final_unmet = max(0.0, effective_demand - total_supply - load_shed_mw)
    if final_unmet < 1.0:   # within 1 MW tolerance
        state.steps_demand_met += 1

    duck_curve_stress = compute_duck_curve_stress(state)

    # Voltage stability index (Feature 10)
    voltage_stability_idx = compute_voltage_stability_index(state.coal, state.hydro, state.nuclear, solar_out, wind_out)
    
    required_reserve = compute_required_spinning_reserve(state)
    actual_reserve = _compute_spinning_reserve(state)

    # 17. Reward
    reward = compute_reward(
        state=state,
        supply_mw=total_supply,
        demand_mw=effective_demand,
        battery_discharged_mw=battery_discharged,
        battery_charged_mw=battery_charged,
        load_shed_mw=load_shed_mw,
        blackout=blackout,
        spillage_occurred=spillage_occurred,
        task_id=task_id,
        feedin_mw=feedin_mw,
        demand_response_mw=instant_dr,
        scheduled_dr=scheduled_dr,
        emergency_coal_boost=emergency_coal_boost,
        duck_curve_stress=duck_curve_stress,
        voltage_stability_index=voltage_stability_idx,
        actual_reserve=actual_reserve,
        required_reserve=required_reserve 
    )

    # 18. Economics & emissions
    coal_mwh = state.coal.output_mw
    state.cumulative_cost += (
        coal_mwh * state.coal_price * 0.001
        + (state.nuclear.output_mw * NUCLEAR_FUEL_COST * 0.0001
           if state.nuclear.available else 0.0)
    )
    state.cumulative_emissions += coal_mwh * COAL_EMISSION_FACTOR

    if task_id == "hard":
        update_coal_price(state, state.rng)

    # 19. Advance time
    state.step += 1
    if state.step % 24 == 0:
        state.day += 1
        # Panel degradation per simulated day
        if state.solar.available:
            state.solar.degradation_factor = max(
                0.9,
                state.solar.degradation_factor - SOLAR_DEGRADATION_PER_DAY,
            )

    done = state.episode_ended or (state.step >= state.total_steps)
    
    # ── Spinning reserve outputs (Feature 2) ─────────────────────────

    # Temporarily override demand for correct reserve calculation
    original_demand = state.demand_mw
    state.demand_mw = effective_demand
    
    state.demand_mw = original_demand

    spot_price = compute_spot_price(
        state,
        state.coal_price,
        state.carbon_price,
        task_id,
        actual_reserve,
        required_reserve,
    )
    
    # Rate of change of frequency (for grid stability metrics)
    rate_of_change_hz = state.frequency.rocof
    
    shortfall_steps = compute_shortfall_projection(state)

    return {
        "reward": reward,
        "done": done,
        "blackout": blackout,
        "load_shed_mw": load_shed_mw,
        "unmet_demand_mw": final_unmet,
        "overproduction_mw": max(0.0, total_supply - effective_demand),
        "total_supply_mw": total_supply,
        "system_inertia": system_inertia,
        "blackout_risk": classify_blackout_risk(state.frequency),
        "solar_weather": state.solar_weather,
        "spillage_occurred": spillage_occurred,
        "demand_response_applied_mw": total_dr,
        "coal_health_pct": state.coal.health_pct,
        "anomaly_score": anomaly_score,
        "carbon_price_per_ton": state.carbon_price,
        "rate_of_change_hz_per_step": rate_of_change_hz,
        "voltage_stability_index": voltage_stability_idx,
        "spinning_reserve_required_mw": required_reserve,
        "spinning_reserve_mw": actual_reserve,
        "baseline_demand_mw": state.baseline_demand_mw,
        "industrial_demand_mw": state.industrial_demand_mw,
        "datacenter_demand_mw": state.datacenter_demand_mw,
        "scheduled_dr_mw": scheduled_dr,
        "effective_demand_mw": effective_demand,
        "steps_until_shortfall": shortfall_steps,
        "spot_price": spot_price,
        "duck_curve_stress_mw_per_step": duck_curve_stress,
    }


# ---------------------------------------------------------------------------
# State initialiser (used by reset())
# ---------------------------------------------------------------------------

def build_initial_state(
    seed: int,
    total_steps: int,
    season: str = "spring",
    config: Dict[str, Any] = None,
) -> GridSimState:
    """
    Construct and return a fully initialised GridSimState.
    Uses config dictionary for starting conditions (from tasks.py).
    """
    rng = random.Random(seed)
    state = GridSimState(rng=rng, total_steps=total_steps, season=season)

    if config:
        # Basic plants
        state.coal.output_mw = config.get("coal_start_mw", 400.0)
        state.coal.online = True
        state.coal_price = config.get("coal_start_price", 1.0)
        state.battery.level_mwh = config.get("battery_start_mwh", 100.0)
        state.capital_budget = config.get("capital_budget", 0.0)
        state.hydro.reservoir_mwh = config.get("hydro_start_reservoir_mwh", 600.0)

        # Availability
        available = config.get("plants_available", ["coal", "battery"])
        state.solar.available = "solar" in available
        state.wind.available = "wind" in available
        state.hydro.available = "hydro" in available
        state.nuclear.available = "nuclear" in available

        # Special case: if hydro not available, set reservoir to 0 to avoid misleading observations
        if not state.hydro.available and "hydro" not in config.get("buildable_plants", []):
            state.hydro.reservoir_mwh = 0.0

    return state