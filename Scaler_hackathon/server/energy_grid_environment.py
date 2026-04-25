"""
Energy Grid Management Environment — Multi-Agent OpenEnv Implementation.

Extends the single-agent environment to support three specialized agents:

    PlanningAgent  — capital investment decisions (infrequent, long-horizon)
    DispatchAgent  — real-time generation control (every step)
    MarketAgent    — economic optimization + grid trading (every step)

Multi-agent step protocol:
    Each simulation step requires actions from all three agents before
    the physics engine advances. The environment buffers partial actions
    and advances the simulator only when all three have been submitted.

    Order within a step:
        1. planning action received  → buffered
        2. dispatch action received  → buffered
        3. market action received    → merge all three → simulator_step()
                                    → return observation to all agents

    Single-agent backward compatibility:
        POST /step with EnergyGridAction triggers immediate simulator_step()
        using the unified action directly. No buffering needed.

Reward decomposition:
    The unified simulator reward is decomposed into per-agent signals:
        dispatch_reward = reliability + frequency + spinning reserve
        planning_reward = emissions + capital efficiency
        market_reward   = cost efficiency + trading profit
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        EnergyGridAction,
        EnergyGridObservation,
        PlanningAgentAction,
        DispatchAgentAction,
        MarketAgentAction,
    )
except ImportError:
    from models import (
        EnergyGridAction,
        EnergyGridObservation,
        PlanningAgentAction,
        DispatchAgentAction,
        MarketAgentAction,
        PlanningAgentObservation,
        DispatchAgentObservation,
        MarketAgentObservation,
    )

try:
    from .normalization import normalize_observation
except ImportError:
    from server.normalization import normalize_observation

try:
    from .simulator import (
        GridSimState,
        build_initial_state,
        schedule_events,
        simulator_step,
        EVENT_DURATIONS,
        HYDRO_RESERVOIR_CAP_MWH,
    )
    from .tasks import get_task, TASK_ORDER
    from .grader import (
        EpisodeLog,
        StepLog,
        grade_episode,
        grade_result_to_dict,
        GradeResult,
    )
except ImportError:
    from server.simulator import (
        GridSimState,
        build_initial_state,
        schedule_events,
        simulator_step,
        EVENT_DURATIONS,
        HYDRO_RESERVOIR_CAP_MWH,
    )
    from server.tasks import get_task, TASK_ORDER
    from server.grader import (
        EpisodeLog,
        StepLog,
        grade_episode,
        grade_result_to_dict,
        GradeResult,
    )


# ---------------------------------------------------------------------------
# Per-agent reward decomposition weights
# These split the unified simulator reward into three per-agent signals.
# Weights sum to 1.0 across each component.
# ---------------------------------------------------------------------------

DISPATCH_REWARD_COMPONENTS = {
    "unmet_demand",       # -0.25 * unmet_demand_mw
    "frequency",          # -0.2 * freq_error + 0.2 bonus
    "spinning_reserve",   # -0.05 * shortfall
    "emergency_boost",    # -3.0 if used
    "load_shedding",      # -0.30 * load_shed_mw
}

PLANNING_REWARD_COMPONENTS = {
    "emissions",          # -0.001 * coal_mw * emission_factor
    "renewable_bonus",    # +0.015*wind + 0.020*solar + 0.010*hydro
    "hydro_spillage",     # -0.05 if spillage
    "hydro_critical",     # -0.10 if reservoir critical
}

MARKET_REWARD_COMPONENTS = {
    "coal_cost",          # -0.003 * coal_mw * coal_price
    "overproduction",     # -0.001 * over
    "demand_response",    # penalty/bonus based on DR usage
    "feedin",             # +0.002 * feedin_mw
    "trading",            # grid import/export credits (new)
}


# ---------------------------------------------------------------------------
# Action buffer — holds partial multi-agent actions within one step
# ---------------------------------------------------------------------------

class StepActionBuffer:
    """
    Buffers partial actions from each agent within a single simulation step.

    A step advances only when all required agents have submitted.
    In single-agent mode the buffer is bypassed entirely.
    """

    def __init__(self) -> None:
        self.planning: Optional[PlanningAgentAction] = None
        self.dispatch: Optional[DispatchAgentAction] = None
        self.market: Optional[MarketAgentAction] = None

    def reset(self) -> None:
        self.planning = None
        self.dispatch = None
        self.market = None

    @property
    def is_complete(self) -> bool:
        """True when all three agents have submitted actions."""
        return (
            self.planning is not None
            and self.dispatch is not None
            and self.market is not None
        )

    def to_unified_action(self) -> EnergyGridAction:
        """
        Merge buffered agent actions into a unified EnergyGridAction
        for the simulator. Called only when is_complete is True.
        """
        assert self.is_complete, "Cannot merge incomplete action buffer"
        return EnergyGridAction.from_agents(
            dispatch=self.dispatch,
            planning=self.planning,
            market=self.market,
        )





# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class EnergyGridEnvironment(Environment):
    """
    Multi-agent Energy Grid Management Environment.

    Supports two modes:

    Single-agent mode (backward compatible):
        env.step(EnergyGridAction) → advances simulator immediately.
        Used by /step endpoint and all existing baseline/training scripts.

    Multi-agent mode:
        env.step_planning(PlanningAgentAction)  → buffers
        env.step_dispatch(DispatchAgentAction)  → buffers
        env.step_market(MarketAgentAction)      → completes step, advances simulator
        Returns observation with per-agent reward breakdown.

    All three agents share the same EnergyGridObservation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, normalize: bool = False) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "easy"
        self._sim: Optional[GridSimState] = None
        self._episode_log: Optional[EpisodeLog] = None
        self._event_schedule: Dict[int, List[str]] = {}
        self._event_end_schedule: Dict[int, List[str]] = {}
        self._last_grade: Optional[GradeResult] = None
        self._last_step_result: Dict[str, Any] = {}
        self._plants_built: List[str] = []
        self._normalize = normalize

        # Multi-agent state
        self._action_buffer = StepActionBuffer()
        self._last_obs: Optional[EnergyGridObservation] = None

        # Grid trading state (new for market agent)
        self._trading_credits: float = 0.0
        self._grid_export_mw: float = 0.0
        self._grid_import_mw: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    @property
    def current_task_id(self) -> str:
        return self._task_id

    def reset(self, task_id: str = "easy", seed: int = None) -> EnergyGridObservation:
        if task_id not in TASK_ORDER:
            task_id = "easy"

        task = get_task(task_id)
        self._task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)

        episode_seed = seed if seed is not None else task["seed"]
        self._sim = build_initial_state(
            task_id=task_id,
            seed=episode_seed,
            total_steps=task["total_steps"],
            season=task["season"],
        )
        self._sim.capital_budget = task["capital_budget"]

        raw_schedule = schedule_events(
            task_id=task_id,
            total_steps=task["total_steps"],
            rng=self._sim.rng,
        )
        self._event_schedule = raw_schedule
        self._event_end_schedule = self._build_end_schedule(raw_schedule)

        self._episode_log = EpisodeLog(
            task_id=task_id,
            total_steps=task["total_steps"],
        )
        self._episode_log.initial_capital_budget = task["capital_budget"]

        self._last_grade = None
        self._last_step_result = {}
        self._plants_built = []
        self._action_buffer.reset()

        # Reset trading state
        self._trading_credits = 0.0
        self._grid_export_mw = 0.0
        self._grid_import_mw = 0.0

        # Initial demand calculation
        result = simulator_step(
            state=self._sim,
            coal_delta=0,
            hydro_delta=0,
            nuclear_delta=0,
            battery_mode="idle",
            emergency_coal_boost=False,
            demand_response_mw=0,
            plant_action="none",
            event_schedule=self._event_schedule,
            event_end_schedule=self._event_end_schedule,
            task_id=task_id,
        )
        self._sim.step = 0
        self._sim.day = 1
        self._last_step_result = result

        obs = self._build_observation(reward=0.0, done=False)
        self._last_obs = obs
        return obs

# ---------------------------------------------------------------------------
# Observation Filtering (Asymmetric Information)
# ---------------------------------------------------------------------------

    def _filter_observation_for_agent(self, obs, agent_type: str):
        data = obs.model_dump()

        if agent_type == "planning":
            allowed = [
                "demand_mw",
                "hour",
                "day",
                "season",
                "coal_price",
                "capital_budget",
                "cumulative_cost",
                "cumulative_emissions_tons",
            ]

        elif agent_type == "dispatch":
            allowed = [
                "demand_mw",
                "coal_mw",
                "solar_mw",
                "wind_mw",
                "hydro_mw",
                "nuclear_mw",
                "battery_mwh",
                "frequency_hz",
                "load_shedding_mw",
                "spinning_reserve_mw",
                "duck_curve_stress",
                "voltage_stability_index",
                "anomaly_score",
            ]

        elif agent_type == "market":
            allowed = [
                "demand_mw",
                "coal_price",
                "spot_price",
                "grid_export_mw",
                "grid_import_mw",
                "trading_credits",
            ]

        else:
            return obs

        filtered = {k: data[k] for k in allowed if k in data}

        filtered_obs = type(obs)(**filtered)

        # Apply FDI AFTER filtering
        filtered_obs = self._apply_fdi(filtered_obs, agent_type)

        return filtered_obs

    def _apply_fdi(self, obs: EnergyGridObservation, agent_type: str) -> EnergyGridObservation:
        """
        Apply False Data Injection (FDI) attack to observation.

        Only affects DISPATCH agent.
        Does NOT affect underlying simulator physics.
        """

        # Only dispatch agent is affected
        if agent_type != "dispatch":
            return obs

        # Only active during FDI event
        if "fdi_attack" not in self._sim.active_events:
            return obs

        data = obs.model_dump()

        rng = self._sim.rng

        # Corrupt key signals
        if "demand_mw" in data:
            data["demand_mw"] *= rng.uniform(0.9, 1.1)

        if "solar_mw" in data:
            data["solar_mw"] *= rng.uniform(0.7, 1.3)

        if "wind_mw" in data:
            data["wind_mw"] *= rng.uniform(0.7, 1.3)

        if "frequency_hz" in data:
            data["frequency_hz"] += rng.uniform(-0.3, 0.3)

        # Optional: noise on reserve signal
        if "spinning_reserve_mw" in data:
            data["spinning_reserve_mw"] *= rng.uniform(0.8, 1.2)

        data["fdi_active"] = True

        return type(obs)(**data)

    # ------------------------------------------------------------------
    # Single-agent step (backward compatible)
    # ------------------------------------------------------------------
    
    def step(self, action: EnergyGridAction) -> EnergyGridObservation:
        """
        Single-agent step. Accepts unified EnergyGridAction and advances
        the simulator immediately. Fully backward compatible with all
        existing code (baseline.py, training scripts, /step endpoint).
        """
        if self._sim is None:
            return self.reset(self._task_id)
        if self._sim.episode_ended:
            return self._build_observation(reward=0.0, done=True)

        result = simulator_step(
            state=self._sim,
            coal_delta=action.coal_delta,
            hydro_delta=action.hydro_delta,
            nuclear_delta=action.nuclear_delta,
            battery_mode=action.battery_mode,
            emergency_coal_boost=action.emergency_coal_boost,
            demand_response_mw=action.demand_response_mw,
            plant_action=action.plant_action,
            event_schedule=self._event_schedule,
            event_end_schedule=self._event_end_schedule,
            task_id=self._task_id,
        )
        self._last_step_result = result
        self._track_plants()
        self._log_step(result)
        self._state.step_count += 1

        reward = result["reward"]
        done = result["done"]

        if done:
            self._finalise_episode()

        obs = self._build_observation(reward=reward, done=done)
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------
    # Multi-agent steps
    # ------------------------------------------------------------------

    def step_planning(
        self, action: PlanningAgentAction
    ) -> EnergyGridObservation:
        """
        Receive planning agent action. Buffers until all agents submit.

        Returns the CURRENT observation (unchanged) — the simulator has
        not advanced yet. The observation only updates after step_market()
        completes the step.

        The planning agent typically fires once at episode start and then
        reactively when events change the capacity outlook.
        """
        if self._sim is None:
            return self.reset(self._task_id)

        self._action_buffer.planning = action

        # If already complete (e.g. planning submitted last), advance
        if self._action_buffer.is_complete:
            return self._advance_multi_agent_step()

        # Return current obs — step not yet complete
        obs = self._last_obs or self._build_observation(
    reward=0.0,
    done=False
)
        return self._filter_observation_for_agent(obs, "planning")

    def step_dispatch(
        self, action: DispatchAgentAction
    ) -> EnergyGridObservation:
        """
        Receive dispatch agent action. Buffers until all agents submit.

        Returns current observation if step not yet complete.
        Advances simulator if this completes the action buffer.
        """
        if self._sim is None:
            return self.reset(self._task_id)

        self._action_buffer.dispatch = action

        if self._action_buffer.is_complete:
            return self._advance_multi_agent_step()

        obs = self._last_obs or self._build_observation(
    reward=0.0,
    done=False
)
        return self._filter_observation_for_agent(obs, "dispatch")

    def step_market(
        self, action: MarketAgentAction
    ) -> EnergyGridObservation:
        """
        Receive market agent action.

        This is typically the LAST action submitted each step (market
        agent responds to what dispatch and planning have decided).
        Advances the simulator when the buffer is complete.

        Also handles grid import/export economics before calling
        simulator_step().
        """
        if self._sim is None:
            return self.reset(self._task_id)

        self._action_buffer.market = action

        if self._action_buffer.is_complete:
            return self._advance_multi_agent_step()

        obs = self._last_obs or self._build_observation(
    reward=0.0,
    done=False
)
        return self._filter_observation_for_agent(obs, "market")

    def _advance_multi_agent_step(self) -> EnergyGridObservation:
        """
        Internal: advance simulator using the completed action buffer.
        Called when all three agents have submitted actions for this step.
        """
        if self._sim.episode_ended:
            return self._build_observation(reward=0.0, done=True)

        unified = self._action_buffer.to_unified_action()
        market = self._action_buffer.market

        # ── Grid trading (new market agent feature) ──────────────────────
        # Apply import/export BEFORE simulator_step so demand is adjusted
        net_import = max(0.0, market.grid_import_mw - market.grid_export_mw)
        net_export = max(0.0, market.grid_export_mw - market.grid_import_mw)

        # Trading economics
        coal_price = self._sim.coal_price
        import_cost = net_import * coal_price * 1.2   # import at 20% premium
        export_revenue = net_export * 0.8             # export at 80% spot

        trading_delta = export_revenue - import_cost
        self._trading_credits += trading_delta
        self._sim.capital_budget = max(0.0, self._sim.capital_budget - import_cost)
        self._grid_export_mw = net_export
        self._grid_import_mw = net_import

        # Adjust demand for import (adds to supply side)
        # We inject net_import as a demand_response reduction equivalent
        effective_dr = unified.demand_response_mw + net_import
        effective_dr = min(effective_dr, 150.0)

        # ── Run simulator ─────────────────────────────────────────────────
        result = simulator_step(
            state=self._sim,
            coal_delta=unified.coal_delta,
            hydro_delta=unified.hydro_delta,
            nuclear_delta=unified.nuclear_delta,
            battery_mode=unified.battery_mode,
            emergency_coal_boost=unified.emergency_coal_boost,
            demand_response_mw=effective_dr,
            plant_action=unified.plant_action,
            event_schedule=self._event_schedule,
            event_end_schedule=self._event_end_schedule,
            task_id=self._task_id,
        )
        self._last_step_result = result
        self._track_plants()
        self._log_step(result)
        self._state.step_count += 1

        # ── Decompose reward ──────────────────────────────────────────────
        reward = result["reward"]
        dispatch_r, planning_r, market_r = self._decompose_reward(
            result=result,
            trading_delta=trading_delta,
            used_emergency_boost=unified.emergency_coal_boost,
        )

        done = result["done"]
        if done:
            self._finalise_episode()

        # Clear buffer for next step
        self._action_buffer.reset()

        obs = self._build_observation(
            reward=reward,
            done=done,
            dispatch_reward=dispatch_r,
            planning_reward=planning_r,
            market_reward=market_r,
        )
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------
    # Reward decomposition
    # ------------------------------------------------------------------

    def _decompose_reward(
        self,
        result: Dict[str, Any],
        trading_delta: float,
        used_emergency_boost: bool,
    ) -> tuple[float, float, float]:
        """
        Split the unified simulator reward into per-agent signals.

        Returns (dispatch_reward, planning_reward, market_reward).

        These are approximations — the simulator computes one scalar
        reward. We decompose it by attributing each penalty/bonus to
        the agent whose action caused it.
        """
        sim = self._sim
        unmet = result.get("unmet_demand_mw", 0.0)
        over = result.get("overproduction_mw", 0.0)
        load_shed = result.get("load_shed_mw", 0.0)
        freq_error = abs(sim.frequency.frequency - 50.0)
        required_reserve = result.get("spinning_reserve_required_mw", 0.0)
        actual_reserve = result.get("spinning_reserve_mw", 0.0)

        reserve_shortfall = max(
            0.0,
            required_reserve - actual_reserve,
        )

        # Dispatch reward: reliability + frequency + reserve
        dispatch_r = 0.0

        # ---- Reliability (PRIMARY) ----
        dispatch_r -= 0.30 * unmet
        dispatch_r -= 0.25 * load_shed

        # ---- Frequency stability ----
        dispatch_r -= 0.25 * freq_error
        if freq_error < 0.1:
            dispatch_r += 0.25

        # ---- Spinning reserve (IMPORTANT) ----
        reserve_shortfall = max(0.0, required_reserve - actual_reserve)
        dispatch_r -= 0.08 * reserve_shortfall / max(1.0, sim.demand_mw)

        # ---- Voltage stability (STRONGER) ----
        voltage_stability_index = result.get("voltage_stability_index", 100.0)

        if voltage_stability_index < 50:
            dispatch_r -= 0.4 * (50 - voltage_stability_index)
        elif voltage_stability_index < 70:
            dispatch_r -= 0.1 * (70 - voltage_stability_index)

        # ---- Duck curve smoothness ----
        duck_curve_stress = result.get("duck_curve_stress", 0.0)
        dispatch_r -= 0.004 * abs(duck_curve_stress)

        # ---- Emergency usage ----
        if used_emergency_boost:
            dispatch_r -= 3.0

        # Planning reward: emissions + renewable bonus
        wind_out = sim.wind.output_mw if sim.wind.available else 0.0
        solar_out = sim.solar.output_mw if sim.solar.available else 0.0
        hydro_out = sim.hydro.output_mw if sim.hydro.available else 0.0
        coal_mw = sim.coal.output_mw

        renewable_bonus = min(0.30,
            0.015 * wind_out + 0.020 * solar_out + 0.010 * hydro_out
        )
        planning_r = (
            - 0.001 * coal_mw * 0.9     # emissions
            + renewable_bonus
        )

        # Market reward: cost + trading
        market_r = 0.0

        # ---- Coal cost ----
        coal_mw = sim.coal.output_mw
        market_r -= 0.004 * coal_mw * sim.coal_price

        # ---- Overproduction ----
        market_r -= 0.002 * over

        # ---- Demand response ----
        dr = result.get("demand_response_mw", 0.0)
        market_r -= 0.01 * dr

        # ---- Trading (already good) ----
        market_r += trading_delta

        # ---- Dynamic feed-in (FIXED) ----
        spot_price = result.get("spot_price", 1.0)
        feedin_mw = result.get("grid_export_mw", 0.0)

        market_r += 0.05 * feedin_mw * spot_price

        return dispatch_r, planning_r, market_r

    # ------------------------------------------------------------------
    # Grading interface
    # ------------------------------------------------------------------

    def get_last_grade(self) -> Optional[Dict[str, Any]]:
        if self._last_grade is None:
            return None
        return grade_result_to_dict(self._last_grade)

    def grade_current_episode(self) -> Optional[Dict[str, Any]]:
        if self._episode_log is None or not self._episode_log.steps_logged:
            return None

        temp_log = EpisodeLog(
            task_id=self._episode_log.task_id,
            total_steps=self._episode_log.total_steps,
        )
        temp_log.steps_logged = list(self._episode_log.steps_logged)
        temp_log.blackout_occurred = self._episode_log.blackout_occurred
        temp_log.early_termination_step = self._episode_log.early_termination_step
        temp_log.initial_capital_budget = self._episode_log.initial_capital_budget

        if self._sim is not None:
            temp_log.finalise(
                battery_level=self._sim.battery.level_mwh,
                battery_capacity=self._sim.battery.capacity_mwh,
                reservoir_level=self._sim.hydro.reservoir_mwh,
                reservoir_capacity=HYDRO_RESERVOIR_CAP_MWH,
                capital_remaining=self._sim.capital_budget,
                total_cost=self._sim.cumulative_cost,
                total_emissions=self._sim.cumulative_emissions,
                plants_built=list(self._plants_built),
            )

        result = grade_episode(temp_log)
        return grade_result_to_dict(result)

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _track_plants(self) -> None:
        """Detect newly completed plants."""
        for ptype in ["solar", "wind", "hydro", "nuclear"]:
            plant = getattr(self._sim, ptype)
            if plant.available and ptype not in self._plants_built:
                self._plants_built.append(ptype)

    def _build_end_schedule(
        self,
        start_schedule: Dict[int, List[str]],
    ) -> Dict[int, List[str]]:
        end_schedule: Dict[int, List[str]] = {}
        for start_step, events in start_schedule.items():
            for event in events:
                duration = EVENT_DURATIONS.get(event, 3)
                end_step = start_step + duration
                end_schedule.setdefault(end_step, [])
                end_schedule[end_step].append(event)
        return end_schedule

    def _build_observation(
        self,
        reward: float,
        done: bool,
        dispatch_reward: float = 0.0,
        planning_reward: float = 0.0,
        market_reward: float = 0.0,
    ) -> EnergyGridObservation:
        sim = self._sim
        if sim is None:
            return EnergyGridObservation(done=done, reward=reward)

        hour = sim.step % 24
        result = self._last_step_result

        spinning_reserve = result.get("spinning_reserve_mw", 0.0)
        voltage_index = result.get("voltage_stability_index", 100.0)
        anomaly_score = result.get("anomaly_score", 0.0)

        construction_list = [
            {
                "type": entry.plant_type,
                "steps_remaining": entry.steps_remaining,
                "capacity_mw": entry.capacity_mw,
            }
            for entry in sim.construction_queue
        ]

        obs = EnergyGridObservation(
            demand_mw=round(sim.demand_mw, 2),
            hour=hour,
            day=sim.day,
            step=sim.step,
            season=sim.season,

            coal_mw=round(sim.coal.output_mw, 2),
            coal_online=sim.coal.online,
            coal_max_mw=round(sim.coal.max_mw, 2),
            coal_startup_remaining=sim.coal.startup_steps_remaining,
            coal_price=round(sim.coal_price, 3),

            solar_mw=round(sim.solar.output_mw, 2),
            solar_weather=result.get("solar_weather", sim.solar_weather),

            wind_mw=round(sim.wind.output_mw, 2),
            wind_speed_ms=round(sim.wind.wind_speed_ms, 2),

            hydro_mw=round(sim.hydro.output_mw, 2),
            reservoir_mwh=round(sim.hydro.reservoir_mwh, 2),
            reservoir_capacity_mwh=HYDRO_RESERVOIR_CAP_MWH,

            nuclear_mw=round(sim.nuclear.output_mw, 2),
            nuclear_online=sim.nuclear.online,
            nuclear_trip_remaining=sim.nuclear.trip_steps_remaining,

            battery_mwh=round(sim.battery.level_mwh, 2),
            battery_capacity_mwh=round(sim.battery.capacity_mwh, 2),

            unmet_demand_mw=round(result.get("unmet_demand_mw", 0.0), 2),
            frequency_hz=round(sim.frequency.frequency, 4),
            load_shedding_mw=round(result.get("load_shed_mw", 0.0), 2),
            blackout_risk=result.get("blackout_risk", "none"),
            spinning_reserve_mw=round(spinning_reserve, 2),
            spinning_reserve_required_mw=round(result.get("spinning_reserve_required_mw", 0.0), 2),

            active_events=list(sim.active_events),
            plants_building=construction_list,
            steps_until_shortfall=result.get("steps_until_shortfall", 999),
            fdi_active="fdi_attack" in self._sim.active_events,

            capital_budget=round(sim.capital_budget, 2),
            cumulative_cost=round(sim.cumulative_cost, 4),
            cumulative_emissions_tons=round(sim.cumulative_emissions, 2),

            # Market agent fields
            grid_export_mw=round(self._grid_export_mw, 2),
            grid_import_mw=round(self._grid_import_mw, 2),
            trading_credits=round(self._trading_credits, 4),

            done=done,
            reward=reward,
            episode_ended_early=sim.blackout_this_step,
            task_id=self._task_id,

            # Per-agent reward breakdown
            dispatch_reward=round(dispatch_reward, 4),
            planning_reward=round(planning_reward, 4),
            market_reward=round(market_reward, 4),
            
            # Phase 1: New fields
            coal_health_pct=round(sim.coal.health_pct, 1),
            duck_curve_stress_mw_per_step=round(result.get("duck_curve_stress_mw_per_step", 0.0), 2),
            spot_price=round(result.get("spot_price", 1.0), 3),
            carbon_price_per_ton=round(result.get("carbon_price_per_ton", 45.0), 2),
            rate_of_change_hz_per_step=round(result.get("rate_of_change_hz_per_step", 0.0), 4),
            duck_curve_stress=result.get("duck_curve_stress", 0.0),
            voltage_stability_index=result.get("voltage_stability_index", 100.0),
            spot_price=result.get("spot_price", 1.0),
            anomaly_score=result.get("anomaly_score", 0.0),
        )

        # Apply per-agent corruption
        if "fdi_attack" in self._sim.active_events:
            obs = self._apply_fdi(obs, agent_type)

        if self._normalize:
            obs_dict = obs.model_dump()
            normalized_dict = normalize_observation(obs_dict, self._task_id)
            obs = EnergyGridObservation(**normalized_dict)

        return obs

    def _log_step(self, result: Dict[str, Any]) -> None:
        if self._episode_log is None or self._sim is None:
            return

        sim = self._sim
        log_entry = StepLog(
            step=sim.step,
            demand_mw=sim.demand_mw,
            total_supply_mw=result.get("total_supply_mw", 0.0),
            unmet_demand_mw=result.get("unmet_demand_mw", 0.0),
            frequency_hz=sim.frequency.frequency,
            coal_mw=sim.coal.output_mw,
            solar_mw=sim.solar.output_mw,
            wind_mw=sim.wind.output_mw,
            hydro_mw=sim.hydro.output_mw,
            nuclear_mw=sim.nuclear.output_mw,
            battery_mwh=sim.battery.level_mwh,
            battery_capacity_mwh=sim.battery.capacity_mwh,
            reservoir_level_mwh=sim.hydro.reservoir_mwh,
            reservoir_capacity_mwh=HYDRO_RESERVOIR_CAP_MWH,
            cumulative_cost=sim.cumulative_cost,
            cumulative_emissions_tons=sim.cumulative_emissions,
            feedin_credits_mwh=round(sim.cumulative_feedin_credits, 2),
            coal_price=sim.coal_price,
            load_shedding_mw=result.get("load_shed_mw", 0.0),
            active_events=list(sim.active_events),
            capital_budget_remaining=sim.capital_budget,
            plants_built=list(self._plants_built),
            blackout=result.get("blackout", False),
            reward=result.get("reward", 0.0),
        )
        self._episode_log.log_step(log_entry)

    def _finalise_episode(self) -> None:
        if self._episode_log is None or self._sim is None:
            return

        self._episode_log.finalise(
            battery_level=self._sim.battery.level_mwh,
            battery_capacity=self._sim.battery.capacity_mwh,
            reservoir_level=self._sim.hydro.reservoir_mwh,
            reservoir_capacity=HYDRO_RESERVOIR_CAP_MWH,
            capital_remaining=self._sim.capital_budget,
            total_cost=self._sim.cumulative_cost,
            total_emissions=self._sim.cumulative_emissions,
            plants_built=list(self._plants_built),
        )

        self._last_grade = grade_episode(self._episode_log)

    @property
    def state(self) -> State:
        return self._state