"""
Energy Grid Management Environment — OpenEnv Environment Implementation.

This module implements the OpenEnv Environment interface for the energy
grid simulation. It wires together:
    - simulator.py  (physics engine)
    - tasks.py      (task configurations)
    - grader.py     (episode scoring)
    - models.py     (typed Pydantic action/observation models)

Public interface (OpenEnv spec):
    reset(task_id="easy")  → EnergyGridObservation
    step(action)           → EnergyGridObservation
    state                  → State

The environment maintains full episode state including the event schedule
(pre-computed at reset for determinism), the EpisodeLog (for grading),
and the GridSimState (physics).

Concurrent sessions are supported — each WebSocket client gets its own
environment instance via factory mode in app.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EnergyGridAction, EnergyGridObservation
except ImportError:
    from models import EnergyGridAction, EnergyGridObservation

try:
    from .simulator import (
        GridSimState,
        build_initial_state,
        schedule_events,
        simulator_step,
        EVENT_DURATIONS,
        HYDRO_RESERVOIR_CAP_MWH,
        SPINNING_RESERVE_RATIO,
        _compute_spinning_reserve,
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
        SPINNING_RESERVE_RATIO,
        _compute_spinning_reserve,
    )
    from server.tasks import get_task, TASK_ORDER
    from server.grader import (
        EpisodeLog,
        StepLog,
        grade_episode,
        grade_result_to_dict,
        GradeResult,
    )


class EnergyGridEnvironment(Environment):
    """
    OpenEnv-compliant Energy Grid Management Environment.

    Simulates operating a national electricity grid over 1–3 simulated
    days. The agent dispatches generation sources, manages battery
    storage, responds to stochastic weather and fault events, and in the
    Hard task makes long-term plant investment decisions.

    Supports three tasks:
        easy   — coal + battery, 1 day, no events
        medium — coal + solar + wind + battery, 2 days, weather events
        hard   — all sources buildable, 3 days, full event roster

    Usage:
        >>> env = EnergyGridEnvironment()
        >>> obs = env.reset("medium")
        >>> action = EnergyGridAction(coal_delta=50.0, battery_mode="idle")
        >>> obs = env.step(action)
        >>> score = env.get_last_grade()
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "easy"
        self._sim: Optional[GridSimState] = None
        self._episode_log: Optional[EpisodeLog] = None
        self._event_schedule: Dict[int, List[str]] = {}
        self._event_end_schedule: Dict[int, List[str]] = {}
        self._last_grade: Optional[GradeResult] = None
        self._last_step_result: Dict[str, Any] = {}
        self._plants_built: List[str] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    @property
    def current_task_id(self) -> str:
        """Return the current task_id for the active episode."""
        return self._task_id

    def reset(self, task_id: str = "easy") -> EnergyGridObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'easy', 'medium', 'hard'.

        Returns:
            Initial EnergyGridObservation with starting grid state.
        """
        if task_id not in TASK_ORDER:
            task_id = "easy"

        task = get_task(task_id)
        self._task_id = task_id

        # Fresh OpenEnv state
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Build physics state
        self._sim = build_initial_state(
            task_id=task_id,
            seed=task["seed"],
            total_steps=task["total_steps"],
            season=task["season"],
        )
        self._sim.capital_budget = task["capital_budget"]

        # Pre-schedule all events for this episode (deterministic)
        raw_schedule = schedule_events(
            task_id=task_id,
            total_steps=task["total_steps"],
            rng=self._sim.rng,
        )
        self._event_schedule = raw_schedule
        self._event_end_schedule = self._build_end_schedule(raw_schedule)

        # Fresh episode log
        self._episode_log = EpisodeLog(
            task_id=task_id,
            total_steps=task["total_steps"],
        )
        self._episode_log.initial_capital_budget = task["capital_budget"]

        self._last_grade = None
        self._last_step_result = {}
        self._plants_built = []

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: EnergyGridAction) -> EnergyGridObservation:  # type: ignore[override]
        """
        Execute one simulation step.

        Args:
            action: EnergyGridAction with generation adjustments and
                    optional plant/emergency actions.

        Returns:
            EnergyGridObservation reflecting the new grid state,
            plus reward and done flag.
        """
        if self._sim is None:
            # Auto-reset if step called before reset
            return self.reset(self._task_id)

        # Guard: prevent stepping after episode has ended
        if self._sim.episode_ended:
            return self._build_observation(reward=0.0, done=True)

        # Run physics
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

        # Track newly completed plants
        for entry in self._sim.construction_queue:
            pass  # construction_queue only holds in-progress items
        # Detect newly available plants by comparing before/after
        # (advance_construction is called inside simulator_step)
        for ptype in ["solar", "wind", "hydro", "nuclear"]:
            plant = getattr(self._sim, ptype)
            if plant.available and ptype not in self._plants_built:
                self._plants_built.append(ptype)

        # Log this step for grading
        self._log_step(result)

        # Update OpenEnv step counter
        self._state.step_count += 1

        reward = result["reward"]
        done = result["done"]

        # If episode ended, finalise the log and grade
        if done:
            self._finalise_episode()

        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        """Return current OpenEnv State (episode_id + step_count)."""
        return self._state

    # ------------------------------------------------------------------
    # Grading interface (used by /grader endpoint)
    # ------------------------------------------------------------------

    def get_last_grade(self) -> Optional[Dict[str, Any]]:
        """
        Return the graded result for the most recently completed episode.

        Returns None if no episode has been completed yet.
        """
        if self._last_grade is None:
            return None
        return grade_result_to_dict(self._last_grade)

    def grade_current_episode(self) -> Optional[Dict[str, Any]]:
        """
        Grade the current episode mid-run (partial grade).

        Useful for inspecting progress without ending the episode.
        Finalises the log temporarily without ending the episode.
        """
        if self._episode_log is None or not self._episode_log.steps_logged:
            return None

        # Snapshot current state into log without permanently finalising
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_end_schedule(
        self,
        start_schedule: Dict[int, List[str]],
    ) -> Dict[int, List[str]]:
        """
        Build a step → events_ending dict from the start schedule.

        Each event ends at start_step + duration.
        """
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
    ) -> EnergyGridObservation:
        """
        Construct an EnergyGridObservation from current sim state.
        """
        sim = self._sim
        if sim is None:
            return EnergyGridObservation(done=done, reward=reward)

        # Use (step - 1) because step was already incremented in simulator_step
        # This ensures time_of_day matches demand/solar computed at that hour
        hour = (sim.step - 1) % 24
        result = self._last_step_result

        # Spinning reserve
        spinning_reserve = result.get(
            "spinning_reserve_mw",
            _compute_spinning_reserve(sim),
        )
        spinning_reserve_required = result.get(
            "spinning_reserve_required_mw",
            sim.demand_mw * SPINNING_RESERVE_RATIO,
        )

        # Construction queue for observation
        construction_list = [
            {
                "type": entry.plant_type,
                "steps_remaining": entry.steps_remaining,
                "capacity_mw": entry.capacity_mw,
            }
            for entry in sim.construction_queue
        ]

        return EnergyGridObservation(
            # Demand & time
            demand_mw=round(sim.demand_mw, 2),
            time_of_day=hour,
            day=sim.day,
            step=sim.step,
            season=sim.season,

            # Coal
            coal_output_mw=round(sim.coal.output_mw, 2),
            coal_online=sim.coal.online,
            coal_startup_steps_remaining=sim.coal.startup_steps_remaining,
            coal_max_mw=round(sim.coal.max_mw, 2),
            coal_price=round(sim.coal_price, 3),

            # Solar
            solar_output_mw=round(sim.solar.output_mw, 2),
            solar_available=sim.solar.available,
            solar_weather=result.get("solar_weather", sim.solar_weather),

            # Wind
            wind_output_mw=round(sim.wind.output_mw, 2),
            wind_available=sim.wind.available,
            wind_speed_ms=round(sim.wind.wind_speed_ms, 2),

            # Hydro
            hydro_output_mw=round(sim.hydro.output_mw, 2),
            hydro_available=sim.hydro.available,
            reservoir_level_mwh=round(sim.hydro.reservoir_mwh, 2),
            reservoir_capacity_mwh=HYDRO_RESERVOIR_CAP_MWH,
            natural_inflow_mwh=round(sim.hydro.natural_inflow_mwh, 2),

            # Nuclear
            nuclear_output_mw=round(sim.nuclear.output_mw, 2),
            nuclear_available=sim.nuclear.available,
            nuclear_online=sim.nuclear.online,
            nuclear_trip_steps_remaining=sim.nuclear.trip_steps_remaining,

            # Battery
            battery_level_mwh=round(sim.battery.level_mwh, 2),
            battery_capacity_mwh=round(sim.battery.capacity_mwh, 2),

            # Grid health
            unmet_demand_mw=round(result.get("unmet_demand_mw", 0.0), 2),
            overproduction_mw=round(result.get("overproduction_mw", 0.0), 2),
            grid_frequency=round(sim.frequency.frequency, 4),
            rate_of_change_hz_per_step=round(sim.frequency.rocof, 4),
            system_inertia_seconds=round(
                result.get("system_inertia", 4.0), 3
            ),
            primary_response_active=sim.frequency.primary_response_active,
            load_shedding_mw=round(result.get("load_shed_mw", 0.0), 2),
            blackout_risk=result.get("blackout_risk", "none"),
            spinning_reserve_mw=round(spinning_reserve, 2),
            spinning_reserve_required_mw=round(spinning_reserve_required, 2),
            transmission_capacity_mw=round(sim.transmission_capacity_mw, 2),

            # Events
            active_events=list(sim.active_events),

            # Construction
            plants_under_construction=construction_list,

            # Economics
            capital_budget=round(sim.capital_budget, 2),
            cumulative_cost=round(sim.cumulative_cost, 4),
            cumulative_emissions_tons=round(sim.cumulative_emissions, 2),
            feedin_credits_mwh=round(sim.cumulative_feedin_credits, 2),

            # Episode metadata
            step_reward=round(reward, 4),
            done=done,
            reward=reward,
            episode_ended_early=sim.blackout_this_step,
            task_id=self._task_id,
        )

    def _log_step(self, result: Dict[str, Any]) -> None:
        """Append a StepLog entry for the current step."""
        if self._episode_log is None or self._sim is None:
            return

        sim = self._sim
        log_entry = StepLog(
            step=sim.step,
            demand_mw=sim.demand_mw,
            total_supply_mw=result.get("total_supply_mw", 0.0),
            unmet_demand_mw=result.get("unmet_demand_mw", 0.0),
            grid_frequency=sim.frequency.frequency,
            coal_output_mw=sim.coal.output_mw,
            solar_output_mw=sim.solar.output_mw,
            wind_output_mw=sim.wind.output_mw,
            hydro_output_mw=sim.hydro.output_mw,
            nuclear_output_mw=sim.nuclear.output_mw,
            battery_level_mwh=sim.battery.level_mwh,
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
        """Finalise the episode log and compute the grade."""
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