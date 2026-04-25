"""
Energy Grid Environment Clients.

Two client classes:

    EnergyGridEnv         — original single-agent client (backward compatible)
    MultiAgentGridEnv     — new multi-agent client with typed per-agent methods

Example — single-agent (unchanged):
    >>> with EnergyGridEnv(base_url="http://localhost:8000") as client:
    ...     result = client.reset()
    ...     action = EnergyGridAction(coal_delta=50.0, battery_mode="discharge")
    ...     result = client.step(action)

Example — multi-agent:
    >>> env = MultiAgentGridEnv(base_url="http://localhost:8000")
    >>> obs = env.reset("hard")
    >>> # Each agent submits independently
    >>> env.submit_planning(PlanningAgentAction(plant_action="build_nuclear"))
    >>> env.submit_dispatch(DispatchAgentAction(coal_delta=50.0, battery_mode="idle"))
    >>> obs = env.submit_market(MarketAgentAction(demand_response_mw=0.0))
    >>> # obs is updated only after all three submit
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
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
    )


# ---------------------------------------------------------------------------
# Original single-agent client (unchanged)
# ---------------------------------------------------------------------------

class EnergyGridEnv(
    EnvClient[EnergyGridAction, EnergyGridObservation, State]
):
    """
    WebSocket client for the Energy Grid Environment (single-agent mode).
    Fully backward compatible with all existing code.
    """

    def _step_payload(self, action: EnergyGridAction) -> Dict[str, Any]:
        return {
            "coal_delta": action.coal_delta,
            "hydro_delta": action.hydro_delta,
            "nuclear_delta": action.nuclear_delta,
            "battery_mode": action.battery_mode,
            "plant_action": action.plant_action,
            "emergency_coal_boost": action.emergency_coal_boost,
            "demand_response_mw": action.demand_response_mw,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[EnergyGridObservation]:
        obs_data: Dict[str, Any] = payload.get("observation", payload)
        construction: List[Dict[str, Any]] = obs_data.get("plants_building", [])

        observation = EnergyGridObservation(
            demand_mw=obs_data.get("demand_mw", 0.0),
            hour=obs_data.get("hour", obs_data.get("time_of_day", 0)),
            day=obs_data.get("day", 1),
            step=obs_data.get("step", 0),
            season=obs_data.get("season", "spring"),
            coal_mw=obs_data.get("coal_mw", 0.0),
            coal_online=obs_data.get("coal_online", True),
            coal_startup_remaining=obs_data.get("coal_startup_remaining",
                obs_data.get("coal_startup_steps_remaining", 0)),
            coal_max_mw=obs_data.get("coal_max_mw", 600.0),
            coal_price=obs_data.get("coal_price", 1.0),
            solar_mw=obs_data.get("solar_mw", 0.0),
            solar_weather=obs_data.get("solar_weather", "clear"),
            wind_mw=obs_data.get("wind_mw", 0.0),
            wind_speed_ms=obs_data.get("wind_speed_ms", 0.0),
            hydro_mw=obs_data.get("hydro_mw", 0.0),
            reservoir_mwh=obs_data.get("reservoir_mwh",
                obs_data.get("reservoir_level_mwh", 600.0)),
            reservoir_capacity_mwh=obs_data.get("reservoir_capacity_mwh", 1000.0),
            nuclear_mw=obs_data.get("nuclear_mw", 0.0),
            nuclear_online=obs_data.get("nuclear_online", False),
            nuclear_trip_remaining=obs_data.get("nuclear_trip_remaining",
                obs_data.get("nuclear_trip_steps_remaining", 0)),
            battery_mwh=obs_data.get("battery_mwh", 100.0),
            battery_capacity_mwh=obs_data.get("battery_capacity_mwh", 200.0),
            unmet_demand_mw=obs_data.get("unmet_demand_mw", 0.0),
            frequency_hz=obs_data.get("frequency_hz", 50.0),
            load_shedding_mw=obs_data.get("load_shedding_mw", 0.0),
            blackout_risk=obs_data.get("blackout_risk", "none"),
            spinning_reserve_mw=obs_data.get("spinning_reserve_mw", 0.0),
            active_events=obs_data.get("active_events", []),
            plants_building=construction,
            capital_budget=obs_data.get("capital_budget", 0.0),
            cumulative_cost=obs_data.get("cumulative_cost", 0.0),
            cumulative_emissions_tons=obs_data.get("cumulative_emissions_tons", 0.0),
            grid_export_mw=obs_data.get("grid_export_mw", 0.0),
            grid_import_mw=obs_data.get("grid_import_mw", 0.0),
            trading_credits=obs_data.get("trading_credits", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            episode_ended_early=obs_data.get("episode_ended_early", False),
            task_id=obs_data.get("task_id", "easy"),
            dispatch_reward=obs_data.get("dispatch_reward", 0.0),
            planning_reward=obs_data.get("planning_reward", 0.0),
            market_reward=obs_data.get("market_reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )


# ---------------------------------------------------------------------------
# Multi-agent HTTP client (new)
# ---------------------------------------------------------------------------

class MultiAgentGridEnv:
    """
    HTTP client for the Energy Grid Environment in multi-agent mode.

    Each agent (planning, dispatch, market) submits actions independently
    via separate HTTP calls. The simulator advances only when all three
    have submitted.

    Usage:
        env = MultiAgentGridEnv(base_url="http://localhost:8000")
        obs = env.reset("hard")

        for step in range(72):
            # Each agent can run in parallel (e.g. separate threads/processes)
            env.submit_planning(PlanningAgentAction(plant_action="build_nuclear"))
            env.submit_dispatch(DispatchAgentAction(coal_delta=50.0))
            obs = env.submit_market(MarketAgentAction(demand_response_mw=0.0))

            print(f"Dispatch reward: {obs.dispatch_reward:.3f}")
            print(f"Planning reward: {obs.planning_reward:.3f}")
            print(f"Market reward:   {obs.market_reward:.3f}")

            if obs.done:
                break
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._last_obs: Optional[EnergyGridObservation] = None

    def reset(self, task_id: str = "easy") -> EnergyGridObservation:
        """Reset the environment and return initial observation."""
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        obs = self._parse_obs(payload)
        self._last_obs = obs
        return obs

    def submit_planning(
        self, action: PlanningAgentAction
    ) -> EnergyGridObservation:
        """
        Submit planning agent action.
        Returns current obs (step not yet advanced).
        """
        resp = self._session.post(
            f"{self.base_url}/step/planning",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def submit_dispatch(
        self, action: DispatchAgentAction
    ) -> EnergyGridObservation:
        """
        Submit dispatch agent action.
        Returns current obs (step not yet advanced unless market already submitted).
        """
        resp = self._session.post(
            f"{self.base_url}/step/dispatch",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def submit_market(
        self, action: MarketAgentAction
    ) -> EnergyGridObservation:
        """
        Submit market agent action.
        If this completes the buffer, returns the NEW observation
        after simulator advance. Otherwise returns current obs.
        """
        resp = self._session.post(
            f"{self.base_url}/step/market",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        obs = self._parse_response(resp.json())
        self._last_obs = obs
        return obs

    def buffer_status(self) -> Dict[str, Any]:
        """Check which agents have submitted for the current step."""
        resp = self._session.get(
            f"{self.base_url}/step/buffer_status",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> Dict[str, Any]:
        """Get grader score for the most recent completed episode."""
        resp = self._session.post(
            f"{self.base_url}/grader",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "MultiAgentGridEnv":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _parse_response(self, payload: Dict[str, Any]) -> EnergyGridObservation:
        return self._parse_obs(payload.get("observation", payload))

    def _parse_obs(self, obs_data: Dict[str, Any]) -> EnergyGridObservation:
        construction = obs_data.get("plants_building", [])
        return EnergyGridObservation(
            demand_mw=obs_data.get("demand_mw", 0.0),
            hour=obs_data.get("hour", obs_data.get("time_of_day", 0)),
            day=obs_data.get("day", 1),
            step=obs_data.get("step", 0),
            season=obs_data.get("season", "spring"),
            coal_mw=obs_data.get("coal_mw", 0.0),
            coal_online=obs_data.get("coal_online", True),
            coal_startup_remaining=obs_data.get("coal_startup_remaining", 0),
            coal_max_mw=obs_data.get("coal_max_mw", 600.0),
            coal_price=obs_data.get("coal_price", 1.0),
            solar_mw=obs_data.get("solar_mw", 0.0),
            solar_weather=obs_data.get("solar_weather", "clear"),
            wind_mw=obs_data.get("wind_mw", 0.0),
            wind_speed_ms=obs_data.get("wind_speed_ms", 0.0),
            hydro_mw=obs_data.get("hydro_mw", 0.0),
            reservoir_mwh=obs_data.get("reservoir_mwh", 600.0),
            reservoir_capacity_mwh=obs_data.get("reservoir_capacity_mwh", 1000.0),
            nuclear_mw=obs_data.get("nuclear_mw", 0.0),
            nuclear_online=obs_data.get("nuclear_online", False),
            nuclear_trip_remaining=obs_data.get("nuclear_trip_remaining", 0),
            battery_mwh=obs_data.get("battery_mwh", 100.0),
            battery_capacity_mwh=obs_data.get("battery_capacity_mwh", 200.0),
            unmet_demand_mw=obs_data.get("unmet_demand_mw", 0.0),
            frequency_hz=obs_data.get("frequency_hz", 50.0),
            load_shedding_mw=obs_data.get("load_shedding_mw", 0.0),
            blackout_risk=obs_data.get("blackout_risk", "none"),
            spinning_reserve_mw=obs_data.get("spinning_reserve_mw", 0.0),
            active_events=obs_data.get("active_events", []),
            plants_building=construction,
            capital_budget=obs_data.get("capital_budget", 0.0),
            cumulative_cost=obs_data.get("cumulative_cost", 0.0),
            cumulative_emissions_tons=obs_data.get("cumulative_emissions_tons", 0.0),
            grid_export_mw=obs_data.get("grid_export_mw", 0.0),
            grid_import_mw=obs_data.get("grid_import_mw", 0.0),
            trading_credits=obs_data.get("trading_credits", 0.0),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            episode_ended_early=obs_data.get("episode_ended_early", False),
            task_id=obs_data.get("task_id", "easy"),
            dispatch_reward=obs_data.get("dispatch_reward", 0.0),
            planning_reward=obs_data.get("planning_reward", 0.0),
            market_reward=obs_data.get("market_reward", 0.0),
            # Phase 1 fields
            coal_health_pct=obs_data.get("coal_health_pct", 100.0),
            duck_curve_stress_mw_per_step=obs_data.get("duck_curve_stress_mw_per_step", 0.0),
            spot_price=obs_data.get("spot_price", 1.0),
            carbon_price_per_ton=obs_data.get("carbon_price_per_ton", 45.0),
            rate_of_change_hz_per_step=obs_data.get("rate_of_change_hz_per_step", 0.0),
            voltage_stability_index=obs_data.get("voltage_stability_index", 100.0),
        )