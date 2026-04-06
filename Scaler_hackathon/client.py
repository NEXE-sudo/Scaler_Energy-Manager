"""
Energy Grid Environment Client.

WebSocket client for the EnergyGridEnvironment server. Maintains a
persistent connection for efficient multi-step interactions with lower
latency than repeated HTTP calls.

Each client instance gets its own dedicated environment session on the
server (when max_concurrent_envs > 1 in app.py).

Example — basic usage:
    >>> from client import EnergyGridEnv
    >>> from models import EnergyGridAction
    >>>
    >>> with EnergyGridEnv(base_url="http://localhost:8000") as client:
    ...     result = client.reset()
    ...     obs = result.observation
    ...     print(f"Demand: {obs.demand_mw} MW")
    ...
    ...     action = EnergyGridAction(
    ...         coal_delta=50.0,
    ...         battery_mode="discharge",
    ...     )
    ...     result = client.step(action)
    ...     print(f"Reward: {result.reward}")

Example — from Docker image:
    >>> client = EnergyGridEnv.from_docker_image("energy-grid-openenv:latest")
    >>> try:
    ...     result = client.reset()
    ...     for _ in range(24):
    ...         action = EnergyGridAction(coal_delta=0.0, battery_mode="idle")
    ...         result = client.step(action)
    ...         if result.done:
    ...             break
    ... finally:
    ...     client.close()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import EnergyGridAction, EnergyGridObservation
except ImportError:
    from models import EnergyGridAction, EnergyGridObservation


class EnergyGridEnv(
    EnvClient[EnergyGridAction, EnergyGridObservation, State]
):
    """
    WebSocket client for the Energy Grid Management Environment.

    Wraps the OpenEnv EnvClient base class with typed payload
    serialisation and response parsing for the energy grid models.

    The client handles:
        - Serialising EnergyGridAction to the wire format
        - Deserialising server responses into EnergyGridObservation
        - Parsing the OpenEnv State object
    """

    def _step_payload(self, action: EnergyGridAction) -> Dict[str, Any]:
        """
        Serialise EnergyGridAction to JSON payload for the step message.

        All fields are included explicitly so the server always receives
        a complete action even if the client is constructed with defaults.

        Args:
            action: EnergyGridAction instance from the agent.

        Returns:
            Dict suitable for JSON encoding and WebSocket transmission.
        """
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
        """
        Parse server response into StepResult[EnergyGridObservation].

        Handles both flat and nested observation payloads — the server
        may return the observation fields at the top level or nested
        under an "observation" key depending on the OpenEnv version.

        Args:
            payload: JSON response dict from the server.

        Returns:
            StepResult with fully populated EnergyGridObservation.
        """
        # Support both flat and nested observation formats
        obs_data: Dict[str, Any] = payload.get("observation", payload)

        # Construction queue — list of dicts
        construction: List[Dict[str, Any]] = obs_data.get(
            "plants_under_construction", []
        )

        observation = EnergyGridObservation(
            # Demand & time
            demand_mw=obs_data.get("demand_mw", 0.0),
            time_of_day=obs_data.get("time_of_day", 0),
            day=obs_data.get("day", 1),
            step=obs_data.get("step", 0),
            season=obs_data.get("season", "spring"),

            # Coal
            coal_output_mw=obs_data.get("coal_output_mw", 0.0),
            coal_online=obs_data.get("coal_online", True),
            coal_startup_steps_remaining=obs_data.get(
                "coal_startup_steps_remaining", 0
            ),
            coal_max_mw=obs_data.get("coal_max_mw", 600.0),
            coal_price=obs_data.get("coal_price", 1.0),

            # Solar
            solar_output_mw=obs_data.get("solar_output_mw", 0.0),
            solar_available=obs_data.get("solar_available", False),
            solar_weather=obs_data.get("solar_weather", "clear"),

            # Wind
            wind_output_mw=obs_data.get("wind_output_mw", 0.0),
            wind_available=obs_data.get("wind_available", False),
            wind_speed_ms=obs_data.get("wind_speed_ms", 0.0),

            # Hydro
            hydro_output_mw=obs_data.get("hydro_output_mw", 0.0),
            hydro_available=obs_data.get("hydro_available", False),
            reservoir_level_mwh=obs_data.get("reservoir_level_mwh", 600.0),
            reservoir_capacity_mwh=obs_data.get(
                "reservoir_capacity_mwh", 1000.0
            ),
            natural_inflow_mwh=obs_data.get("natural_inflow_mwh", 15.0),

            # Nuclear
            nuclear_output_mw=obs_data.get("nuclear_output_mw", 0.0),
            nuclear_available=obs_data.get("nuclear_available", False),
            nuclear_online=obs_data.get("nuclear_online", False),
            nuclear_trip_steps_remaining=obs_data.get(
                "nuclear_trip_steps_remaining", 0
            ),

            # Battery
            battery_level_mwh=obs_data.get("battery_level_mwh", 100.0),
            battery_capacity_mwh=obs_data.get("battery_capacity_mwh", 200.0),

            # Grid health
            unmet_demand_mw=obs_data.get("unmet_demand_mw", 0.0),
            overproduction_mw=obs_data.get("overproduction_mw", 0.0),
            grid_frequency=obs_data.get("grid_frequency", 50.0),
            rate_of_change_hz_per_step=obs_data.get(
                "rate_of_change_hz_per_step", 0.0
            ),
            system_inertia_seconds=obs_data.get("system_inertia_seconds", 4.0),
            primary_response_active=obs_data.get(
                "primary_response_active", False
            ),
            load_shedding_mw=obs_data.get("load_shedding_mw", 0.0),
            blackout_risk=obs_data.get("blackout_risk", "none"),
            spinning_reserve_mw=obs_data.get("spinning_reserve_mw", 0.0),
            spinning_reserve_required_mw=obs_data.get(
                "spinning_reserve_required_mw", 0.0
            ),
            transmission_capacity_mw=obs_data.get(
                "transmission_capacity_mw", 1200.0
            ),

            # Events
            active_events=obs_data.get("active_events", []),

            # Construction queue
            plants_under_construction=construction,

            # Economics
            capital_budget=obs_data.get("capital_budget", 0.0),
            cumulative_cost=obs_data.get("cumulative_cost", 0.0),
            cumulative_emissions_tons=obs_data.get(
                "cumulative_emissions_tons", 0.0
            ),
            feedin_credits_mwh=obs_data.get("feedin_credits_mwh", 0.0),

            # Episode metadata
            step_reward=obs_data.get("step_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            episode_ended_early=obs_data.get("episode_ended_early", False),
            task_id=obs_data.get("task_id", "easy"),

            # Pass through metadata if present
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into OpenEnv State object.

        Args:
            payload: JSON response from the /state endpoint.

        Returns:
            State with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )