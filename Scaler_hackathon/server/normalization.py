"""
Observation Normalization Utilities for Energy Grid Environment.

Provides normalization functions to rescale raw grid observations into
[0, 1] range for improved agent generalization and training stability.

Typical usage in environment:
    from .normalization import normalize_observation
    normalized_obs = normalize_observation(raw_obs, task_id)
"""

from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class NormalizationBounds:
    """Physical bounds for normalizing different observation features."""
    
    # Generation sources (MW)
    MIN_GENERATION: float = 0.0
    MAX_GENERATION: float = 1200.0  # Peak possible supply
    
    # Demand (MW)
    MIN_DEMAND: float = 200.0
    MAX_DEMAND: float = 1100.0
    
    # Frequency (Hz)
    MIN_FREQUENCY: float = 48.5
    MAX_FREQUENCY: float = 50.5
    
    # Battery (MWh)
    MIN_BATTERY_MWH: float = 0.0
    MAX_BATTERY_MWH: float = 200.0
    
    # Reservoir (MWh)
    MIN_RESERVOIR_MWH: float = 0.0
    MAX_RESERVOIR_MWH: float = 1000.0
    
    # Price ($/MWh)
    MIN_PRICE: float = 20.0
    MAX_PRICE: float = 200.0
    
    # Emissions (tons CO2)
    MIN_EMISSIONS: float = 0.0
    MAX_EMISSIONS: float = 5000.0
    
    # Cost ($)
    MIN_COST: float = 0.0
    MAX_COST: float = 500.0
    
    # Capital budget ($1000s)
    MIN_CAPITAL: float = 0.0
    MAX_CAPITAL: float = 500.0


def normalize_value(
    value: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    clip: bool = True
) -> float:
    """
    Normalize a scalar value to [0, 1] range.
    
    Args:
        value: Raw value to normalize
        min_val: Minimum physical bound
        max_val: Maximum physical bound
        clip: If True, clamp result to [0, 1]
    
    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    
    if clip:
        normalized = max(0.0, min(1.0, normalized))
    
    return normalized


def normalize_observation(
    obs_dict: Dict[str, Any],
    task_id: str = "easy"
) -> Dict[str, Any]:
    """
    Normalize raw EnergyGridObservation to [0, 1] range for better
    agent generalization. Returns a new dict with normalized values.
    
    Normalization strategy:
    - Generation / demand: [200 MW, 1100 MW] → [0, 1]
    - Frequency: [48.5 Hz, 50.5 Hz] → [0, 1]
    - Battery: [0 MWh, 200 MWh] → [0, 1]
    - Reservoir: [0 MWh, 1000 MWh] → [0, 1]
    - Price: [20 $/MWh, 200 $/MWh] → [0, 1]
    - Costs/Budget: task-dependent scaling
    - Categorical features: preserved as-is
    
    Args:
        obs_dict: Raw observation as dictionary
        task_id: Task identifier (easy/medium/hard) for task-specific scaling
    
    Returns:
        Dict with same structure but normalized numerical values
    """
    bounds = NormalizationBounds()
    normalized = obs_dict.copy()
    
    # Task-specific max cost references
    task_max_costs = {
        "easy": 20.0,
        "medium": 100.0,
        "hard": 300.0,
    }
    max_cost = task_max_costs.get(task_id, 100.0)
    
    # ---- Generation sources (MW) ----
    if "demand_mw" in normalized:
        normalized["demand_mw"] = normalize_value(
            normalized["demand_mw"],
            bounds.MIN_DEMAND,
            bounds.MAX_DEMAND
        )
    
    for key in [
        "coal_output_mw",
        "solar_output_mw", 
        "wind_output_mw",
        "hydro_output_mw",
        "nuclear_output_mw",
    ]:
        if key in normalized:
            normalized[key] = normalize_value(
                normalized[key],
                bounds.MIN_GENERATION,
                bounds.MAX_GENERATION
            )
    
    # Coal max capacity
    if "coal_max_mw" in normalized:
        normalized["coal_max_mw"] = normalize_value(
            normalized["coal_max_mw"],
            0.0,
            600.0  # coal max
        )
    
    # ---- Frequency (Hz) ----
    if "grid_frequency" in normalized:
        normalized["grid_frequency"] = normalize_value(
            normalized["grid_frequency"],
            bounds.MIN_FREQUENCY,
            bounds.MAX_FREQUENCY
        )
    
    if "rate_of_change_hz_per_step" in normalized:
        # Rate of change: [-0.5, +0.5] Hz/step
        normalized["rate_of_change_hz_per_step"] = normalize_value(
            normalized["rate_of_change_hz_per_step"],
            -0.5,
            0.5
        )
    
    # ---- Battery (MWh and %) ----
    if "battery_level_mwh" in normalized:
        normalized["battery_level_mwh"] = normalize_value(
            normalized["battery_level_mwh"],
            bounds.MIN_BATTERY_MWH,
            bounds.MAX_BATTERY_MWH
        )
    
    if "battery_capacity_mwh" in normalized:
        # Capacity is typically 200 MWh
        normalized["battery_capacity_mwh"] = normalize_value(
            normalized["battery_capacity_mwh"],
            100.0,
            200.0
        )
    
    # ---- Hydro reservoir (MWh and %) ----
    if "reservoir_level_mwh" in normalized:
        normalized["reservoir_level_mwh"] = normalize_value(
            normalized["reservoir_level_mwh"],
            bounds.MIN_RESERVOIR_MWH,
            bounds.MAX_RESERVOIR_MWH
        )
    
    if "reservoir_capacity_mwh" in normalized:
        normalized["reservoir_capacity_mwh"] = normalize_value(
            normalized["reservoir_capacity_mwh"],
            900.0,
            1100.0
        )
    
    if "natural_inflow_mwh" in normalized:
        # Natural inflow: typically [5, 50] MWh/step
        normalized["natural_inflow_mwh"] = normalize_value(
            normalized["natural_inflow_mwh"],
            0.0,
            100.0
        )
    
    # ---- Unmet demand (MW) ----
    if "unmet_demand_mw" in normalized:
        normalized["unmet_demand_mw"] = normalize_value(
            normalized["unmet_demand_mw"],
            0.0,
            300.0  # worst case
        )
    
    if "overproduction_mw" in normalized:
        normalized["overproduction_mw"] = normalize_value(
            normalized["overproduction_mw"],
            0.0,
            200.0
        )
    
    if "load_shedding_mw" in normalized:
        normalized["load_shedding_mw"] = normalize_value(
            normalized["load_shedding_mw"],
            0.0,
            300.0
        )
    
    # ---- Spinning reserve (MW) ----
    if "spinning_reserve_mw" in normalized:
        normalized["spinning_reserve_mw"] = normalize_value(
            normalized["spinning_reserve_mw"],
            0.0,
            200.0
        )
    
    if "spinning_reserve_required_mw" in normalized:
        normalized["spinning_reserve_required_mw"] = normalize_value(
            normalized["spinning_reserve_required_mw"],
            0.0,
            200.0
        )
    
    # ---- Transmission capacity (MW) ----
    if "transmission_capacity_mw" in normalized:
        normalized["transmission_capacity_mw"] = normalize_value(
            normalized["transmission_capacity_mw"],
            800.0,
            1200.0
        )
    
    # ---- Prices ($/MWh) ----
    if "coal_price" in normalized:
        normalized["coal_price"] = normalize_value(
            normalized["coal_price"],
            bounds.MIN_PRICE,
            bounds.MAX_PRICE
        )
    
    # ---- Economics ----
    if "cumulative_cost" in normalized:
        normalized["cumulative_cost"] = normalize_value(
            normalized["cumulative_cost"],
            0.0,
            max_cost
        )
    
    if "cumulative_emissions_tons" in normalized:
        # Max emissions roughly proportional to task difficulty
        max_emissions = {"easy": 1000.0, "medium": 3000.0, "hard": 5000.0}.get(
            task_id, 5000.0
        )
        normalized["cumulative_emissions_tons"] = normalize_value(
            normalized["cumulative_emissions_tons"],
            0.0,
            max_emissions
        )
    
    if "feedin_credits_mwh" in normalized:
        normalized["feedin_credits_mwh"] = normalize_value(
            normalized["feedin_credits_mwh"],
            0.0,
            500.0
        )
    
    # Capital budget (task-aware)
    capital_budgets = {"easy": 0.0, "medium": 0.0, "hard": 500.0}
    max_capital = capital_budgets.get(task_id, 500.0)
    
    if "capital_budget" in normalized and max_capital > 0.0:
        normalized["capital_budget"] = normalize_value(
            normalized["capital_budget"],
            0.0,
            max_capital
        )
    
    # ---- Time features (keep natural scale for cyclic encoding) ----
    # time_of_day, day, step, season — keep as-is for agents to learn cyclic patterns
    
    # ---- Categorical features (preserve as-is) ----
    # active_events, season, task_id, etc.
    
    return normalized


def denormalize_observation(
    normalized_dict: Dict[str, Any],
    task_id: str = "easy"
) -> Dict[str, Any]:
    """
    Reverse the normalization process (denormalize [0, 1] back to physical units).
    Useful for debugging or visualization.
    
    Args:
        normalized_dict: Normalized observation
        task_id: Task identifier for task-specific scaling
    
    Returns:
        Denormalized observation in original physical units
    """
    bounds = NormalizationBounds()
    denormalized = normalized_dict.copy()
    
    task_max_costs = {
        "easy": 20.0,
        "medium": 100.0,
        "hard": 300.0,
    }
    max_cost = task_max_costs.get(task_id, 100.0)
    
    # Reverse each normalization
    if "demand_mw" in denormalized:
        norm_val = denormalized["demand_mw"]
        denormalized["demand_mw"] = bounds.MIN_DEMAND + norm_val * (bounds.MAX_DEMAND - bounds.MIN_DEMAND)
    
    for key in ["coal_output_mw", "solar_output_mw", "wind_output_mw", "hydro_output_mw", "nuclear_output_mw"]:
        if key in denormalized:
            norm_val = denormalized[key]
            denormalized[key] = bounds.MIN_GENERATION + norm_val * (bounds.MAX_GENERATION - bounds.MIN_GENERATION)
    
    if "grid_frequency" in denormalized:
        norm_val = denormalized["grid_frequency"]
        denormalized["grid_frequency"] = bounds.MIN_FREQUENCY + norm_val * (bounds.MAX_FREQUENCY - bounds.MIN_FREQUENCY)
    
    if "battery_level_mwh" in denormalized:
        norm_val = denormalized["battery_level_mwh"]
        denormalized["battery_level_mwh"] = bounds.MIN_BATTERY_MWH + norm_val * (bounds.MAX_BATTERY_MWH - bounds.MIN_BATTERY_MWH)
    
    if "cumulative_cost" in denormalized:
        norm_val = denormalized["cumulative_cost"]
        denormalized["cumulative_cost"] = norm_val * max_cost
    
    return denormalized
