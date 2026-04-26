"""
LLM Adapter Layer: Converts observations to natural language and extracts
structured actions from LLM reasoning outputs.

Enables LLM agents to perform reasoning (Thought:) followed by structured
action output (Action: {...JSON...}), improving interpretability and
decision quality over raw JSON-only outputs.
"""

import json
import re
from typing import Any, Dict, Optional

# Action bounds (same as ppo_agent.py for consistency)
ACTION_BOUNDS = {
    "coal_delta": (-100.0, 100.0),
    "hydro_delta": (-80.0, 80.0),
    "nuclear_delta": (-10.0, 10.0),
    "demand_response_mw": (0.0, 150.0),
}

SAFE_DEFAULT_ACTION = {
    "coal_delta": 0.0,
    "hydro_delta": 0.0,
    "nuclear_delta": 0.0,
    "battery_mode": "idle",
    "plant_action": "none",
    "emergency_coal_boost": False,
    "demand_response_mw": 0.0,
    "grid_export_mw": 0.0,
    "grid_import_mw": 0.0,
    "coal_price_bid": None,
}


def observation_to_text(obs: dict) -> str:
    demand = obs.get("demand_mw", 0)
    supply = (
        obs.get("coal_mw", 0)
        + obs.get("solar_mw", 0)
        + obs.get("wind_mw", 0)
        + obs.get("hydro_mw", 0)
        + obs.get("nuclear_mw", 0)
    )

    gap = demand - supply
    freq = obs.get("frequency_hz", 50.0)

    reserve = obs.get("spinning_reserve_mw", 0)
    required_reserve = obs.get("spinning_reserve_required_mw", 0)

    battery_pct = int(
        100 * obs.get("battery_mwh", 0)
        / max(1, obs.get("battery_capacity_mwh", 1))
    )

    shortfall_steps = obs.get("steps_until_shortfall", 999)
    blackout_risk = obs.get("blackout_risk", "none")

    # ---- Market + dynamics ----

    duck = obs.get("duck_curve_stress", 0.0)
    spot = obs.get("spot_price", 1.0)

    market_msg = f"Market: Price={spot:.2f}, Ramp={duck:+.1f}\n"

    # ---- Negotiation History (Round 2) ----
    negotiation_msg = ""
    history = obs.get("negotiation_history")
    if history:
        negotiation_msg = "\nNEGOTIATION (Last 3):\n"
        for entry in history[-3:]:
            agent = entry.get("agent", "unknown").upper()
            thought = entry.get("thought", "No thought provided.")
            action = entry.get("proposal", {})
            action_sum = ", ".join([f"{k}={v}" for k, v in action.items() if v not in (0.0, "none", "idle", False, None)])
            negotiation_msg += f"- {agent}: {thought} [{action_sum}]\n"

    # ---- Final prompt ----

    return f"""Grid Status:
- Demand: {demand:.0f}MW | Supply: {supply:.0f}MW
- Freq: {freq:.2f}Hz | Reserve: {reserve:.0f}/{required_reserve:.0f}MW
- Battery: {battery_pct}% | Risk: {blackout_risk}
{market_msg}{negotiation_msg}
Goal: Maintain 50Hz, avoid blackout.
"""

def extract_action_from_llm_output(text: str) -> dict:
    if not text:
        return dict(SAFE_DEFAULT_ACTION)

    # Strip markdown fences
    text = re.sub(r"```[a-z]*\n?|```", "", text).strip()

    # Find the last JSON object (more likely to be the final action)
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    if not matches:
        # Try multi-level braces
        start = text.rfind("{")
        end   = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return dict(SAFE_DEFAULT_ACTION)
        matches = [type("m", (), {"group": lambda self, x=0: text[start:end+1]})()]

    for match in reversed(matches):
        raw = match.group(0)

        # Clean Python literals → JSON
        raw = re.sub(r"\bNone\b",  "0",     raw)
        raw = re.sub(r"\bTrue\b",  "true",  raw)
        raw = re.sub(r"\bFalse\b", "false", raw)
        raw = raw.replace("'", '"')
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        raw = re.sub(r"//.*",  "",  raw)

        # Skip placeholder text
        if "valid JSON" in raw or "<" in raw:
            continue

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
        except json.JSONDecodeError:
            pass

    return dict(SAFE_DEFAULT_ACTION)

def _dict_to_action(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parsed JSON dict to validated action dict.
    Clips numeric fields to ACTION_BOUNDS, validates string fields.
    """
    def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return default
    
    # Battery mode
    battery_mode = str(data.get("battery_mode", "idle")).lower().strip()
    if battery_mode not in ("charge", "discharge", "idle"):
        battery_mode = "idle"
    
    # Plant action
    plant_action = str(data.get("plant_action", "none")).lower().strip()
    valid_actions = {
        "none", "build_solar", "build_wind", "build_hydro", "build_nuclear", "close_coal"
    }
    if plant_action not in valid_actions:
        plant_action = "none"
    
    # Emergency boost
    emergency_boost = data.get("emergency_coal_boost", False)
    if isinstance(emergency_boost, str):
        emergency_boost = emergency_boost.lower() in ("true", "1", "yes")
    else:
        emergency_boost = bool(emergency_boost)
    
    # Demand response (check for both single and multi-agent variants)
    demand_response = _clamp(
        data.get("demand_response_mw", 0.0),
        ACTION_BOUNDS["demand_response_mw"][0],
        ACTION_BOUNDS["demand_response_mw"][1],
        0.0
    )
    
    return {
        "coal_delta": _clamp(
            data.get("coal_delta", 0.0),
            ACTION_BOUNDS["coal_delta"][0],
            ACTION_BOUNDS["coal_delta"][1],
            0.0
        ),
        "hydro_delta": _clamp(
            data.get("hydro_delta", 0.0),
            ACTION_BOUNDS["hydro_delta"][0],
            ACTION_BOUNDS["hydro_delta"][1],
            0.0
        ),
        "nuclear_delta": _clamp(
            data.get("nuclear_delta", 0.0),
            ACTION_BOUNDS["nuclear_delta"][0],
            ACTION_BOUNDS["nuclear_delta"][1],
            0.0
        ),
        "battery_mode": battery_mode,
        "plant_action": plant_action,
        "emergency_coal_boost": emergency_boost,
        "demand_response_mw": demand_response,
        "grid_export_mw": _clamp(data.get("grid_export_mw", 0.0), 0.0, 100.0, 0.0),
        "grid_import_mw": _clamp(data.get("grid_import_mw", 0.0), 0.0, 100.0, 0.0),
        "coal_price_bid": data.get("coal_price_bid") if data.get("coal_price_bid") is not None else None,
    }

def build_compact_obs(obs):
    """
    Converts observation object into compact text prompt.
    """

    try:
        return f"""Grid Status:
    - Demand: {obs.demand_mw}MW | Supply: {obs.supply_mw}MW
    - Freq: {obs.frequency_hz:.2f}Hz | Reserve: {obs.reserve_margin_mw}/{obs.reserve_margin_mw}MW
    - Battery: {int(obs.battery_soc * 100)}% | Risk: {getattr(obs, 'risk_level', 'none')}
    Market: Price={getattr(obs, 'market_price', 1.0):.2f}

    Goal: Maintain 50Hz, avoid blackout."""
    except Exception:
            # fallback if object is dict-like
            d = obs if isinstance(obs, dict) else obs.__dict__

            return f"""Grid Status:
    - Demand: {d.get('demand_mw')}MW | Supply: {d.get('supply_mw')}MW
    - Freq: {d.get('frequency_hz')}Hz | Reserve: {d.get('reserve_margin_mw')}MW
    - Battery: {int(d.get('battery_soc', 0) * 100)}% | Risk: {d.get('risk_level', 'none')}
    Market: Price={d.get('market_price', 1.0)}

    Goal: Maintain 50Hz, avoid blackout."""

def build_multi_agent_prompt(obs: Dict[str, Any], ask_agent: str = "dispatch") -> str:
    """
    Optional dialogue-style prompt for multi-agent reasoning.
    Can be injected into system prompt for enhanced context awareness.
    
    ask_agent: "dispatch", "planning", or "market" — which agent to prompt
    """
    demand = obs.get("demand_mw", 0.0)
    coal = obs.get("coal_mw", 0.0)
    solar = obs.get("solar_mw", 0.0)
    wind = obs.get("wind_mw", 0.0)
    hydro = obs.get("hydro_mw", 0.0)
    nuclear = obs.get("nuclear_mw", 0.0)
    
    total_gen = coal + solar + wind + hydro + nuclear
    gap = demand - total_gen
    
    frequency = obs.get("frequency_hz", 50.0)
    spinning_reserve = obs.get("spinning_reserve_mw", 0.0)
    reserve_required = obs.get("spinning_reserve_required_mw", 0.0)
    
    spot_price = obs.get("spot_price", 1.0)
    duck_stress = obs.get("duck_curve_stress_mw_per_step", 0.0)
    
    lines = [
        "=== Multi-Agent Grid Coordination ===",
        "",
        "PLANNER (long-term perspective):",
        f"  • Capital available: {obs.get('capital_budget', 0):.0f} units",
        f"  • Cumulative emissions: {obs.get('cumulative_emissions_tons', 0):.0f} tons",
        f"  • Construction: {len(obs.get('plants_building', []))} projects in flight",
        "",
        "MARKET (economics):",
        f"  • Spot price: £{spot_price:.2f}/MWh",
        f"  • Demand response budget: 150 MW max",
        f"  • Grid trading: available for import/export",
        "",
        "DISPATCH (real-time control):",
        f"  • Frequency: {frequency:.2f} Hz (need ±0.1 Hz accuracy)",
        f"  • Spinning reserve: {spinning_reserve:.0f} / {reserve_required:.0f} MW",
        f"  • Ramp stress: {duck_stress:+.1f} MW/step",
        "",
    ]
    
    if ask_agent == "dispatch":
        lines.append("DISPATCH: Based on planner's strategy and market conditions,")
        lines.append("what generation controls stabilize frequency right now?")
    elif ask_agent == "planning":
        lines.append("PLANNER: Given the 72-step horizon and current events,")
        lines.append("what capital investments secure long-term reliability?")
    elif ask_agent == "market":
        lines.append("MARKET: Considering price trends and reserve margins,")
        lines.append("should we increase demand response or grid trading?")
    
    return "\n".join(lines)
