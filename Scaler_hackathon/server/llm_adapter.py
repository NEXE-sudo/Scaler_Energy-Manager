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
    voltage = obs.get("voltage_stability_index", 100)

    # ---- Semantic reasoning ----
    gap_msg = f"{gap:+.0f} MW"
    freq_msg = f"{freq:.2f} Hz"
    risk_msg = blackout_risk.upper()
    future_msg = f"In {shortfall_steps} steps" if shortfall_steps < 15 else "Clear"

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



def extract_action_from_llm_output(text: str) -> Dict[str, Any]:
    """
    Extract JSON action from LLM output containing both reasoning and action.
    
    Expected format:
        Thought:
        <reasoning>
        
        Action:
        { ... JSON ... }
    
    Returns a dict with clipped values and safe defaults for invalid fields.
    On parse failure, returns SAFE_DEFAULT_ACTION with a warning.
    """

    if not text or not text.strip():
        print("[WARN] extract_action_from_llm_output: empty response")
        return SAFE_DEFAULT_ACTION.copy()
    
    # Find JSON block after "Action:" if present
    action_match = re.search(r'(?:^|\n)\s*Action\s*:\s*({.*?})\s*(?:\n|$)', text, re.DOTALL | re.IGNORECASE)
    
    if action_match:
        json_block = action_match.group(1)
    else:
        # Fallback: try to find any JSON object in the response
        def extract_json(s: str) -> Optional[str]:
            start = s.find('{')
            if start == -1:
                return None
            depth = 0
            for i, ch in enumerate(s[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
            return None
        
        json_block = extract_json(text)
    
    if not json_block:
        print("[WARN] extract_action_from_llm_output: no JSON found")
        return SAFE_DEFAULT_ACTION.copy()
    
    # Normalize JSON
    json_block = json_block.replace("'", '"')  # Single quotes to double
    json_block = re.sub(r',\s*([}\]])', r'\1', json_block)  # Trailing commas
    json_block = re.sub(r'\btrue\b', 'true', json_block, flags=re.IGNORECASE)
    json_block = re.sub(r'\bfalse\b', 'false', json_block, flags=re.IGNORECASE)
    json_block = re.sub(r'\bTrue\b', 'true', json_block)
    json_block = re.sub(r'\bFalse\b', 'false', json_block)
    json_block = re.sub(r'\bNone\b', 'null', json_block)
    
    # Task: Fix missing commas between fields (e.g. "a": 1 "b": 2)
    json_block = re.sub(r'("[\w]+"\s*:\s*[\d\.]+\s*)\n\s*("[\w]+"\s*:\s*)', r'\1,\n\2', json_block)
    # Fix missing commas between string fields
    json_block = re.sub(r'("[\w]+"\s*:\s*"[\w]+"\s*)\n\s*("[\w]+"\s*:\s*)', r'\1,\n\2', json_block)
    
    try:
        data = json.loads(json_block)
        return _dict_to_action(data)
    except json.JSONDecodeError as e:
        print(f"[WARN] extract_action_from_llm_output: JSON decode error: {e}")
        return SAFE_DEFAULT_ACTION.copy()
    except Exception as e:
        print(f"[WARN] extract_action_from_llm_output: unexpected error: {e}")
        return SAFE_DEFAULT_ACTION.copy()


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
    }


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
