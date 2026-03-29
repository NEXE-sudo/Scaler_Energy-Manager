# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for the Energy Grid Management Environment.

Architecture:
    Easy / Medium — executor with 4-turn rolling conversation history.
    Hard          — one-shot strategic planner at episode start, then
                    executor with plan injected into every system prompt.

Environment variables (hackathon spec):
    API_BASE_URL   — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — Model identifier  (e.g. openai/gpt-oss-20b)
    HF_TOKEN       — Auth key (also accepts OPENAI_API_KEY or API_KEY)

Usage:
    # From project root
    python -m server.baseline

    # Via uv
    uv run python server/baseline.py

    # Triggered by /baseline endpoint
    from server.baseline import run_baseline_agent
    results = run_baseline_agent(task_ids=["easy", "medium", "hard"])
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file if present (cross-platform: works on Windows, Bazzite, Docker)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv optional; env vars can be set directly

from openai import OpenAI

try:
    from .energy_grid_environment import EnergyGridEnvironment
    from .grader import grade_result_to_dict
    from .tasks import get_task, TASK_ORDER
except ImportError:
    from server.energy_grid_environment import EnergyGridEnvironment
    from server.grader import grade_result_to_dict
    from server.tasks import get_task, TASK_ORDER

try:
    from ..models import EnergyGridAction, EnergyGridObservation
except ImportError:
    from models import EnergyGridAction, EnergyGridObservation

# ---------------------------------------------------------------------------
# Rate limiter — keeps requests within Groq TPM limits
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
MIN_CALL_INTERVAL: float = 5.5  # seconds — safe for 8K TPM at ~700 tokens/call

def _rate_limited_sleep(verbose: bool = False) -> None:
    """Sleep only the remaining gap since the last API call."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    remaining = MIN_CALL_INTERVAL - elapsed
    if remaining > 0:
        if verbose:
            print(f"  [RATE] Sleeping {remaining:.1f}s to respect TPM limit...")
        time.sleep(remaining)
    _last_call_time = time.time()

# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenAI, str]:
    """
    Build OpenAI-compatible client.

    Priority (Hackathon-compliant):
        1. API_BASE_URL + MODEL_NAME (required by hackathon spec)
        2. HF_TOKEN or API_KEY for authentication
        3. Fall back to legacy env vars (GROQ_API_KEY, etc.) for backward compatibility
    """
    # Hackathon-required variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )
    api_key = hf_token
    
    # Legacy variables (backward compatibility)
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    base_url_legacy = os.getenv("BASELINE_BASE_URL")
    model_legacy = os.getenv("BASELINE_MODEL")

    # Use hackathon variables if available
    if api_base_url and model_name:
        auth_key = hf_token or api_key or groq_key or openai_key or "dummy"
        client = OpenAI(api_key=auth_key, base_url=api_base_url)
        return client, model_name
    
    # Fall back to legacy configuration
    if base_url_legacy or groq_key or openai_key:
        if base_url_legacy:
            # Fully custom endpoint (legacy)
            auth_key = hf_token or groq_key or openai_key or "dummy"
            client = OpenAI(api_key=auth_key, base_url=base_url_legacy)
            model = model_legacy or "llama-3.3-70b-versatile"
            return client, model

        if groq_key:
            client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
            model = model_legacy or "llama-3.3-70b-versatile"
            return client, model

        if openai_key:
            client = OpenAI(api_key=openai_key)
            model = model_legacy or "gpt-4o-mini"
            return client, model

    raise EnvironmentError(
        "Missing required API configuration. Set either:\n"
        "  1. API_BASE_URL + MODEL_NAME (hackathon spec), or\n"
        "  2. GROQ_API_KEY or OPENAI_API_KEY (legacy)\n"
        "\nSee .env file for examples."
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(task_id: str = "easy", plan: str = "") -> str:
    """
    Build the system prompt for the executor.
    For Hard task, injects the planner output.
    """
    base = """You are an expert electricity grid operator.

You MUST always output BOTH:
1. REASON
2. ACTION

Never skip ACTION.

HARD CONSTRAINTS (violating these causes blackouts or wasted steps):
- Coal min stable = 200 MW. Going below shuts it down — 3-step restart, zero output during restart.
- Coal max = 600 MW normally. Ramp limit = ±100 MW per step.
- Nuclear min = 300 MW (cannot go lower once online). Max = 500 MW. Ramp = ±10 MW only.
- Nuclear cannot be adjusted during a SCRAM event (output drops to 0 for 8 steps).
- Battery: pick exactly ONE of charge / discharge / idle per step. Max ±50 MW.
- Hydro depletes reservoir at 1 MWh per MWh generated. Conserve during drought events.
- Blackout at frequency < 47.5 Hz or > 51.5 Hz — episode ends immediately.
- Load shedding starts at < 49.0 Hz (100 MW auto-shed), then < 48.5 Hz (200 MW).

DEMAND CURVE (approximate, winter multiplier is x1.3):
- Night  00:00-05:00: 370-480 MW base
- Ramp   06:00-11:00: 480-880 MW base
- Peak   12:00-17:00: 840-880 MW base
- Eve    18:00-23:00: 560-870 MW base

Gap = demand - total_generation (positive = shortage, negative = overproduction)

STRATEGY:
- Gap > 0 (shortfall) → increase coal, discharge battery, or use emergency boost.
- Gap < 0 (overproduction) → reduce coal or charge battery.
- Emergency boost: +200 MW instant but reduces coal_max by 50 MW for 5 steps. Last resort only.
- Keep battery above 20% as emergency buffer.
- Pre-ramp coal before morning peak (start increasing at step/hour 5–6).
- At night reduce coal toward 300–350 MW to save cost.
- Avoid large oscillations: make smaller adjustments when close to balance.
- If Gap is near zero (|Gap| < 20 MW), keep changes minimal to maintain stability.
- Do not use battery unless needed for balancing or reserve protection.
- Anticipate demand trend: rising demand → pre-increase coal, falling demand → reduce coal early.

PRIORITY UNDER SHORTFALL:
1. Increase coal within limits
2. Use battery discharge
3. Use hydro if available
4. Use emergency coal boost ONLY if blackout is imminent

Never leave a positive Gap unresolved if avoidable — it risks blackout.

STRICT OUTPUT REQUIREMENTS (CRITICAL):
- ACTION must be valid JSON.
- Do NOT truncate JSON.
- Do NOT include trailing commas.
- Ensure all brackets are properly closed.
- Do NOT include text after the JSON.
- If JSON is invalid, your action will be ignored and treated as no-op.

EXAMPLE (correct format):
REASON: Increase coal to meet rising demand.
ACTION: {"coal_delta": 40, "hydro_delta": 0, "nuclear_delta": 0, "battery_mode": "idle", "plant_action": "none", "emergency_coal_boost": false, "demand_response_mw": 0}

FINAL OUTPUT RULES (STRICT):
- Output must be EXACTLY 2 lines.
- Line 1 must start with: REASON:
- Line 2 must start with: ACTION:
- ACTION must be valid JSON.
- Do NOT include any text before or after these two lines.
- Do NOT use code blocks.
- Do NOT truncate JSON.
- Do NOT include trailing commas.
- Do NOT output multiple ACTION blocks.

If this format is violated, your action will be ignored.
"""

    if task_id == "hard" and plan:
        base += f"\n\nSTRATEGIC PLAN (follow this unless state forces deviation):\n{plan}"

    return base

def _build_planner_prompt(obs: "EnergyGridObservation") -> str:
    """
    One-shot strategic planning prompt for the Hard task.
    Runs once at step 0. Output is injected into every executor call.
    """
    battery_pct = int(100 * obs.battery_level_mwh / max(1, obs.battery_capacity_mwh))

    return f"""You are a strategic planner for a 72-step electricity grid simulation (Hard task, Winter).

KNOWN CONSTANTS (these are guaranteed facts, not estimates):
- Winter peak demand ~1100 MW. Coal max = 600 MW → structural deficit ~500 MW → new capacity is REQUIRED.
- Coal outage GUARANTEED at steps 24–27 (day 2). Coal max drops to 300 MW for 3 steps.
- Nuclear: costs 1000, build time 15 steps, output 500 MW baseload (min 300 MW once online).
- Wind:    costs 400,  build time 6 steps,  output 250 MW (available day & night).
- Solar:   costs 500,  build time 8 steps,  output 300 MW (zero at night).
- Hydro:   costs 600,  build time 10 steps, output 200 MW dispatchable (reservoir-limited).
- Total capital budget: 2000 units.

INITIAL STATE:
- Coal output: {obs.coal_output_mw:.0f} MW (online)
- Battery: {battery_pct}% ({obs.battery_level_mwh:.0f}/{obs.battery_capacity_mwh:.0f} MWh)
- Capital available: {obs.capital_budget:.0f} units

IMPORTANT:
- Solar and wind are NOT installed initially — they must be built.
- Build decisions are irreversible and delayed — timing matters.

TIMING CONSTRAINTS:
- You MUST commit key build actions within the first 3 steps.
- Nuclear started at step 0 → online at step 15 (safe before outage).
- Nuclear started at step 5 → online at step 20 (tight but acceptable).
- Nuclear started at step 10 → online at step 25 (too late — overlaps outage).
- Wind started at step 0 → online at step 6 (provides early buffer before peak/outage).

RECOMMENDED STRATEGY (strong baseline, may deviate with justification):
- Step 0: build_wind (250 MW online at step 6 — stabilises early demand)
- Step 1: build_nuclear (500 MW online at step 16 — critical for outage)
- Step 2+: optional hydro or solar depending on remaining budget

GUIDELINES:
- Ensure sufficient capacity BEFORE the coal outage — do not rely only on battery.
- Use battery as a buffer, not a primary supply source.
- Avoid overbuilding — unused capital improves capital efficiency score.
- Prefer stable and predictable generation (nuclear + wind) over purely variable sources.
- Plan for both reliability and cost — not just survival.

OUTPUT YOUR PLAN in this format:

PLAN:
- Build Order: <what to build and at which steps>
- Coal Strategy: <how coal is ramped and used over time>
- Battery Strategy: <when to conserve vs discharge>
- Outage Plan: <how to survive steps 24–27>
- Emissions Strategy: <how and when to reduce coal dependency>

Be concise, decisive, and forward-looking.
"""

def _build_user_prompt(obs: EnergyGridObservation, task_id: str) -> str:
    task = get_task(task_id)
    steps_remaining = task["total_steps"] - obs.step
    events_str = ", ".join(obs.active_events) if obs.active_events else "none"
    construction_str = (
        ", ".join(f"{p['type']}({p['steps_remaining']})" for p in obs.plants_under_construction)
        if obs.plants_under_construction else "none"
    )
    battery_pct = int(100 * obs.battery_level_mwh / max(1, obs.battery_capacity_mwh))
    freq_diff = obs.grid_frequency - 50.0
    total_gen = (obs.coal_output_mw + obs.solar_output_mw + obs.wind_output_mw +
                 obs.hydro_output_mw + obs.nuclear_output_mw)
    gap = obs.demand_mw - total_gen

    warnings = []
    if obs.unmet_demand_mw > 10:
        warnings.append(f"SHORTFALL {obs.unmet_demand_mw:.0f}MW")
    if obs.blackout_risk in ("high", "critical"):
        warnings.append(f"RISK={obs.blackout_risk.upper()}")
    if obs.primary_response_active:
        warnings.append("GOVERNOR_ACTIVE")
    if battery_pct < 15:
        warnings.append("BATTERY_LOW")
    if obs.hydro_available and obs.reservoir_level_mwh < 100:
        warnings.append("RESERVOIR_LOW")
    if (task_id == "hard" 
        and obs.capital_budget > 600 
        and steps_remaining > 15
        and not obs.plants_under_construction):
        warnings.append(f"BUILD_OPPORTUNITY({obs.capital_budget:.0f}cap)")
    warn_str = " | ".join(warnings) if warnings else "OK"

    lines = [
        f"Step {obs.step}/{task['total_steps']} {task_id.upper()} Day{obs.day} {obs.time_of_day:02d}:00 {obs.season}",
        f"Demand={obs.demand_mw:.0f} Gap={gap:+.0f} Unmet={obs.unmet_demand_mw:.0f} [{warn_str}]",
        f"Coal={obs.coal_output_mw:.0f}/{obs.coal_max_mw:.0f}MW online={obs.coal_online} price={obs.coal_price:.2f}x",
        f"Solar={obs.solar_output_mw:.0f}MW wind={obs.wind_output_mw:.0f}MW({obs.wind_speed_ms:.1f}m/s) hydro={obs.hydro_output_mw:.0f}MW nuclear={obs.nuclear_output_mw:.0f}MW",
        f"Battery={battery_pct}%({obs.battery_level_mwh:.0f}/{obs.battery_capacity_mwh:.0f}MWh) reservoir={obs.reservoir_level_mwh:.0f}MWh inflow={obs.natural_inflow_mwh:.1f}",
        f"Freq={obs.grid_frequency:.3f}Hz({freq_diff:+.3f}) RoCoF={obs.rate_of_change_hz_per_step:+.3f} inertia={obs.system_inertia_seconds:.1f}s reserve={obs.spinning_reserve_mw:.0f}/{obs.spinning_reserve_required_mw:.0f}MW",
        f"Events={events_str} Construction={construction_str} Capital={obs.capital_budget:.0f} Cost={obs.cumulative_cost:.2f} CO2={obs.cumulative_emissions_tons:.0f}t",
        f"StepsLeft={steps_remaining}",
        "",
        "REASON: <one sentence>",
        'ACTION: {"coal_delta": <-100..100>, "hydro_delta": <-80..80>, "nuclear_delta": <-10..10>, "battery_mode": "<charge|discharge|idle>", "plant_action": "<none|build_solar|build_wind|build_hydro|build_nuclear|close_coal>", "emergency_coal_boost": <true|false>, "demand_response_mw": <0..150>}',
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_action(response_text: str) -> EnergyGridAction:
    """
    Extract JSON ACTION block using balanced-brace extraction.
    Tolerates markdown fences, single quotes, trailing commas,
    Python-style booleans, and line-breaks inside the JSON.
    """
    # Strip markdown fences
    txt = re.sub(r'^```[a-z]*\n|```$', '', response_text.strip(), flags=re.MULTILINE)

    # Balanced-brace extraction — finds the first complete JSON object
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

    # Try from after ACTION: first, then anywhere in text
    action_pos = txt.upper().find('ACTION')
    search_text = txt[action_pos:] if action_pos != -1 else txt
    raw = extract_json(search_text) or extract_json(txt)

    if raw is None:
        print(f"  [WARN] Parser failed — no JSON found")
        print(f"  [WARN] Raw response: {txt[:300]}")
        return EnergyGridAction()

    # Normalise: single quotes → double, trailing commas, Python booleans
    raw = raw.replace("'", '"')
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r'\bTrue\b', 'true', raw)
    raw = re.sub(r'\bFalse\b', 'false', raw)

    try:
        return _dict_to_action(json.loads(raw))
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] Parser failed — JSON decode error: {e}")
        print(f"  [WARN] Attempted to parse: {raw[:200]}")
        return EnergyGridAction()

def _dict_to_action(data: Dict[str, Any]) -> EnergyGridAction:
    """
    Convert parsed JSON dict to EnergyGridAction.
    Clamps to valid ranges only — no snapping to round numbers.
    """
    def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return default

    emergency_boost = data.get("emergency_coal_boost", False)
    if isinstance(emergency_boost, str):
        emergency_boost = emergency_boost.lower() in ("true", "1", "yes")

    battery_mode = str(data.get("battery_mode", "idle")).lower().strip()
    if battery_mode not in ("charge", "discharge", "idle"):
        battery_mode = "idle"

    plant_action = str(data.get("plant_action", "none")).lower().strip()
    valid_plant_actions = {
        "none", "build_solar", "build_wind",
        "build_hydro", "build_nuclear", "close_coal",
    }
    if plant_action not in valid_plant_actions:
        plant_action = "none"

    return EnergyGridAction(
        coal_delta=_clamp(data.get("coal_delta", 0.0), -100.0, 100.0, 0.0),
        hydro_delta=_clamp(data.get("hydro_delta", 0.0), -80.0, 80.0, 0.0),
        nuclear_delta=_clamp(data.get("nuclear_delta", 0.0), -10.0, 10.0, 0.0),
        battery_mode=battery_mode,
        plant_action=plant_action,
        emergency_coal_boost=bool(emergency_boost),
        demand_response_mw=_clamp(data.get("demand_response_mw", 0.0), 0.0, 150.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Control layer (safety guards)
# ---------------------------------------------------------------------------

def _apply_control_layer(
    action: EnergyGridAction,
    obs: EnergyGridObservation,
) -> EnergyGridAction:
    """
    Safety guards applied after LLM action parsing.
    Prevents the most common catastrophic step-0 and emergency mistakes.
    """
    # Step 0: never charge battery or reduce coal — prevents instant shortfall
    if obs.step == 0:
        action.coal_delta = max(action.coal_delta, 50.0)   # always ramp up
        if obs.unmet_demand_mw > 0:
            action.battery_mode = "discharge"
        elif action.battery_mode == "charge":
            action.battery_mode = "idle"

    # Any step: if coal is restarting, don't try to reduce it further
    if not obs.coal_online:
        action.coal_delta = max(action.coal_delta, 0.0)

    # Any step: critical blackout risk → force battery to discharge if idle
    if obs.blackout_risk == "critical" and action.battery_mode == "idle":
        action.battery_mode = "discharge"

    return action

# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(
    env: EnergyGridEnvironment,
    client: OpenAI,
    model: str,
    task_id: str,
    max_steps_override: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete task episode with the LLM agent.

    Hard task: runs planner at step 0, injects plan into every executor call.
    Easy / Medium: single-prompt executor with 4-turn rolling history.
    """
    task = get_task(task_id)
    total_steps = max_steps_override or task["total_steps"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task['name']} ({task_id})")
        print(f"Steps: {total_steps} | Season: {task['season']}")
        print(f"{'='*60}")

    # Reset environment
    obs = env.reset(task_id)

    # ------------------------------------------------------------------
    # Hard task: one-shot planner before step 0
    # ------------------------------------------------------------------
    plan = ""
    if task_id == "hard":
        if verbose:
            print("  [PLANNER] Generating strategic plan...")
        planner_response = _call_llm_with_retry(
            client=client,
            model=model,
            system="You are a strategic planner. Output a concise operational plan only.",
            messages=[{"role": "user", "content": _build_planner_prompt(obs)}],
            max_retries=3,
            max_tokens = 700,
            verbose=verbose,
        )
        plan = planner_response.strip()
        if verbose:
            print(f"  [PLANNER] Plan generated ({len(plan)} chars):")
            # Print first 300 chars so it's readable without flooding terminal
            print(f"  {plan[:300]}{'...' if len(plan) > 300 else ''}")

    system_prompt = _build_system_prompt(task_id=task_id, plan=plan)
    total_reward = 0.0
    step_count = 0
    reason_log = []
    conversation_history: List[Dict[str, str]] = []   # rolling 4-turn window

    for step in range(total_steps):
        user_prompt = _build_user_prompt(obs, task_id)

        # Append current state to history
        conversation_history.append({"role": "user", "content": user_prompt})

        # LLM call with full rolling history
        response_text = _call_llm_with_retry(
            client=client,
            model=model,
            system=system_prompt,
            messages=conversation_history,
            max_retries=3,
            verbose=verbose,
        )

        # Append assistant response to history before next iteration
        conversation_history.append({"role": "assistant", "content": response_text})
        if len(conversation_history) > 8:
            conversation_history = conversation_history[-8:]

        # Extract reasoning for logging
        reason_match = re.search(r"REASON\s*:\s*(.+?)(?:ACTION|$)", response_text, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()[:120]
            reason_log.append(f"Step {step:02d}: {reason}")
            if verbose:
                print(f"  Step {step:02d} | REASON: {reason}")

        # Parse + sanitise action
        action = _parse_action(response_text)

        # Apply safety guards
        action = _apply_control_layer(action, obs)

        if verbose:
            print(
                f"  Step {step:02d} | coal_delta={action.coal_delta:+.0f} "
                f"battery={action.battery_mode} "
                f"plant={action.plant_action} "
                f"boost={action.emergency_coal_boost}"
            )

        # Execute step
        obs = env.step(action)
        total_reward += obs.step_reward
        step_count += 1

        if verbose:
            print(
                f"           | demand={obs.demand_mw:.0f} MW "
                f"unmet={obs.unmet_demand_mw:.0f} MW "
                f"freq={obs.grid_frequency:.3f} Hz "
                f"reward={obs.step_reward:.3f} "
                f"risk={obs.blackout_risk}"
            )

        if obs.done:
            if verbose:
                if obs.episode_ended_early:
                    print(f"  ⚡ BLACKOUT at step {step} — episode ended early")
                else:
                    print(f"  ✓ Episode completed at step {step}")
            break

    # Grade the episode
    grade = env.get_last_grade()
    if grade is None:
        grade = env.grade_current_episode() or {}

    if verbose:
        print(f"\n  📊 GRADE: {grade.get('total_score', 0.0):.4f}")
        print(f"  Components: {grade.get('component_scores', {})}")
        print(f"  Total reward: {total_reward:.2f}")

    return {
        "task_id": task_id,
        "task_name": task["name"],
        "score": grade.get("total_score", 0.0),
        "component_scores": grade.get("component_scores", {}),
        "weighted_components": grade.get("weighted_components", {}),
        "blackout_occurred": grade.get("blackout_occurred", False),
        "steps_completed": step_count,
        "total_steps": total_steps,
        "total_reward": round(total_reward, 4),
        "metadata": grade.get("metadata", {}),
        "reasoning_samples": reason_log[:5],
    }


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    system: str,
    messages: list,
    max_retries: int = 3,
    verbose: bool = False,
    max_tokens = 700
) -> str:
    """
    Call the LLM API with exponential backoff on rate limit errors.
    """
    for attempt in range(max_retries):
        try:
            _rate_limited_sleep(verbose=verbose)
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    *messages,
                ],
            )
            response_text = response.choices[0].message.content or ""
            if not response_text.strip():
                raise ValueError("Empty response from model — possible TPM burst")
            return response_text

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "limit" in error_str

            if is_rate_limit and attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5   # 5s, 10s, 20s
                if verbose:
                    print(f"  [RATE LIMIT] Waiting {wait_time}s before retry {attempt + 1}...")
                time.sleep(wait_time)
                continue

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                if verbose:
                    print(f"  [ERROR] {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Final attempt failed
            print(f"  [ERROR] LLM call failed after {max_retries} attempts: {e}")
            return ""   # return empty string — parser will use safe default action

    return ""


# ---------------------------------------------------------------------------
# Main runner (all tasks)
# ---------------------------------------------------------------------------

def run_baseline_agent(
    task_ids: Optional[List[str]] = None,
    max_steps_override: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the baseline LLM agent on specified tasks.

    This is the primary public interface, called by:
        - The /baseline endpoint
        - Direct CLI execution

    Args:
        task_ids: List of task IDs to run. Defaults to all three.
        max_steps_override: Cap episode length (useful for quick tests).
        verbose: Print step-by-step output.

    Returns:
        Dict mapping task_id → result dict with score and metadata.
    """
    if task_ids is None:
        task_ids = list(TASK_ORDER)

    if verbose:
        print("\n" + "="*60)
        print("  ENERGY GRID OPENENV — BASELINE AGENT")
        print("="*60)

    # Build LLM client
    client, model = _build_client()
    if verbose:
        print(f"  Model: {model}")
        print(f"  Tasks: {task_ids}")

    # Create environment
    env = EnergyGridEnvironment()

    results: Dict[str, Any] = {}
    summary_scores: Dict[str, float] = {}

    for task_id in task_ids:
        if task_id not in ("easy", "medium", "hard"):
            print(f"  [SKIP] Unknown task: {task_id}")
            continue

        result = run_task(
            env=env,
            client=client,
            model=model,
            task_id=task_id,
            max_steps_override=max_steps_override,
            verbose=verbose,
        )
        results[task_id] = result
        summary_scores[task_id] = result["score"]

        # Small pause between tasks to respect rate limits
        if task_id != task_ids[-1]:
            time.sleep(10)

    # Summary
    if verbose and len(summary_scores) > 1:
        print("\n" + "="*60)
        print("  BASELINE SUMMARY")
        print("="*60)
        for tid, score in summary_scores.items():
            bar = "█" * int(score * 20)
            print(f"  {tid:8s}: {score:.4f}  {bar}")
        avg = sum(summary_scores.values()) / len(summary_scores)
        print(f"  {'average':8s}: {avg:.4f}")
        print("="*60)

    # Save results to outputs/ directory
    import datetime
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = outputs_dir / f"baseline_{timestamp}.json"
    output_file.write_text(json.dumps({
        "results": results,
        "summary_scores": summary_scores,
        "average_score": round(sum(summary_scores.values()) / max(1, len(summary_scores)), 4),
        "model": model,
        "timestamp": timestamp,
    }, indent=2))
    if verbose:
        print(f"\n  Results saved to {output_file}")

    return {
        "results": results,
        "summary_scores": summary_scores,
        "average_score": round(
            sum(summary_scores.values()) / max(1, len(summary_scores)), 4
        ),
        "model": model,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run baseline LLM agent on Energy Grid OpenEnv"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Tasks to run (default: all three)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override episode length (useful for quick tests)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    results = run_baseline_agent(
        task_ids=args.tasks,
        max_steps_override=args.max_steps,
        verbose=not args.quiet,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output_path}")