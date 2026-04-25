#!/usr/bin/env python3
"""
Baseline inference script for the Energy Grid Management Environment.

Architecture:
    Easy / Medium — executor (stateless, no conversation history).
    Hard          — one‑shot strategic planner at episode start, then
                    executor with plan injected into every system prompt.

Environment variables (hackathon spec):
    API_BASE_URL   — LLM API endpoint (e.g. https://api.groq.com/openai/v1)
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

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional .env loading (helps locally)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv optional; env vars can be set directly

from openai import OpenAI

# ---------------------------------------------------------------------------
# Local imports (work both when run from repo root or as installed pkg)
# ---------------------------------------------------------------------------
# Local imports
try:
    from .energy_grid_environment import EnergyGridEnvironment
    from .tasks import get_task, TASK_ORDER
    from .llm_adapter import observation_to_text, extract_action_from_llm_output
    from ..models import EnergyGridAction, EnergyGridObservation
except (ImportError, ValueError):
    from server.energy_grid_environment import EnergyGridEnvironment
    from server.tasks import get_task, TASK_ORDER
    from server.llm_adapter import observation_to_text, extract_action_from_llm_output
    from models import EnergyGridAction, EnergyGridObservation

# ---------------------------------------------------------------------------
# Rate limiter — keeps requests within Groq TPM limits
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
MIN_CALL_INTERVAL: float = 0.5  # Faster evaluation

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
# Client configuration (hackathon‑spec only)
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenAI, str]:
    """
    Build an OpenAI‑compatible client.

    Required env vars:
        API_BASE_URL   - endpoint (e.g. https://api.groq.com/openai/v1)
        MODEL_NAME      - model identifier
        OPENAI_API_KEY / HF_TOKEN - auth token (checked in that priority order)
    """
    api_base_url = os.getenv("API_BASE_URL")
    model_name   = os.getenv("MODEL_NAME")
    api_key      = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN")
    )

    if not (api_base_url and model_name and api_key):
        raise EnvironmentError(
            f"Missing required API configuration. URL={bool(api_base_url)}, "
            f"Model={bool(model_name)}, Key={bool(api_key)}"
        )

    print(f"[DEBUG] Using API Key starting with: {api_key[:5]}...")
    
    try:
        client = OpenAI(api_key=api_key, base_url=api_base_url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize OpenAI client: {type(e).__name__}: {e}. "
            f"Check your API_BASE_URL and credentials."
        ) from e
    
    return client, model_name

# ---------------------------------------------------------------------------
# Model Routing & Token budget
# ---------------------------------------------------------------------------

PLANNING_MODEL = os.getenv("PLANNING_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL     = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")

PLANNING_MAX_TOKENS = 600  # Detailed reasoning for infrastructure
FAST_MAX_TOKENS     = 150  # Quick, directive response for real-time control

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-Agent Role Prompts
# ---------------------------------------------------------------------------

AGENT_PROMPTS = {
    "planning": "Planner. Goal: Long-term reliability. Output JSON with 'plant_action' (none, build_solar, build_wind, build_hydro, build_nuclear, close_coal).",
    "dispatch": "Dispatch. Goal: 50Hz. Output JSON with 'coal_delta' (-100 to 100), 'hydro_delta' (-80 to 80), 'nuclear_delta' (-10 to 10), 'battery_mode' (charge, discharge, idle), 'emergency_coal_boost' (bool).",
    "market": "Market. Goal: Efficiency. Output JSON with 'demand_response_mw' (0 to 150), 'grid_export_mw' (0 to 100), 'grid_import_mw' (0 to 100), 'coal_price_bid' (0.5 to 3.0)."
}

MAX_EVAL_STEPS = 20

def _build_system_prompt(
    task_id: str = "easy",
    plan: str = "",
    step: int = 0,
    obs_dict: dict = None,
    agent_type: str = "unified"
) -> str:
    """
    Build system prompt for LLM.
    Includes optional multi-agent context and planner injection.
    """

    # Use role-specific prompt if provided, else use the unified base
    role_prompt = AGENT_PROMPTS.get(agent_type, """You are an expert electricity grid operator.
Your priorities:
1. Prevent blackout
2. Maintain frequency near 50 Hz
3. Ensure sufficient reserve
4. Optimise cost only when stable
""")

    base = f"""{role_prompt}

Think step-by-step before acting.

Respond EXACTLY in this format:

Thought:
<Your reasoning here>

Action:
{{
  // Only the fields relevant to your role
}}
"""

    # ---- Multi-agent context (every 3 steps only) ----
    if task_id == "hard" and obs_dict is not None and step % 3 == 0:
        try:
            from .llm_adapter import build_multi_agent_prompt
            base += "\n\n" + build_multi_agent_prompt(obs_dict)
        except Exception:
            pass  # fail silently to avoid breaking execution

    # ---- Strategic plan injection (early phase only) ----
    if task_id == "hard" and plan and step < 40:
        base += f"\n\nSTRATEGIC PLAN (follow this unless state forces deviation):\n{plan}"

    return base

def _build_planner_prompt(obs: "EnergyGridObservation") -> str:
    """
    One‑shot strategic planning prompt for the Hard task.
    Runs once at step 0. Output is injected into every executor call.
    """
    battery_pct = int(100 * obs.battery_mwh / max(1, obs.battery_capacity_mwh))

    return f"""You are a strategic planner for a 72‑step electricity grid simulation (Hard task, Winter).

KNOWN CONSTANTS (these are guaranteed facts, not estimates):
- Winter peak demand ~1100 MW. Coal max = 600 MW → structural deficit ~500 MW → new capacity is REQUIRED.
- Coal outage GUARANTEED at steps 23-25 (day 2). Coal max drops to 300 MW for 3 steps.
- Nuclear: costs 1000, build time 15 steps, output 500 MW baseload (min 300 MW once online).
- Wind:    costs 400, build time 6 steps, output 250 MW (available day & night).
- Solar:   costs 500, build time 8 steps, output 300 MW (zero at night).
- Hydro:   costs 600, build time 10 steps, output 200 MW dispatchable (reservoir‑limited).
- Total capital budget: 2000 units.

INITIAL STATE:
- Coal output: {obs.coal_mw:.0f} MW (online)
- Battery: {battery_pct}% ({obs.battery_mwh:.0f}/{obs.battery_capacity_mwh:.0f} MWh)
- Capital available: {obs.capital_budget:.0f} units

IMPORTANT:
- Solar and wind ARE available from the start at no cost.
- Hydro and nuclear must be built. Build decisions are irreversible and delayed.

TIMING CONSTRAINTS:
- You MUST commit key build actions within the first 3 steps.
- Nuclear started at step 0 → online at step 15 (safe before outage).
- Nuclear started at step 5 → online at step 20 (tight but acceptable).
- Nuclear started at step 10 → online at step 25 (too late — overlaps outage).
- Wind started at step 0 → online at step 6 (provides early buffer before peak/outage).

Example strong strategy (you may deviate if justified):
- Step 0: build_nuclear (500 MW online at step 15 — critical baseload, online before outage)
- Step 1: build_hydro (200 MW online at step 11 — dispatchable, helps offset coal outage)
- Step 2+: preserve remaining 400 units as reserve capital or invest in wind if needed

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
- Outage Plan: <how to survive steps 23-25>
- Emissions Strategy: <how and when to reduce coal dependency>

Be concise, decisive, and forward‑looking.
"""

def _parse_action(response_text: str) -> EnergyGridAction:
    """
    Extract action from LLM response using the adapter.
    
    Now expects "Thought:" followed by "Action:" format, but gracefully handles
    raw JSON responses for backward compatibility.
    
    Returns a fully‑validated EnergyGridAction; on failure returns a
    safe default (all zeros / idle).
    """
    if not response_text or not response_text.strip():
        return EnergyGridAction() # SAFE_DEFAULT_ACTION equivalent
    
    # Use adapter to extract action dict from LLM output
    action_dict = extract_action_from_llm_output(response_text)
    
    # Convert dict to EnergyGridAction
    return _dict_to_action(action_dict)

def _dict_to_action(data: Dict[str, Any]) -> EnergyGridAction:
    """
    Convert action dict to EnergyGridAction.
    Data dict should already be clipped by the adapter, but we validate once more.
    """
    def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return default

    # Bool handling
    emergency_boost = data.get("emergency_coal_boost", False)
    if isinstance(emergency_boost, str):
        emergency_boost = emergency_boost.lower() in ("true", "1", "yes")

    # Battery mode
    battery_mode = str(data.get("battery_mode", "idle")).lower().strip()
    if battery_mode not in ("charge", "discharge", "idle"):
        battery_mode = "idle"

    # Plant action
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
    Prevents the most common catastrophic step‑0 and emergency mistakes.
    """

    # If coal is restarting, never command a negative delta
    if not obs.coal_online:
        action.coal_delta = max(action.coal_delta, 0.0)

    # Critical blackout risk → force battery discharge
    if obs.blackout_risk == "critical" and action.battery_mode == "idle":
        action.battery_mode = "discharge"

    # Block plant actions on easy and medium tasks entirely
    if obs.task_id in ("easy", "medium"):
        action.plant_action = "none"

    # Allow coal closure in Hard task (now has capital recovery ~50% salvage)
    # Easy/Medium still block it since no salvage value system there
    if action.plant_action == "close_coal" and obs.task_id != "hard":
        action.plant_action = "none"

    # Block emergency boost spam — only if coal is damaged (not during outage)
    # Check: coal is not starting up AND max_mw is below recovery threshold (550 = 600 - boost_damage)
    if action.emergency_coal_boost and obs.coal_startup_remaining == 0 and obs.coal_max_mw < 550.0:
        action.emergency_coal_boost = False

    return action

# ---------------------------------------------------------------------------
# Single‑task runner
# ---------------------------------------------------------------------------

def _is_major_event(obs: EnergyGridObservation, prev_obs: Optional[EnergyGridObservation]) -> bool:
    """Detect significant grid changes requiring Planning Agent attention."""
    if prev_obs is None: return True
    # Coal or Nuclear Outage
    if prev_obs.coal_online and not obs.coal_online: return True
    if prev_obs.nuclear_online and not obs.nuclear_online: return True
    # Demand Spike (> 15% increase)
    if obs.demand_mw > prev_obs.demand_mw * 1.15: return True
    return False

def run_task(
    env: EnergyGridEnvironment,
    client: OpenAI,
    model: str,
    task_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete task episode with the LLM agent.
    Hard task: planner at step 0, then executor with 4‑turn rolling history.
    Easy / Medium: single‑prompt executor with 4‑turn history.
    """
    task = get_task(task_id)
    total_steps = task["total_steps"]

    # Emit structured log: START
    print(f"[START] task={task_id} env=energy-grid-openenv model={model}", flush=True)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Task: {task['name']} ({task_id})")
        print(f"Steps: {total_steps} | Season: {task['season']}")
        print("=" * 60)

    # Reset environment
    obs = env.reset(task_id)

    # Wall-clock timeout (18 min hard cap, leaves 2 min margin before 20-min budget)
    episode_start = time.time()
    EPISODE_TIMEOUT = 18 * 60

    # ------------------------------------------------------------------
    # Hard task - one‑shot planner
    # ------------------------------------------------------------------
    plan = ""
    if task_id == "hard":
        if verbose:
            print("  [PLANNER] Generating strategic plan...")
        planner_response = _call_llm_with_retry(
            client=client,
            model=PLANNING_MODEL,
            system="You are a strategic planner. Output a concise operational plan only.",
            messages=[{"role": "user", "content": _build_planner_prompt(obs)}],
            max_retries=2,
            max_tokens=PLANNING_MAX_TOKENS,
            agent_type="planning",
            verbose=verbose,
        )
        plan = planner_response.strip()
        if verbose:
            print(f"  [PLANNER] Plan generated ({len(plan)} chars).")
            print(f"  {plan}")

    total_reward = 0.0
    step_count = 0
    rewards_list: List[float] = []  # Track all rewards for structured logging

    last_planning_action = EnergyGridAction(plant_action="none", thought="Maintaining plan.")
    prev_obs = None
    history = []

    for step in range(min(total_steps, MAX_EVAL_STEPS)):
        # Wall-clock timeout check before each LLM call
        if time.time() - episode_start > EPISODE_TIMEOUT:
            print(f"[WARN] Episode timeout reached at step {step}, stopping early", flush=True)
            break

        # --- ROUND 1: PROPOSALS ---
        # 1. Gated Planning Agent
        if _is_major_event(obs, prev_obs):
            if verbose: print(f"  [EVENT] Triggering Planning Agent at step {step}")
            sys_p = _build_system_prompt(task_id, plan, step, agent_type="planning")
            resp_p = _call_llm_with_retry(client, PLANNING_MODEL, sys_p, [{"role": "user", "content": observation_to_text(obs.__dict__)}], agent_type="planning", verbose=verbose)
            last_planning_action = _parse_action(resp_p)
        
        # 2. Dispatch and Market Proposals (Always called)
        sys_d = _build_system_prompt(task_id, plan, step, agent_type="dispatch")
        resp_d = _call_llm_with_retry(client, model, sys_d, [{"role": "user", "content": observation_to_text(obs.__dict__)}], agent_type="dispatch", verbose=verbose)
        prop_d = _parse_action(resp_d)

        sys_m = _build_system_prompt(task_id, plan, step, agent_type="market")
        resp_m = _call_llm_with_retry(client, model, sys_m, [{"role": "user", "content": observation_to_text(obs.__dict__)}], agent_type="market", verbose=verbose)
        prop_m = _parse_action(resp_m)

        if task_id == "easy":
            # Task 1: Proposal becomes final immediately for easy
            last_planning_action.proposal_type = "revision"
            prop_d.proposal_type = "revision"
            prop_m.proposal_type = "revision"
            
            env.step_planning(last_planning_action)
            env.step_dispatch(prop_d)
            obs = env.step_market(prop_m) # Advances simulator
        else:
            # Round 1 proposals submitted
            env.step_planning(last_planning_action)
            env.step_dispatch(prop_d)
            obs_mid = env.step_market(prop_m)

            # --- ROUND 2: REVISIONS (Dispatch & Market Only) ---
            sys_rd = _build_system_prompt(task_id, plan, step, agent_type="dispatch")
            resp_rd = _call_llm_with_retry(client, model, sys_rd, [{"role": "user", "content": observation_to_text(obs_mid.__dict__)}], agent_type="dispatch", verbose=verbose)
            rev_d = _parse_action(resp_rd)
            rev_d.proposal_type = "revision"

            sys_rm = _build_system_prompt(task_id, plan, step, agent_type="market")
            resp_rm = _call_llm_with_retry(client, model, sys_rm, [{"role": "user", "content": observation_to_text(obs_mid.__dict__)}], agent_type="market", verbose=verbose)
            rev_m = _parse_action(resp_rm)
            rev_m.proposal_type = "revision"

            # Round 2 submitted (advances simulator)
            last_planning_action.proposal_type = "revision"
            env.step_planning(last_planning_action)
            env.step_dispatch(rev_d)
            obs = env.step_market(rev_m)

        # Collect history
        final_d = rev_d if task_id != "easy" else prop_d
        final_m = rev_m if task_id != "easy" else prop_m

        history.append({
            "step": step_count + 1,
            "demand": float(obs.demand_mw),
            "supply": float(obs.coal_mw + obs.solar_mw + obs.wind_mw + obs.hydro_mw + obs.nuclear_mw),
            "frequency": float(obs.frequency_hz),
            "blackoutRisk": obs.blackout_risk,
            "planning": {
                "thought": last_planning_action.thought or "Continuing baseline strategy.",
                "action": last_planning_action.plant_action
            },
            "dispatch": {
                "thought": final_d.thought or "Optimizing real-time dispatch.",
                "controls": {
                    "coal_delta": float(final_d.coal_delta),
                    "hydro_delta": float(final_d.hydro_delta),
                    "battery_mode": final_d.battery_mode
                }
            },
            "market": {
                "thought": final_m.thought or "Managing economic efficiency.",
                "controls": {
                    "demand_response": float(final_m.demand_response_mw),
                    "import_export": f"{float(final_m.grid_import_mw)} MW import"
                }
            },
            "finalAction": {
                "coal_delta": float(final_d.coal_delta),
                "hydro_delta": float(final_d.hydro_delta),
                "battery_mode": final_d.battery_mode,
                "demand_response": float(final_m.demand_response_mw),
            },
            "reward": float(reward),
            "status": "stable" if obs.frequency_hz > 49.8 and obs.frequency_hz < 50.2 else "warning" if obs.frequency_hz > 49.0 else "failure"
        })

        prev_obs = obs
        step_count += 1
        reward = obs.reward or 0.0
        rewards_list.append(reward)

        # Detailed state logging
        if verbose:
            print(
                f"  Step {step_count:02d} | "
                f"Coal: {obs.coal_mw:.0f}/{obs.coal_max_mw:.0f} MW | "
                f"Demand: {obs.demand_mw:.0f} MW | "
                f"Unmet: {obs.unmet_demand_mw:.0f} MW | "
                f"Freq: {obs.frequency_hz:.2f} Hz | "
                f"Reward: {reward:.2f}"
            )

        # Emit structured log for evaluation parsing
        log_msg = f"[STEP] step={step_count} reward={reward:.2f} done={str(obs.done).lower()} unmet={obs.unmet_demand_mw:.0f}MW freq={obs.frequency_hz:.3f}Hz"
        print(log_msg, flush=True)

        # Accumulate total reward (always, not just in verbose mode)
        total_reward += reward

        if verbose:
            print(
                f"           | Demand={obs.demand_mw:.0f}MW Unmet={obs.unmet_demand_mw:.0f}MW "
                f"Coal={obs.coal_mw:.0f}/{obs.coal_max_mw:.0f}MW "
                f"Solar={obs.solar_mw:.0f}MW Wind={obs.wind_mw:.0f}MW "
                f"Hydro={obs.hydro_mw:.0f}MW Nuc={obs.nuclear_mw:.0f}MW "
                f"Batt={int(100*obs.battery_mwh/max(1,obs.battery_capacity_mwh))}% "
                f"Freq={obs.frequency_hz:.3f}Hz | Step Reward={reward:.2f} | Total Reward={total_reward:.2f} | Risk={obs.blackout_risk}"
            )

        if obs.done:
            if verbose:
                if obs.episode_ended_early:
                    print(f"  ⚡ BLACKOUT at step {step} — episode ended early")
                else:
                    print(f"  ✓ Episode completed at step {step}")
            break

    # --------------------------------------------------------------
    # Grading
    # --------------------------------------------------------------
    grade = env.get_last_grade()
    if grade is None:
        grade = env.grade_current_episode() or {}

    # Emit structured log: END
    score = grade.get("total_score", 0.0)
    success = score >= 0.5  # threshold for "success"
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)
    if rewards_list:
        print(f"[METRIC] avg_reward={sum(rewards_list)/len(rewards_list):.3f}", flush=True)

    if verbose:
        print("\n  📊 GRADE:", grade.get("total_score", 0.0))
        print("  Components:", grade.get("component_scores", {}))
        print("  Total reward:", total_reward)

    return {
        "task_id": task_id,
        "task_name": task["name"],
        "score": score,
        "component_scores": grade.get("component_scores", {}),
        "weighted_components": grade.get("weighted_components", {}),
        "blackout_occurred": grade.get("blackout_occurred", False),
        "steps_completed": step_count,
        "total_steps": total_steps,
        "total_reward": round(total_reward, 4),
        "metadata": grade.get("metadata", {}),
        "history": history
    }

# ---------------------------------------------------------------------------
# LLM call with retry (now uses tiered model routing)
# ---------------------------------------------------------------------------

def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    system: str,
    messages: list,
    max_retries: int = 3,
    agent_type: str = "dispatch",
    verbose: bool = True
) -> str:
    """
    Call LLM with tiered model routing and retries.
    """
    # Override model based on agent_type
    target_model = PLANNING_MODEL if agent_type == "planning" else FAST_MODEL
    max_tokens   = PLANNING_MAX_TOKENS if agent_type == "planning" else FAST_MAX_TOKENS
    
    for attempt in range(max_retries):
        try:
            _rate_limited_sleep(verbose=verbose)
            response = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            # Extract the assistant's content
            content = response.choices[0].message.content or ""
            
            # If content is empty but reasoning exists, use reasoning as the response
            if not content.strip() and hasattr(response.choices[0].message, 'reasoning'):
                reasoning = response.choices[0].message.reasoning or ""
                if reasoning.strip():
                    content = reasoning
                    if verbose:
                        print(f"  [DEBUG] Using reasoning field ({len(reasoning)} chars) - finish_reason='{response.choices[0].finish_reason}'")
            
            # ✅ PRINT TOKENS FIRST (before any validation checks)
            # This ensures we see token usage even if content is empty/invalid
            if verbose and hasattr(response, "usage"):
                usage = response.usage
                print(
                    f"  [DEBUG] Tokens - prompt:{usage.prompt_tokens} "
                    f"completion:{usage.completion_tokens} total:{usage.total_tokens}"
                )
            
            # DEBUG: If content is still empty, log raw response
            if not content.strip() and verbose:
                print(f"  [DEBUG] Raw response choices[0]: {response.choices[0]}")
                print(f"  [DEBUG] Content value: {repr(content)}")
            
            # ✅ NOW validate content (after logging tokens)
            if not content.strip():
                raise ValueError("Empty response from model — possible TPM burst")
                    
            return content

        except Exception as e:
            err = str(e).lower()
            
            # ✅ ALSO PRINT TOKENS IN ERROR PATH if we got a response
            # This shows token usage even when the response was invalid
            if verbose and 'response' in locals() and hasattr(response, "usage"):
                usage = response.usage
                print(
                    f"  [DEBUG] Error path - Tokens - prompt:{usage.prompt_tokens} "
                    f"completion:{usage.completion_tokens} total:{usage.total_tokens}"
                )
            
            is_rate_limit = "rate" in err or "429" in err or "limit" in err

            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** attempt * 5   # 5s, 10s, 20s
                if verbose:
                    print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)
                continue

            if attempt < max_retries - 1:
                wait = 2 ** attempt
                if verbose:
                    print(f"  [ERROR] {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue

            # Final failure
            print(f"  [ERROR] LLM call failed after {max_retries} attempts: {e}")
            return ""   # parser will fall back to safe default action

    return ""

# ---------------------------------------------------------------------------
# Main runner (all tasks)
# ---------------------------------------------------------------------------

def run_baseline_agent(
    task_ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the baseline LLM agent on the specified tasks.
    Returns a dict mapping task_id → result dict (score, metadata, …).
    """
    if task_ids is None:
        task_ids = list(TASK_ORDER)

    if verbose:
        print("\n" + "=" * 60)
        print("  ENERGY GRID OPENENV — BASELINE AGENT")
        print("=" * 60)

    try:
        client, model = _build_client()
    except (EnvironmentError, RuntimeError) as e:
        print(f"  [ERROR] Client initialization failed: {e}")
        raise  # Re-raise for caller to handle
    
    if verbose:
        print(f"  Model: {model}")
        print(f"  Tasks: {task_ids}")

    env = EnergyGridEnvironment(normalize=False)

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
            verbose=verbose,
        )
        results[task_id] = result
        summary_scores[task_id] = result["score"]

    # --------------------------------------------------------------
    # Summary output
    # --------------------------------------------------------------
    if verbose and len(summary_scores) > 1:
        print("\n" + "=" * 60)
        print("  BASELINE SUMMARY")
        print("=" * 60)
        for tid, score in summary_scores.items():
            bar = "█" * int(score * 20)
            print(f"  {tid:8s}: {score:.4f}  {bar}")
        avg = sum(summary_scores.values()) / len(summary_scores)
        print(f"  {'average':8s}: {avg:.4f}")
        print("=" * 60)

    # Save results to outputs/ (wrap in try/except for HF Spaces read-only filesystem)
    import datetime
    try:
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
    except (OSError, PermissionError) as e:
        # HF Spaces or other read-only filesystems — silently skip file write
        if verbose:
            print(f"  (Could not write results file: {e})")

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

def signal_handler(sig, frame):
    """Gracefully handle Ctrl+C — save partial results if possible."""
    print("\n\n[INTERRUPTED] Ctrl+C received. Exiting gracefully...")
    import sys
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
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
        "--quiet",
        action="store_true",
        help="Suppress step‑by‑step output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a JSON file",
    )

    args = parser.parse_args()

    results = run_baseline_agent(
        task_ids=args.tasks,
        verbose=not args.quiet,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")
