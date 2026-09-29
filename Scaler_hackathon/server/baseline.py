#!/usr/bin/env python3
"""
Baseline inference script for the Energy Grid Management Environment.

Architecture:
    Easy / Medium  executor (stateless, no conversation history).
    Hard           oneshot strategic planner at episode start, then
                    executor with plan injected into every system prompt.

Environment variables (hackathon spec):
    API_BASE_URL    LLM API endpoint (e.g. https://api.groq.com/openai/v1)
    MODEL_NAME      Model identifier  (e.g. openai/gpt-oss-20b)
    HF_TOKEN        Auth key (also accepts OPENAI_API_KEY or API_KEY)

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
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    from .trace import TraceRecorder, git_sha, base_url_host
    from .llm_adapter import observation_to_text, extract_action_from_llm_output, try_extract_action_from_llm_output
    from ..models import EnergyGridAction, EnergyGridObservation
except (ImportError, ValueError):
    from server.energy_grid_environment import EnergyGridEnvironment
    from server.tasks import get_task, TASK_ORDER
    from server.trace import TraceRecorder, git_sha, base_url_host
    from server.llm_adapter import observation_to_text, extract_action_from_llm_output, try_extract_action_from_llm_output
    from models import EnergyGridAction, EnergyGridObservation

# ---------------------------------------------------------------------------
# Rate limiter  keeps requests within Groq TPM limits
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
# Client configuration (hackathonspec only)
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenAI, str]:
    """
    Build an OpenAIcompatible client.

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

PLANNING_MODEL = os.getenv("PLANNING_MODEL", "openai/gpt-oss-120b")
FAST_MODEL     = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")

PLANNING_MAX_TOKENS = 600  # Detailed reasoning for infrastructure
FAST_MAX_TOKENS     = 150  # Quick, directive response for real-time control

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-Agent Role Prompts
# ---------------------------------------------------------------------------

def get_agent_prompt(agent_type: str) -> str:
    if agent_type == "planning":
        return """You are the Planning Agent.

Your role:

Make long-term infrastructure decisions
Ensure reliability before future demand spikes

CRITICAL RULES:

Return ONLY valid JSON
Do NOT add text outside the format

FORMAT:

Thought:
<short reasoning>

Action:
{
"plant_action": "none | build_solar | build_wind | build_hydro | build_nuclear | close_coal"
}
"""

    elif agent_type == "dispatch":
        return """You are the Dispatch Agent.

Your role:

Maintain grid frequency at 50 Hz
Prevent blackouts
Balance supply and demand

CRITICAL RULES:

Return ONLY valid JSON
Include ALL fields
Do NOT output all zeros unless necessary

FORMAT:

Thought:
<short reasoning>

Action:
{
"coal_delta": number,
"hydro_delta": number,
"nuclear_delta": number,
"battery_mode": "charge | discharge | idle",
"emergency_coal_boost": true | false
}
"""

    elif agent_type == "market":
        return """You are the Market Agent.

Your role:

Optimise cost and efficiency
Use demand response and trading

CRITICAL RULES:

Return ONLY valid JSON
Include ALL fields

FORMAT:

Thought:
<short reasoning>

Action:
{
"demand_response_mw": number,
"grid_export_mw": number,
"grid_import_mw": number,
"coal_price_bid": number | null
}
"""

    else:
        return "Return valid JSON only."


# Cap on steps per episode. Default 20 keeps the historical evaluation budget.
# Set MAX_EVAL_STEPS=0 (or negative) to run full-length episodes. Results always
# report `truncated` and `eval_step_cap` so capped runs are never mistaken for
# full-episode scores.
MAX_EVAL_STEPS = int(os.getenv("MAX_EVAL_STEPS", "20"))

def _build_system_prompt(
    task_id: str = "easy",
    plan: str = "",
    step: int = 0,
    obs_dict: dict = None,
    agent_type: str = "dispatch",
) -> str:
    """
    Build system prompt for LLM.
    Includes optional multi-agent context and planner injection.
    """

    # Use role-specific prompt if provided, else use the unified base
    role_prompt = get_agent_prompt(agent_type)

    base = f"""{role_prompt}

Think step-by-step before acting.
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
    Oneshot strategic planning prompt for the Hard task.
    Runs once at step 0. Output is injected into every executor call.
    """
    battery_pct = int(100 * obs.battery_mwh / max(1, obs.battery_capacity_mwh))

    return f"""You are a strategic planner for a 72step electricity grid simulation (Hard task, {obs.season.capitalize()}).

KNOWN CONSTANTS (these are guaranteed facts, not estimates):
- Winter peak demand ~1100 MW. Coal max = 600 MW  structural deficit ~500 MW  new capacity is REQUIRED.
- Coal outage GUARANTEED at steps 23-25 (day 2). Coal max drops to 300 MW for 3 steps.
- Nuclear: costs 1000, build time 15 steps, output 500 MW baseload (min 300 MW once online).
- Wind:    costs 400, build time 6 steps, output 250 MW (available day & night).
- Solar:   costs 500, build time 8 steps, output 300 MW (zero at night).
- Hydro:   costs 600, build time 10 steps, output 200 MW dispatchable (reservoirlimited).
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
- Nuclear started at step 0  online at step 15 (safe before outage).
- Nuclear started at step 5  online at step 20 (tight but acceptable).
- Nuclear started at step 10  online at step 25 (too late  overlaps outage).
- Wind started at step 0  online at step 6 (provides early buffer before peak/outage).

Example strong strategy (you may deviate if justified):
- Step 0: build_nuclear (500 MW online at step 15  critical baseload, online before outage)
- Step 1: build_hydro (200 MW online at step 11  dispatchable, helps offset coal outage)
- Step 2+: preserve remaining 400 units as reserve capital or invest in wind if needed

GUIDELINES:
- Ensure sufficient capacity BEFORE the coal outage  do not rely only on battery.
- Use battery as a buffer, not a primary supply source.
- Avoid overbuilding  unused capital improves capital efficiency score.
- Prefer stable and predictable generation (nuclear + wind) over purely variable sources.
- Plan for both reliability and cost  not just survival.

OUTPUT YOUR PLAN in this format:

PLAN:
- Build Order: <what to build and at which steps>
- Coal Strategy: <how coal is ramped and used over time>
- Battery Strategy: <when to conserve vs discharge>
- Outage Plan: <how to survive steps 23-25>
- Emissions Strategy: <how and when to reduce coal dependency>

Be concise, decisive, and forwardlooking.
"""

def _parse_action_checked(response_text: Optional[str]) -> Tuple[EnergyGridAction, str]:
    """Parse model output into an action and report whether parsing succeeded.

    Returns (action, status) with status one of:
        "ok"          - a JSON action object was found and parsed
        "empty"       - no text (e.g. reasoning models returning null content)
        "unparseable" - text present but no usable JSON action
    On "empty"/"unparseable" the action is the safe default (all zeros / idle);
    callers must treat that as a failed call, not as a model decision.
    """
    if not response_text or not response_text.strip():
        return EnergyGridAction(), "empty"
    if try_extract_action_from_llm_output(response_text) is None:
        return _dict_to_action(extract_action_from_llm_output(response_text)), "unparseable"
    return _dict_to_action(extract_action_from_llm_output(response_text)), "ok"


def _parse_action(response_text: str) -> EnergyGridAction:
    """Backward-compatible wrapper: action only, failures become the safe default."""
    return _parse_action_checked(response_text)[0]

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
        grid_export_mw=_clamp(data.get("grid_export_mw", 0.0), 0.0, 100.0, 0.0),
        grid_import_mw=_clamp(data.get("grid_import_mw", 0.0), 0.0, 100.0, 0.0),
        coal_price_bid=data.get("coal_price_bid") if data.get("coal_price_bid") is not None else None,
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
    Prevents the most common catastrophic step0 and emergency mistakes.
    """

    # If coal is restarting, never command a negative delta
    if not obs.coal_online:
        action.coal_delta = max(action.coal_delta, 0.0)

    # Critical blackout risk  force battery discharge
    if obs.blackout_risk == "critical" and action.battery_mode == "idle":
        action.battery_mode = "discharge"

    # Block plant actions on easy and medium tasks entirely
    if obs.task_id in ("easy", "medium"):
        action.plant_action = "none"

    # Allow coal closure in Hard task (now has capital recovery ~50% salvage)
    # Easy/Medium still block it since no salvage value system there
    if action.plant_action == "close_coal" and obs.task_id != "hard":
        action.plant_action = "none"

    # Block emergency boost spam  only if coal is damaged (not during outage)
    # Check: coal is not starting up AND max_mw is below recovery threshold (550 = 600 - boost_damage)
    if action.emergency_coal_boost and obs.coal_startup_remaining == 0 and obs.coal_max_mw < 550.0:
        action.emergency_coal_boost = False

    return action

# ---------------------------------------------------------------------------
# Singletask runner
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

@dataclass
class AgentResult:
    """One agent invocation: what the model said, what we parsed, what we applied."""
    agent: str
    action: EnergyGridAction          # action after the control layer (what is submitted)
    ok: bool                          # False on call failure or parse failure
    status: str                       # ok | empty | unparseable | call_failed
    call: CallResult
    proposed: Dict[str, Any] = field(default_factory=dict)         # parsed, before control layer
    control_changes: Dict[str, List[Any]] = field(default_factory=dict)  # field -> [proposed, applied]
    call_id: Optional[str] = None


def _new_stats() -> Dict[str, Any]:
    return {
        "calls": 0, "call_failures": 0, "parse_failures": 0, "control_overrides": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
    }


def _account_call(stats: Dict[str, Any], call: CallResult) -> None:
    stats["calls"] += 1
    stats["latency_s"] = round(stats["latency_s"] + call.latency_s, 4)
    stats["prompt_tokens"] += call.prompt_tokens or 0
    stats["completion_tokens"] += call.completion_tokens or 0
    if not call.ok:
        stats["call_failures"] += 1


_OBS_SUMMARY_FIELDS = (
    "demand_mw", "coal_mw", "solar_mw", "wind_mw", "hydro_mw", "nuclear_mw", "frequency_hz",
    "unmet_demand_mw", "blackout_risk", "spinning_reserve_mw", "spinning_reserve_required_mw",
    "battery_mwh", "battery_capacity_mwh", "spot_price", "coal_online",
)


def _obs_summary(obs: Any) -> Dict[str, Any]:
    return {k: getattr(obs, k, None) for k in _OBS_SUMMARY_FIELDS}


def _record_call(trace: Optional[TraceRecorder], call_id: Optional[str], step: Optional[int],
                 round_name: str, agent: str, model: str, system: str, user_text: str,
                 call: CallResult, status: str, proposed: Dict[str, Any], applied: Dict[str, Any],
                 changes: Dict[str, List[Any]], state: Optional[Dict[str, Any]],
                 inputs_from: Optional[List[str]] = None, plan_call_id: Optional[str] = None) -> None:
    if trace is None:
        return
    trace.record(
        "agent_call", call_id=call_id, step=step, round=round_name, agent=agent, model=model,
        system_prompt=system, user_prompt=user_text, raw_response=call.content,
        finish_reason=call.finish_reason, attempts=call.attempts, latency_s=round(call.latency_s, 4),
        prompt_tokens=call.prompt_tokens, completion_tokens=call.completion_tokens,
        call_error=call.error, parse_status=status, ok=(status == "ok"),
        proposed_action=proposed, applied_action=applied, control_changes=changes,
        state=state, inputs_from=inputs_from or [], plan_call_id=plan_call_id,
    )


def _invoke_agent(
    client,
    model: str,
    agent: str,
    system: str,
    user_text: str,
    obs: EnergyGridObservation,
    stats: Dict[str, Any],
    max_tokens: int = 256,
    verbose: bool = True,
    trace: Optional[TraceRecorder] = None,
    run_id: Optional[str] = None,
    step: Optional[int] = None,
    round_name: str = "proposal",
    inputs_from: Optional[List[str]] = None,
    plan_call_id: Optional[str] = None,
) -> AgentResult:
    """Call one agent, parse its output, apply the control layer, and account for it.

    A failed call or unparseable output yields the safe default action but is
    flagged (ok=False) and counted in `stats`; it is never reported as a success.
    """
    call = _call_llm_detailed(
        client, model, system, [{"role": "user", "content": user_text}], max_tokens=max_tokens
    )
    _account_call(stats, call)
    if not call.ok:
        action, status = EnergyGridAction(), "call_failed"
    else:
        action, status = _parse_action_checked(call.content)
        if status != "ok":
            stats["parse_failures"] += 1
    proposed = action.model_dump()
    action = _apply_control_layer(action, obs)
    applied = action.model_dump()
    changes = {k: [proposed.get(k), applied[k]] for k in applied if proposed.get(k) != applied[k]}
    if changes:
        stats["control_overrides"] += 1
    if verbose and status != "ok":
        detail = f" ({call.error})" if call.error else ""
        print(f"  [WARN] {agent} agent: {status}{detail}; using safe default action", flush=True)
    call_id = f"{run_id}:{step}:{round_name}:{agent}" if run_id is not None else None
    _record_call(trace, call_id, step, round_name, agent, model, system, user_text, call, status,
                 proposed, applied, changes, _obs_summary(obs), inputs_from, plan_call_id)
    return AgentResult(agent, action, status == "ok", status, call, proposed, changes, call_id)


def run_task(
    env: EnergyGridEnvironment,
    client: OpenAI,
    model: str,
    task_id: str,
    verbose: bool = True,
    trace: Optional[TraceRecorder] = None,
) -> Dict[str, Any]:
    """
    Run one complete task episode with the LLM agent.
    Hard task: planner at step 0, then executor with 4turn rolling history.
    Easy / Medium: singleprompt executor with 4turn history.
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

    run_id = f"{task_id}-{uuid.uuid4().hex[:8]}" if trace is not None else None
    if trace is not None:
        models = {"planning": PLANNING_MODEL, "dispatch": model, "market": model}
        trace.record(
            "run_start", run_id=run_id, task_id=task_id, protocol="p1_free_text",
            models=models, heterogeneous=len(set(models.values())) > 1,
            provider_host=base_url_host(client), seed=task.get("seed"),
            total_steps=task["total_steps"], max_eval_steps=MAX_EVAL_STEPS,
            temperature=0.2, max_tokens={"planning": PLANNING_MAX_TOKENS, "default": 256},
            git_sha=git_sha(),
        )

    # Wall-clock timeout (18 min hard cap, leaves 2 min margin before 20-min budget)
    episode_start = time.time()
    EPISODE_TIMEOUT = 18 * 60

    # ------------------------------------------------------------------
    # Hard task - oneshot planner
    # ------------------------------------------------------------------
    stats = _new_stats()
    plan = ""
    plan_call_id: Optional[str] = None
    last_planning_call_id: Optional[str] = None
    if task_id == "hard":
        if verbose:
            print("  [PLANNER] Generating strategic plan...")
        plan_call = _call_llm_detailed(
            client=client,
            model=PLANNING_MODEL,
            system="You are a strategic planner. Output a concise operational plan only.",
            messages=[{"role": "user", "content": _build_planner_prompt(obs)}],
            max_retries=2,
            max_tokens=PLANNING_MAX_TOKENS,
        )
        _account_call(stats, plan_call)
        plan_call_id = f"{run_id}:plan" if run_id is not None else None
        _record_call(trace, plan_call_id, None, "plan", "planning", PLANNING_MODEL,
                     "You are a strategic planner. Output a concise operational plan only.",
                     _build_planner_prompt(obs), plan_call, "ok" if plan_call.ok and plan_call.content else ("empty" if plan_call.ok else "call_failed"),
                     {}, {}, {}, _obs_summary(obs))
        if plan_call.ok and plan_call.content:
            plan = plan_call.content.strip()
        else:
            stats["parse_failures"] += 1 if plan_call.ok else 0
            print(f"  [WARN] planner: no plan produced ({plan_call.error or 'empty content'}); continuing without a plan", flush=True)
        if verbose:
            print(f"  [PLANNER] Plan generated ({len(plan)} chars).")
            print(f"  {plan}")

    total_reward = 0.0
    step_count = 0
    rewards_list: List[float] = []  # Track all rewards for structured logging

    last_planning_action = EnergyGridAction(plant_action="none", thought="Maintaining plan.")
    prev_obs = None
    history = []

    step_cap = min(total_steps, MAX_EVAL_STEPS) if MAX_EVAL_STEPS > 0 else total_steps
    for step in range(step_cap):
        # Wall-clock timeout check before each LLM call
        if time.time() - episode_start > EPISODE_TIMEOUT:
            print(f"[WARN] Episode timeout reached at step {step}, stopping early", flush=True)
            break

        # --- ROUND 1: PROPOSALS ---
        obs_text = observation_to_text(obs.model_dump())
        plan_id_now = plan_call_id if (task_id == "hard" and plan and step < 40) else None

        # 1. Gated Planning Agent (keeps its previous action if the call fails)
        if _is_major_event(obs, prev_obs):
            if verbose: print(f"  [EVENT] Triggering Planning Agent at step {step}")
            res_p = _invoke_agent(client, PLANNING_MODEL, "planning",
                                  _build_system_prompt(task_id, plan, step, agent_type="planning"),
                                  obs_text, obs, stats, verbose=verbose,
                                  trace=trace, run_id=run_id, step=step, round_name="proposal",
                                  plan_call_id=plan_id_now)
            if res_p.ok:
                last_planning_action = res_p.action
                last_planning_call_id = res_p.call_id

        # 2. Dispatch and Market Proposals (Always called)
        res_d = _invoke_agent(client, model, "dispatch",
                              _build_system_prompt(task_id, plan, step, agent_type="dispatch"),
                              obs_text, obs, stats, verbose=verbose,
                              trace=trace, run_id=run_id, step=step, round_name="proposal",
                              plan_call_id=plan_id_now)
        prop_d = res_d.action

        res_m = _invoke_agent(client, model, "market",
                              _build_system_prompt(task_id, plan, step, agent_type="market"),
                              obs_text, obs, stats, verbose=verbose,
                              trace=trace, run_id=run_id, step=step, round_name="proposal",
                              plan_call_id=plan_id_now)
        prop_m = res_m.action

        rev_d = prop_d
        rev_m = prop_m

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
            env.step_market(prop_m)  # submit round 1; returned view is market-filtered, so not used

            # --- ROUND 2: REVISIONS (Dispatch & Market Only) ---
            # Handoff provenance: the negotiation history each reviser sees is built from these
            # round-1 calls (the planning entry may date from an earlier step if not re-invoked).
            r1_ids = [i for i in (last_planning_call_id, res_d.call_id, res_m.call_id) if i]
            # Prompts come from the canonical state plus round-1 negotiation history.
            obs_rd = env.get_agent_observation("dispatch")
            res_rd = _invoke_agent(client, model, "dispatch",
                                   _build_system_prompt(task_id, plan, step, agent_type="dispatch"),
                                   observation_to_text(obs_rd.model_dump()), obs_rd, stats, verbose=verbose,
                                   trace=trace, run_id=run_id, step=step, round_name="revision",
                                   inputs_from=r1_ids, plan_call_id=plan_id_now)
            rev_d = res_rd.action
            rev_d.proposal_type = "revision"

            obs_rm = env.get_agent_observation("market")
            res_rm = _invoke_agent(client, model, "market",
                                   _build_system_prompt(task_id, plan, step, agent_type="market"),
                                   observation_to_text(obs_rm.model_dump()), obs_rm, stats, verbose=verbose,
                                   trace=trace, run_id=run_id, step=step, round_name="revision",
                                   inputs_from=r1_ids, plan_call_id=plan_id_now)
            rev_m = res_rm.action
            rev_m.proposal_type = "revision"

            # Round 2 submitted (advances simulator)
            last_planning_action.proposal_type = "revision"
            env.step_planning(last_planning_action)
            env.step_dispatch(rev_d)
            obs = env.step_market(rev_m)

        # Collect history
        final_d = rev_d if task_id != "easy" else prop_d
        final_m = rev_m if task_id != "easy" else prop_m

        reward = obs.reward or 0.0

        if trace is not None:
            trace.record(
                "env_step", step=step, reward=float(reward), done=bool(obs.done),
                submitted={"planning": last_planning_action.model_dump(),
                           "dispatch": final_d.model_dump(), "market": final_m.model_dump()},
                observation=_obs_summary(obs),
            )

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
                    print(f"   BLACKOUT at step {step}  episode ended early")
                else:
                    print(f"   Episode completed at step {step}")
            break

    # --------------------------------------------------------------
    # Grading
    # --------------------------------------------------------------
    grade = env.get_last_grade() or env.grade_current_episode() or {}

    # Emit structured log: END
    score = grade.get("total_score", 0.0)
    success = score >= 0.5  # threshold for "success"
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)
    if rewards_list:
        print(f"[METRIC] avg_reward={sum(rewards_list)/len(rewards_list):.3f}", flush=True)

    if verbose:
        print("\n  [GRADE]:", grade.get("total_score", 0.0))
        print("  Components:", grade.get("component_scores", {}))
        print("  Total reward:", total_reward)

    truncated = step_count < total_steps and not obs.done
    valid_run = stats["call_failures"] == 0 and stats["parse_failures"] == 0
    if trace is not None:
        trace.record("run_end", run_id=run_id, score=score, steps=step_count, truncated=truncated,
                     valid_run=valid_run, blackout=bool(grade.get("blackout_occurred", False)), stats=stats)
    if not valid_run:
        print(f"[WARN] task={task_id} call_failures={stats['call_failures']} "
              f"parse_failures={stats['parse_failures']} - safe-default actions were substituted", flush=True)
    if truncated:
        print(f"[WARN] task={task_id} truncated at {step_count}/{total_steps} steps "
              f"(MAX_EVAL_STEPS={MAX_EVAL_STEPS}); score is not a full-episode score", flush=True)

    return {
        "task_id": task_id,
        "task_name": task["name"],
        "score": score,
        "truncated": truncated,
        "eval_step_cap": step_cap,
        "valid_run": valid_run,
        "llm_stats": stats,
        "trace_path": str(trace.path) if trace is not None and trace.path is not None else None,
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

@dataclass
class CallResult:
    """Outcome of one logical LLM call (including retries)."""
    content: Optional[str] = None
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    attempts: int = 0
    latency_s: float = 0.0            # wall time across all attempts
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _call_llm_detailed(
    client,
    model,
    system,
    messages,
    max_retries=3,
    max_tokens=256,
) -> CallResult:
    """Call the model with bounded retries. Never raises; failure is in the result."""
    result = CallResult()
    start = time.time()
    for attempt in range(max_retries):
        result.attempts = attempt + 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=0.2,
                max_tokens=max_tokens,
            )
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"[:300]
            result.exception = e
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        choice = response.choices[0]
        result.content = choice.message.content
        result.finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        result.prompt_tokens = getattr(usage, "prompt_tokens", None)
        result.completion_tokens = getattr(usage, "completion_tokens", None)
        result.error = None
        result.exception = None
        break
    result.latency_s = time.time() - start
    return result


def _call_llm_with_retry(
    client,
    model,
    system,
    messages,
    max_retries=3,
    max_tokens=256,
    agent_type="dispatch",
    verbose=True,
):
    """Backward-compatible wrapper: returns content, raises after retries are exhausted."""
    result = _call_llm_detailed(client, model, system, messages, max_retries=max_retries, max_tokens=max_tokens)
    if result.exception is not None:
        raise result.exception
    return result.content


# ---------------------------------------------------------------------------
# Main runner (all tasks)
# ---------------------------------------------------------------------------

def run_baseline_agent(
    task_ids: Optional[List[str]] = None,
    verbose: bool = True,
    trace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the baseline LLM agent on the specified tasks.
    Returns a dict mapping task_id  result dict (score, metadata, ).
    """
    if task_ids is None:
        task_ids = list(TASK_ORDER)
    trace_dir = trace_dir or os.getenv("SCALER_TRACE_DIR") or None

    if verbose:
        print("\n" + "=" * 60)
        print("  ENERGY GRID OPENENV  BASELINE AGENT")
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

        trace = None
        if trace_dir:
            trace = TraceRecorder(Path(trace_dir) / f"{task_id}-{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        result = run_task(
            env=env,
            client=client,
            model=model,
            task_id=task_id,
            verbose=verbose,
            trace=trace,
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
            bar = "" * int(score * 20)
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
        # HF Spaces or other read-only filesystems  silently skip file write
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
    """Gracefully handle Ctrl+C  save partial results if possible."""
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
        help="Suppress stepbystep output",
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
