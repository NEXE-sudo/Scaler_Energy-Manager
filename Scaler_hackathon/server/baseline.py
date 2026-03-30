#!/usr/bin/env python3
"""
Baseline inference script for the Energy Grid Management Environment.

Architecture:
    Easy / Medium — executor with 4‑turn rolling conversation history.
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
try:
    from .energy_grid_environment import EnergyGridEnvironment
    from .grader import grade_result_to_dict
    from .tasks import get_task, TASK_ORDER
except ImportError:
    try:
        # If running as installed package
        from energy_grid_openenv.server.energy_grid_environment import EnergyGridEnvironment
        from energy_grid_openenv.server.grader import grade_result_to_dict
        from energy_grid_openenv.server.tasks import get_task, TASK_ORDER
    except ImportError:
        # If running locally
        from server.energy_grid_environment import EnergyGridEnvironment
        from server.grader import grade_result_to_dict
        from server.tasks import get_task, TASK_ORDER

try:
    from ..models import EnergyGridAction, EnergyGridObservation
except ImportError:
    try:
        # If running as installed package
        from energy_grid_openenv.models import EnergyGridAction, EnergyGridObservation
    except ImportError:
        # If running locally
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
# Client configuration (hackathon‑spec only)
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenAI, str]:
    """
    Build an OpenAI‑compatible client.

    Required env vars:
        API_BASE_URL   – endpoint (e.g. https://api.groq.com/openai/v1)
        MODEL_NAME      – model identifier
        HF_TOKEN / OPENAI_API_KEY / API_KEY – auth token
    """
    api_base_url = os.getenv("API_BASE_URL")
    model_name   = os.getenv("MODEL_NAME")
    api_key      = (
        os.getenv("HF_TOKEN")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )

    if not (api_base_url and model_name and api_key):
        raise EnvironmentError(
            "Missing required API configuration. Set API_BASE_URL, MODEL_NAME, "
            "and an API key (HF_TOKEN / OPENAI_API_KEY / API_KEY)."
        )
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    return client, model_name

# ---------------------------------------------------------------------------
# Token budget (small enough to force a complete ACTION line)
# ---------------------------------------------------------------------------

MAX_TOKENS = 4000  # enough for REASON + full JSON ACTION

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(task_id: str = "easy", plan: str = "") -> str:
    """
    Build the system prompt for the executor.
    For Hard task, injects the planner output.
    """
    # -------------------------------------------------------------------
    # Compact prompt – ~500 tokens (still contains every hard constraint)
    # -------------------------------------------------------------------
    base = """You are an expert electricity grid operator. ALWAYS output exactly two lines:

REASON: <one‑sentence explanation>
ACTION: {"coal_delta":<-100..100>,"hydro_delta":<-80..80>,"nuclear_delta":<-10..10>,
         "battery_mode":"charge|discharge|idle",
         "plant_action":"none|build_solar|build_wind|build_hydro|build_nuclear|close_coal",
         "emergency_coal_boost":true|false,
         "demand_response_mw":<0..150>}

Constraints:
- Coal: min 200 MW, max 600 MW, ramp ±100 MW/step.
- Nuclear: min 300 MW, ramp ±10 MW/step, SCRAM → 0 for 8 steps.
- Battery: charge/discharge ≤ 50 MW, cannot do both.
- Frequency: keep 49.5 – 50.5 Hz, blackout < 47.5 Hz or > 51.5 Hz.
- Gap = demand – total_generation (positive = shortfall).

If gap > 0 → use battery discharge first, then increase coal, then emergency boost (last resort only).
If gap < 0 (oversupply > 20 MW) → reduce coal immediately, or charge battery.
Never leave a positive gap unresolved.
Operational strategy:
- Never leave unmet demand.
- Avoid overproduction (>20 MW).
- Prefer battery discharge over increasing coal whenever battery > 20%.
- Use battery actively for short-term balancing, not just emergencies.
- Preserve battery above 20% unless preventing blackout.
- Minimise coal usage aggressively — high cost and CO2 emissions.
- Use renewables whenever available.
- If demand is met, reduce coal before anything else.
"""

    if task_id == "hard" and plan:
        base += f"\n\nSTRATEGIC PLAN (follow this unless state forces deviation):\n{plan}"
    return base

def _build_planner_prompt(obs: "EnergyGridObservation") -> str:
    """
    One‑shot strategic planning prompt for the Hard task.
    Runs once at step 0. Output is injected into every executor call.
    """
    battery_pct = int(100 * obs.battery_level_mwh / max(1, obs.battery_capacity_mwh))

    return f"""You are a strategic planner for a 72‑step electricity grid simulation (Hard task, Winter).

KNOWN CONSTANTS (these are guaranteed facts, not estimates):
- Winter peak demand ~1100 MW. Coal max = 600 MW → structural deficit ~500 MW → new capacity is REQUIRED.
- Coal outage GUARANTEED at steps 24–27 (day 2). Coal max drops to 300 MW for 3 steps.
- Nuclear: costs 1000, build time 15 steps, output 500 MW baseload (min 300 MW once online).
- Wind:    costs 400, build time 6 steps, output 250 MW (available day & night).
- Solar:   costs 500, build time 8 steps, output 300 MW (zero at night).
- Hydro:   costs 600, build time 10 steps, output 200 MW dispatchable (reservoir‑limited).
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

Be concise, decisive, and forward‑looking.
"""

# ---------------------------------------------------------------------------
# User prompt (state summary) – unchanged apart from minor formatting
# ---------------------------------------------------------------------------

def _build_user_prompt(obs: EnergyGridObservation, task_id: str) -> str:
    """Ultra-compact state: ~400 tokens instead of ~1000. Keeps only decision-critical fields."""
    task = get_task(task_id)
    
    # Compact status line
    alerts = []
    if obs.unmet_demand_mw > 10: alerts.append(f"SHORT_{obs.unmet_demand_mw:.0f}")
    if obs.blackout_risk in ("high", "critical"): alerts.append(f"RISK_{obs.blackout_risk.upper()}")
    if obs.battery_level_mwh / max(1, obs.battery_capacity_mwh) < 0.15: alerts.append("BATT_LOW")
    if obs.primary_response_active: alerts.append("GOV")
    
    alert_str = " ".join(alerts) if alerts else "OK"
    
    avail = ",".join(k for k in ["coal","solar","wind","hydro","nuclear"] 
                     if (k=="coal" or getattr(obs, f"{k}_available", False)))
    
    total = (obs.coal_output_mw + obs.solar_output_mw + obs.wind_output_mw + 
             obs.hydro_output_mw + obs.nuclear_output_mw)
    gap = obs.demand_mw - total
    
    # Single compact line format (all essential info)
    prompt = (
        f"S{obs.step}/{task['total_steps']} {task_id.upper()} "
        f"D{obs.day} {obs.time_of_day:02d}:00 {obs.season} | "
        f"Avail:{avail} | "
        f"Demand:{obs.demand_mw:.0f} Unmet:{obs.unmet_demand_mw:.0f} "
        f"[{alert_str}] | "
        f"Coal:{obs.coal_output_mw:.0f}/{obs.coal_max_mw:.0f} "
        f"Sol:{obs.solar_output_mw:.0f} "
        f"Wind:{obs.wind_output_mw:.0f} "
        f"Hydro:{obs.hydro_output_mw:.0f} "
        f"Nuc:{obs.nuclear_output_mw:.0f} | "
        f"Batt:{int(100*obs.battery_level_mwh/max(1,obs.battery_capacity_mwh))}% "
        f"Freq:{obs.grid_frequency:.2f}Hz "
        f"Steps:{task['total_steps']-obs.step} | "
        f"REASON: <1 sentence decision> | "
        f'ACTION: {{"coal_delta":<-100..100>,"hydro_delta":<-80..80>,"nuclear_delta":<-10..10>,"battery_mode":"idle|charge|discharge","plant_action":"none|...","emergency_coal_boost":false,"demand_response_mw":0}}'
    )
    return prompt


def _parse_action(response_text: str) -> EnergyGridAction:
    """
    Extract a well‑formed JSON ACTION block from the LLM response.
    Handles:
        - markdown fences (```, ```json)
        - single quotes → double quotes
        - trailing commas
        - Python booleans (True/False)
        - multi‑line JSON
    Returns a fully‑validated EnergyGridAction; on failure returns a
    safe default (all zeros / idle) and prints a warning.
    """
    # 1️⃣ strip markdown fences and surrounding whitespace
    txt = re.sub(r'^```[a-z]*\n|```$', '', response_text.strip(), flags=re.MULTILINE)

    # 2️⃣ balanced‑brace extraction (find the first complete JSON object)
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

    # Prefer the JSON after the word ACTION (if present)
    action_pos = txt.upper().find('ACTION')
    search_text = txt[action_pos:] if action_pos != -1 else txt
    raw = extract_json(search_text) or extract_json(txt)

    if raw is None:
        print("[WARN] Parser failed – no JSON object found")
        print("[WARN] Raw response (first 300 chars):")
        print(txt[:300])
        return EnergyGridAction()      # safe default (no‑op)

    # 3️⃣ normalise the JSON text
    raw = raw.replace("'", '"')
    raw = re.sub(r',\s*}', '}', raw)                     # strip trailing commas
    raw = re.sub(r'\bTrue\b', 'true', raw, flags=re.I)   # Python bool → JSON bool
    raw = re.sub(r'\bFalse\b', 'false', raw, flags=re.I)

    try:
        payload = json.loads(raw)
        return _dict_to_action(payload)
    except (json.JSONDecodeError, Exception) as exc:
        print("[WARN] JSON decode error while parsing ACTION:", exc)
        print("[WARN] Attempted JSON (first 200 chars):")
        print(raw[:200])
        return EnergyGridAction()      # safe default

def _dict_to_action(data: Dict[str, Any]) -> EnergyGridAction:
    """
    Convert parsed JSON dict to EnergyGridAction.
    Clamps every numeric field to the model‑defined limits.
    """
    def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return default

    # Bool handling (accepts true/false, "true", "1", "yes")
    emergency_boost = data.get("emergency_coal_boost", False)
    if isinstance(emergency_boost, str):
        emergency_boost = emergency_boost.lower() in ("true", "1", "yes")

    # Battery mode normalisation
    battery_mode = str(data.get("battery_mode", "idle")).lower().strip()
    if battery_mode not in ("charge", "discharge", "idle"):
        battery_mode = "idle"

    # Plant‑action normalisation
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

    # Block close_coal on any task — too dangerous to allow
    if action.plant_action == "close_coal":
        action.plant_action = "none"

    # Block emergency boost spam — if coal is already damaged, don't boost again
    if action.emergency_coal_boost and obs.coal_max_mw < 580.0:
        action.emergency_coal_boost = False

    # Don't waste battery on small shortfalls coal can handle
    if (action.battery_mode == "discharge"
            and obs.unmet_demand_mw < 80.0
            and obs.coal_output_mw < obs.coal_max_mw - 80.0):
        action.battery_mode = "idle"

    return action

# ---------------------------------------------------------------------------
# Single‑task runner
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
    Hard task: planner at step 0, then executor with 4‑turn rolling history.
    Easy / Medium: single‑prompt executor with 4‑turn history.
    """
    task = get_task(task_id)
    total_steps = max_steps_override or task["total_steps"]

    if verbose:
        print("\n" + "=" * 60)
        print(f"Task: {task['name']} ({task_id})")
        print(f"Steps: {total_steps} | Season: {task['season']}")
        print("=" * 60)

    # Reset environment
    obs = env.reset(task_id)

    # ------------------------------------------------------------------
    # Hard task – one‑shot planner
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
            max_retries=2,  # faster failure recovery
            max_tokens=MAX_TOKENS,
            verbose=verbose,
        )
        plan = planner_response.strip()
        if verbose:
            print(f"  [PLANNER] Plan generated ({len(plan)} chars).")
            print(f"  {plan[:300]}{'...' if len(plan) > 300 else ''}")

    system_prompt = _build_system_prompt(task_id=task_id, plan=plan)
    total_reward = 0.0
    step_count = 0
    reason_log: List[str] = []
    conversation_history: List[Dict[str, str]] = []   # rolling 4‑turn window

    for step in range(total_steps):
        user_prompt = _build_user_prompt(obs, task_id)

        # Append current state to history
        conversation_history.append({"role": "user", "content": user_prompt})

        # LLM call
        response_text = _call_llm_with_retry(
            client=client,
            model=model,
            system=system_prompt,
            messages=conversation_history,
            max_retries=2,  # faster failure recovery
            verbose=verbose,
        )

        # Keep only the most recent 8 messages (4 turns)
        conversation_history.append({"role": "assistant", "content": response_text})
        if len(conversation_history) > 4:
            conversation_history = conversation_history[-4:]

        # Extract REASON for logging
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
                f"plant={action.plant_action} boost={action.emergency_coal_boost}"
            )

        # Execute step
        obs = env.step(action)
        total_reward += obs.step_reward
        step_count += 1

        last_action_summary = (
            f"LastAction: coal_delta={action.coal_delta:+.0f} "
            f"battery={action.battery_mode} "
            f"→ unmet={obs.unmet_demand_mw:.0f}MW freq={obs.grid_frequency:.3f}Hz"
        )

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

    # --------------------------------------------------------------
    # Grading
    # --------------------------------------------------------------
    grade = env.get_last_grade()
    if grade is None:
        grade = env.grade_current_episode() or {}

    if verbose:
        print("\n  📊 GRADE:", grade.get("total_score", 0.0))
        print("  Components:", grade.get("component_scores", {}))
        print("  Total reward:", total_reward)

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
# LLM call with retry (now uses MAX_TOKENS)
# ---------------------------------------------------------------------------

def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    system: str,
    messages: list,
    max_retries: int = 3,
    verbose: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Call the LLM API with exponential back‑off on rate‑limit errors.
    Returns the raw `content` string of the assistant message.
    """
    for attempt in range(max_retries):
        try:
            _rate_limited_sleep(verbose=verbose)
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "system", "content": system}, *messages],
            )
            # Extract the assistant's content
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Empty response from model — possible TPM burst")

            # ----- DEBUG: print token usage (optional) -----
            if verbose and hasattr(response, "usage"):
                usage = response.usage
                print(
                    f"  [DEBUG] Tokens – prompt:{usage.prompt_tokens} "
                    f"completion:{usage.completion_tokens} total:{usage.total_tokens}"
                )
                    
            return content

        except Exception as e:
            err = str(e).lower()
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
    max_steps_override: Optional[int] = None,
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

    client, model = _build_client()
    if verbose:
        print(f"  Model: {model}")
        print(f"  Tasks: {task_ids}")

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

        # Small pause between tasks (respect rate limits)
        if task_id != task_ids[-1]:
            time.sleep(10)

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

    # Save results to outputs/
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
        "--max-steps",
        type=int,
        default=None,
        help="Override episode length (useful for quick tests)",
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
        max_steps_override=args.max_steps,
        verbose=not args.quiet,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")
