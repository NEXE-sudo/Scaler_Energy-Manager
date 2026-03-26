# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for the Energy Grid Management Environment.

Runs a Groq-hosted LLM (llama-3.3-70b-versatile) against all three
tasks using the OpenAI-compatible client. Uses a hybrid chain-of-thought
prompt: one sentence of reasoning followed by a JSON action block.

This design:
    - Improves decision quality over pure JSON-only prompting
    - Keeps token usage low enough for Groq free tier
    - Produces interpretable reasoning traces for debugging

Environment variables required:
    GROQ_API_KEY   — Groq API key (get free at console.groq.com)

Optional:
    OPENAI_API_KEY — Falls back to OpenAI gpt-4o-mini if GROQ_API_KEY
                     is not set (requires credits)
    BASELINE_MODEL — Override the model name
    BASELINE_BASE_URL — Override the API base URL

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
# Client configuration
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenAI, str]:
    """
    Build OpenAI-compatible client.

    Priority:
        1. Groq (GROQ_API_KEY) — free tier, recommended
        2. OpenAI (OPENAI_API_KEY) — fallback, requires credits
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    base_url_override = os.getenv("BASELINE_BASE_URL")
    model_override = os.getenv("BASELINE_MODEL")

    if base_url_override:
        # Fully custom endpoint
        api_key = groq_key or openai_key or "dummy"
        client = OpenAI(api_key=api_key, base_url=base_url_override)
        model = model_override or "llama-3.3-70b-versatile"
        return client, model

    if groq_key:
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = model_override or "llama-3.3-70b-versatile"
        return client, model

    if openai_key:
        client = OpenAI(api_key=openai_key)
        model = model_override or "gpt-4o-mini"
        return client, model

    raise EnvironmentError(
        "No API key found. Set GROQ_API_KEY (recommended, free) or "
        "OPENAI_API_KEY in your environment or .env file."
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are an expert electricity grid operator. "
        "Your job is to dispatch generation sources and manage storage "
        "to meet demand reliably at minimum cost while keeping the grid stable. "
        "At each step you will receive the current grid state and must decide "
        "how to adjust generation.\n\n"
        "Key rules:\n"
        "- Grid frequency must stay near 50.0 Hz. Below 49.0 Hz triggers "
        "load shedding. Below 47.5 Hz causes a full blackout and ends the episode.\n"
        "- Coal has a minimum stable output of 200 MW when online. "
        "Ramping below this shuts it down (takes 3 steps to restart).\n"
        "- Nuclear (if available) cannot go below 300 MW and ramps very slowly (±10 MW).\n"
        "- Battery cannot charge and discharge in the same step.\n"
        "- Hydro uses reservoir water — conserve it during drought events.\n"
        "- In Hard task, build plants early to benefit from them before the episode ends.\n"
    )


def _build_user_prompt(obs: EnergyGridObservation, task_id: str) -> str:
    """
    Build a structured state prompt for the LLM.

    Uses clear sections so the model can parse the state quickly.
    Includes a one-sentence reasoning request before the JSON action.
    """
    task = get_task(task_id)

    # Format active events nicely
    events_str = ", ".join(obs.active_events) if obs.active_events else "none"

    # Format construction queue
    if obs.plants_under_construction:
        construction_str = ", ".join(
            f"{p['type']} ({p['steps_remaining']} steps left)"
            for p in obs.plants_under_construction
        )
    else:
        construction_str = "none"

    # Frequency risk indicator
    freq_diff = obs.grid_frequency - 50.0
    freq_str = f"{obs.grid_frequency:.3f} Hz ({freq_diff:+.3f} from nominal)"

    # Build prompt sections
    lines = [
        f"=== ENERGY GRID — Step {obs.step}/{task['total_steps']} | "
        f"Task: {task_id.upper()} | Day {obs.day} | {obs.time_of_day:02d}:00 | "
        f"Season: {obs.season} ===",
        "",
        "--- DEMAND & SUPPLY ---",
        f"  Demand:          {obs.demand_mw:.1f} MW",
        f"  Unmet demand:    {obs.unmet_demand_mw:.1f} MW  "
        f"{'⚠ SHORTFALL' if obs.unmet_demand_mw > 0 else '✓ MET'}",
        "",
        "--- GENERATION ---",
        f"  Coal:    {obs.coal_output_mw:.1f} / {obs.coal_max_mw:.0f} MW  "
        f"[online={obs.coal_online}, startup_steps={obs.coal_startup_steps_remaining}]",
        f"  Solar:   {obs.solar_output_mw:.1f} MW  "
        f"[available={obs.solar_available}, weather={obs.solar_weather}]",
        f"  Wind:    {obs.wind_output_mw:.1f} MW  "
        f"[available={obs.wind_available}, speed={obs.wind_speed_ms:.1f} m/s]",
        f"  Hydro:   {obs.hydro_output_mw:.1f} / 200 MW  "
        f"[available={obs.hydro_available}]",
        f"  Nuclear: {obs.nuclear_output_mw:.1f} MW  "
        f"[available={obs.nuclear_available}, online={obs.nuclear_online}, "
        f"trip_steps={obs.nuclear_trip_steps_remaining}]",
        "",
        "--- STORAGE ---",
        f"  Battery: {obs.battery_level_mwh:.1f} / {obs.battery_capacity_mwh:.1f} MWh  "
        f"({100*obs.battery_level_mwh/max(1,obs.battery_capacity_mwh):.0f}%)",
        f"  Hydro reservoir: {obs.reservoir_level_mwh:.1f} / "
        f"{obs.reservoir_capacity_mwh:.0f} MWh  "
        f"({100*obs.reservoir_level_mwh/max(1,obs.reservoir_capacity_mwh):.0f}%)  "
        f"[inflow={obs.natural_inflow_mwh:.1f} MWh/step]",
        "",
        "--- GRID STABILITY ---",
        f"  Frequency:         {freq_str}",
        f"  RoCoF:             {obs.rate_of_change_hz_per_step:+.4f} Hz/step",
        f"  System inertia:    {obs.system_inertia_seconds:.2f} s",
        f"  Blackout risk:     {obs.blackout_risk.upper()}",
        f"  Load shedding:     {obs.load_shedding_mw:.1f} MW",
        f"  Spinning reserve:  {obs.spinning_reserve_mw:.1f} / "
        f"{obs.spinning_reserve_required_mw:.1f} MW required  "
        f"{'✓' if obs.spinning_reserve_mw >= obs.spinning_reserve_required_mw else '⚠ BELOW REQUIREMENT'}",
        f"  Transmission cap:  {obs.transmission_capacity_mw:.0f} MW",
        f"  Primary response:  {'ACTIVE — act within 3 steps' if obs.primary_response_active else 'inactive'}",
        "",
        "--- EVENTS ---",
        f"  Active: {events_str}",
        "",
        "--- ECONOMICS ---",
        f"  Coal price:        {obs.coal_price:.3f}x",
        f"  Cumulative cost:   {obs.cumulative_cost:.3f}",
        f"  Cumulative CO₂:    {obs.cumulative_emissions_tons:.1f} tons",
        f"  Capital budget:    {obs.capital_budget:.0f} units",
        f"  Construction:      {construction_str}",
        "",
        "--- LAST STEP REWARD ---",
        f"  {obs.step_reward:.4f}",
        "",
    ]

    # Task-specific hints
    steps_remaining = task["total_steps"] - obs.step
    lines.append(f"--- SITUATION ({steps_remaining} steps remaining) ---")

    if obs.blackout_risk in ("high", "critical"):
        lines.append(
            "  ⚠ CRITICAL: Grid stability emergency. "
            "Prioritise frequency recovery over cost."
        )
    if obs.unmet_demand_mw > 50:
        lines.append(
            f"  ⚠ SHORTFALL of {obs.unmet_demand_mw:.0f} MW. "
            "Increase generation or discharge battery."
        )
    if obs.primary_response_active:
        lines.append(
            "  ⚠ Governor response active. Secondary response needed within 3 steps."
        )
    if obs.battery_level_mwh < 20:
        lines.append("  ⚠ Battery critically low. Consider charging during low-demand hours.")
    if obs.hydro_available and obs.reservoir_level_mwh < 100:
        lines.append("  ⚠ Hydro reservoir critically low. Reduce hydro output.")
    if task_id == "hard" and obs.capital_budget > 300 and steps_remaining > 15:
        lines.append(
            f"  💡 {obs.capital_budget:.0f} capital available. "
            "Consider building renewable plants for long-term benefit."
        )
    if task_id == "hard" and steps_remaining <= 10 and obs.plants_under_construction:
        lines.append(
            "  ⚠ Plants under construction may not complete before episode ends."
        )

    lines.append("")

    # Action schema reminder
    lines.append("--- YOUR ACTION ---")
    lines.append(
        "In ONE sentence explain your reasoning, then output your action as JSON."
    )
    lines.append("Format your response EXACTLY as:")
    lines.append(
        'REASON: <one sentence explaining your decision>'
    )
    lines.append(
        'ACTION: {"coal_delta": <-100 to 100>, '
        '"hydro_delta": <-80 to 80>, '
        '"nuclear_delta": <-10 to 10>, '
        '"battery_mode": "<charge|discharge|idle>", '
        '"plant_action": "<none|build_solar|build_wind|build_hydro|build_nuclear|close_coal>", '
        '"emergency_coal_boost": <true|false>, '
        '"demand_response_mw": <0 to 150>}'
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_action(response_text: str) -> EnergyGridAction:
    """
    Extract JSON action from hybrid chain-of-thought response.

    Handles:
        - Standard format: ACTION: {...}
        - JSON code blocks: ```json ... ```
        - Raw JSON fallback
        - Partial JSON with missing fields (uses defaults)
    """
    text = response_text.strip()

    # Strategy 1: Extract ACTION: line
    action_match = re.search(r"ACTION\s*:\s*(\{[^}]+\})", text, re.DOTALL)
    if action_match:
        raw = action_match.group(1)
        try:
            data = json.loads(raw)
            return _dict_to_action(data)
        except json.JSONDecodeError:
            pass

    # Strategy 2: JSON code block
    code_block_match = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            return _dict_to_action(data)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Any JSON object in the response
    json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return _dict_to_action(data)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Complete fallback — do nothing this step
    print(f"  [WARN] Could not parse action from response. Using safe default.")
    print(f"  [WARN] Response was: {text[:200]}")
    return EnergyGridAction()


def _dict_to_action(data: Dict[str, Any]) -> EnergyGridAction:
    """
    Convert a parsed dict to EnergyGridAction with safe defaults
    and validation.
    """
    # Normalise boolean fields that LLMs sometimes return as strings
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
        coal_delta=float(data.get("coal_delta", 0.0)),
        hydro_delta=float(data.get("hydro_delta", 0.0)),
        nuclear_delta=float(data.get("nuclear_delta", 0.0)),
        battery_mode=battery_mode,
        plant_action=plant_action,
        emergency_coal_boost=bool(emergency_boost),
        demand_response_mw=float(data.get("demand_response_mw", 0.0)),
    )


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

    Returns a dict with the grade result and episode statistics.
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
    conversation_history = []

    system_prompt = _build_system_prompt()
    total_reward = 0.0
    step_count = 0
    reason_log = []

    for step in range(total_steps):
        user_prompt = _build_user_prompt(obs, task_id)

        # Add to conversation history (rolling window of last 3 turns
        # to keep tokens manageable on free tier)
        conversation_history.append({"role": "user", "content": user_prompt})
        if len(conversation_history) > 6:   # 3 turns × 2 messages each
            conversation_history = conversation_history[-6:]

        # LLM call with retry on rate limit
        response_text = _call_llm_with_retry(
            client=client,
            model=model,
            system=system_prompt,
            messages=conversation_history,
            max_retries=3,
            verbose=verbose,
        )

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response_text})

        # Extract reasoning for logging
        reason_match = re.search(r"REASON\s*:\s*(.+?)(?:ACTION|$)", response_text, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()[:120]
            reason_log.append(f"Step {step:02d}: {reason}")
            if verbose:
                print(f"  Step {step:02d} | REASON: {reason}")

        # Parse action
        action = _parse_action(response_text)

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
        # Force grade if episode didn't naturally end
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
        "reasoning_samples": reason_log[:5],   # first 5 for brevity
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
) -> str:
    """
    Call the LLM API with exponential backoff on rate limit errors.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=256,     # enough for one sentence + JSON
                temperature=0.2,    # low temperature for consistent decisions
                messages=[
                    {"role": "system", "content": system},
                    *messages,
                ],
            )
            return response.choices[0].message.content or ""

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
            time.sleep(3)

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