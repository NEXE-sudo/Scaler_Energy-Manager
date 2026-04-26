"""
data_generation.py

Runs the existing baseline LLM agent across all tasks and collects
per-step (prompt, response, reward, done) tuples into a JSONL file.

Reuses the prompt-building and action-parsing logic from server/baseline.py
without modifying it — this is purely a data collection wrapper.

Usage:
    python data_generation.py --output dataset_raw.jsonl --episodes 5
    python data_generation.py --tasks easy --episodes 20 --quiet
"""

import argparse
import json
import os
import sys
import time
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from openai import OpenAI

# Import from existing baseline (reuse all prompt logic, no duplication)
from server.baseline import (
    _build_client,
    _build_system_prompt,
    _build_planner_prompt,
    _parse_action,
    _call_llm_with_retry,
)
from server.energy_grid_environment import EnergyGridEnvironment
from server.tasks import get_task, TASK_ORDER
from models import (
    EnergyGridAction, 
    EnergyGridObservation,
    PlanningAgentAction,
    DispatchAgentAction,
    MarketAgentAction
)

# ─────────────────────────────────────────────────────────────────────────────


from server.llm_adapter import observation_to_text

def format_response(response_text: str) -> str:
    if not response_text or not response_text.strip():
        return ""

    if "Thought:" not in response_text:
        return ""

    if "Action:" not in response_text:
        return ""

    return response_text.strip()

def run_episode_with_collection(
    env: EnergyGridEnvironment,
    client: OpenAI,
    model_map: Dict[str, str], # Mapping of agent_type -> model_name
    task_id: str,
    plan: str = "",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run one episode and return a list of per-step multi-agent records.
    """
    task = get_task(task_id)
    total_steps = task["total_steps"]
    obs = env.reset(task_id)
    prev_obs = None
    last_planning_action = None
    records: List[Dict[str, Any]] = []

    MAX_STEPS = 20
    for step in range(total_steps):
        step_records = []
        
        # --- ROUND 1: PROPOSALS ---
        proposals = {}
        for agent_type in ["planning", "dispatch", "market"]:
            system_prompt = ""
            # Task 2: Gated Planning
            if agent_type == "planning":
                from server.baseline import _is_major_event
                if not _is_major_event(obs, prev_obs) and last_planning_action is not None:
                    proposals[agent_type] = {"action": action_dict, "response": response_text, "prompt": user_prompt, "system": data.get("system", ""), "called": True}
                    obs_negotiation = env.step_planning(last_planning_action)
                    proposals[agent_type] = {"action": last_planning_action, "called": False, "system": "", "response": "", "prompt": ""}
                    continue

            model_to_use = model_map.get(agent_type, model_map.get("default", "llama-3.1-8b-instant"))
            system_prompt = _build_system_prompt(task_id=task_id, plan=plan, step=step, agent_type=agent_type)
            filtered_obs = env._filter_observation_for_agent(obs, agent_type)
            user_prompt = observation_to_text(filtered_obs if isinstance(filtered_obs, dict) else filtered_obs.model_dump())

            response_text = _call_llm_with_retry(
                client=client, model=model_to_use, system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                verbose=verbose
            )
            
            # Task 7: Failsafe for empty response
            if not response_text or not response_text.strip():
                unified_action = EnergyGridAction()
            else:
                unified_action = _parse_action(response_text)
                
            from server.baseline import _apply_control_layer
            unified_action = _apply_control_layer(unified_action, obs)
            
            try:
                if agent_type == "planning":
                    action_dict = PlanningAgentAction(**unified_action.model_dump())
                    last_planning_action = action_dict
                elif agent_type == "dispatch":
                    action_dict = DispatchAgentAction(**unified_action.model_dump())
                else:
                    action_dict = MarketAgentAction(**unified_action.model_dump())
            except Exception:
                if agent_type == "planning":
                    action_dict = PlanningAgentAction(plant_action="none")
                    last_planning_action = action_dict
                elif agent_type == "dispatch":
                    action_dict = DispatchAgentAction(coal_delta=0.0, hydro_delta=0.0, nuclear_delta=0.0, battery_mode="idle", emergency_coal_boost=False)
                else:
                    action_dict = MarketAgentAction(demand_response_mw=0.0, grid_export_mw=0.0, grid_import_mw=0.0, coal_price_bid=0.0)
            
            proposals[agent_type] = {"action": action_dict, "response": response_text, "prompt": user_prompt, "system": system_prompt, "called": True}
            
            if agent_type == "planning": obs_negotiation = env.step_planning(action_dict)
            elif agent_type == "dispatch": obs_negotiation = env.step_dispatch(action_dict)
            else: obs_negotiation = env.step_market(action_dict)

        # Store Round 1 data (only for agents that were actually called)
        for agent_type, data in proposals.items():
            if data.get("called"):
                step_records.append({
                    "agent": agent_type, "phase": "proposal", "step": step,
                    "prompt": data["prompt"], "response": data["response"],
                    "system": data.get("system", ""),
                    "reward": 0.0,
                    "task_id": task_id,
                    "blackout": getattr(obs_negotiation, "episode_ended_early", False)
                })

        # Step 3 agents for Round 2
        if task_id == "easy":
            # Task 8: Skip revision round for easy tasks
            last_planning_action.proposal_type = "revision"
            proposals["dispatch"]["action"].proposal_type = "revision"
            proposals["market"]["action"].proposal_type = "revision"
            env.step_planning(last_planning_action)
            env.step_dispatch(proposals["dispatch"]["action"])
            next_obs = env.step_market(proposals["market"]["action"])
        else:
            # --- ROUND 2: REVISIONS ---
            revisions = {}
            for agent_type in ["dispatch", "market"]: # Skip Planning in Round 2
                model_to_use = model_map.get(agent_type, model_map.get("default", "llama-3.1-8b-instant"))
                system_prompt = _build_system_prompt(task_id=task_id, plan=plan, step=step, agent_type=agent_type)
                filtered_obs = env._filter_observation_for_agent(obs_negotiation, agent_type)
                user_prompt = observation_to_text(filtered_obs if isinstance(filtered_obs, dict) else filtered_obs.model_dump())

                response_text = _call_llm_with_retry(
                    client=client, model=model_to_use, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    verbose=verbose, agent_type=agent_type
                )
                
                response_text = format_response(response_text)

                if not response_text:
                    continue
                
                # Task 7: Failsafe for empty response
                if not response_text or not response_text.strip():
                    unified_action = EnergyGridAction()
                else:
                    unified_action = _parse_action(response_text)
                
                from server.baseline import _apply_control_layer
                unified_action = _apply_control_layer(unified_action, obs_negotiation)
                
                try:
                    if agent_type == "dispatch":
                        action_dict = DispatchAgentAction(**unified_action.model_dump())
                    else:
                        action_dict = MarketAgentAction(**unified_action.model_dump())
                except Exception:
                    if agent_type == "dispatch":
                        action_dict = DispatchAgentAction(coal_delta=0.0, hydro_delta=0.0, nuclear_delta=0.0, battery_mode="idle", emergency_coal_boost=False)
                    else:
                        action_dict = MarketAgentAction(demand_response_mw=0.0, grid_export_mw=0.0, grid_import_mw=0.0, coal_price_bid=0.0)
                revisions[agent_type] = {"action": action_dict, "response": response_text, "prompt": user_prompt, "system": system_prompt}

            last_planning_action.proposal_type = "revision"
            env.step_planning(last_planning_action)
            env.step_dispatch(revisions["dispatch"]["action"])
            next_obs = env.step_market(revisions["market"]["action"]) # ADVANCES SIMULATOR

            # Store Round 2 data
            reward = float(next_obs.reward or 0.0)
            for agent_type, data in revisions.items():
                # Get agent-specific reward if available
                agent_reward = getattr(next_obs, f"{agent_type}_reward", reward)
                step_records.append({
                    "agent": agent_type, "phase": "revision", "step": step,
                    "prompt": data["prompt"], "response": data["response"],
                    "system": data.get("system", ""),
                    "reward": agent_reward, "done": next_obs.done, "task_id": task_id,
                    "blackout": getattr(next_obs, "episode_ended_early", False)
                })

        records.extend(step_records)
        prev_obs = obs
        obs = next_obs
        if next_obs.done: break

    return records


def generate_dataset(
    task_ids:    List[str],
    n_episodes:  int,
    output_path: Path,
    model_map:   Optional[Dict[str, str]] = None,
    verbose:     bool = True,
) -> int:
    """
    Main loop to run episodes and save multi-agent data.
    """
    if model_map is None:
        # Default mixed-model approach
        model_map = {
            "planning": "openai/gpt-oss-120b",
            "dispatch": "llama-3.1-8b-instant",
            "market":   "llama-3.1-8b-instant",
            "default":  "llama-3.1-8b-instant"
        }

    client, _ = _build_client()
    env = EnergyGridEnvironment(normalize=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for task_id in task_ids:
            print(f"\n{'='*60}\nCollecting {n_episodes} episodes for task: {task_id}\n{'='*60}")

            for ep_idx in range(n_episodes):
                print(f"\n  Episode {ep_idx + 1}/{n_episodes}")

                # Hard task: generate a plan at episode start (mirrors baseline.py)
                plan = ""
                if task_id == "hard":
                    obs_init = env.reset(task_id)
                    plan_resp = _call_llm_with_retry(
                        client=client,
                        model=model_map.get("planning", model_map.get("default", "llama-3.3-70b-versatile")),
                        system="You are a strategic planner. Output a concise operational plan only.",
                        messages=[{"role": "user", "content": _build_planner_prompt(obs_init)}],
                        max_retries=2,
                        verbose=verbose,
                        agent_type="planning"
                    )
                    plan = plan_resp.strip()

                records = run_episode_with_collection(
                    env=env,
                    client=client,
                    model_map=model_map,
                    task_id=task_id,
                    plan=plan,
                    verbose=verbose,
                )

                for rec in records:
                    # Validate before writing: skip empty responses
                    if not rec.get("response") or not rec["response"].strip():
                        continue
                    if not rec.get("response", "").strip():
                        continue
                    if "Action:" not in rec["response"]:
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush() # Incremental save
                    total_records += 1

                print(f"  Episode {ep_idx+1} done — {len(records)} steps collected")
                time.sleep(1.0)  # brief pause between episodes

    print(f"\n[DONE] Dataset saved to {output_path} ({total_records} records)")
    return total_records


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect LLM rollout data from energy grid env")
    parser.add_argument("--output",   type=str,   default="dataset_raw.jsonl")
    parser.add_argument("--episodes", type=int,   default=2,  help="Episodes per task")
    parser.add_argument("--tasks",    nargs="+",  default=["easy", "medium", "hard"],
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    # Resolve output path relative to project root (parent of Scaler_hackathon)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # If relative path, save to project root
        project_root = ROOT.parent
        output_path = ROOT / output_path

    generate_dataset(
        task_ids   = args.tasks,
        n_episodes = args.episodes,
        output_path = output_path,
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()