#!/usr/bin/env python3
"""
compare_models.py

Runs inference twice:
  1. Base model (llama-3.1-8b-instant via Groq API)
  2. Fine-tuned LoRA model (TinyLlama + ./lora_energy_grid)

Prints a side-by-side comparison table of scores per task.

Usage:
    python compare_models.py --tasks easy medium hard
"""
import argparse
import os
import sys
import json
import re

from server.tasks import TASKS
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL_NAME  = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")
LORA_MODEL_DIR   = "./lora_energy_grid"
BASE_MODEL_LABEL = f"Base ({BASE_MODEL_NAME})"
LORA_MODEL_LABEL = "Fine-tuned (TinyLlama + LoRA)"

# ─── Run baseline agent and capture scores ────────────────────────────────────
def run_baseline(task_ids, model_override=None):
    """
    Run the Groq-based baseline agent and return per-task score dicts.
    Returns list of result dicts: {task_id, score, avg_reward, steps, blackout}
    """
    if model_override:
        os.environ["MODEL_NAME"] = model_override

    from server.baseline import run_baseline_agent
    results = run_baseline_agent(task_ids=task_ids, verbose=False)
    return results


def run_lora_inference(task_ids):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from server.energy_grid_environment import EnergyGridEnvironment
    from server.llm_adapter import extract_action_from_llm_output, build_compact_obs
    from models import EnergyGridAction

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[LoRA] Loading TinyLlama + LoRA from {LORA_MODEL_DIR} ...")
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )

    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    for task_id in task_ids:
        print(f"\n[LoRA] Running task: {task_id}")

        env = EnergyGridEnvironment()
        obs = env.reset(task_id)

        total_reward = 0.0
        rewards_list = []
        step = 0

        # ✅ Clean step mapping (NO TASKS dependency)
        steps_map = {
            "easy": 24,
            "medium": 48,
            "hard": 72
        }
        total_steps = steps_map.get(task_id, 24)

        while step < total_steps and not obs.done:
            obs_text = build_compact_obs(obs)

            prompt = f"""<|system|>
            You are the Dispatch Agent.

            STRICT RULES:
            - Output MUST contain valid JSON
            - Do NOT skip Action
            - Do NOT add extra text

            FORMAT:

            Thought:
            <reasoning>

            Action:
            {{ valid JSON }}

            <|user|>
            {obs_text}

            <|assistant|>
            Thought:
            """

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,  # 🔥 faster
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            full_text = "Thought:\n" + generated

            action_dict = extract_action_from_llm_output(full_text)

            try:
                action = EnergyGridAction(**action_dict)
            except Exception:
                action = EnergyGridAction()

            obs = env.step(action)

            rewards_list.append(obs.reward)
            total_reward += obs.reward
            step += 1

        grade = env.grade_current_episode() or {}

        score = grade.get("total_score", 0.0)
        avg_reward = total_reward / max(len(rewards_list), 1)

        print(
            f"[LoRA] Task={task_id} | "
            f"Score={score:.3f} | AvgReward={avg_reward:.3f} | Steps={step}"
        )

        save_rewards(task_id, rewards_list, prefix="lora")
        
        results.append({
            "task_id": task_id,
            "score": score,
            "avg_reward": avg_reward,
            "steps": step,
            "blackout": grade.get("blackout_occurred", False),
        })

    return results

# ─── Comparison table printer ─────────────────────────────────────────────────
def print_comparison(base_results, lora_results, task_ids):
    base_by_task = {}

    if isinstance(base_results, list):
        for r in base_results:
            if isinstance(r, dict) and "task_id" in r:
                base_by_task[r["task_id"]] = r
                
    lora_by_task = {r["task_id"]: r for r in (lora_results or [])}

    COL = 22
    SEP = "-" * (10 + COL * 2 + 6)

    print("\n")
    print("=" * (10 + COL * 2 + 6))
    print(f"  MODEL COMPARISON  —  Energy Grid Baseline")
    print("=" * (10 + COL * 2 + 6))
    print(f"  {'TASK':<10} {BASE_MODEL_LABEL:>{COL}} {LORA_MODEL_LABEL:>{COL}}")
    print(SEP)

    total_base_score = 0.0
    total_lora_score = 0.0
    count = 0

    for task_id in task_ids:
        b = base_by_task.get(task_id, {})
        l = lora_by_task.get(task_id, {})

        b_score = b.get("score", float("nan"))
        l_score = l.get("score", float("nan"))
        b_avg   = b.get("avg_reward", float("nan"))
        l_avg   = l.get("avg_reward", float("nan"))

        delta = l_score - b_score if (b_score == b_score and l_score == l_score) else float("nan")
        arrow = ("[+]" if delta > 0 else "[-]") if delta == delta else "[ ]"

        print(f"  {task_id.upper():<10} {'Score':>{COL-9}} {b_score:>8.3f}    {'Score':>{COL-9}} {l_score:>8.3f}  {arrow} {abs(delta):.3f}")
        print(f"  {'':10} {'AvgRew':>{COL-9}} {b_avg:>8.3f}    {'AvgRew':>{COL-9}} {l_avg:>8.3f}")
        print(f"  {'':10} {'Blackout':>{COL-9}} {str(b.get('blackout', '?')):>8}    {'Blackout':>{COL-9}} {str(l.get('blackout', '?')):>8}")
        print(SEP)

        if b_score == b_score:
            total_base_score += b_score
            count += 1
        if l_score == l_score:
            total_lora_score += l_score

    if count:
        print(f"  {'AVERAGE':<10} {'Score':>{COL-9}} {total_base_score/count:>8.3f}    {'Score':>{COL-9}} {total_lora_score/count:>8.3f}")
        improvement = ((total_lora_score - total_base_score) / max(abs(total_base_score), 1e-9)) * 100
        print(f"\n  Overall improvement: {improvement:+.1f}%")
    print("=" * (10 + COL * 2 + 6))

import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
# 1. Plot single reward curve
# ─────────────────────────────────────────────
def plot_rewards(rewards, title="Reward over Time", save_path=None):
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# 2. Plot base vs LoRA scores
# ─────────────────────────────────────────────
def plot_comparison(base_results, lora_results, save_path=None):
    import matplotlib.pyplot as plt

    # ✅ Safe fallback if base is missing
    if not isinstance(base_results, list):
        print("[WARN] No valid base_results — using fallback values")
        base_results = [
            {"task_id": r["task_id"], "score": 0.24}
            for r in lora_results
        ]

    tasks = [r["task_id"] for r in lora_results]

    base_scores = []
    for t in tasks:
        match = next(
            (b for b in base_results if isinstance(b, dict) and b.get("task_id") == t),
            None
        )
        base_scores.append(match["score"] if match else 0)

    lora_scores = [r["score"] for r in lora_results]

    x = range(len(tasks))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], base_scores, width=width, label="Base Model")
    plt.bar([i + width/2 for i in x], lora_scores, width=width, label="LoRA Model")

    plt.xticks(x, tasks)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(axis='y')

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")

    plt.show()

# ─────────────────────────────────────────────
# 3. Plot dual reward comparison
# ─────────────────────────────────────────────
def plot_dual_rewards(base_rewards, lora_rewards, title="Reward Comparison", save_path=None):
    plt.figure()

    plt.plot(base_rewards, label="Base Model", linestyle='--')
    plt.plot(lora_rewards, label="LoRA Model")

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")

    plt.show()
    
def plot_comparison(base_results, lora_results, save_path=None):
    import matplotlib.pyplot as plt

    tasks = [r["task_id"] for r in lora_results]

    base_scores = [
        next((b["score"] for b in base_results if b["task_id"] == t), 0)
        for t in tasks
    ]

    lora_scores = [r["score"] for r in lora_results]

    x = range(len(tasks))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], base_scores, width=width, label="Base Model")
    plt.bar([i + width/2 for i in x], lora_scores, width=width, label="LoRA Model")

    plt.xticks(x, tasks)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(axis='y')

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# 4. Save rewards automatically
# ─────────────────────────────────────────────
def save_rewards(task_id, rewards, prefix="lora"):
    os.makedirs("plots", exist_ok=True)
    path = f"plots/{prefix}_{task_id}_rewards.png"
    plot_rewards(rewards, f"{prefix.upper()} Rewards - {task_id}", save_path=path)
    
# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model on energy grid tasks")
    parser.add_argument("--tasks",     nargs="+", default=["easy"],
                        help="Task IDs to benchmark (default: easy)")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the base model run (use saved scores)")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip the LoRA model run")
    parser.add_argument("--base-scores", type=str, default=None,
                        help="JSON file with pre-recorded base scores (optional)")
    args = parser.parse_args()

    task_ids = args.tasks

    # ── Base model run ──
    base_results = None
    if not args.skip_base:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Running BASE model ({BASE_MODEL_NAME})")
        print(f"{'='*60}")
        base_results = run_baseline(task_ids, model_override=BASE_MODEL_NAME)
    elif args.base_scores:
        with open(args.base_scores) as f:
            base_results = json.load(f)
        print(f"[INFO] Loaded base scores from {args.base_scores}")

    # Save base scores for reuse
    if base_results:
        with open("base_scores.json", "w") as f:
            json.dump(base_results, f, indent=2)
        print("\n[INFO] Base scores saved to base_scores.json")

    # ── LoRA model run ──
    lora_results = None
    if not args.skip_lora:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Running FINE-TUNED LoRA model")
        print(f"{'='*60}")
        lora_results = run_lora_inference(task_ids)

    # Save lora scores for reuse
    if lora_results:
        with open("lora_scores.json", "w") as f:
            json.dump(lora_results, f, indent=2)
        print("\n[INFO] LoRA scores saved to lora_scores.json")

    # ── Print comparison ──
    # If we already have the base from inference.py run, inject it
    if base_results is None and Path("base_scores.json").exists():
        with open("base_scores.json") as f:
            base_results = json.load(f)

    print_comparison(base_results, lora_results, task_ids)
    if base_results and lora_results:
        plot_comparison(base_results, lora_results, save_path="plots/comparison.png")


if __name__ == "__main__":
    main()
