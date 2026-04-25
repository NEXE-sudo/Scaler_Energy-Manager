"""
PPO Training loop for Energy Grid Dispatch Agent.

Curriculum: easy → medium → hard
Each phase trains for a fixed number of environment steps.

Usage:
    python train.py
    python train.py --tasks easy --steps 50000
    python train.py --resume checkpoints/agent_easy_final.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Local imports ────────────────────────────────────────────────────────────
# Adjust sys.path so this can be run from any directory.
ROOT = Path(__file__).parent.parent  # project root
sys.path.insert(0, str(ROOT))

from ppo_agent import PPOAgent, flatten_obs, OBS_DIM, CONT_ACT_DIM
from buffer import RolloutBuffer

# Import environment  
try:
    from server.energy_grid_environment import EnergyGridEnvironment
except ImportError:
    from Scaler_hackathon.server.energy_grid_environment import EnergyGridEnvironment

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # PPO
    "lr":            3e-4,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_eps":      0.2,
    "vf_coef":       0.5,
    "ent_coef":      0.01,
    "max_grad_norm": 0.5,
    "n_epochs":      8,          # PPO update epochs per rollout
    "batch_size":    256,
    "rollout_len":   2048,       # steps per rollout collection
    # Reward
    "reward_scale":  0.01,       # divide raw reward to keep value targets ~1
    # Action EMA smoothing (prevents jitter)
    "action_ema":    0.8,        # 0 = no smoothing, 1 = frozen
    # Curriculum steps per task
    "steps_easy":    200_000,
    "steps_medium":  300_000,
    "steps_hard":    400_000,
    # Checkpointing
    "save_every":    50_000,
    "checkpoint_dir": "checkpoints",
}


# ─────────────────────────────────────────────────────────────────────────────
# PPO Update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    agent:      PPOAgent,
    buffer:     RolloutBuffer,
    optimizer:  torch.optim.Optimizer,
    cfg:        Dict[str, Any],
) -> Dict[str, float]:
    """
    Run `n_epochs` of PPO updates over the filled buffer.
    Returns a dict of mean losses for logging.
    """
    buffer.normalise_advantages()

    total_pg_loss   = 0.0
    total_vf_loss   = 0.0
    total_ent_loss  = 0.0
    total_clip_frac = 0.0
    n_updates       = 0

    for _ in range(cfg["n_epochs"]):
        for obs_b, cont_b, boost_b, old_lp_b, ret_b, adv_b in buffer.get_batches(cfg["batch_size"]):

            # Re-evaluate actions under current policy
            log_prob, entropy, value = agent.evaluate_actions(obs_b, cont_b, boost_b)

            # PPO clipped objective
            ratio = torch.exp(log_prob - old_lp_b)
            pg_loss1 = -adv_b * ratio
            pg_loss2 = -adv_b * ratio.clamp(1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"])
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (clipped, following PPO paper)
            vf_loss  = 0.5 * (value - ret_b).pow(2).mean()

            # Entropy bonus
            ent_loss = -entropy.mean()

            loss = pg_loss + cfg["vf_coef"] * vf_loss + cfg["ent_coef"] * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean().item()

            total_pg_loss   += pg_loss.item()
            total_vf_loss   += vf_loss.item()
            total_ent_loss  += ent_loss.item()
            total_clip_frac += clip_frac
            n_updates       += 1

    d = max(1, n_updates)
    return {
        "pg_loss":   total_pg_loss   / d,
        "vf_loss":   total_vf_loss   / d,
        "ent_loss":  total_ent_loss  / d,
        "clip_frac": total_clip_frac / d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rollout(
    agent:      PPOAgent,
    env:        EnergyGridEnvironment,
    buffer:     RolloutBuffer,
    task_id:    str,
    cfg:        Dict[str, Any],
    prev_action: Optional[Dict[str, Any]] = None,
) -> Tuple[float, int, int, Optional[Dict[str, Any]]]:
    """
    Collect `rollout_len` steps into buffer.

    Handles:
        - Episode resets when done
        - Action EMA smoothing
        - Reward scaling

    Returns:
        mean_ep_reward — mean reward of completed episodes
        n_episodes     — number of completed episodes
        steps_done     — steps collected
        prev_action    — last action (for EMA continuity across rollouts)
    """
    buffer.reset()
    obs = env.reset(task_id)
    obs_np = flatten_obs(obs)

    ep_reward    = 0.0
    ep_rewards:  List[float] = []
    ema_action:  Optional[np.ndarray] = None

    # Initialise EMA from previous rollout if available
    if prev_action is not None:
        ema_action = np.array([
            prev_action["coal_delta"],
            prev_action["hydro_delta"],
            prev_action["nuclear_delta"],
            prev_action["demand_response_mw"],
        ], dtype=np.float32)

    steps = 0
    while steps < cfg["rollout_len"]:
        # Get action
        action_dict, log_prob, value = agent.act(obs_np)

        # Action EMA smoothing (reduces high-frequency oscillation)
        raw_cont = np.array([
            action_dict["coal_delta"],
            action_dict["hydro_delta"],
            action_dict["nuclear_delta"],
            action_dict["demand_response_mw"],
        ], dtype=np.float32)

        if ema_action is None:
            ema_action = raw_cont.copy()
        else:
            alpha = cfg["action_ema"]
            ema_action = alpha * ema_action + (1.0 - alpha) * raw_cont

        action_dict["coal_delta"]         = float(np.clip(ema_action[0], -100, 100))
        action_dict["hydro_delta"]        = float(np.clip(ema_action[1], -80, 80))
        action_dict["nuclear_delta"]      = float(np.clip(ema_action[2], -10, 10))
        action_dict["demand_response_mw"] = float(np.clip(ema_action[3], 0, 150))

        # Environment step
        next_obs = env.step(
            type("EnergyGridAction", (), action_dict)()
            if not hasattr(env, "_action_class")
            else _dict_to_action(action_dict)
        )
        done   = next_obs.done
        reward = float(next_obs.reward or 0.0) * cfg["reward_scale"]

        ep_reward += float(next_obs.reward or 0.0)

        # Retrieve raw cont_raw from agent's last forward pass
        # We need the PRE-tanh values; recompute from log_prob is complex,
        # so we store the raw mean directly for the update.
        # Simple approach: use atanh of the scaled action as approx of cont_raw.
        def scaled_to_raw(vals):
            """Inverse of (tanh(x)+1)/2 * (hi-lo) + lo → atanh(2*(v-lo)/(hi-lo)-1)"""
            lo = np.array([-100., -80., -10., 0.], dtype=np.float32)
            hi = np.array([ 100.,  80.,  10., 150.], dtype=np.float32)
            norm = 2.0 * (vals - lo) / (hi - lo) - 1.0
            norm = np.clip(norm, -0.9999, 0.9999)
            return np.arctanh(norm)

        cont_raw_stored = scaled_to_raw(raw_cont)

        # Update obs normalisation stats periodically
        if steps % 512 == 0:
            agent.update_obs_stats(
                torch.tensor(obs_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
            )

        buffer.add(
            obs       = obs_np,
            cont_raw  = cont_raw_stored,
            boost     = float(action_dict["emergency_coal_boost"]),
            log_prob  = log_prob.item(),
            reward    = reward,
            value     = value.item(),
            done      = float(done),
        )

        steps += 1
        obs_np = flatten_obs(next_obs)

        if done:
            ep_rewards.append(ep_reward)
            ep_reward  = 0.0
            ema_action = None   # reset smoothing on episode boundary
            next_obs   = env.reset(task_id)
            obs_np     = flatten_obs(next_obs)

    # Bootstrap value for incomplete episode
    with torch.no_grad():
        _, _, last_value = agent.forward(
            torch.tensor(obs_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
        )
    last_val = 0.0 if done else last_value.item()
    buffer.flush(last_value=last_val)

    mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else ep_reward
    return mean_ep_reward, len(ep_rewards), steps, action_dict


def _dict_to_action(d: Dict[str, Any]):
    """Convert action dict to EnergyGridAction. Import lazily to avoid circular deps."""
    try:
        from models import EnergyGridAction
    except ImportError:
        from Scaler_hackathon.models import EnergyGridAction
    return EnergyGridAction(**{k: v for k, v in d.items()})


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum training
# ─────────────────────────────────────────────────────────────────────────────

def train_task(
    agent:     PPOAgent,
    optimizer: torch.optim.Optimizer,
    task_id:   str,
    total_env_steps: int,
    cfg:       Dict[str, Any],
    checkpoint_dir: Path,
    resume_steps:   int = 0,
) -> None:
    """Train agent on a single task for `total_env_steps` environment steps."""
    env    = EnergyGridEnvironment(normalize=False)
    buffer = RolloutBuffer(
        capacity     = cfg["rollout_len"],
        obs_dim      = OBS_DIM,
        cont_act_dim = CONT_ACT_DIM,
        gamma        = cfg["gamma"],
        gae_lambda   = cfg["gae_lambda"],
        device       = agent.device,
    )

    env_steps        = resume_steps
    update_count     = 0
    prev_action      = None
    rollout_rewards  = []
    t_start          = time.time()
    next_save        = resume_steps + cfg["save_every"]

    print(f"\n{'='*60}")
    print(f"  Training task: {task_id.upper()} | Target: {total_env_steps:,} steps")
    print(f"{'='*60}")

    while env_steps < total_env_steps:
        # Collect rollout
        agent.eval()
        mean_rew, n_eps, steps, prev_action = collect_rollout(
            agent, env, buffer, task_id, cfg, prev_action
        )
        env_steps += steps
        rollout_rewards.append(mean_rew)

        # PPO update
        agent.train()
        stats = ppo_update(agent, buffer, optimizer, cfg)
        update_count += 1

        # Logging
        elapsed = time.time() - t_start
        fps = env_steps / max(elapsed, 1)
        print(
            f"  [{task_id}] steps={env_steps:>8,} | "
            f"ep_rew={mean_rew:>8.1f} | "
            f"n_eps={n_eps:>3} | "
            f"pg={stats['pg_loss']:>6.3f} vf={stats['vf_loss']:>6.3f} "
            f"ent={stats['ent_loss']:>6.3f} clip={stats['clip_frac']:>5.3f} | "
            f"fps={fps:>5.0f}"
        )

        # Checkpoint
        if env_steps >= next_save:
            ckpt_path = checkpoint_dir / f"agent_{task_id}_{env_steps}.pt"
            _save(agent, optimizer, ckpt_path, env_steps, task_id)
            next_save += cfg["save_every"]

    # Final checkpoint for this task
    final_path = checkpoint_dir / f"agent_{task_id}_final.pt"
    _save(agent, optimizer, final_path, env_steps, task_id)
    print(f"  [✓] {task_id} complete | mean last-10 ep reward: {np.mean(rollout_rewards[-10:]):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(agent, optimizer, path: Path, steps: int, task_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_mean":  agent._obs_mean,
        "obs_var":   agent._obs_var,
        "obs_count": agent._obs_count,
        "steps":     steps,
        "task_id":   task_id,
    }, path)
    print(f"  [✓] Saved checkpoint → {path}")


def _load(agent, optimizer, path: Path) -> Tuple[int, str]:
    ckpt = torch.load(path, map_location=agent.device)
    agent.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    agent._obs_mean  = ckpt.get("obs_mean")
    agent._obs_var   = ckpt.get("obs_var")
    agent._obs_count = ckpt.get("obs_count", 0)
    return ckpt.get("steps", 0), ckpt.get("task_id", "easy")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PPO training for Energy Grid Dispatch")
    p.add_argument("--tasks",    nargs="+", default=["easy", "medium", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--steps",    type=int,  default=None,
                   help="Override steps per task (same for all)")
    p.add_argument("--resume",   type=str,  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--lr",       type=float, default=DEFAULTS["lr"])
    p.add_argument("--hidden",   type=int,   default=256)
    p.add_argument("--device",   type=str,   default=None,
                   help="cpu / cuda / cuda:0")
    p.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = {**DEFAULTS}
    if args.lr:
        cfg["lr"] = args.lr

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build agent & optimizer
    agent = PPOAgent(
        obs_dim  = OBS_DIM,
        hidden   = args.hidden,
        device   = args.device,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)

    print(f"Device  : {agent.device}")
    print(f"Obs dim : {OBS_DIM}")
    print(f"Agent   : {sum(p.numel() for p in agent.parameters()):,} params")

    # Optionally resume
    resume_steps = 0
    resume_task  = args.tasks[0]
    if args.resume:
        resume_steps, resume_task = _load(agent, optimizer, Path(args.resume))
        print(f"Resumed from {args.resume} at {resume_steps:,} steps (task={resume_task})")

    # Curriculum
    task_steps = {
        "easy":   args.steps or cfg["steps_easy"],
        "medium": args.steps or cfg["steps_medium"],
        "hard":   args.steps or cfg["steps_hard"],
    }

    for task_id in args.tasks:
        # Skip already-completed tasks if resuming
        total = task_steps[task_id]
        rstart = resume_steps if task_id == resume_task else 0
        if rstart >= total:
            print(f"  [skip] {task_id} already complete ({rstart}/{total} steps)")
            continue

        train_task(
            agent         = agent,
            optimizer     = optimizer,
            task_id       = task_id,
            total_env_steps = total,
            cfg           = cfg,
            checkpoint_dir = ckpt_dir,
            resume_steps  = rstart,
        )

        # After easy, anneal learning rate slightly for harder tasks
        if task_id == "easy":
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            print(f"  [lr anneal] LR → {cfg['lr'] * 0.5:.2e}")
        elif task_id == "medium":
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            print(f"  [lr anneal] LR → {cfg['lr'] * 0.25:.2e}")

        # Reset EMA / obs stats between tasks for better transfer
        agent._obs_mean  = None
        agent._obs_var   = None
        agent._obs_count = 0

    print("\n[✓] Training complete.")


if __name__ == "__main__":
    main()
