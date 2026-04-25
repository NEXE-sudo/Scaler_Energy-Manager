"""
Rollout buffer for PPO trajectory storage.

Stores one complete rollout (N steps across M episodes).
Computes GAE advantages on flush.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple


class RolloutBuffer:
    """
    Fixed-size circular buffer for PPO rollouts.

    Stores:
        obs        — flattened observation vectors
        cont_raw   — raw (pre-tanh) continuous action samples
        boosts     — binary emergency_coal_boost samples
        log_probs  — log π(a|s) at collection time
        rewards    — step rewards
        values     — critic estimates V(s)
        dones      — episode termination flags

    Call flush() after collection to compute returns & advantages (GAE).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        cont_act_dim: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity    = capacity
        self.obs_dim     = obs_dim
        self.cont_act_dim = cont_act_dim
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.device      = device or torch.device("cpu")

        self._ptr  = 0   # write pointer
        self._size = 0   # number of stored transitions

        # Pre-allocate on CPU; move batch to GPU during update
        self.obs       = np.zeros((capacity, obs_dim),       dtype=np.float32)
        self.cont_raw  = np.zeros((capacity, cont_act_dim),  dtype=np.float32)
        self.boosts    = np.zeros((capacity,),               dtype=np.float32)
        self.log_probs = np.zeros((capacity,),               dtype=np.float32)
        self.rewards   = np.zeros((capacity,),               dtype=np.float32)
        self.values    = np.zeros((capacity,),               dtype=np.float32)
        self.dones     = np.zeros((capacity,),               dtype=np.float32)

        # Computed after flush
        self.returns    = np.zeros((capacity,), dtype=np.float32)
        self.advantages = np.zeros((capacity,), dtype=np.float32)

    # ── Write ───────────────────────────────────────────────────────────────

    def add(
        self,
        obs:       np.ndarray,
        cont_raw:  np.ndarray,
        boost:     float,
        log_prob:  float,
        reward:    float,
        value:     float,
        done:      bool,
    ) -> None:
        idx = self._ptr % self.capacity
        self.obs[idx]       = obs
        self.cont_raw[idx]  = cont_raw
        self.boosts[idx]    = float(boost)
        self.log_probs[idx] = float(log_prob)
        self.rewards[idx]   = float(reward)
        self.values[idx]    = float(value)
        self.dones[idx]     = float(done)
        self._ptr  += 1
        self._size  = min(self._size + 1, self.capacity)

    # ── GAE ─────────────────────────────────────────────────────────────────

    def flush(self, last_value: float = 0.0) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        last_value — bootstrap value for incomplete episodes (V(s_{T+1})).
                     Set to 0.0 if the episode ended cleanly.
        """
        n = self._size
        gae = 0.0
        # Walk backwards through stored transitions
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    def normalise_advantages(self) -> None:
        """Normalise advantages to zero mean, unit variance."""
        n = self._size
        adv = self.advantages[:n]
        self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ── Iterate mini-batches ─────────────────────────────────────────────────

    def get_batches(self, batch_size: int):
        """
        Yield random mini-batches as tensors on self.device.
        Shuffles indices each call for SGD noise.
        """
        n = self._size
        indices = np.random.permutation(n)

        obs_t      = torch.tensor(self.obs[:n],       dtype=torch.float32)
        cont_raw_t = torch.tensor(self.cont_raw[:n],  dtype=torch.float32)
        boosts_t   = torch.tensor(self.boosts[:n],    dtype=torch.float32)
        lp_t       = torch.tensor(self.log_probs[:n], dtype=torch.float32)
        ret_t      = torch.tensor(self.returns[:n],   dtype=torch.float32)
        adv_t      = torch.tensor(self.advantages[:n],dtype=torch.float32)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                obs_t[idx].to(self.device),
                cont_raw_t[idx].to(self.device),
                boosts_t[idx].to(self.device),
                lp_t[idx].to(self.device),
                ret_t[idx].to(self.device),
                adv_t[idx].to(self.device),
            )

    def reset(self) -> None:
        self._ptr  = 0
        self._size = 0

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    @property
    def size(self) -> int:
        return self._size
