"""
PPO Agent for Energy Grid Dispatch.

Handles structured dict observations (flattened internally),
outputs continuous actions for coal_delta, hydro_delta, nuclear_delta,
demand_response_mw, and discrete emergency_coal_boost.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any, List, Optional

# ── Observation key ordering (must be consistent across all calls) ──────────
# Only numerical scalar features; lists/strings excluded and handled separately.
OBS_KEYS = [
    "demand_mw", "hour", "day", "step",
    "coal_mw", "coal_online", "coal_max_mw", "coal_startup_remaining", "coal_price",
    "solar_mw", "wind_mw", "wind_speed_ms",
    "hydro_mw", "reservoir_mwh", "reservoir_capacity_mwh",
    "nuclear_mw", "nuclear_online", "nuclear_trip_remaining",
    "battery_mwh", "battery_capacity_mwh",
    "unmet_demand_mw", "frequency_hz", "load_shedding_mw",
    "spinning_reserve_mw", "spinning_reserve_required_mw",
    "capital_budget", "cumulative_cost", "cumulative_emissions_tons",
    "coal_health_pct", "duck_curve_stress_mw_per_step",
    "spot_price", "carbon_price_per_ton", "rate_of_change_hz_per_step",
    "voltage_stability_index",
    # LLM Strategist Inputs (for Hybrid Architecture)
    "llm_coal_delta", "llm_hydro_delta", "llm_nuclear_delta", "llm_dr_mw",
]

# Action bounds for clipping
ACTION_BOUNDS = {
    "coal_delta":          (-100.0, 100.0),
    "hydro_delta":         (-80.0,  80.0),
    "nuclear_delta":       (-10.0,  10.0),
    "demand_response_mw":  (0.0,    150.0),
}

# Blackout risk string → scalar (for potential use in reward shaping)
BLACKOUT_RISK_MAP = {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}


def flatten_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Convert observation dict to a fixed-length float32 numpy array.
    Missing keys default to 0.0. Bool fields are cast to float.
    """
    vec = []
    for key in OBS_KEYS:
        val = obs.get(key, 0.0)
        if isinstance(val, bool):
            vec.append(float(val))
        elif isinstance(val, (int, float)):
            vec.append(float(val))
        else:
            vec.append(0.0)

    # Encode blackout_risk as a scalar
    risk = obs.get("blackout_risk", "none")
    vec.append(BLACKOUT_RISK_MAP.get(risk, 0.0))

    # Solar weather as scalar: clear=1.0, partial=0.6, cloudy=0.3, storm=0.0
    weather_map = {"clear": 1.0, "partial": 0.6, "cloudy": 0.3, "storm": 0.0}
    vec.append(weather_map.get(obs.get("solar_weather", "clear"), 1.0))

    return np.array(vec, dtype=np.float32)


OBS_DIM = len(OBS_KEYS) + 2   # +2 for blackout_risk, solar_weather encodings
# Continuous actions: coal_delta, hydro_delta, nuclear_delta, demand_response_mw
CONT_ACT_DIM = 4


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """Build a simple MLP with LayerNorm for training stability."""
    mods: List[nn.Module] = []
    prev = in_dim
    for _ in range(layers):
        mods += [nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        prev = hidden
    mods.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*mods)


class PPOAgent(nn.Module):
    """
    Actor-Critic PPO agent.

    Actor outputs:
        - mean for 4 continuous actions (tanh-squashed to [-1, 1] then scaled)
        - log_std (learned, state-independent, per-action)
        - logit for emergency_coal_boost (Bernoulli)

    Critic outputs:
        - scalar state value
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden: int = 256,
        mlp_layers: int = 2,
        log_std_init: float = -0.5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Shared feature extractor (helps with sample efficiency)
        self.shared = _mlp(obs_dim, hidden, hidden, layers=mlp_layers)

        # Actor head — continuous actions
        self.actor_mean = nn.Linear(hidden, CONT_ACT_DIM)
        # State-independent log_std (one per action dim)
        self.log_std = nn.Parameter(torch.full((CONT_ACT_DIM,), log_std_init))

        # Actor head — discrete emergency boost (binary)
        self.boost_head = nn.Linear(hidden, 1)

        # Critic head
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal initialisation — standard PPO practice
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small gain for output layers
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.boost_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

        self.to(self.device)

    # ── Normalisation (running mean/var) ────────────────────────────────────
    # Simple running stats; updated during trajectory collection.

    def update_obs_stats(self, obs_batch: torch.Tensor) -> None:
        """Welford online update of observation statistics."""
        batch_mean = obs_batch.mean(0)
        batch_var  = obs_batch.var(0, unbiased=False)
        n = obs_batch.shape[0]
        if self._obs_mean is None:
            self._obs_mean = batch_mean
            self._obs_var  = batch_var + 1e-8
            self._obs_count = n
        else:
            total = self._obs_count + n
            delta = batch_mean - self._obs_mean
            self._obs_mean = self._obs_mean + delta * n / total
            m_a   = self._obs_var * self._obs_count
            m_b   = batch_var * n
            m_2   = m_a + m_b + delta ** 2 * self._obs_count * n / total
            self._obs_var  = m_2 / total + 1e-8
            self._obs_count = total

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self._obs_mean is None:
            return obs
        return (obs - self._obs_mean) / (self._obs_var.sqrt() + 1e-8)

    # ── Core forward ────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> Tuple[
        torch.distributions.Distribution,
        torch.distributions.Bernoulli,
        torch.Tensor,
    ]:
        """
        Returns:
            cont_dist  — MultivariateNormal (diagonal) for continuous actions
            boost_dist — Bernoulli for emergency_coal_boost
            value      — scalar state value
        """
        obs = self.normalize_obs(obs)
        feat = self.shared(obs)

        # Continuous policy
        mean    = self.actor_mean(feat)        # (B, 4), unconstrained
        std     = self.log_std.exp().expand_as(mean)
        cont_dist = torch.distributions.Normal(mean, std)

        # Boost policy
        boost_logit = self.boost_head(feat).squeeze(-1)
        boost_dist  = torch.distributions.Bernoulli(logits=boost_logit)

        # Value
        value = self.critic(feat).squeeze(-1)

        return cont_dist, boost_dist, value

    # ── Action sampling & conversion ────────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy given a raw numpy observation.

        Returns:
            action_dict  — ready for env.step()
            log_prob     — sum of log-probs across all action dims
            value        — critic estimate
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        cont_dist, boost_dist, value = self.forward(obs_t)

        if deterministic:
            cont_raw = cont_dist.mean
            boost    = (boost_dist.probs > 0.5).float()
        else:
            cont_raw = cont_dist.sample()
            boost    = boost_dist.sample()

        # Log-probs (sum over action dims for continuous)
        lp_cont  = cont_dist.log_prob(cont_raw).sum(-1)
        lp_boost = boost_dist.log_prob(boost)
        log_prob = lp_cont + lp_boost

        # Scale continuous actions from unconstrained → env range using tanh squash
        # tanh maps ℝ → (-1, 1); then scale to action bounds
        cont_scaled = torch.tanh(cont_raw)   # (1, 4)
        cont_np     = cont_scaled.squeeze(0).cpu().numpy()

        bounds_lo = np.array([-100.0, -80.0, -10.0, 0.0],   dtype=np.float32)
        bounds_hi = np.array([ 100.0,  80.0,  10.0, 150.0], dtype=np.float32)
        cont_clipped = (cont_np + 1.0) / 2.0 * (bounds_hi - bounds_lo) + bounds_lo

        action_dict = {
            "coal_delta":          float(cont_clipped[0]),
            "hydro_delta":         float(cont_clipped[1]),
            "nuclear_delta":       float(cont_clipped[2]),
            "demand_response_mw":  float(cont_clipped[3]),
            "emergency_coal_boost": bool(boost.item() > 0.5),
            # Fixed (dispatch agent doesn't plan)
            "battery_mode":  "idle",
            "plant_action":  "none",
        }

        return action_dict, log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self,
        obs_t: torch.Tensor,
        cont_raw: torch.Tensor,
        boost_t:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under current policy parameters.
        Used during PPO update.

        Returns:
            log_prob  — (B,)
            entropy   — (B,)
            value     — (B,)
        """
        cont_dist, boost_dist, value = self.forward(obs_t)

        lp_cont  = cont_dist.log_prob(cont_raw).sum(-1)
        lp_boost = boost_dist.log_prob(boost_t)
        log_prob = lp_cont + lp_boost

        entropy = cont_dist.entropy().sum(-1) + boost_dist.entropy()

        return log_prob, entropy, value


class HybridNegotiator:
    """
    Orchestrates the LLM Strategist + PPO Executor flow.
    Used in the 2-round negotiation protocol.
    """
    def __init__(self, ppo_agent: PPOAgent, llm_client: Any, model_name: str):
        _obs_mean: Optional[torch.Tensor] = None
        _obs_var:  Optional[torch.Tensor] = None
        _obs_count: int = 0
        
        self.ppo = ppo_agent
        self.llm_client = llm_client
        self.model = model_name

    def get_dispatch_revision(self, obs: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes Round 1 proposal (LLM) and produces Round 2 revision (PPO).
        """
        # Inject LLM proposal into observation for PPO
        obs["llm_coal_delta"] = proposal.get("coal_delta", 0.0)
        obs["llm_hydro_delta"] = proposal.get("hydro_delta", 0.0)
        obs["llm_nuclear_delta"] = proposal.get("nuclear_delta", 0.0)
        obs["llm_dr_mw"] = proposal.get("demand_response_mw", 0.0)
        
        obs_np = flatten_obs(obs)
        action_dict, _, _ = self.ppo.act(obs_np, deterministic=True)
        
        # Mark as revision
        action_dict["proposal_type"] = "revision"
        action_dict["thought"] = f"[PPO Executor] Refined LLM strategy for stability."
        
        return action_dict
