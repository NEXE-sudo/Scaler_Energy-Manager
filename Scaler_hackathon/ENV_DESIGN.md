# Environment Design Documentation

## Overview

This document explains the design decisions for the Energy Grid Management environment, focusing on four key improvements:

1. **Observation Normalization**
2. **Task-Specific Reward Weighting**
3. **Variable Stochasticity (Seed Override)**
4. **Action Space Scaling Justification**

These improvements enhance agent generalization and support diverse evaluation methodologies.

---

## 1. Observation Normalization

### Rationale

Raw observations span vastly different scales:
- Demand: 200–1100 MW
- Grid frequency: 48.5–50.5 Hz
- Battery: 0–200 MWh
- Prices: 20–200 $/MWh

Agents trained on unnormalized observations often:
- Struggle with feature scaling variance
- Learn brittle policies sensitive to absolute values
- Have longer training times
- Perform poorly on slight distribution shifts

### Implementation

**Location**: `server/normalization.py`

**Module**: `normalize_observation(obs_dict, task_id)` function

**Normalization Bounds** (per feature type):

| Feature | Min | Max | Physical Justification |
|---------|-----|-----|------------------------|
| Demand (MW) | 200 | 1100 | Spring min to winter peak |
| Generation (MW) | 0 | 1200 | Off to all plants max |
| Frequency (Hz) | 48.5 | 50.5 | Safe operational limits |
| Battery (MWh) | 0 | 200 | Typical capacity |
| Reservoir (MWh) | 0 | 1000 | Hydro reservoir max |
| Price ($/MWh) | 20 | 200 | Historical range |
| Unmet demand (MW) | 0 | 300 | Worst-case load shed |
| Emissions (tons) | 0 | 5000 | Task-dependent max |

### Usage

```python
from server.normalization import normalize_observation
from server.energy_grid_environment import EnergyGridEnvironment

env = EnergyGridEnvironment()
obs = env.reset("medium")

# Get normalized observation (all features in [0, 1])
obs_dict = obs.dict()
normalized = normalize_observation(obs_dict, task_id="medium")

# Agent now sees reliable [0, 1] signals
demand_norm = normalized["demand_mw"]  # e.g. 0.45 instead of 500 MW
freq_norm = normalized["grid_frequency"]  # e.g. 0.50 instead of 49.5 Hz
```

### Features NOT Normalized

Categorical and cyclic features are preserved as-is for agents to learn patterns:
- `time_of_day` (0–23): Agents learn daily seasonality directly
- `day` (1–3): Multi-day pattern learning
- `season`: Categorical string preserved
- `active_events`: List of event names
- `task_id`: Task identifier

### Denormalization

For debugging/visualization, reverse the transformation:

```python
from server.normalization import denormalize_observation

physical_units = denormalize_observation(normalized, task_id="medium")
# Restores original units: demand_mw, frequency_hz, etc.
```

---

## 2. Task-Specific Reward Weighting

### Rationale

Different tasks emphasize different objectives:

- **Easy**: Focus on reliable dispatch with cost efficiency
- **Medium**: Reliability under weather variability
- **Hard**: Multi-objective optimization (reliability, emissions, capital ROI)

Step-by-step reward components already reflect these priorities, but can be further tuned per task.

### Current Grader Weights

These weights determine **final episode scores** (see `server/tasks.py`):

#### Easy Task

```python
{
    "reliability": 0.60,        # Demand met (primary)
    "cost_efficiency": 0.40,    # Operational cost
    # Others: 0.0 (not evaluated)
}
```

**Interpretation**: Easy task is ~ 60% reliability, 40% cost. Agent must balance meeting demand against coal dispatch cost.

#### Medium Task

```python
{
    "reliability": 0.60,        # Demand met
    "cost_efficiency": 0.30,    # Operational cost
    "battery_health": 0.10,     # Final battery %
    # Others: 0.0
}
```

**Interpretation**: Reliability still primary. Cost slightly less important (renewable variability). Battery health matters (shows effective charging strategy).

#### Hard Task

```python
{
    "reliability": 0.40,           # Demand met (lower weight due to difficulty)
    "cost_efficiency": 0.20,       # Operational cost
    "emissions": 0.10,             # CO2 reduction
    "battery_health": 0.10,        # Final battery %
    "reservoir_management": 0.10,  # Hydro efficiency
    "capital_efficiency": 0.10,    # Did building plants pay off?
}
```

**Interpretation**: Multi-objective. Reliability weighted lower (full success unlikely). Capital efficiency highly valued (plants should have been worthwhile).

### Step-by-Step Reward Shaping (During Episode)

Actual per-step rewards (computed in `server/simulator.py:compute_reward()`) emphasize:

1. **Reliability** (primary penalty)
   - -0.25 MW per MW unmet demand
   - Shows immediate impact of dispatch decisions

2. **Frequency Stability**
   - -0.2 × frequency_error (Hz)
   - +0.2 bonus if error < 0.1 Hz

3. **Cost Discipline**
   - Encourages low coal output (fuel price dependent)
   - Penalizes unnecessary battery cycling

4. **Blackout Penalty** (catastrophic)
   - -500 per episode if unmet_demand > 50 MW for 3+ steps

### Extension: Custom Per-Task Step Reward Weights

To make step rewards reflect task emphasis, extend `compute_reward()`:

```python
# Example: Task-specific emphasis
TASK_REWARD_CONFIG = {
    "easy": {
        "unmet_demand_weight": 0.30,      # 30% of step reward from reliability
        "frequency_weight": 0.10,         # 10% from frequency
        "cost_weight": 0.60,              # 60% from cost efficiency
    },
    "medium": {
        "unmet_demand_weight": 0.40,      # Weather makes reliability harder
        "frequency_weight": 0.20,
        "cost_weight": 0.40,
    },
    "hard": {
        "unmet_demand_weight": 0.50,      # Hardest → reliability most important
        "emissions_weight": 0.20,         # Penalize coal-heavy solutions
        "frequency_weight": 0.15,
        "cost_weight": 0.15,
    },
}

# Usage in simulator:
config = TASK_REWARD_CONFIG[task_id]
unmet_penalty = config["unmet_demand_weight"] * unmet
freq_penalty = config["frequency_weight"] * freq_error
cost_penalty = config["cost_weight"] * coal_cost
```

This makes step rewards align with grader weights, giving agents consistent guidance.

---

## 3. Variable Stochasticity (Seed Override)

### Rationale

Fixed seeds ensure reproducibility, but:
- Agents may overfit to deterministic event schedules
- Real-world grid operation requires robustness to weather variance
- Evaluation should test generalization beyond the canonical seed

### Implementation

**Updated**: `server/energy_grid_environment.py:reset()`

**New parameter**: `seed: int = None`

```python
def reset(self, task_id: str = "easy", seed: int = None) -> EnergyGridObservation:
    """
    Reset environment.
    
    Args:
        task_id: Task ID (easy/medium/hard)
        seed: Optional RNG seed. If None, uses task default.
              Override to generate episode variants with same task 
              parameters but different stochastic weather/events.
    """
    episode_seed = seed if seed is not None else task["seed"]
    self._sim = build_initial_state(..., seed=episode_seed, ...)
```

### Usage Examples

```python
env = EnergyGridEnvironment()

# Standard reproducible episode (task default seed)
obs = env.reset("medium")

# Variant of medium task (different weather)
obs = env.reset("medium", seed=42)
obs = env.reset("medium", seed=100)  # Another variant

# Sweep over seed range for robustness testing
scores = []
for seed in range(200, 210):
    obs = env.reset("hard", seed=seed)
    # Run episode...
    scores.append(episode_score)

avg_robustness = sum(scores) / len(scores)
```

### Impact on Evaluation

- **Canonical runs**: Use `seed=None` → Reproduces baseline scores (0.18/0.23/0.33)
- **Robustness eval**: Sweep 10+ seeds → Measures generalization
- **Cross-validation**: Train on seeds [x, x+10), eval on [x+100, x+110)

---

## 4. Action Space Scaling Justification

### Rationale

Action limits (coal ±100 MW, hydro ±80 MW, nuclear ±10 MW) are not arbitrary—they reflect:
- Physical ramp-rate constraints
- Realistic grid operator training
- Pedagogical difficulty (slow nuclear teaches long-horizon planning)

### Scaling Justification

#### Coal Delta: ±100 MW

**Physical basis**:
- Real steam generators ramp at ~100 MW/min (Gen IV plants)
- Timestep = 1 hour = 60 minutes
- Achievable ramp: ~100 MW/step (hour) is realistic limit
- Realistic operators cannot ramp faster without damaging turbines

**What agents learn**:
- Cannot instantaneously replace lost solar (must plan ahead)
- Gradual coal ramp-down needed to avoid blackouts
- Balanced dispatch requires anticipation

**Example**: During 200 MW solar loss, agent cannot instantly add 200 MW coal (±100 limit). Must:
- Ramp +100 MW coal
- Discharge 50 MW battery
- Shed 50 MW load (emergency)
- Or accept blackout if unprepared

#### Hydro Delta: ±80 MW

**Physical basis**:
- Pump-hydro response faster than coal (minutes, not hours)
- Typical pump-hydro: 80–150 MW ramp
- ±80 MW chosen as conservative, realistic for medium plant

**What agents learn**:
- Hydro better than coal for fast response
- Depletes/replenishes reservoir over episodes
- Reservoir management becomes strategic
- Agents learn "save hydro for peaks" pattern

#### Nuclear Delta: ±10 MW

**Physical basis**:
- Nuclear plants designed for stable baseload
- Real operators rarely adjust >10 MW/hour
- Emergency ramps risk SCRAM event (physics failure)
- Intentionally constrained to teach long-horizon planning

**What agents learn**:
- Nuclear is slow-moving baseline (set and forget)
- Must plan nuclear build 15+ steps ahead
- Cannot use nuclear for fast load balancing
- Creates tension: coal is flexible but dirty, nuclear is clean but slow

### Command Summary Table

| Source | Delta Range | Response Time | Use Case | Agent Lesson |
|--------|-------------|---------------|----------|--------------|
| **Coal** | ±100 MW | 1 hour | Flexible load following | Plan ahead for ramps |
| **Hydro** | ±80 MW | 10 minutes | Peak shaving & reserve | Manage reservoir pressure |
| **Nuclear** | ±10 MW | 10 hours | Baseload foundation | Build early or regret |
| **Battery** | 50 MW mode | Instant | Emergency buffer | Use wisely, degrades |
| **Solar/Wind** | N/A (weather driven) | — | Forecast variability | Plan for lulls |

### Observation: Physical Realism Improves Training

Agents trained on realistic ramp limits learn:
1. **Hierarchical dispatch** (nuclear base → coal middle → hydro peak → battery emergency)
2. **Predictive planning** (build plants, avoid stranded assets)
3. **Risk management** (buffer demand response for emergencies)
4. **Operational discipline** (respect ramp rates, avoid oscillation)

Unrealistically fast limits (e.g., "coal ±500 MW") allow trivial solutions and don't generalize.

---

## Integration Example

```python
from server.energy_grid_environment import EnergyGridEnvironment
from server.normalization import normalize_observation
from server.models import EnergyGridAction

# Initialize
env = EnergyGridEnvironment()

# Reset with optional seed variant (e.g., robustness test)
obs = env.reset("hard", seed=42)

# Get normalized observation
obs_dict = obs.dict()
obs_norm = normalize_observation(obs_dict, task_id="hard")

# Agent makes decision based on:
# - Normalized [0, 1] features (reliable scaling)
# - Realistic action limits (±100 coal, ±10 nuclear)
# - Task-specific grader weights (hint: reliability critical in hard)
agent_action = agent.predict(obs_norm)

# Execute action
obs = env.step(EnergyGridAction(**agent_action))

# After episode:
score = env.get_last_grade()
print(f"Hard task score: {score.total_score:.3f}")
# Grader emphasizes capital_efficiency (10%) + emissions (10%)
# Step rewards emphasized reliability + cost
# Agent learned task-specific behavior
```

---

## Performance Impact

These improvements are **backward compatible**:
- Existing code still works (defaults to task seeds)
- Normalization is optional (agents can use raw observations)
- Grader weights unchanged (backward compatible)
- Action limits unchanged (maintains benchmark alignment)

**Expected benefit**:
- Better generalization (normalization)
- Clearer agent guidance (aligned rewards)
- Robustness testing possible (seed override)
- Physic-based learning (action scaling rationale)

---

## References

- `server/normalization.py` — Normalization utilities
- `server/tasks.py` — Grader weights per task
- `server/energy_grid_environment.py` — Environment API with seed override
- `server/simulator.py` — Per-step reward computation
- `models.py` — Action space definition with scaling justification
