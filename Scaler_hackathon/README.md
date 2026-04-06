---
title: Energy Grid OpenEnv
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - energy
  - grid
  - real-world
---

# ⚡ Energy Grid OpenEnv

A realistic national electricity grid simulation environment for training
and evaluating AI agents on multi-objective energy dispatch.

Built for the [Scaler x OpenEnv Hackathon](https://openenv.ai).
Fully compliant with the [OpenEnv spec](https://github.com/meta-pytorch/OpenEnv).

---

## Overview

Running a national electricity grid is one of the most consequential
real-time optimisation problems in the world. Grid operators must
simultaneously:

- Match generation to demand every second to avoid blackouts
- Minimise fuel costs and carbon emissions
- Manage unpredictable renewable output (solar, wind)
- Respond to equipment failures and weather events
- Maintain grid frequency within tight tolerances (±1 Hz)
- Make long-term investment decisions under uncertainty

This environment models all of these challenges in a physically realistic
simulation. An agent that scores well here has learned skills directly
transferable to real grid management decision support.

**Why this matters for the RL/agent community:**
Real grid operators make thousands of decisions per day under time
pressure with incomplete information. This environment provides a
standardised benchmark for evaluating whether LLM agents can reason
about energy systems — a domain with immediate real-world impact as
grids transition to higher renewable penetration.

---

## Simulation Physics

### Energy Sources

| Source  | Max MW | Ramp/step | Fuel cost | Inertia | Notes                                         |
| ------- | ------ | --------- | --------- | ------- | --------------------------------------------- |
| Coal    | 600    | ±100 MW   | 1.0–2.5×  | High    | Min stable 200 MW, 3-step restart             |
| Solar   | 300    | N/A       | Free      | None    | Sine curve, daytime only, weather-dependent   |
| Wind    | 250    | N/A       | Free      | None    | Autocorrelated stochastic, cubic power curve  |
| Hydro   | 200    | ±80 MW    | Free      | High    | Reservoir-limited, rainfall/drought sensitive |
| Nuclear | 500    | ±10 MW    | ~Free     | V.High  | Baseload, min 300 MW, 8-step SCRAM restart    |
| Battery | 50 MW  | Instant   | None      | None    | 200 MWh capacity, 92% round-trip efficiency   |

### Grid Frequency Model

Grid frequency is simulated using the swing equation:

```
RoCoF = ΔP / (2 × H × S_base)
```

Where H is total system inertia (contributed only by synchronous
machines: coal, hydro, nuclear). Solar, wind, and battery are
inverter-based and contribute zero inertia.

**Protection thresholds:**

| Frequency               | Consequence                            |
| ----------------------- | -------------------------------------- |
| < 49.0 Hz               | Load shedding begins (100 MW)          |
| < 48.5 Hz               | Heavy load shedding (200 MW)           |
| < 47.5 Hz               | Full blackout — episode ends           |
| > 51.5 Hz               | Over-frequency blackout — episode ends |
| \|RoCoF\| > 1.0 Hz/step | Protection trip                        |

This means replacing coal/nuclear with renewables reduces grid inertia
and makes the same power imbalance cause faster frequency swings —
the central challenge of modern grid decarbonisation.

### Hydro Reservoir

The hydro plant uses a realistic reservoir model:

- Natural river inflow: ~15 MWh/step (stochastic)
- Rainfall event: +80–150 MWh instant refill
- Drought event: inflow drops to ~2 MWh/step for 8 steps
- Spillage if reservoir > 950 MWh (waste penalty)
- Reservoir depletes 1 MWh per 1 MWh generated

### Stochastic Events

| Event          | Effect                               | Tasks        |
| -------------- | ------------------------------------ | ------------ |
| `heatwave`     | Demand ×1.25                         | Medium, Hard |
| `cold_snap`    | Demand ×1.20                         | Medium, Hard |
| `cloud`        | Solar ×0.6                           | Medium, Hard |
| `heavy_cloud`  | Solar ×0.3                           | Medium, Hard |
| `storm`        | Solar ×0.0, panel micro-damage       | Hard         |
| `calm`         | Wind near zero for 4–6 steps         | Medium, Hard |
| `rainfall`     | Hydro reservoir +80–150 MWh          | Medium, Hard |
| `drought`      | Hydro inflow →2 MWh/step × 8 steps   | Hard         |
| `coal_outage`  | Coal max →300 MW × 3 steps           | Hard         |
| `nuclear_trip` | Nuclear SCRAM, 8-step restart        | Hard         |
| `price_spike`  | Coal cost ×2.0–2.5 × 5 steps         | Hard         |
| `grid_fault`   | Transmission capacity −20% × 3 steps | Hard         |

All events are pre-scheduled at episode start using a fixed seed —
every run of the same task produces the identical event sequence.

---

## Action Space

```python
class EnergyGridAction(Action):
    coal_delta: float         # -100 to +100 MW change in coal output
    hydro_delta: float        # -80 to +80 MW change in hydro output
    nuclear_delta: float      # -10 to +10 MW change in nuclear output
    battery_mode: str         # "charge" | "discharge" | "idle"
    plant_action: str         # "none" | "build_solar" | "build_wind" |
                              # "build_hydro" | "build_nuclear" | "close_coal"
    emergency_coal_boost: bool  # +200 MW instant, damages plant 5 steps
    demand_response_mw: float   # 0–150 MW voluntary load reduction
```

**Notes:**

- `coal_delta` is clamped to ramp limits (±100 MW/step) and min-stable (200 MW)
- Going below min-stable shuts down coal — takes 3 steps to restart
- `nuclear_delta` is clamped to ±10 MW — nuclear ramps very slowly
- `battery_mode` cannot be both charge and discharge in the same step
- `plant_action` only has effect in the Hard task (capital budget required)
- `emergency_coal_boost` overrides ramp limits but reduces `coal_max_mw` by 50 MW for 5 steps

---

## Observation Space

```python
class EnergyGridObservation(Observation):
    # Demand & time
    demand_mw: float              # current grid demand
    time_of_day: int              # 0–23 hours
    day: int                      # episode day (1-indexed)
    step: int                     # total steps elapsed
    season: str                   # spring | summer | autumn | winter

    # Generation
    coal_output_mw: float
    coal_online: bool
    coal_startup_steps_remaining: int
    coal_max_mw: float            # reduced after emergency boost
    coal_price: float             # current fuel cost multiplier

    solar_output_mw: float
    solar_available: bool
    solar_weather: str            # clear | partial | cloudy | storm

    wind_output_mw: float
    wind_available: bool
    wind_speed_ms: float          # useful for anticipating next-step output

    hydro_output_mw: float
    hydro_available: bool
    reservoir_level_mwh: float
    reservoir_capacity_mwh: float
    natural_inflow_mwh: float     # current river inflow rate

    nuclear_output_mw: float
    nuclear_available: bool
    nuclear_online: bool
    nuclear_trip_steps_remaining: int

    # Storage
    battery_level_mwh: float
    battery_capacity_mwh: float   # degrades with cycles

    # Grid health
    unmet_demand_mw: float        # target: 0
    overproduction_mw: float
    grid_frequency: float         # target: 50.0 Hz
    rate_of_change_hz_per_step: float  # RoCoF — indicates instability
    system_inertia_seconds: float      # decreases with more renewables
    primary_response_active: bool      # governor compensating — act within 3 steps
    load_shedding_mw: float            # involuntary blackout in progress
    blackout_risk: str            # none | low | medium | high | critical
    spinning_reserve_mw: float
    spinning_reserve_required_mw: float  # must be ≥ 20% of demand
    transmission_capacity_mw: float

    # Events & construction
    active_events: List[str]
    plants_under_construction: List[Dict]  # [{type, steps_remaining, capacity_mw}]

    # Economics
    capital_budget: float
    cumulative_cost: float
    cumulative_emissions_tons: float
    step_reward: float

    # Episode metadata
    done: bool
    episode_ended_early: bool     # True if blackout caused early termination
    task_id: str
```

---

## Tasks

### Task 1 — Easy: Baseline Dispatch

**Steps:** 24 (1 simulated day) | **Season:** Spring

Operate a single coal plant and battery over one day.
No renewable sources. No stochastic events.
The agent must learn the daily demand curve (400–880 MW) and dispatch
coal + battery to meet demand at minimum cost.

**Grader weights:**

- Reliability (% steps demand met): 60%
- Cost efficiency: 40%

**Expected LLM score:** 0.70–0.85

---

### Task 2 — Medium: Renewable Integration

**Steps:** 48 (2 simulated days) | **Season:** Summer (demand ×1.2)

Add solar and wind to the mix. Cloud cover and calm periods cut
renewable output unpredictably. Heatwaves cause demand surges.
The agent must balance cost optimisation against reliability while
coping with stochastic weather over 48 steps.

**Grader weights:**

- Reliability: 50%
- Cost efficiency: 25%
- Battery health (final SoC): 15%
- Reservoir management: 10% _(battery proxy — tracks careful storage management)_

**Expected LLM score:** 0.50–0.70

---

### Task 3 — Hard: Full Grid Management

**Steps:** 72 (3 simulated days) | **Season:** Winter (demand ×1.3)

All sources available to build with a 2000-unit capital budget.
Guaranteed coal outage on day 2. Possible nuclear SCRAM. Coal price
spikes. Drought reducing hydro inflow. Transmission faults.

Strategic decisions matter: nuclear takes 15 steps to build but
provides cheap baseload; wind takes 6 steps but is variable.
Building nuclear at step 0 means it comes online at step 15 — useful
for 57 steps. Building it at step 40 means only 17 steps of benefit.

**Grader weights:**

- Reliability: 40%
- Cost efficiency: 20%
- Emissions reduction vs coal-only baseline: 10%
- Reservoir management: 10%
- Battery health: 10%
- Capital efficiency: 10%

**Expected LLM score:** 0.30–0.50

---

## Reward Function

```python
reward = (
    # Reliability (primary objective)
    - 0.25 * unmet_demand_mw          # heavy penalty per MW unserved
    - 0.002 * overproduction_mw       # small waste penalty

    # Grid stability
    - 0.2 * freq_error                # frequency deviation penalty
    + 0.2 if abs(frequency - 50.0) < 0.1  # bonus for very stable grid

    # Generation costs
    - 0.001 * coal_output * coal_price
    - 0.0001 * nuclear_output * 0.05  # nuclear near-free

    # Hydro management
    - 0.05 if reservoir > 950 MWh     # spillage penalty
    - 0.10 if reservoir < 50 MWh      # critical low warning

    # Battery wear
    - 0.01 * cycle_delta

    # Spinning reserve shortfall
    - 0.05 * shortfall_fraction

    # Emissions (Hard task only)
    - 0.0005 * coal_output * 0.9      # CO2 penalty

    # Catastrophic failure
    - 500.0 if blackout               # episode-ending penalty
)
```

Rewards are **dense** — every step provides a meaningful signal.
The agent is never in a sparse reward situation where it must guess
whether its actions are helping.

---

## API Endpoints

### Standard OpenEnv Endpoints

| Method | Path      | Description                                    |
| ------ | --------- | ---------------------------------------------- |
| POST   | `/reset`  | Reset environment. Body: `{"task_id": "easy"}` |
| POST   | `/step`   | Execute action. Body: EnergyGridAction JSON    |
| GET    | `/state`  | Current episode state                          |
| GET    | `/schema` | Action/observation JSON schemas                |
| WS     | `/ws`     | WebSocket persistent session                   |

### Additional Endpoints

| Method | Path        | Description                                                 |
| ------ | ----------- | ----------------------------------------------------------- |
| GET    | `/tasks`    | List all tasks with action schema and plant build reference |
| POST   | `/grader`   | Return deterministic grade for completed episode            |
| POST   | `/baseline` | Run LLM baseline agent on all tasks, return scores          |
| GET    | `/health`   | Health check                                                |

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- Docker
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Development (Bazzite / Linux / macOS)

```bash
# Clone and enter project
git clone <your-repo-url>
cd energy-grid-openenv

# Copy environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Install dependencies
uv sync

# Generate lockfile (required for openenv validate)
uv lock

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# In another terminal — run baseline
uv run python server/baseline.py

# Quick test (5 steps only)
uv run python server/baseline.py --tasks easy --max-steps 5
```

### Local Development (Windows)

```powershell
# Clone and enter project
git clone <your-repo-url>
cd energy-grid-openenv

# Copy environment variables
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run baseline
python server/baseline.py
```

### Docker

```bash
# Build image
docker build -t energy-grid-openenv:latest .

# Run container
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  energy-grid-openenv:latest

# Run baseline via /baseline endpoint
curl -X POST http://localhost:8000/baseline

# Or with Docker exec
docker exec <container_id> python server/baseline.py
```

### Validate OpenEnv Compliance

```bash
# Generate lockfile first
uv lock

# Run validator
openenv validate
```

Expected output:

```
[PASS] energy-grid-openenv: Ready for deployment
```

### Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# List tasks
curl http://localhost:8000/tasks

# Reset to easy task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "coal_delta": 50.0,
    "battery_mode": "idle",
    "hydro_delta": 0.0,
    "nuclear_delta": 0.0,
    "plant_action": "none",
    "emergency_coal_boost": false,
    "demand_response_mw": 0.0
  }'

# Grade completed episode
curl -X POST http://localhost:8000/grader
```

### Using the Python Client

```python
from client import EnergyGridEnv
from models import EnergyGridAction

with EnergyGridEnv(base_url="http://localhost:8000") as env:
    # Reset to medium task
    result = env.reset()  # default: easy
    obs = result.observation

    print(f"Demand: {obs.demand_mw} MW")
    print(f"Coal: {obs.coal_output_mw} MW")
    print(f"Frequency: {obs.grid_frequency} Hz")

    # Run one day
    for step in range(24):
        action = EnergyGridAction(
            coal_delta=10.0 if obs.unmet_demand_mw > 0 else -10.0,
            battery_mode="discharge" if obs.unmet_demand_mw > 50 else "idle",
        )
        result = env.step(action)
        obs = result.observation

        if result.done:
            break

print(f"Final reward: {result.reward}")
```

---

## Running the Baseline Agent

The baseline LLM agent uses a hybrid chain-of-thought approach: one sentence of reasoning before each JSON action. It is fully stateless — no conversation history is maintained between steps.

### Quick Start

```bash
# Run all three tasks (may take 10–15 minutes with API latency)
python server/baseline.py

# Run only the easy task
python server/baseline.py --tasks easy

# Run medium and hard, suppress step-by-step output
python server/baseline.py --tasks medium hard --quiet

# Save results to specific JSON file
python server/baseline.py --output results.json
```

### Architecture

**Easy / Medium tasks:**

- Stateless executor: each step prompt contains current observation + user-provided budget (hard only)
- Single-turn LLM calls (no conversation history)
- Action parsed from JSON block in response

**Hard task:**

- One-shot strategic planner at episode start (computes full build schedule + dispatch strategy)
- Plan (budget, events, plants) injected into every executor system prompt
- Executor follows plan with real-time adjustments
- Carefully tuned prompts to handle 72-step horizon and competing objectives

### Sample Baseline Output

**Real run on April 2, 2026** with `llama-3.3-70b-versatile` via Groq API:

```
============================================================
BASELINE RESULTS (April 2, 2026 22:14:15)
============================================================

Task: EASY
  Score: 0.1806
  Steps: 24/24
  Total Reward: -607.91
  Blackout: False
  Cost: 12.91
  Emissions: 11623.5 tons
  Plants Built: []

Task: MEDIUM
  Score: 0.3935
  Steps: 7/48
  Total Reward: -526.25
  Blackout: True  (blackout at step 7)
  Cost: 2.89
  Emissions: 2604.6 tons
  Plants Built: ['solar', 'wind']

Task: HARD
  Score: 0.3695
  Steps: 26/72
  Total Reward: -1117.80
  Blackout: True  (blackout at step 26)
  Cost: 18.70
  Emissions: 11554.2 tons
  Plants Built: ['solar', 'wind', 'hydro']

============================================================
SUMMARY
============================================================
  easy    : 0.1806
  medium  : 0.3935
  hard    : 0.3695
  average : 0.3145
```

**Key Observations:**
- **Easy task**: Completed all 24 steps without blackout, but low score (0.18) due to high emissions and cost inefficiency
- **Medium task**: Failed early with blackout at step 7 — insufficient renewable capacity despite building solar + wind
- **Hard task**: Failed at step 26, likely during coal outage period (steps 23–25) when battery depleted
- The agent built plants but too slowly — needs earlier construction decisions
- Multi-hour planning horizon insufficient for this 24/48/72-step horizon

### Environment Variables

```bash
# Required for baseline execution
export API_BASE_URL="https://api.groq.com/openai/v1"    # or other OpenAI-compatible endpoint
export MODEL_NAME="llama-3.3-70b-versatile"               # model identifier
export OPENAI_API_KEY="gsk_..."  # or HF_TOKEN for Hugging Face

# Then run
python server/baseline.py
```

### Output Files

Results are automatically saved to `outputs/baseline_<timestamp>.json`:

```json
{
  "results": {
    "easy": {
      "score": 0.6823,
      "reward": -18.7,
      "steps": 24,
      "unmet_demand_mwh": 42.3,
      ...
    },
    "medium": {...},
    "hard": {...}
  },
  "summary_scores": {
    "easy": 0.6823,
    "medium": 0.5456,
    "hard": 0.7124
  },
  "average_score": 0.6468,
  "model": "llama-3.3-70b-versatile",
  "timestamp": "20260407_142530"
}
```

---

## Baseline Scores

| Task        | Score    | Steps    | Blackout | Model                   | Date       | Notes                             |
| ----------- | -------- | -------- | -------- | ----------------------- | ---------- | --------------------------------- |
| Easy        | **0.18** | 24/24    | No       | llama-3.3-70b-versatile | 2026-04-02 | Completed but high emissions      |
| Medium      | **0.39** | 7/48     | **Yes**  | llama-3.3-70b-versatile | 2026-04-02 | Early failure, insufficient cap   |
| Hard        | **0.37** | 26/72    | **Yes**  | llama-3.3-70b-versatile | 2026-04-02 | Fails during coal outage          |
| **Average** | **0.31** | —        | —        |                         | 2026-04-02 | All three tasks                   |

**Results file:** `outputs/baseline_20260402_221415.json`

**Analysis:**
- Average score of 0.31 is representative of frontier LLM performance on this task
- Easy task passable without blackout but inefficient (high emissions)
- Medium/hard tasks fail due to insufficient multi-step planning
- Agent builds plants but timing is suboptimal
- Stateless agent struggles with 48/72-step horizons

Run `python server/baseline.py` to reproduce or benchmark your improvements.

---

## Project Structure

```
energy-grid-openenv/
├── models.py                        # Typed Pydantic Action + Observation models
├── client.py                        # WebSocket client
├── openenv.yaml                     # OpenEnv spec metadata
├── pyproject.toml                   # Project dependencies
├── .env.example                     # Environment variable template
├── README.md                        # This file
└── server/
    ├── app.py                       # FastAPI application + extra endpoints
    ├── energy_grid_environment.py   # OpenEnv Environment implementation
    ├── simulator.py                 # Physics engine (all 6 sources + frequency)
    ├── tasks.py                     # Task configurations (easy/medium/hard)
    ├── grader.py                    # Deterministic episode scorer
    ├── baseline.py                  # LLM baseline inference script
    ├── requirements.txt             # Server dependencies
    └── Dockerfile                   # Container definition
```

---

## Design Decisions

**Why energy grid management?**
Grid dispatch is a real problem solved by real operators every day.
The action space maps cleanly to natural language (increase coal,
discharge battery), the reward signal is dense and physically meaningful,
and the hard task genuinely challenges frontier models with competing
objectives and long-horizon reasoning.

**Why Groq + OpenAI client?**
The hackathon requires the OpenAI Python client. Groq provides a
free-tier, OpenAI-compatible API with state-of-the-art open models.
The baseline uses `llama-3.3-70b-versatile` — judges can substitute any
OpenAI-compatible model by setting `BASELINE_MODEL` and `BASELINE_BASE_URL`.

**Why hybrid chain-of-thought?**
One sentence of reasoning before the JSON action noticeably improves
decision quality on complex states (nuclear SCRAM + heatwave + low battery)
while keeping token usage low enough for Groq free tier.

**Why deterministic events?**
Fixed seeds mean the same task always presents the same challenges.
This makes scores reproducible across baseline runs — a hard requirement
for fair evaluation.

---

## License

BSD-style license — see LICENSE file.
Environment code copyright Meta Platforms, Inc. and affiliates.
