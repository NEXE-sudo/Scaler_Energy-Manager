"""
FastAPI application — Multi-Agent Energy Grid Environment.

New endpoints (multi-agent):
    POST /step/planning   — receive PlanningAgentAction, buffer
    POST /step/dispatch   — receive DispatchAgentAction, buffer
    POST /step/market     — receive MarketAgentAction, complete step

Original endpoints (unchanged, backward compatible):
    POST /reset
    POST /step            — single-agent unified action
    GET  /state
    GET  /schema
    WS   /ws

Additional endpoints (unchanged):
    GET  /tasks
    POST /grader
    POST /baseline
    GET  /health
"""

from __future__ import annotations

import gradio as gr

import asyncio
import json
import os
import traceback
import warnings
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from .energy_grid_environment import EnergyGridEnvironment
    from .tasks import get_tasks_summary, PLANT_BUILD_REFERENCE
    from .grader import grade_result_to_dict
    from ..models import (
        EnergyGridAction,
        EnergyGridObservation,
        PlanningAgentAction,
        DispatchAgentAction,
        MarketAgentAction,
    )
except (ImportError, ModuleNotFoundError, ValueError):
    from server.energy_grid_environment import EnergyGridEnvironment
    from server.tasks import get_tasks_summary, PLANT_BUILD_REFERENCE
    from server.grader import grade_result_to_dict
    from models import (
        EnergyGridAction,
        EnergyGridObservation,
        PlanningAgentAction,
        DispatchAgentAction,
        MarketAgentAction,
    )


# ---------------------------------------------------------------------------
# Base OpenEnv app
# ---------------------------------------------------------------------------

app: FastAPI = create_app(
    EnergyGridEnvironment,
    EnergyGridAction,
    EnergyGridObservation,
    env_name="energy-grid-openenv",
    max_concurrent_envs=4,
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",      # Frontend dev server
        "http://127.0.0.1:8080",
        "http://localhost:3000",       # Alternative dev port
        "http://127.0.0.1:3000",
        "http://localhost:5173",       # Vite default
        "http://127.0.0.1:5173",
        "*",                           # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class GraderRequest(BaseModel):
    task_id: Optional[str] = None


class BaselineRequest(BaseModel):
    tasks: list[str] = []


# ---------------------------------------------------------------------------
# Shared HTTP environment instance
# ---------------------------------------------------------------------------

_http_env: Optional[EnergyGridEnvironment] = None

def get_http_env() -> EnergyGridEnvironment:
    global _http_env
    if _http_env is None:
        _http_env = EnergyGridEnvironment()
        workers = os.getenv("WEB_CONCURRENCY")
        if workers and int(workers) > 1:
            warnings.warn(
                "_http_env is process-local. With multiple uvicorn workers, "
                "/grader results may not match the session that ran /reset. "
                "Use a single worker or WebSocket sessions for reliable grading."
            )
    return _http_env

# ---------------------------------------------------------------------------
# Original endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.get("/reset", include_in_schema=False)
async def reset_ping() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.post("/reset", tags=["OpenEnv Extensions"])
async def reset_episode(request: ResetRequest = ResetRequest()) -> JSONResponse:
    """
    Reset the environment to start a new episode.
    
    Request body:
        {"task_id": "easy|medium|hard"}
    
    Returns the initial observation for the task.
    """
    try:
        env = get_http_env()
        obs = env.reset(request.task_id)
        return JSONResponse(content=obs.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to reset environment for task {request.task_id}: {str(e)}"
        )


@app.get("/state", tags=["OpenEnv Extensions"])
async def get_state() -> JSONResponse:
    """Get current environment state/observation."""
    try:
        env = get_http_env()
        if env._sim is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call POST /reset first."
            )
        obs = env.get_observation()
        return JSONResponse(content=obs.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema", tags=["OpenEnv Extensions"])
async def get_schema() -> JSONResponse:
    """Get environment schema/metadata."""
    try:
        env = get_http_env()
        return JSONResponse(content={
            "action_space": "EnergyGridAction",
            "observation_space": "EnergyGridObservation",
            "tasks": ["easy", "medium", "hard"],
            "multi_agent": True,
            "agents": ["planning", "dispatch", "market"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", tags=["OpenEnv Extensions"])
async def get_tasks() -> JSONResponse:
    return JSONResponse(content={
        "tasks": get_tasks_summary(),
        "plant_build_reference": PLANT_BUILD_REFERENCE,
        "multi_agent_endpoints": {
            "planning": "POST /step/planning — PlanningAgentAction",
            "dispatch": "POST /step/dispatch — DispatchAgentAction",
            "market": "POST /step/market — MarketAgentAction",
            "note": (
                "Step advances only when all 3 agents submit. "
                "Use POST /step for single-agent backward-compatible mode."
            ),
        },
    })


@app.post("/grader", tags=["OpenEnv Extensions"])
async def grade_episode(request: GraderRequest = GraderRequest()) -> JSONResponse:
    env = get_http_env()
    if request.task_id is not None and request.task_id != env.current_task_id:
        raise HTTPException(status_code=400, detail="task_id does not match active episode")

    grade = env.get_last_grade() or env.grade_current_episode()
    if grade is None:
        raise HTTPException(status_code=404, detail="No episode data. Call POST /reset first.")

    return JSONResponse(content=grade)

@app.post("/train")
async def train_model():
    import subprocess

    try:
        result = subprocess.run(
            ["python", "train_llm.py", "--max-steps", "100"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )

        return {
            "status": "completed",
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/baseline", tags=["OpenEnv Extensions"])
async def run_baseline(request: BaselineRequest = BaselineRequest()) -> JSONResponse:
    try:
        try:
            from .baseline import run_baseline_agent
        except ImportError:
            from server.baseline import run_baseline_agent

        task_ids = request.tasks if request.tasks else ["easy", "medium", "hard"]
        valid_tasks = {"easy", "medium", "hard"}
        invalid = [t for t in task_ids if t not in valid_tasks]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid task IDs: {invalid}")

        results = await asyncio.to_thread(run_baseline_agent, task_ids=task_ids)
        return JSONResponse(content={
            "status": "completed",
            "baseline_scores": results,
            "model": results.get("model", "unknown"),
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


@app.get("/health", tags=["Infrastructure"])
async def health_check() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "environment": "energy-grid-openenv",
        "openenv_spec_version": 1,
        "multi_agent": True,
    })


@app.get("/")
async def root():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Multi-agent endpoints (new)
# ---------------------------------------------------------------------------

@app.post(
    "/step/planning",
    summary="Planning agent action",
    description=(
        "Submit a PlanningAgentAction. Buffered until dispatch and market "
        "agents also submit. Returns current observation (simulator not yet advanced). "
        "Step advances only when all three agents have submitted."
    ),
    tags=["Multi-Agent"],
)
async def step_planning(action: PlanningAgentAction) -> JSONResponse:
    """
    Receive planning agent action.

    Example body:
        {"plant_action": "build_nuclear", "target_step": 15, "rationale": "baseload before coal outage"}
    """
    env = get_http_env()
    if env._sim is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )

    obs = env.step_planning(action)
    step_complete = env._action_buffer.planning  # buffer cleared = step advanced

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "step_complete": step_complete,
        "buffer_status": {
            "planning": "submitted",
            "dispatch": "submitted" if env._action_buffer.dispatch is not None else "pending",
            "market": "submitted" if env._action_buffer.market is not None else "pending",
        },
        "done": obs.done,
        "reward": obs.reward,
        "dispatch_reward": obs.dispatch_reward,
        "planning_reward": obs.planning_reward,
        "market_reward": obs.market_reward,
    })


@app.post(
    "/step/dispatch",
    summary="Dispatch agent action",
    description=(
        "Submit a DispatchAgentAction. Buffered until planning and market "
        "agents also submit. Returns current observation."
    ),
    tags=["Multi-Agent"],
)
async def step_dispatch(action: DispatchAgentAction) -> JSONResponse:
    """
    Receive dispatch agent action.

    Example body:
        {"coal_delta": 100.0, "battery_mode": "discharge", "hydro_delta": 0.0,
         "nuclear_delta": 0.0, "emergency_coal_boost": false}
    """
    env = get_http_env()
    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    obs = env.step_dispatch(action)
    step_complete = env._action_buffer.dispatch

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "step_complete": step_complete,
        "buffer_status": {
            "planning": "submitted" if env._action_buffer.planning is not None else "pending",
            "dispatch": "submitted",
            "market": "submitted" if env._action_buffer.market is not None else "pending",
        },
        "done": obs.done,
        "reward": obs.reward,
        "dispatch_reward": obs.dispatch_reward,
        "planning_reward": obs.planning_reward,
        "market_reward": obs.market_reward,
    })


@app.post(
    "/step/market",
    summary="Market agent action",
    description=(
        "Submit a MarketAgentAction. When all three agents have submitted, "
        "the simulator advances and the new observation is returned. "
        "This is typically the last action submitted each step."
    ),
    tags=["Multi-Agent"],
)
async def step_market(action: MarketAgentAction) -> JSONResponse:
    """
    Receive market agent action.

    Example body:
        {"demand_response_mw": 50.0, "grid_export_mw": 0.0,
         "grid_import_mw": 0.0, "coal_price_bid": null}

    When this completes the action buffer, the simulator advances.
    The response contains the NEW observation after physics update.
    """
    env = get_http_env()
    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    obs = env.step_market(action)
    step_complete = env._action_buffer.market  # buffer cleared = advanced

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "step_complete": step_complete,
        "buffer_status": {
            "planning": "submitted" if env._action_buffer.planning is not None else "pending",
            "dispatch": "submitted" if env._action_buffer.dispatch is not None else "pending",
            "market": "submitted",
        },
        "done": obs.done,
        "reward": obs.reward,
        "dispatch_reward": obs.dispatch_reward,
        "planning_reward": obs.planning_reward,
        "market_reward": obs.market_reward,
        "trading_credits": obs.trading_credits,
    })


@app.get(
    "/step/buffer_status",
    summary="Check action buffer status",
    description="Shows which agents have submitted actions for the current step.",
    tags=["Multi-Agent"],
)
async def buffer_status() -> JSONResponse:
    """Useful for debugging multi-agent coordination."""
    env = get_http_env()
    buf = env._action_buffer
    return JSONResponse(content={
        "planning": "submitted" if buf.planning is not None else "pending",
        "dispatch": "submitted" if buf.dispatch is not None else "pending",
        "market": "submitted" if buf.market is not None else "pending",
        "step_ready": buf.is_complete,
        "current_step": env._sim.step if env._sim else None,
        "task_id": env.current_task_id,
    })

def run_demo():
    return "Demo running (connect your model here)"

with gr.Blocks() as demo:

    gr.Markdown("""
# Energy Grid AI — A Simple System with a Real Problem

## Why this project exists

This project did not start as an “AI for energy systems” idea.

It started with a much simpler frustration.

While building larger projects using language models, I kept running into the same issue:
- one model would generate a large chunk of code,
- then I would switch models (because of rate limits or cost),
- and everything would break.

The second model did not understand what the first one had done. It lost context. It made wrong assumptions. The system became unstable.

This led to a simple question:

> What if different “agents” could cooperate instead of replace each other?

That idea became the foundation of this project.

---

## From code instability to real-world systems

Around the same time, there were large-scale blackouts in Cuba.

Electricity systems are not very different from what I was facing:
- multiple components,
- each making decisions,
- all needing to stay in sync.

If one part behaves unpredictably, the whole system can fail.

So I reframed the problem:

> Can we build a system where multiple decision-makers coordinate to keep something stable?

---

## The core idea (in simple terms)

Imagine a city’s electricity system like a group of people running a kitchen:

- One person decides how much food is needed (planning)
- One person actually cooks (dispatch)
- One person manages costs and supplies (market)

If they don’t communicate properly:
- too much food → waste
- too little → shortage
- bad timing → chaos

The same thing happens in power grids.

The goal is simple:
- keep supply equal to demand
- keep the system stable
- avoid failure (blackouts)

---

## What was built

A simulation of an electricity grid where:

- demand changes over time
- supply must be adjusted continuously
- decisions are made step-by-step

On top of this, I trained a lightweight AI model (LoRA fine-tuned TinyLlama) to make these decisions.

Each step, the model:
1. reads the current grid situation
2. reasons about what to do
3. outputs an action in a structured format

---

## What improved

The fine-tuned model performed better than the base model:

- Score improved from **0.24 to 0.408**
- About **70% improvement in task performance**

This means:
- better decisions
- more consistent behaviour
- clearer outputs

---

## But something important showed up

Even though the score improved, the system sometimes became unstable.

In simple terms:
- it tried to optimise too aggressively
- and ended up causing failures (blackouts)

This reveals something important:

> Optimising for performance is not the same as ensuring stability.

This is true not just for AI, but for real-world systems as well.

---

## What this project shows

1. Multiple agents (or decision-makers) need coordination, not replacement  
2. Structured outputs make systems more reliable  
3. Optimisation without constraints can lead to failure  
4. Small models, when fine-tuned properly, can behave surprisingly well  

---

## Visual results

Below are two key results:

- Reward over time (how well the system behaves step-by-step)
- Comparison between base model and fine-tuned model

---

## Demo

You can run the model below to see how it behaves in the simulated environment.

Each step, it will:
- analyse the grid
- produce a decision
- affect the system

The goal is always the same:
keep the system stable and avoid collapse.
""")

    gr.Button("Run Demo").click(run_demo)

demo.launch()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = None) -> None:
    import uvicorn
    if port is None:
        port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()