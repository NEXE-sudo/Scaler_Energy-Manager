"""
FastAPI application for the Energy Grid Management Environment.

Creates an HTTP + WebSocket server exposing the EnergyGridEnvironment
over the OpenEnv spec endpoints, plus the three hackathon-required
additional endpoints.

Standard OpenEnv endpoints (provided by create_app()):
    POST /reset          Reset environment, optionally with task_id
    POST /step           Execute an action
    GET  /state          Current episode state
    GET  /schema         Action / observation JSON schemas
    WS   /ws             WebSocket for persistent sessions

Additional endpoints (hackathon requirement):
    GET  /tasks          List all tasks with action schema
    POST /grader         Return grader score for completed episode
    POST /baseline       Run baseline LLM agent on all 3 tasks, return scores

Usage:
    # Development
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Direct
    python -m server.app
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import warnings
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import EnergyGridAction, EnergyGridObservation
    from .energy_grid_environment import EnergyGridEnvironment
    from .tasks import get_tasks_summary, PLANT_BUILD_REFERENCE
    from .grader import grade_result_to_dict
except (ImportError, ModuleNotFoundError):
    from models import EnergyGridAction, EnergyGridObservation
    from server.energy_grid_environment import EnergyGridEnvironment
    from server.tasks import get_tasks_summary, PLANT_BUILD_REFERENCE
    from server.grader import grade_result_to_dict


# ---------------------------------------------------------------------------
# Create the base OpenEnv app
# ---------------------------------------------------------------------------

app: FastAPI = create_app(
    EnergyGridEnvironment,
    EnergyGridAction,
    EnergyGridObservation,
    env_name="energy-grid-openenv",
    max_concurrent_envs=4,
)

# ---------------------------------------------------------------------------
# Request / response models for additional endpoints
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "easy"


class GraderRequest(BaseModel):
    """
    Request body for /grader.

    The environment instance graded is the one attached to the current
    HTTP session. For WebSocket sessions, grading happens automatically
    at episode end and can be retrieved via /grader with no body.
    """
    task_id: Optional[str] = None


class BaselineRequest(BaseModel):
    """
    Optional configuration for /baseline endpoint.
    If tasks is empty, all three tasks are run.
    """
    tasks: list[str] = []


# ---------------------------------------------------------------------------
# Shared environment instance for HTTP (non-WebSocket) endpoints
# ---------------------------------------------------------------------------
# This single instance handles the /tasks, /grader, /baseline endpoints.
# WebSocket sessions each get their own instance via factory mode.

_http_env: Optional[EnergyGridEnvironment] = None

def get_http_env() -> EnergyGridEnvironment:
    global _http_env
    if _http_env is None:
        _http_env = EnergyGridEnvironment()
        workers = os.getenv("WEB_CONCURRENCY")
        if workers and int(workers) > 1:
            warnings.warn(
                "_http_env is process-local. With multiple uvicorn workers, "
                "/grader results may not match the session that ran /reset and /step. "
                "Use a single worker or WebSocket sessions for reliable grading."
            )
    return _http_env


# ---------------------------------------------------------------------------
# /reset (GET) — health ping endpoint
# ---------------------------------------------------------------------------

@app.get("/reset", include_in_schema=False)
async def reset_ping() -> JSONResponse:
    """Health ping endpoint — validators use GET /reset to check liveness."""
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# /tasks — list tasks and action schema
# ---------------------------------------------------------------------------

@app.get(
    "/tasks",
    summary="List all tasks",
    description=(
        "Returns all available tasks with their descriptions, difficulty, "
        "episode length, action schema, and plant build reference. "
        "Use this to understand what actions are valid before calling /step."
    ),
    tags=["OpenEnv Extensions"],
)
async def get_tasks() -> JSONResponse:
    return JSONResponse(
        content={
            "tasks": get_tasks_summary(),
            "plant_build_reference": PLANT_BUILD_REFERENCE,
            "notes": {
                "battery_mode": "charge | discharge | idle",
                "plant_action": (
                    "none | build_solar | build_wind | "
                    "build_hydro | build_nuclear | close_coal"
                ),
                "emergency_coal_boost": (
                    "true/false — overrides ramp limits, damages plant for 5 steps"
                ),
                "demand_response_mw": (
                    "0–150 MW — reduces effective demand, costs 0.5 capital/MW "
                    "(Hard task only deducts capital)"
                ),
            },
        }
    )


# ---------------------------------------------------------------------------
# /grader — return score for completed episode
# ---------------------------------------------------------------------------

@app.post(
    "/grader",
    summary="Grade completed episode",
    description=(
        "Returns the deterministic grader score (0.0–1.0) for the most "
        "recently completed episode on this server instance. "
        "Also returns per-component score breakdown. "
        "If the episode is still running, returns a partial grade."
    ),
    tags=["OpenEnv Extensions"],
)
async def grade_episode(request: GraderRequest = GraderRequest()) -> JSONResponse:
    env = get_http_env()

    # Validate task_id if provided
    if request.task_id and request.task_id != env.current_task_id:
        raise HTTPException(
            status_code=400,
            detail="task_id does not match the active episode",
        )

    # Try completed grade first
    grade = env.get_last_grade()

    if grade is None:
        # Try partial grade of in-progress episode
        grade = env.grade_current_episode()

    if grade is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No episode data available. "
                "Call POST /reset to start an episode first."
            ),
        )

    return JSONResponse(content=grade)


# ---------------------------------------------------------------------------
# /baseline — run LLM agent on all tasks and return scores
# ---------------------------------------------------------------------------

@app.post(
    "/baseline",
    summary="Run baseline LLM agent",
    description=(
        "Runs the baseline LLM agent against all three tasks "
        "(or a subset specified in the request body). "
        "Requires API_BASE_URL, MODEL_NAME, and HF_TOKEN environment variables. "
        "Returns reproducible scores for each task. "
        "This endpoint blocks until all tasks complete — expect 2–5 minutes."
    ),
    tags=["OpenEnv Extensions"],
)
async def run_baseline(request: BaselineRequest = BaselineRequest()) -> JSONResponse:
    """
    Trigger the baseline inference script programmatically.

    Runs in the same process to avoid subprocess complexity in Docker.
    The baseline agent uses a planner/executor architecture with rolling
    conversation history. Model and endpoint are configured via environment
    variables (API_BASE_URL, MODEL_NAME, HF_TOKEN).
    """
    try:
        # Import here to avoid circular imports at module load
        try:
            from .baseline import run_baseline_agent
        except ImportError:
            from server.baseline import run_baseline_agent

        task_ids = request.tasks if request.tasks else ["easy", "medium", "hard"]

        # Validate task ids
        valid_tasks = {"easy", "medium", "hard"}
        invalid = [t for t in task_ids if t not in valid_tasks]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task IDs: {invalid}. Valid: {list(valid_tasks)}",
            )

        # Run baseline (async-friendly — uses asyncio.to_thread for blocking calls)
        results = await asyncio.to_thread(
            run_baseline_agent,
            task_ids=task_ids,
        )

        return JSONResponse(
            content={
                "status": "completed",
                "baseline_scores": results,
                "model": results.get("model", "unknown"),
                "provider": "openai-compatible",
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": tb,
                "hint": (
                    "Ensure API_BASE_URL, MODEL_NAME, and HF_TOKEN are set. "
                    "Check server logs for full traceback."
                ),
            },
        )


# ---------------------------------------------------------------------------
# /health — explicit health check (supplements Dockerfile HEALTHCHECK)
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health check",
    tags=["Infrastructure"],
)
async def health_check() -> JSONResponse:
    return JSONResponse(
        content={
            "status": "ok",
            "environment": "energy-grid-openenv",
            "openenv_spec_version": 1,
        }
    )

@app.get("/")
async def root():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = None) -> None:
    """
    Entry point for direct execution.

        uv run --project . server
        uv run --project . server --port 8001
        python -m server.app
    """
    import uvicorn
    if port is None:
        port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()