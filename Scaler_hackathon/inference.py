#!/usr/bin/env python3
"""
Energy Grid OpenEnv Baseline Inference Script

This script is required by the Scaler x OpenEnv Hackathon Phase 1 validation.
It runs the baseline LLM agent against all three energy grid tasks and outputs
reproducible scores in the mandatory structured format.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM (e.g., https://api.groq.com/openai/v1)
    MODEL_NAME    — The model identifier to use (e.g., llama-3.3-70b-versatile)
    HF_TOKEN      — Your Hugging Face / API authentication key

STDOUT FORMAT (MANDATORY):
    [START] task=<task_id> env=energy-grid-openenv model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="your_token_here"
    python inference.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server.baseline import run_baseline_agent


def main() -> int:
    """Run baseline agent on all tasks with structured logging."""
    
    # Validate required environment variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    
    if not api_base_url or not model_name or not hf_token:
        print("ERROR: Missing required environment variables", flush=True)
        print("Required:", flush=True)
        print("  API_BASE_URL  — API endpoint", flush=True)
        print("  MODEL_NAME    — Model identifier", flush=True)
        print("  HF_TOKEN      — Authentication key", flush=True)
        return 1
    
    # Debug: Show what was read
    print(f"[DEBUG] API_BASE_URL={api_base_url}", flush=True)
    print(f"[DEBUG] MODEL_NAME={model_name}", flush=True)
    print(f"[DEBUG] HF_TOKEN set: {bool(hf_token)}", flush=True)
    
    # Bridge environment variables (API key handling will be in baseline.py)
    os.environ["API_BASE_URL"] = api_base_url
    os.environ["MODEL_NAME"] = model_name
    os.environ["HF_TOKEN"] = hf_token
    
    # Run baseline agent on all three tasks
    # (Will emit structured logs directly from run_baseline_agent)
    try:
        results = run_baseline_agent(
            task_ids=["easy", "medium", "hard"],
            verbose=True  # Show detailed output including REASON, state changes, cumulative rewards
        )
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

