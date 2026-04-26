#!/usr/bin/env python3
"""
Energy Grid OpenEnv Baseline Inference Script

This script is required by the Scaler x OpenEnv Hackathon Phase 1 validation.
It runs the baseline LLM agent against all three energy grid tasks and outputs
reproducible scores in the mandatory structured format.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM (e.g., https://api.groq.com/openai/v1)
    MODEL_NAME     The model identifier to use (e.g., llama-3.3-70b-versatile)
    HF_TOKEN       Your Hugging Face / API authentication key

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

# Load .env file if it exists (optional but recommended)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv optional; env vars can be set directly

from server.baseline import run_baseline_agent


def main() -> int:
    """Run baseline agent on all tasks with structured logging."""
    
    # Load environment variables with defaults for API_BASE_URL and MODEL_NAME only
    api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # HF_TOKEN is required (no default)
    if not hf_token:
        print("ERROR: Missing required HF_TOKEN environment variable", flush=True)
        print("Optional (have defaults):", flush=True)
        print("  API_BASE_URL   defaults to: https://api.groq.com/openai/v1", flush=True)
        print("  MODEL_NAME     defaults to: llama-3.3-70b-versatile", flush=True)
        print("Required:", flush=True)
        print("  HF_TOKEN       Authentication key for API (required)", flush=True)
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
        
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user", flush=True)
        sys.stdout.flush()
        return 130
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        
        # Output minimal valid structured format on error
        print("\n[INFO] Outputting error-state results...", flush=True)
        try:
            for task_id in ["easy", "medium", "hard"]:
                print(f"[START] task={task_id} env=energy-grid-openenv model={model_name}", flush=True)
                print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
        except Exception as inner_e:
            print(f"[ERROR] Even fallback output failed: {inner_e}", flush=True)
        
        sys.stdout.flush()
        return 2


if __name__ == "__main__":
    sys.exit(main())

