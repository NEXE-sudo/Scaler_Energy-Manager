#!/usr/bin/env python3
"""
Energy Grid OpenEnv Baseline Inference Script

This script is required by the Scaler x OpenEnv Hackathon Phase 1 validation.
It runs the baseline LLM agent against all three energy grid tasks and outputs
reproducible scores.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM (e.g., https://api.groq.com/openai/v1)
    MODEL_NAME    — The model identifier to use (e.g., llama-3.3-70b-versatile)
    HF_TOKEN      — Your Hugging Face / API authentication key

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
    """Run baseline agent and display results."""
    
    # Validate required environment variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    
    if not api_base_url or not model_name:
        print("ERROR: Missing required environment variables")
        print()
        print("Required:")
        print("  API_BASE_URL  — API endpoint (e.g., https://api.groq.com/openai/v1)")
        print("  MODEL_NAME    — Model identifier (e.g., llama-3.3-70b-versatile)")
        print("  HF_TOKEN      — Authentication key (optional if using API_KEY)")
        print()
        print("Set them with:")
        print("  export API_BASE_URL='...'")
        print("  export MODEL_NAME='...'")
        print("  export HF_TOKEN='...'")
        return 1
    
    print("=" * 70)
    print("Energy Grid OpenEnv — Baseline Inference")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  API Base: {api_base_url}")
    print(f"  Model:    {model_name}")
    print()
    
    # Run baseline agent on all three tasks
    try:
        print("Running baseline agent on all tasks...")
        print("-" * 70)
        results = run_baseline_agent(
            task_ids=["easy", "medium", "hard"],
            verbose=True
        )
        
        # Display summary
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        total_score = 0.0
        task_count = len(results)
        
        for task_id, result in results.items():
            score = result.get("score", 0.0)
            total_score += score
            print(f"  {task_id.upper():8s}: {score:.4f}")
        
        if task_count > 0:
            avg_score = total_score / task_count
            print("-" * 70)
            print(f"  AVERAGE  : {avg_score:.4f}")
        
        print("=" * 70)
        print()
        print("✓ Inference completed successfully")
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR DURING INFERENCE")
        print("=" * 70)
        print(f"  {e}")
        print()
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
