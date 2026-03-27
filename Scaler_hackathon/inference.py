#!/usr/bin/env python3
"""
Energy Grid OpenEnv — Inference Script

This is the root-level inference script required by the Scaler x OpenEnv Hackathon.
It validates required environment variables and runs the baseline agent across all
three task difficulties (easy, medium, hard).

Required Environment Variables:
    API_KEY         — API key for the model provider
    API_BASE_URL    — Base URL for the API endpoint (e.g., https://api.openai.com/v1)
    MODEL_NAME      — Model identifier (e.g., gpt-4, llama-3.3-70b-versatile)

Usage:
    # Set environment variables
    export API_KEY="your-api-key"
    export API_BASE_URL="https://api.provider.com/v1"
    export MODEL_NAME="your-model"
    
    # Run inference
    python inference.py

Exit codes:
    0   — Success: all tasks completed
    1   — Missing environment variables
    2   — Execution error
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server.baseline import run_baseline_agent


def validate_environment() -> None:
    """Validate that required environment variables are set."""
    required = ["API_KEY", "API_BASE_URL", "MODEL_NAME"]
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print("ERROR: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nSet them with:")
        print(f"  export {' '.join(missing[0:1])}")
        sys.exit(1)


def main() -> None:
    """Run baseline agent across all task difficulties."""
    print("=" * 70)
    print("Energy Grid OpenEnv — Baseline Inference")
    print("=" * 70)
    print()
    
    # Validate environment
    validate_environment()
    
    # Display configuration
    print("Configuration:")
    print(f"  Model: {os.getenv('MODEL_NAME')}")
    print(f"  API Base: {os.getenv('API_BASE_URL')}")
    print()
    
    # Run baseline agent
    try:
        print("Running baseline agent across all tasks...")
        print("-" * 70)
        results = run_baseline_agent(
            task_ids=["easy", "medium", "hard"],
            verbose=True
        )
        
        # Display results summary
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        total_score = 0.0
        task_count = 0
        
        for task_id, result in results.items():
            score = result.get("score", 0.0)
            total_score += score
            task_count += 1
            steps = result.get("total_steps", 0)
            print(f"{task_id.upper():8} | Score: {score:.4f}  | Steps: {steps}")
        
        if task_count > 0:
            avg_score = total_score / task_count
            print("-" * 70)
            print(f"{'AVERAGE':8} | Score: {avg_score:.4f}")
        
        print("=" * 70)
        print("\nInference completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
