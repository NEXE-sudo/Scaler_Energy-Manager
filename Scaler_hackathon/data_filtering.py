"""
data_filtering.py

Filters raw collected rollout data to keep only high-quality training samples,
then formats them for TRL SFTTrainer.

Filtering criteria:
  - Response must contain a valid JSON action block
  - No blackout steps (catastrophic failures are bad demonstrations)
  - reward > threshold OR in top-N percentile (configurable)
  - Response must not be empty or whitespace-only

Output format for TRL:
  {
    "prompt":     "<system>\n<user observation>",
    "completion": "<LLM response with Thought + Action>"
  }

Usage:
    python data_filtering.py --input dataset_raw.jsonl --output dataset_clean.jsonl
    python data_filtering.py --input dataset_raw.jsonl --top-pct 50
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REQUIRED_KEYS = {
    "planning": {"plant_action"},
    "dispatch": {"coal_delta", "hydro_delta", "nuclear_delta", "battery_mode", "emergency_coal_boost"},
    "market": {"demand_response_mw", "grid_export_mw", "grid_import_mw", "coal_price_bid"},
}

VALID_BATTERY_MODES  = {"charge", "discharge", "idle"}
VALID_PLANT_ACTIONS  = {
    "none", "build_solar", "build_wind",
    "build_hydro", "build_nuclear", "close_coal",
}


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object from LLM response text.
    Returns parsed dict or None if not found / invalid.
    """
    if not text or not text.strip():
        return None

    # Strip markdown fences
    cleaned = re.sub(r"```[a-z]*\n?|```", "", text).strip()

    # Find first balanced brace
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                # Normalise and parse
                candidate = re.sub(r",\s*}", "}", candidate)
                candidate = re.sub(r",\s*]", "]", candidate)
                candidate = re.sub(r"\bTrue\b",  "true",  candidate)
                candidate = re.sub(r"\bFalse\b", "false", candidate)
                candidate = re.sub(r"\bNone\b",  "null",  candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def is_valid_action(action_dict: Optional[Dict[str, Any]], agent_type: str = "unified") -> bool:
    """Return True if the parsed action dict has all required keys for the given agent."""
    if action_dict is None:
        return False

    # Multi-agent vs Unified validation
    required_keys = AGENT_REQUIRED_KEYS.get(agent_type, AGENT_REQUIRED_KEYS["dispatch"].union(AGENT_REQUIRED_KEYS["planning"]))
    
    if not required_keys.issubset(action_dict.keys()):
        return False

    # Specific checks based on agent type
    if agent_type in ("dispatch", "unified"):
        try:
            cd = float(action_dict["coal_delta"])
            if not (-101 <= cd <= 101): return False
            if str(action_dict.get("battery_mode", "")).lower() not in VALID_BATTERY_MODES: return False
        except (TypeError, ValueError): return False

    if agent_type in ("planning", "unified"):
        if str(action_dict.get("plant_action", "")).lower() not in VALID_PLANT_ACTIONS:
            return False

    return True


def is_valid_response(response: str, agent_type: str = "unified") -> bool:
    """Return True if response is non-empty and contains a parseable action."""
    if not response or not response.strip():
        return False
    parsed = _extract_json_block(response)
    return is_valid_action(parsed, agent_type)


# ─────────────────────────────────────────────────────────────────────────────
# Core filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_dataset(
    records:     List[Dict[str, Any]],
    reward_threshold: float = 0.0,
    top_percentile:   float = 100.0,  # keep top-N% by reward; 100 = no percentile filter
    exclude_blackout: bool  = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter records to retain high-quality training samples.

    Args:
        records:          raw records from JSONL
        reward_threshold: minimum reward to keep a sample
        top_percentile:   if < 100, keep only records in top-N% by reward
        exclude_blackout: remove any step where blackout=True

    Returns:
        (filtered_records, stats_dict)
    """
    stats = {
        "total":            len(records),
        "removed_empty":    0,
        "removed_invalid":  0,
        "removed_blackout": 0,
        "removed_reward":   0,
        "kept":             0,
    }

    # Step 1: Remove empty / invalid responses
    valid = []
    for rec in records:
        agent_type = rec.get("agent", "unified")
        if not is_valid_response(rec.get("response", ""), agent_type):
            stats["removed_invalid"] += 1
            continue
        valid.append(rec)

    # Step 2: Remove blackout steps
    if exclude_blackout:
        non_blackout = []
        for rec in valid:
            if rec.get("blackout", False):
                stats["removed_blackout"] += 1
            else:
                non_blackout.append(rec)
        valid = non_blackout

    # Step 3: Reward filtering
    rewards = [rec["reward"] for rec in valid]

    if top_percentile < 100.0 and rewards:
        import numpy as np
        cutoff = float(np.percentile(rewards, 100.0 - top_percentile))
        reward_threshold = max(reward_threshold, cutoff)

    reward_filtered = []
    for rec in valid:
        if rec["reward"] < reward_threshold:
            stats["removed_reward"] += 1
        else:
            reward_filtered.append(rec)

    stats["kept"] = len(reward_filtered)
    return reward_filtered, stats


# ─────────────────────────────────────────────────────────────────────────────
# TRL format conversion
# ─────────────────────────────────────────────────────────────────────────────

def format_for_trl(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a raw record into TRL SFTTrainer format.

    The 'prompt' field combines system + user text separated by a clear delimiter.
    The 'completion' field is the raw LLM response (Thought + Action JSON).

    TRL's SFTTrainer will train on completion tokens only when using
    DataCollatorForCompletionOnlyLM with the response template.
    """
    system_text = record.get("system", "").strip()
    user_text   = record.get("prompt", "").strip()
    agent_type  = record.get("agent", "operator")
    phase       = record.get("phase", "proposal")

    # Inject agent identity if not in system text
    identity = f" [Agent: {agent_type.upper()} | Phase: {phase.upper()}]"
    
    # Format as a clean chat-style prompt
    if system_text:
        full_prompt = f"<|system|>\n{system_text}\n<|user|>\n{identity}\n{user_text}\n<|assistant|>\n"
    else:
        full_prompt = f"<|user|>\n{identity}\n{user_text}\n<|assistant|>\n"

    completion = record.get("response", "").strip()

    return {
        "prompt":     full_prompt,
        "completion": completion,
    }


def build_trl_dataset(filtered_records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert all filtered records to TRL format."""
    return [format_for_trl(r) for r in filtered_records]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] Skipping malformed line {line_no}: {e}")
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter and format energy grid rollout data")
    parser.add_argument("--input",       type=str,   default="dataset_raw.jsonl")
    parser.add_argument("--output",      type=str,   default="dataset_clean.jsonl")
    parser.add_argument("--reward-min",  type=float, default=0.0,
                        help="Minimum reward threshold (default: 0.0)")
    parser.add_argument("--top-pct",     type=float, default=100.0,
                        help="Keep top-N%% of samples by reward (default: 100 = no filter)")
    parser.add_argument("--keep-blackouts", action="store_true",
                        help="Don't remove blackout steps")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    raw = load_jsonl(Path(args.input))
    print(f"Loaded {len(raw)} records")

    filtered, stats = filter_dataset(
        records          = raw,
        reward_threshold = args.reward_min,
        top_percentile   = args.top_pct,
        exclude_blackout = not args.keep_blackouts,
    )

    print("\nFiltering stats:")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    trl_data = build_trl_dataset(filtered)
    save_jsonl(trl_data, Path(args.output))
    print(f"\n[DONE] Clean dataset saved to {args.output} ({len(trl_data)} samples)")


if __name__ == "__main__":
    main()