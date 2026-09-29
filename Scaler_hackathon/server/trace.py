"""Run trace recorder: makes every agent handoff observable and replayable.

One JSONL file per episode. Each line is one event with a monotonically
increasing `seq`. Event types (schema "scaler.trace.v1"):

    run_start   manifest: models per role, protocol, seed, cap, git sha, ...
    agent_call  one agent invocation: exact prompt in, exact raw text out,
                parsed action, action after the control layer, parse status,
                tokens, latency. Revision-round calls list `inputs_from`
                (call_ids of the round-1 calls whose output they received).
    env_step    the actions submitted to the simulator for one step and the
                resulting canonical observation summary and reward.
    run_end     score, stats, truncation and validity flags.

Traces contain prompts and model output, never API keys. Only the host part of
the API base URL is recorded.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

SCHEMA = "scaler.trace.v1"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def base_url_host(client: Any) -> Optional[str]:
    url = getattr(client, "base_url", None)
    if url is None:
        return None
    return urlparse(str(url)).netloc or None


class TraceRecorder:
    """Collects trace events in memory and optionally appends them to a JSONL file."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path: Optional[Path] = Path(path) if path else None
        self.events: List[Dict[str, Any]] = []
        self._seq = 0
        self._write_ok = self.path is not None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._write_ok = False  # e.g. read-only filesystem; keep in-memory only

    def record(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        self._seq += 1
        event = {"schema": SCHEMA, "seq": self._seq, "type": event_type, "t": round(time.time(), 3), **fields}
        self.events.append(event)
        if self._write_ok:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str) + "\n")
            except OSError:
                self._write_ok = False  # never let tracing break a run
        return event


def load_trace(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def handoff_chain(events: List[Dict[str, Any]], step: int) -> Dict[str, Any]:
    """Everything needed to inspect one step's handoffs, in order.

    Returns {"calls": [...agent_call events...], "env_step": {...} | None}.
    Each revision call's `inputs_from` can be resolved against `calls` by call_id
    to compare what a sender produced with what its receiver was shown.
    """
    calls = [e for e in events if e["type"] == "agent_call" and e.get("step") == step]
    env_step = next((e for e in events if e["type"] == "env_step" and e.get("step") == step), None)
    return {"calls": calls, "env_step": env_step}
