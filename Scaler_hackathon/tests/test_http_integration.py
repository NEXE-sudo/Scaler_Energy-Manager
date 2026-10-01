"""Real `openai` SDK against a local OpenAI-compatible server (no internet, no real keys).

Guards SDK-level behaviour the fake client cannot: extra_body merging, per-request timeout,
Authorization header, `reasoning_content` surfacing on the response message, models.list().
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from server import baseline
from server.energy_grid_environment import EnergyGridEnvironment
from server.model_config import build_team, load_team_config
from server.model_smoke import run_smoke
from server.trace import TraceRecorder
from tests.conftest import DEFAULT_REPLIES, role_of

KEY = "sk-local-canary-42"


class Fake(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.state["auth"].append(self.headers.get("Authorization"))
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "x"}
                                                    for m in self.state["models"]]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["auth"].append(self.headers.get("Authorization"))
        self.state["bodies"].append(body)
        role = role_of(body["messages"][0]["content"])
        thinking_off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        if role == "planning" and not thinking_off:
            msg, finish = {"role": "assistant", "content": None, "reasoning_content": "thinking about the grid"}, "length"
        else:
            msg, finish = {"role": "assistant", "content": DEFAULT_REPLIES[role]}, "stop"
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}})


@pytest.fixture
def server(monkeypatch):
    Fake.state = {"auth": [], "bodies": [], "models": ["org/big", "org/small"]}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LOCAL_TEST_KEY", KEY)
    monkeypatch.setattr(baseline, "MAX_EVAL_STEPS", 1)
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", Fake.state
    srv.shutdown()


def team_env(base_url, planning_extra=None):
    env = {"SCALER_MODEL": "org/small", "SCALER_PLANNING_MODEL": "org/big", "SCALER_PLANNING_MAX_TOKENS": "64",
           "SCALER_PROVIDER": "nebius", "SCALER_PLANNING_BASE_URL": base_url, "SCALER_DISPATCH_BASE_URL": base_url,
           "SCALER_MARKET_BASE_URL": base_url, "SCALER_PLANNING_API_KEY_ENV": "LOCAL_TEST_KEY",
           "SCALER_DISPATCH_API_KEY_ENV": "LOCAL_TEST_KEY", "SCALER_MARKET_API_KEY_ENV": "LOCAL_TEST_KEY",
           "LOCAL_TEST_KEY": KEY}
    if planning_extra:
        env["SCALER_PLANNING_EXTRA_BODY"] = planning_extra
    return env


def test_smoke_diagnoses_reasoning_model_then_passes_once_thinking_is_off(server):
    base_url, state = server
    lines = []
    code = run_smoke(load_team_config(env=team_env(base_url)), env=team_env(base_url), out=lines.append)
    text = "\n".join(lines)
    assert code == 1 and "[FAIL] planning" in text and "parse=reasoning_truncated" in text and "reasoning=yes" in text
    assert "probe B" in text and "-> add extra_body" in text            # thinking-off probe fixed it
    assert "[PASS] dispatch" in text and "[PASS] market" in text and "2 models listed" in text
    assert KEY not in text and set(state["auth"]) == {f"Bearer {KEY}"}  # key only ever in the header

    off = '{"chat_template_kwargs": {"enable_thinking": false}}'
    lines.clear()
    code = run_smoke(load_team_config(env=team_env(base_url, off)), env=team_env(base_url, off), out=lines.append)
    assert code == 0 and "ALL PASS" in "\n".join(lines)


def test_real_sdk_sends_extra_body_and_timeout_and_traces_cleanly(server):
    base_url, state = server
    off = '{"chat_template_kwargs": {"enable_thinking": false}}'
    env = team_env(base_url, off)
    team = build_team(load_team_config(env=env), env=env)
    trace = TraceRecorder()
    result = baseline.run_task(EnergyGridEnvironment(), None, "x", "medium", verbose=False, trace=trace, team=team)

    assert result["valid_run"] is True and result["llm_stats"]["prompt_tokens"] == 20 * result["llm_stats"]["calls"]
    planning_bodies = [b for b in state["bodies"] if b["model"] == "org/big"]
    dispatch_bodies = [b for b in state["bodies"] if b["model"] == "org/small"]
    assert planning_bodies and all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in planning_bodies)
    assert all("chat_template_kwargs" not in b for b in dispatch_bodies)
    assert all(b["max_tokens"] == 64 for b in planning_bodies)
    start = trace.events[0]
    assert start["heterogeneous"] is True and start["roles"]["planning"]["provider_host"].startswith("127.0.0.1")
    assert KEY not in json.dumps(trace.events) + json.dumps(result, default=str)


def test_real_sdk_surfaces_reasoning_content_in_trace(server):
    base_url, _ = server
    env = team_env(base_url)                                            # planner keeps thinking on -> truncated
    team = build_team(load_team_config(env=env), env=env)
    trace = TraceRecorder()
    result = baseline.run_task(EnergyGridEnvironment(), None, "x", "medium", verbose=False, trace=trace, team=team)
    p = [e for e in trace.events if e["type"] == "agent_call" and e["agent"] == "planning"][0]
    assert p["parse_status"] == "reasoning_truncated" and p["reasoning"] == "thinking about the grid"
    assert p["finish_reason"] == "length" and result["valid_run"] is False
