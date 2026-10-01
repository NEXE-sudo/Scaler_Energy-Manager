"""scripts/smoke_test_models.py logic, against fake clients."""
import pytest

from server import baseline
from server.model_config import load_team_config
from server.model_smoke import run_smoke, suggest_models
from tests.conftest import DEFAULT_REPLIES, FakeClient

CANARY = "sk-canary-555"
ENV = {"SCALER_PROVIDER": "nebius", "NEBIUS_API_KEY": CANARY, "SCALER_PLANNING_MODEL": "nvidia/nemotron-big",
       "SCALER_DISPATCH_MODEL": "nvidia/nemotron-small", "SCALER_MARKET_MODEL": "nvidia/nemotron-small"}
LISTED = ["meta/llama-70b", "nvidia/nemotron-big", "nvidia/nemotron-small", "nvidia/nemotron-small-v2"]


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(baseline.time, "sleep", lambda *_: None)


def smoke(client, env=ENV, team_env=None, **kw):
    lines = []
    team = load_team_config(env=team_env or env)
    code = run_smoke(team, client_factory=lambda *a: client, env=env, out=lines.append, **kw)
    return code, "\n".join(lines), lines


def test_all_roles_pass_and_key_is_never_printed():
    client = FakeClient(model_ids=LISTED)
    code, text, _ = smoke(client)
    assert code == 0 and "ALL PASS" in text
    assert text.count("[PASS]") == 3 and "parse=ok" in text and "tokens(in/out)=11/7" in text
    assert CANARY not in text
    assert client.list_calls == 1                       # one shared client -> listed once


def test_missing_key_fails_cleanly_without_calls():
    client = FakeClient(model_ids=LISTED)
    code, text, _ = smoke(client, env={k: v for k, v in ENV.items() if k != "NEBIUS_API_KEY"}, team_env=ENV)
    assert code == 1 and "NEBIUS_API_KEY" in text and "[FAIL]" in text and client.calls == []


def test_unlisted_model_warns_with_suggestions_but_still_tests():
    client = FakeClient(model_ids=["nvidia/nemotron-big", "nvidia/nemotron-small-v2"])
    code, text, _ = smoke(client)
    assert "'nvidia/nemotron-small' is not in the model list" in text and "nvidia/nemotron-small-v2" in text
    assert len(client.calls) == 3                       # warning is advisory: the call is still attempted
    assert code == 0


def test_model_list_failure_is_a_warning_not_a_failure():
    client = FakeClient(list_error=RuntimeError("403 forbidden"))
    code, text, _ = smoke(client)
    assert code == 0 and "could not list models" in text and "403 forbidden" in text


def test_call_failure_is_reported_with_error():
    client = FakeClient(model_ids=LISTED, replies={"dispatch": RuntimeError("401 unauthorized")})
    code, text, _ = smoke(client)
    assert code == 1 and "[FAIL] dispatch" in text and "401 unauthorized" in text and "parse=call_failed" in text


def test_reasoning_truncation_triggers_probes_that_name_the_fix():
    def fn(role, n, kwargs):
        thinking_off = (kwargs.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
        if role == "dispatch" and not thinking_off and kwargs["max_tokens"] < 1000:
            return {"content": None, "reasoning_content": "thinking", "finish_reason": "length"}
        return DEFAULT_REPLIES[role]

    client = FakeClient(model_ids=LISTED, fn=fn)
    code, text, _ = smoke(client)
    assert code == 1 and "parse=reasoning_truncated" in text
    assert "probe A (max_tokens=1024" in text and "raise this role's max_tokens" in text
    assert "probe B" in text and "enable_thinking" in text and "add extra_body" in text
    probed = [c for c in client.calls if c["role"] == "dispatch"]
    assert len(probed) == 3 and probed[2]["kwargs"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_probes_can_be_disabled_and_skip_probe_b_when_extra_body_already_set():
    always_bad = lambda role, n, kw: {"content": None, "reasoning_content": "t", "finish_reason": "length"}
    client = FakeClient(model_ids=LISTED, fn=always_bad)
    code, _, _ = smoke(client, roles=["dispatch"], probe=False)
    assert code == 1 and len(client.calls) == 1

    env = {**ENV, "SCALER_DISPATCH_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}'}
    client = FakeClient(model_ids=LISTED, fn=always_bad)
    _, text, _ = smoke(client, env=env, roles=["dispatch"])
    assert "probe A" in text and "probe B" not in text


def test_roles_filter_and_list_models_only():
    client = FakeClient(model_ids=LISTED)
    code, text, lines = smoke(client, roles=["market"])
    assert code == 0 and text.count("[PASS]") == 1 and "[PASS] market" in text

    client = FakeClient(model_ids=LISTED)
    code, text, lines = smoke(client, list_filter="nemotron", list_only=True)
    assert code == 0 and client.calls == []
    listed = [l.strip() for l in lines if l.startswith("         ")]
    assert listed == ["nvidia/nemotron-big", "nvidia/nemotron-small", "nvidia/nemotron-small-v2"]


def test_suggest_models():
    assert "nvidia/nemotron-small-v2" in suggest_models("nvidia/nemotron-small", LISTED)
    assert suggest_models("totally/unrelated-thing", ["a/b"]) == []
