"""inference.py must still work with the legacy variables, and accept per-role config."""
import pytest

import inference

CANARY = "sk-canary-inference"
CLEAR = ["API_BASE_URL", "MODEL_NAME", "PLANNING_MODEL", "HF_TOKEN", "API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
         "NEBIUS_API_KEY", "SCALER_PROVIDER", "SCALER_MODEL", "SCALER_MODELS_FILE", "SCALER_PLANNING_MODEL",
         "SCALER_DISPATCH_MODEL", "SCALER_MARKET_MODEL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in CLEAR:
        monkeypatch.delenv(k, raising=False)
    calls = []
    monkeypatch.setattr(inference, "run_baseline_agent", lambda **kw: calls.append(kw) or {})
    return calls


def test_legacy_mode_still_requires_a_token(capsys, clean_env):
    assert inference.main() == 1
    assert "HF_TOKEN" in capsys.readouterr().out and clean_env == []


def test_legacy_mode_with_token_runs_and_bridges_env(monkeypatch, clean_env):
    monkeypatch.setenv("HF_TOKEN", "x")
    assert inference.main() == 0 and len(clean_env) == 1
    import os
    assert os.environ["MODEL_NAME"] == "llama-3.3-70b-versatile"      # previous default preserved


def test_per_role_mode_needs_no_hf_token(monkeypatch, capsys, clean_env):
    monkeypatch.setenv("SCALER_PROVIDER", "nebius")
    monkeypatch.setenv("NEBIUS_API_KEY", CANARY)
    monkeypatch.setenv("SCALER_PLANNING_MODEL", "org/big")
    monkeypatch.setenv("SCALER_DISPATCH_MODEL", "org/small")
    monkeypatch.setenv("SCALER_MARKET_MODEL", "org/small")
    assert inference.main() == 0 and len(clean_env) == 1
    out = capsys.readouterr().out
    assert "org/big" in out and "heterogeneous models: True" in out and CANARY not in out


def test_per_role_mode_missing_key_names_the_variable(monkeypatch, capsys, clean_env):
    monkeypatch.setenv("SCALER_PROVIDER", "nebius")
    monkeypatch.setenv("SCALER_MODEL", "org/one")
    assert inference.main() == 1
    assert "NEBIUS_API_KEY" in capsys.readouterr().out and clean_env == []


def test_per_role_mode_bad_config_is_a_clean_error(monkeypatch, capsys, clean_env):
    monkeypatch.setenv("SCALER_PROVIDER", "nebius")       # no models given
    assert inference.main() == 1
    assert "missing" in capsys.readouterr().out.lower() and clean_env == []
