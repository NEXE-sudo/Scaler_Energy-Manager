"""Per-role model configuration: loading, precedence, validation, secret hygiene."""
import json
from pathlib import Path

import pytest

from server.model_config import (
    ConfigError, ROLES, build_team, describe_team, load_team_config, resolve_api_key, team_env_configured,
)

CANARY = "sk-canary-123456"
EXAMPLE = Path(__file__).resolve().parent.parent / "config" / "models.example.yaml"
LEGACY = {"API_BASE_URL": "https://api.groq.com/openai/v1", "MODEL_NAME": "llama-x", "HF_TOKEN": CANARY}


# ---- legacy mode (original env vars must keep working exactly) -------------------------

def test_legacy_mode_matches_previous_behaviour():
    t = load_team_config(env=LEGACY)
    assert t.source == "legacy-env"
    assert t.roles["dispatch"].model == t.roles["market"].model == "llama-x"
    assert t.roles["planning"].model == "openai/gpt-oss-120b"       # previous PLANNING_MODEL default
    assert t.roles["planning"].plan_max_tokens == 600 and t.roles["planning"].max_tokens == 256
    assert all(t.roles[r].temperature == 0.2 and t.roles[r].timeout_s is None for r in ROLES)
    assert all(t.roles[r].api_key_env == "HF_TOKEN" for r in ROLES)


def test_legacy_planning_override_and_key_priority():
    t = load_team_config(env={**LEGACY, "PLANNING_MODEL": "big-planner", "GROQ_API_KEY": "g"})
    assert t.roles["planning"].model == "big-planner"
    assert t.roles["dispatch"].api_key_env == "GROQ_API_KEY"  # GROQ > OPENAI > HF, as in _build_client


def test_legacy_missing_config_is_a_clear_error():
    with pytest.raises(ConfigError, match="API_BASE_URL"):
        load_team_config(env={})


def test_trace_dir_alone_does_not_leave_legacy_mode():
    assert not team_env_configured({"SCALER_TRACE_DIR": "traces"})
    assert load_team_config(env={**LEGACY, "SCALER_TRACE_DIR": "traces"}).source == "legacy-env"


# ---- env-driven config -------------------------------------------------------------------

def test_provider_preset_and_per_role_models():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_PLANNING_MODEL": "org/big",
                              "SCALER_DISPATCH_MODEL": "org/small", "SCALER_MARKET_MODEL": "org/small"})
    c = t.roles["planning"]
    assert c.base_url == "https://api.tokenfactory.nebius.com/v1/" and c.api_key_env == "NEBIUS_API_KEY"
    assert t.heterogeneous is True
    assert t.model_ids() == ["org/big", "org/small", "org/small"]


def test_single_model_for_all_roles_is_homogeneous():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "org/one"})
    assert t.heterogeneous is False and set(t.model_ids()) == {"org/one"}


def test_role_env_overrides_global_env():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "org/one", "SCALER_MAX_TOKENS": "300",
                              "SCALER_MARKET_MODEL": "org/two", "SCALER_MARKET_MAX_TOKENS": "900"})
    assert t.roles["market"].model == "org/two" and t.roles["market"].max_tokens == 900
    assert t.roles["dispatch"].model == "org/one" and t.roles["dispatch"].max_tokens == 300


def test_explicit_base_url_beats_preset_and_extra_body_json():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m",
                              "SCALER_DISPATCH_BASE_URL": "https://other.example/v1", "SCALER_DISPATCH_API_KEY_ENV": "OTHER_KEY",
                              "SCALER_PLANNING_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}'})
    assert t.roles["dispatch"].base_url == "https://other.example/v1" and t.roles["dispatch"].api_key_env == "OTHER_KEY"
    assert t.roles["planning"].extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert t.roles["market"].extra_body is None


# ---- file config -------------------------------------------------------------------------

def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_yaml_and_json_files_load_equivalently(tmp_path):
    y = write(tmp_path, "m.yaml", "provider: nebius\nroles:\n  planning: {model: a/p}\n  dispatch: {model: a/d}\n  market: {model: a/m, max_tokens: 500}\n")
    j = write(tmp_path, "m.json", json.dumps({"provider": "nebius", "roles": {
        "planning": {"model": "a/p"}, "dispatch": {"model": "a/d"}, "market": {"model": "a/m", "max_tokens": 500}}}))
    ty, tj = load_team_config(env={}, path=y), load_team_config(env={}, path=j)
    assert ty.roles == tj.roles and ty.roles["market"].max_tokens == 500 and ty.source == "file:m.yaml"
    assert ty.roles["dispatch"].timeout_s == 60.0  # file/env config defaults to a 60s request timeout


def test_env_overrides_file(tmp_path):
    p = write(tmp_path, "m.yaml", "provider: nebius\ndefaults: {max_tokens: 400}\nroles:\n  planning: {model: a}\n  dispatch: {model: b}\n  market: {model: c}\n")
    t = load_team_config(env={"SCALER_DISPATCH_MODEL": "z", "SCALER_MARKET_MAX_TOKENS": "50"}, path=p)
    assert t.roles["dispatch"].model == "z" and t.roles["planning"].max_tokens == 400
    assert t.roles["market"].max_tokens == 50


def test_models_file_env_var(tmp_path):
    p = write(tmp_path, "m.json", json.dumps({"provider": "groq", "defaults": {"model": "x"}}))
    t = load_team_config(env={"SCALER_MODELS_FILE": str(p)})
    assert t.roles["market"].model == "x" and t.roles["market"].api_key_env == "GROQ_API_KEY"


def test_example_config_loads():
    t = load_team_config(env={}, path=EXAMPLE)
    assert t.roles["planning"].max_tokens == 1024 and t.roles["planning"].plan_max_tokens == 1536
    assert t.roles["dispatch"].plan_max_tokens is None


@pytest.mark.parametrize("text, match", [
    ("provider: nebius\nbogus: 1\nroles: {planning: {model: a}}", "Unknown key"),
    ("provider: nebius\nroles: {judge: {model: a}}", "Unknown key"),
    ("provider: nebius\nroles: {planning: {modle: a}}", "Unknown key"),
    ("provider: nope\nroles: {}", "unknown provider"),
    ("provider: nebius\nroles:\n  planning: {model: a}\n  dispatch: {model: a, plan_max_tokens: 5}\n  market: {model: a}", "plan_max_tokens"),
    ("provider: nebius\ndefaults: {temperature: 9}\nroles: {}", "temperature"),
    ("provider: nebius\ndefaults: {max_tokens: 0}\nroles: {}", "max_tokens"),
    ("provider: nebius\nroles:\n  planning: {model: a}\n  dispatch: {model: a}", "market"),   # market has no model
    ("- just\n- a list", "mapping"),
])
def test_invalid_files_raise_clear_errors(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load_team_config(env={}, path=write(tmp_path, "bad.yaml", text))


def test_missing_file_and_bad_env_values():
    with pytest.raises(ConfigError, match="not found"):
        load_team_config(env={}, path="/nonexistent/models.yaml")
    with pytest.raises(ConfigError, match="extra_body"):
        load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m", "SCALER_PLANNING_EXTRA_BODY": "{not json"})


# ---- secrets -----------------------------------------------------------------------------

def test_key_value_never_appears_in_config_output():
    env = {"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m", "NEBIUS_API_KEY": CANARY}
    t = load_team_config(env=env)
    text = "\n".join(describe_team(t, env)) + json.dumps([t.roles[r].to_manifest() for r in ROLES]) + repr(t)
    assert CANARY not in text
    assert "NEBIUS_API_KEY(set)" in text


def test_missing_key_error_names_the_variable_only():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m"})
    with pytest.raises(ConfigError, match="NEBIUS_API_KEY"):
        resolve_api_key(t.roles["dispatch"], env={})


# ---- client building ---------------------------------------------------------------------

def test_clients_are_shared_per_endpoint_and_key():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m",
                              "SCALER_MARKET_BASE_URL": "https://other.example/v1", "SCALER_MARKET_API_KEY_ENV": "OTHER"})
    made = []

    def factory(base_url, key, timeout):
        made.append((base_url, key, timeout))
        return object()

    team = build_team(t, client_factory=factory, env={"NEBIUS_API_KEY": "k1", "OTHER": "k2"})
    assert len(made) == 2
    assert team["planning"].client is team["dispatch"].client and team["market"].client is not team["dispatch"].client
    assert ("https://other.example/v1", "k2", 60.0) in made


def test_build_team_reports_missing_key_without_creating_clients():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m"})
    with pytest.raises(ConfigError, match="NEBIUS_API_KEY"):
        build_team(t, client_factory=lambda *a: pytest.fail("client must not be created"), env={})


def test_client_factory_error_does_not_leak_key():
    t = load_team_config(env={"SCALER_PROVIDER": "nebius", "SCALER_MODEL": "m"})

    def boom(base_url, key, timeout):
        raise RuntimeError(f"bad credentials {key}")

    with pytest.raises(ConfigError) as exc:
        build_team(t, client_factory=boom, env={"NEBIUS_API_KEY": CANARY})
    assert CANARY not in str(exc.value)
