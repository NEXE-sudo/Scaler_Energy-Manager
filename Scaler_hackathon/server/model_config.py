"""Per-role model configuration.

Each agent role (planning, dispatch, market) can use its own model, provider
endpoint, token budget, temperature, timeout and extra request body. Config
holds the *name* of the environment variable that contains the API key, never
the key itself, so configs can be committed, logged and put in trace manifests.

Sources, lowest to highest precedence:
    built-in defaults
    < config file (JSON or YAML; SCALER_MODELS_FILE or an explicit path)
    < SCALER_* environment variables

Legacy mode: if no SCALER_* configuration variable is present and no file is
given, the original API_BASE_URL / MODEL_NAME / PLANNING_MODEL variables are
used, exactly as before (dispatch and market share MODEL_NAME).

Environment variables (all optional):
    SCALER_MODELS_FILE            path to a JSON/YAML config file
    SCALER_PROVIDER               provider preset: nebius | groq
    SCALER_MODEL                  model for every role (overridden per role)
    SCALER_MAX_TOKENS, SCALER_TEMPERATURE, SCALER_TIMEOUT_S, SCALER_EXTRA_BODY
    SCALER_<ROLE>_<FIELD>         ROLE in PLANNING|DISPATCH|MARKET and FIELD in
                                  PROVIDER, MODEL, BASE_URL, API_KEY_ENV,
                                  MAX_TOKENS, PLAN_MAX_TOKENS (planning only),
                                  TEMPERATURE, TIMEOUT_S, EXTRA_BODY (JSON)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

ROLES: Tuple[str, ...] = ("planning", "dispatch", "market")

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "nebius": {"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key_env": "NEBIUS_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
}

# Defaults equal the values the runner used before this module existed.
DEFAULT_MAX_TOKENS = 256
DEFAULT_PLAN_MAX_TOKENS = 600
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_S = 60.0
LEGACY_PLANNING_MODEL = "openai/gpt-oss-120b"
LEGACY_KEY_ENV_CANDIDATES = ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "API_KEY")

_FIELDS = {"provider", "model", "base_url", "api_key_env", "max_tokens", "plan_max_tokens",
           "temperature", "timeout_s", "extra_body"}
# Env vars that start with SCALER_ but do not configure models.
_NON_CONFIG_ENV = {"SCALER_TRACE_DIR"}


class ConfigError(EnvironmentError):
    """Invalid or incomplete model configuration. Messages never contain secrets."""


@dataclass(frozen=True)
class RoleModelConfig:
    role: str
    model: str
    base_url: str
    api_key_env: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout_s: Optional[float] = DEFAULT_TIMEOUT_S
    extra_body: Optional[Dict[str, Any]] = None
    plan_max_tokens: Optional[int] = None  # planning role only: one-shot plan call budget

    def to_manifest(self) -> Dict[str, Any]:
        """Safe-to-log description (no secrets: only the key's variable name)."""
        from urllib.parse import urlparse
        return {
            "model": self.model, "provider_host": urlparse(self.base_url).netloc or None,
            "api_key_env": self.api_key_env, "max_tokens": self.max_tokens,
            "plan_max_tokens": self.plan_max_tokens, "temperature": self.temperature,
            "timeout_s": self.timeout_s, "extra_body": self.extra_body,
        }


@dataclass(frozen=True)
class TeamConfig:
    roles: Dict[str, RoleModelConfig]
    source: str = "defaults"  # "legacy-env" | "env" | "file:<path>"

    def model_ids(self) -> List[str]:
        return [self.roles[r].model for r in ROLES]

    @property
    def heterogeneous(self) -> bool:
        """True only if the roles use genuinely different model IDs."""
        return len(set(self.model_ids())) > 1


@dataclass
class RoleRuntime:
    """A role's config plus the client that serves it."""
    client: Any
    cfg: RoleModelConfig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def team_env_configured(env: Optional[Mapping[str, str]] = None) -> bool:
    """True if a config file or any SCALER_* model variable is set (i.e. not legacy mode)."""
    env = os.environ if env is None else env
    return any(k.startswith("SCALER_") and k not in _NON_CONFIG_ENV and env[k] != "" for k in env)


def _coerce(field_name: str, value: Any, where: str) -> Any:
    try:
        if field_name in ("max_tokens", "plan_max_tokens"):
            v = int(value)
            if v <= 0:
                raise ValueError("must be > 0")
            return v
        if field_name == "temperature":
            v = float(value)
            if not 0.0 <= v <= 2.0:
                raise ValueError("must be between 0 and 2")
            return v
        if field_name == "timeout_s":
            if value in (None, "", "none", "None", 0, "0"):
                return None
            v = float(value)
            if v <= 0:
                raise ValueError("must be > 0")
            return v
        if field_name == "extra_body":
            if isinstance(value, str):
                value = json.loads(value) if value.strip() else None
            if value is not None and not isinstance(value, dict):
                raise ValueError("must be a JSON object")
            return value
        if field_name == "provider":
            if value not in PROVIDER_PRESETS:
                raise ValueError(f"unknown provider; choose from {sorted(PROVIDER_PRESETS)}")
            return value
        if field_name == "base_url":
            if not str(value).startswith(("http://", "https://")):
                raise ValueError("must start with http:// or https://")
            return str(value)
        if field_name in ("model", "api_key_env"):
            if not str(value).strip():
                raise ValueError("must not be empty")
            return str(value).strip()
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ConfigError(f"Invalid value for '{field_name}' in {where}: {e}") from None
    raise ConfigError(f"Unknown field '{field_name}' in {where}")


def _check_keys(d: Mapping[str, Any], allowed: set, where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}")


def _read_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Model config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # lazy: only needed for YAML files
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except Exception as e:
        raise ConfigError(f"Could not parse {path.name}: {type(e).__name__}: {str(e)[:200]}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    _check_keys(data, {"provider", "defaults", "roles"}, path.name)
    return data


def _layers_from_file(data: Mapping[str, Any], where: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    base: Dict[str, Any] = {}
    if "provider" in data:
        base["provider"] = _coerce("provider", data["provider"], where)
    defaults = data.get("defaults") or {}
    _check_keys(defaults, _FIELDS - {"plan_max_tokens"}, f"{where} defaults")
    for k, v in defaults.items():
        base[k] = _coerce(k, v, f"{where} defaults")
    per_role: Dict[str, Dict[str, Any]] = {}
    roles = data.get("roles") or {}
    _check_keys(roles, set(ROLES), f"{where} roles")
    for role, spec in roles.items():
        spec = spec or {}
        _check_keys(spec, _FIELDS, f"{where} roles.{role}")
        per_role[role] = {k: _coerce(k, v, f"{where} roles.{role}") for k, v in spec.items()}
    return base, per_role


def _layers_from_env(env: Mapping[str, str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    base: Dict[str, Any] = {}
    for field_name in ("provider", "model", "max_tokens", "temperature", "timeout_s", "extra_body"):
        raw = env.get(f"SCALER_{field_name.upper()}")
        if raw not in (None, ""):
            base[field_name] = _coerce(field_name, raw, f"SCALER_{field_name.upper()}")
    per_role: Dict[str, Dict[str, Any]] = {}
    for role in ROLES:
        spec: Dict[str, Any] = {}
        for field_name in _FIELDS:
            name = f"SCALER_{role.upper()}_{field_name.upper()}"
            if env.get(name, "") != "":
                spec[field_name] = _coerce(field_name, env[name], name)
        if spec:
            per_role[role] = spec
    return base, per_role


def _legacy_team(env: Mapping[str, str]) -> TeamConfig:
    base_url = env.get("API_BASE_URL")
    model = env.get("MODEL_NAME")
    if not (base_url and model):
        raise ConfigError(
            f"Missing required API configuration. API_BASE_URL set={bool(base_url)}, "
            f"MODEL_NAME set={bool(model)}. (Or configure models with SCALER_* variables / "
            f"SCALER_MODELS_FILE; see server/model_config.py.)"
        )
    key_env = next((k for k in LEGACY_KEY_ENV_CANDIDATES if env.get(k)), LEGACY_KEY_ENV_CANDIDATES[1])
    planning_model = env.get("PLANNING_MODEL", LEGACY_PLANNING_MODEL)
    roles = {}
    for role in ROLES:
        roles[role] = RoleModelConfig(
            role=role, model=planning_model if role == "planning" else model,
            base_url=base_url, api_key_env=key_env, timeout_s=None,  # legacy: no explicit timeout
            plan_max_tokens=DEFAULT_PLAN_MAX_TOKENS if role == "planning" else None,
        )
    return TeamConfig(roles, source="legacy-env")


def load_team_config(env: Optional[Mapping[str, str]] = None, path: Optional[str | Path] = None) -> TeamConfig:
    env = os.environ if env is None else env
    file_path = path or env.get("SCALER_MODELS_FILE") or None
    if not file_path and not team_env_configured(env):
        return _legacy_team(env)

    file_base: Dict[str, Any] = {}
    file_roles: Dict[str, Dict[str, Any]] = {}
    source = "env"
    if file_path:
        p = Path(file_path)
        file_base, file_roles = _layers_from_file(_read_file(p), p.name)
        source = f"file:{p.name}"
    env_base, env_roles = _layers_from_env(env)

    roles: Dict[str, RoleModelConfig] = {}
    for role in ROLES:
        merged: Dict[str, Any] = {}
        for layer in (file_base, file_roles.get(role, {}), env_base, env_roles.get(role, {})):
            merged.update(layer)
        if role != "planning" and "plan_max_tokens" in merged:
            raise ConfigError("plan_max_tokens is only valid for the planning role")
        preset = PROVIDER_PRESETS.get(merged.get("provider", ""), {})
        model = merged.get("model")
        base_url = merged.get("base_url") or preset.get("base_url")
        key_env = merged.get("api_key_env") or preset.get("api_key_env")
        missing = [n for n, v in (("model", model), ("base_url (or provider)", base_url),
                                  ("api_key_env (or provider)", key_env)) if not v]
        if missing:
            raise ConfigError(f"Role '{role}' is missing: {', '.join(missing)}")
        roles[role] = RoleModelConfig(
            role=role, model=model, base_url=base_url, api_key_env=key_env,
            max_tokens=merged.get("max_tokens", DEFAULT_MAX_TOKENS),
            temperature=merged.get("temperature", DEFAULT_TEMPERATURE),
            timeout_s=merged["timeout_s"] if "timeout_s" in merged else DEFAULT_TIMEOUT_S,
            extra_body=merged.get("extra_body"),
            plan_max_tokens=(merged.get("plan_max_tokens", DEFAULT_PLAN_MAX_TOKENS) if role == "planning" else None),
        )
    return TeamConfig(roles, source=source)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def resolve_api_key(cfg: RoleModelConfig, env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    key = env.get(cfg.api_key_env)
    if not key:
        raise ConfigError(f"Environment variable {cfg.api_key_env} is not set (needed for the {cfg.role} role).")
    return key


def _default_client_factory(base_url: str, api_key: str, timeout: Optional[float]) -> Any:
    from openai import OpenAI
    kwargs: Dict[str, Any] = {"base_url": base_url, "api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def build_team(
    team: TeamConfig,
    client_factory: Optional[Callable[[str, str, Optional[float]], Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, RoleRuntime]:
    """One client per distinct (base_url, key variable, timeout); shared between roles."""
    factory = client_factory or _default_client_factory
    clients: Dict[Tuple[str, str, Optional[float]], Any] = {}
    out: Dict[str, RoleRuntime] = {}
    for role in ROLES:
        cfg = team.roles[role]
        ident = (cfg.base_url, cfg.api_key_env, cfg.timeout_s)
        if ident not in clients:
            try:
                clients[ident] = factory(cfg.base_url, resolve_api_key(cfg, env), cfg.timeout_s)
            except ConfigError:
                raise
            except Exception as e:
                raise ConfigError(f"Failed to initialise client for {cfg.role} ({type(e).__name__}); "
                                  f"check base_url and the key in {cfg.api_key_env}.") from None
        out[role] = RoleRuntime(clients[ident], cfg)
    return out


def describe_team(team: TeamConfig, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Human-readable, secret-free summary (shows whether each key variable is set)."""
    env = os.environ if env is None else env
    lines = [f"config source: {team.source}; heterogeneous models: {team.heterogeneous}"]
    for role in ROLES:
        c = team.roles[role]
        m = c.to_manifest()
        think = ""
        if c.extra_body:
            think = f" extra_body={json.dumps(c.extra_body, separators=(',', ':'))}"
        plan = f" plan_max_tokens={c.plan_max_tokens}" if c.plan_max_tokens else ""
        lines.append(
            f"  {role:9s} model={c.model} host={m['provider_host']} max_tokens={c.max_tokens}{plan} "
            f"temp={c.temperature} timeout={c.timeout_s} key_env={c.api_key_env}"
            f"({'set' if env.get(c.api_key_env) else 'NOT SET'}){think}"
        )
    return lines
