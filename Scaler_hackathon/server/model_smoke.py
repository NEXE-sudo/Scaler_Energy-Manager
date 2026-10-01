"""Smoke test for a model team: verifies keys, model IDs and one realistic call per role.

Uses the same call and response-interpretation code as the real runner, so a PASS
here means the runner can parse that model's output. Never prints key values.

If a role fails because a reasoning model produced no answer, two diagnostic
probes are run (more tokens; reasoning switched off) and the working fix, if any,
is reported as a config suggestion.
"""
from __future__ import annotations

import difflib
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from server.baseline import _build_system_prompt, _call_llm_detailed, _interpret_call
from server.energy_grid_environment import EnergyGridEnvironment
from server.llm_adapter import observation_to_text
from server.model_config import (
    ROLES, ConfigError, RoleModelConfig, TeamConfig, _default_client_factory, describe_team, resolve_api_key,
)

THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
RETRYABLE_STATUSES = {"empty", "reasoning_truncated", "reasoning_only"}


def _list_model_ids(client: Any) -> Tuple[Optional[List[str]], Optional[str]]:
    try:
        return sorted(m.id for m in client.models.list().data), None
    except Exception as e:  # listing may be unsupported or forbidden; not fatal
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def suggest_models(wanted: str, available: Sequence[str], n: int = 5) -> List[str]:
    tail = wanted.split("/")[-1].lower()
    hits = [m for m in available if tail in m.lower() or m.lower().split("/")[-1] in wanted.lower()]
    hits += difflib.get_close_matches(wanted, available, n=n, cutoff=0.5)
    seen: List[str] = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return seen[:n]


def _one_call(client: Any, cfg: RoleModelConfig, system: str, user: str,
              max_tokens: Optional[int] = None, extra_body: Optional[Dict[str, Any]] = None):
    call = _call_llm_detailed(
        client, cfg.model, system, [{"role": "user", "content": user}], max_retries=2,
        max_tokens=max_tokens or cfg.max_tokens, temperature=cfg.temperature,
        extra_body=extra_body if extra_body is not None else cfg.extra_body, timeout=cfg.timeout_s,
    )
    if not call.ok:
        return call, "call_failed", None
    _, status, reasoning = _interpret_call(call)
    return call, status, reasoning


def run_smoke(
    team: TeamConfig,
    client_factory: Optional[Callable[[str, str, Optional[float]], Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    roles: Optional[Sequence[str]] = None,
    list_filter: Optional[str] = None,
    list_only: bool = False,
    probe: bool = True,
    out: Callable[[str], None] = print,
) -> int:
    """Returns 0 if every tested role passes, 1 otherwise."""
    env = os.environ if env is None else env
    factory = client_factory or _default_client_factory
    chosen = [r for r in ROLES if roles is None or r in roles]

    out("Model smoke test")
    for line in describe_team(team, env):
        out("  " + line)

    clients: Dict[Tuple[str, str, Optional[float]], Any] = {}
    listings: Dict[Tuple[str, str, Optional[float]], Optional[List[str]]] = {}
    failures = 0
    ready: List[str] = []
    for role in chosen:
        cfg = team.roles[role]
        ident = (cfg.base_url, cfg.api_key_env, cfg.timeout_s)
        try:
            if ident not in clients:
                clients[ident] = factory(cfg.base_url, resolve_api_key(cfg, env), cfg.timeout_s)
        except ConfigError as e:
            out(f"[FAIL] {role:9s} {e}")
            failures += 1
            continue
        except Exception as e:
            out(f"[FAIL] {role:9s} could not create client ({type(e).__name__})")
            failures += 1
            continue
        ready.append(role)

    for ident, client in clients.items():
        ids, err = _list_model_ids(client)
        listings[ident] = ids
        host = ident[0].split("//")[-1].split("/")[0]
        if ids is None:
            out(f"[WARN] {host}: could not list models ({err}); will still try the calls")
        else:
            out(f"[ OK ] {host}: {len(ids)} models listed")
            if list_filter is not None:
                for m in ids:
                    if list_filter.lower() in m.lower():
                        out(f"         {m}")
    if list_only:
        return 1 if failures else 0

    env_obj = EnergyGridEnvironment()
    user_text = observation_to_text(env_obj.reset("medium").model_dump())

    for role in ready:
        cfg = team.roles[role]
        ident = (cfg.base_url, cfg.api_key_env, cfg.timeout_s)
        client = clients[ident]
        ids = listings.get(ident)
        if ids is not None and cfg.model not in ids:
            sugg = suggest_models(cfg.model, ids)
            out(f"[WARN] {role:9s} '{cfg.model}' is not in the model list"
                + (f"; did you mean: {', '.join(sugg)}" if sugg else ""))
        system = _build_system_prompt("medium", "", 0, agent_type=role)
        call, status, reasoning = _one_call(client, cfg, system, user_text)
        ok = status == "ok"
        tokens = f"{call.prompt_tokens}/{call.completion_tokens}" if call.prompt_tokens is not None else "n/a"
        out(f"[{'PASS' if ok else 'FAIL'}] {role:9s} {cfg.model}  {call.latency_s:.1f}s  tokens(in/out)={tokens}  "
            f"finish={call.finish_reason}  parse={status}  reasoning={'yes' if reasoning else 'no'}"
            + (f"  error={call.error}" if call.error else ""))
        if ok:
            continue
        failures += 1
        if probe and status in RETRYABLE_STATUSES:
            bigger = min(cfg.max_tokens * 4, 4096)
            _, st_a, _ = _one_call(client, cfg, system, user_text, max_tokens=bigger)
            out(f"         probe A (max_tokens={bigger}, reasoning unchanged): parse={st_a}"
                + ("  -> raise this role's max_tokens" if st_a == "ok" else ""))
            if not cfg.extra_body:
                _, st_b, _ = _one_call(client, cfg, system, user_text, extra_body=THINKING_OFF)
                out(f"         probe B (extra_body enable_thinking=false): parse={st_b}"
                    + ("  -> add extra_body: {chat_template_kwargs: {enable_thinking: false}}" if st_b == "ok" else ""))

    out(f"Result: {'ALL PASS' if failures == 0 else f'{failures} problem(s)'}")
    return 0 if failures == 0 else 1
