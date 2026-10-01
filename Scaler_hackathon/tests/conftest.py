"""Shared test fixtures: a scripted fake OpenAI-compatible client (no network)."""
from types import SimpleNamespace

import pytest


def role_of(system_prompt: str) -> str:
    if "Planning Agent" in system_prompt or "strategic planner" in system_prompt:
        return "planning"
    if "Dispatch Agent" in system_prompt:
        return "dispatch"
    if "Market Agent" in system_prompt:
        return "market"
    return "unknown"


DEFAULT_REPLIES = {
    "planning": 'Thought:\nhold\nAction:\n{"plant_action": "none"}',
    "dispatch": (
        'Thought:\nsteady\nAction:\n{"coal_delta": 5, "hydro_delta": 0, '
        '"nuclear_delta": 0, "battery_mode": "idle", "emergency_coal_boost": false}'
    ),
    "market": (
        'Thought:\nhold\nAction:\n{"demand_response_mw": 0, "grid_export_mw": 0, '
        '"grid_import_mw": 0, "coal_price_bid": null}'
    ),
    "unknown": "{}",
}


class FakeClient:
    """Records every chat.completions.create call and replies from a script.

    replies: role -> str | None | dict | Exception | callable(n_for_role)
        dict form: {"content": str|None, "reasoning_content": str|None, "finish_reason": str}
    fn: optional callable(role, n_for_role, kwargs) -> reply, takes precedence over `replies`
    model_ids / list_error: what client.models.list() returns / raises
    base_url: exposed like the real SDK client (used for the trace's provider host)
    """

    def __init__(self, replies=None, usage=(11, 7), fn=None, model_ids=(), list_error=None, base_url=None):
        self.replies = {**DEFAULT_REPLIES, **(replies or {})}
        self.fn = fn
        self.calls = []  # list of dicts: role, model, system, user, kwargs
        self._per_role = {}
        self._usage = usage
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._model_ids = list(model_ids)
        self._list_error = list_error
        self.list_calls = 0
        self.models = SimpleNamespace(list=self._list_models)

    def _list_models(self):
        self.list_calls += 1
        if self._list_error is not None:
            raise self._list_error
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self._model_ids])

    def _create(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        user = messages[-1]["content"]
        role = role_of(system)
        n = self._per_role.get(role, 0)
        self._per_role[role] = n + 1
        self.calls.append(
            {"role": role, "model": kwargs.get("model"), "system": system, "user": user, "kwargs": kwargs}
        )
        reply = self.fn(role, n, kwargs) if self.fn else self.replies[role]
        if callable(reply):
            reply = reply(n)
        if isinstance(reply, Exception):
            raise reply
        usage = SimpleNamespace(prompt_tokens=self._usage[0], completion_tokens=self._usage[1])
        finish = "stop"
        extra = {}
        if isinstance(reply, dict):
            finish = reply.get("finish_reason", "stop")
            if reply.get("reasoning_content") is not None:
                extra["reasoning_content"] = reply["reasoning_content"]
            reply = reply.get("content")
        msg = SimpleNamespace(content=reply, **extra)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage)


@pytest.fixture
def fake_client():
    return FakeClient()
