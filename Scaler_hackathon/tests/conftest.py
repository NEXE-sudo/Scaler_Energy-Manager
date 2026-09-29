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

    replies: role -> str | callable(call_index_for_role) -> str | Exception
    """

    def __init__(self, replies=None, usage=(11, 7)):
        self.replies = {**DEFAULT_REPLIES, **(replies or {})}
        self.calls = []  # list of dicts: role, model, system, user, kwargs
        self._per_role = {}
        self._usage = usage
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

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
        reply = self.replies[role]
        if callable(reply):
            reply = reply(n)
        if isinstance(reply, Exception):
            raise reply
        usage = SimpleNamespace(prompt_tokens=self._usage[0], completion_tokens=self._usage[1])
        msg = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


@pytest.fixture
def fake_client():
    return FakeClient()
