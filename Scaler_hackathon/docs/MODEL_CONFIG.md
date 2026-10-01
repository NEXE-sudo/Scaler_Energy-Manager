# Model configuration (per role)

Each agent role — **planning**, **dispatch**, **market** — can use its own model, endpoint,
token budget, temperature, timeout and extra request body. A team is *heterogeneous* only if
the roles use different model IDs; traces record the actual IDs.

## Quick start: Nebius Token Factory

Endpoint `https://api.tokenfactory.nebius.com/v1/` and key variable `NEBIUS_API_KEY`
(from the Token Factory docs). Model IDs must come from your account's model list.

```bash
export NEBIUS_API_KEY=...                       # never commit; or put it in a git-ignored .env
export SCALER_PROVIDER=nebius
python scripts/smoke_test_models.py --list-models nemotron     # find the exact IDs your key can use

export SCALER_PLANNING_MODEL=<id>               # e.g. a larger model for planning
export SCALER_DISPATCH_MODEL=<id>               # e.g. a smaller, faster model
export SCALER_MARKET_MODEL=<id>
python scripts/smoke_test_models.py             # one real call per role + diagnostics
SCALER_TRACE_DIR=traces MAX_EVAL_STEPS=0 python inference.py
```

One model for every role (homogeneous baseline): `SCALER_MODEL=<id>`.
A config file works too: `SCALER_MODELS_FILE=config/models.example.yaml`
(precedence: defaults < file < `SCALER_*` variables).

## Reasoning models

Some models (reported for Nemotron on Token Factory) spend the token budget on thinking and
return an empty `content`, with the text in `reasoning_content` or inside `<think>` tags.
The runner separates reasoning from the answer, never executes an action found only in
reasoning text, and reports these statuses instead of silently idling:

| parse status | meaning |
|---|---|
| `reasoning_truncated` | budget exhausted while thinking (`finish_reason=length` or unclosed `<think>`) |
| `reasoning_only` | no answer; text only in reasoning |
| `empty` / `unparseable` / `call_failed` | nothing returned / no JSON action / request failed |

Fixes, per role: raise `max_tokens` (and `plan_max_tokens` for planning), or set
`extra_body: {chat_template_kwargs: {enable_thinking: false}}`. Whether a provider honours the
latter must be checked with the smoke test; it tries both when a role fails.

## Legacy variables

With no `SCALER_*` model variables set, `API_BASE_URL`, `MODEL_NAME`, `PLANNING_MODEL` and
`HF_TOKEN`/`GROQ_API_KEY`/`OPENAI_API_KEY` behave exactly as before.

## Secrets

Config stores the *name* of the key variable, never its value. Keys are not printed, traced or
saved to results. Test: `tests/test_model_config.py`, `tests/test_http_integration.py`.
