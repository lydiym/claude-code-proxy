# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A FastAPI proxy that accepts Anthropic Messages API requests, translates them to OpenAI Chat Completions via LiteLLM, and converts the response back. Single backend, single code path — `server.py` is the whole proxy, `tests.py` is the whole test suite.

## Commands

- **Run the server**: `uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload` (uses `uv`; deps are pinned in `uv.lock`)
- **Run with Docker**: `docker build -t proxy . && docker run -p 8082:8082 --env-file .env proxy`
- **Unit tests** (no network): `python tests.py` — default mode
- **Integration tests** (needs server running on `localhost:8082`): `python tests.py --integration`
- **All tests**: `python tests.py --all`
- **Filter integration scenarios**: `--simple` (skip tool tests) or `--tools` (only tool tests)
- **Add a unit test**: write a `test_*` function in `tests.py` — `discover_unit_tests()` collects them automatically, no decorators needed

## Architecture

The proxy is a single FastAPI app (`server.py`) with the following pieces. They're listed in the order a request flows through.

### Request entry

`POST /v1/messages` is the only HTTP endpoint that does work (`GET /` is a stub). It accepts an Anthropic-format `MessagesRequest` and calls `convert_anthropic_to_litellm`, then dispatches to `litellm.completion` (non-streaming) or `litellm.acompletion` (streaming).

### Model mapping (Pydantic validator)

`MessagesRequest.validate_model_field` runs during parsing and **rewrites** the model field to an `openai/<name>` string. Mapping rules:

- Substring match on tier name (`haiku`/`sonnet`/`opus`/`fable`/`mythos`) chooses the tier's target
- The tier's model is resolved by `_default_model_for_tier(tier)` (cached at first call): `{TIER}_MODEL` env → `{BIG|SMALL}_MODEL` env → `[tier].model` → `[bucket].model` (haiku → `small`, others → `big`) → `[global].model` → built-in default
- Bare names in `OPENAI_MODELS` get prefixed with `openai/`; an existing `openai/` prefix passes through; unknown names get prefixed (assumes custom OpenAI-compatible endpoint)
- `anthropic/`, `openai/`, `gemini/` prefixes are stripped before matching

The incoming model name is also captured into `original_model` by `capture_original_model` (a `model_validator(mode="before")`) so the log line can show what the client asked for vs. what was sent upstream.

### Request translation

`convert_anthropic_to_litellm` walks `request.messages` and calls `_convert_message` per message. Notable logic:

- **System messages**: top-level `system` field is hoisted into a single `{"role": "system"}`. In-band `role="system"` messages (which Claude Code 2.1.154+ sometimes injects) are merged in too — in-band first, then top-level, with `\n\n` between them.
- **Tool calls**: `assistant.tool_calls` + `role="tool"` messages are preserved literally rather than flattened to text. Tests assert that small models can still do tool use after this conversion.
- **Dangling tool calls** (no matching `tool_result`): folded into a prose description so the model doesn't re-emit the call verbatim.
- **Orphaned tool results** (no matching `tool_use`): folded into user text.
- `sanitize_messages_for_openai` strips any extra keys not in `{role, content, name, tool_call_id, tool_calls}` before the upstream call.

### Response translation

`convert_litellm_to_anthropic` handles non-streaming. `_build_content_blocks` emits blocks in order: `thinking` (from `reasoning_content`), `text`, then `tool_use` for each call. Tool call `arguments` strings are JSON-parsed; non-JSON falls back to `{"raw": "..."}`.

`handle_streaming` handles streaming. Key collaborators:

- **`BlockTracker`** — manages which Anthropic content block index is currently open. `ensure(kind)` closes a different-kind block first; `open(kind)` is unconditional. The tool_use path relies on this so consecutive parallel tool calls each get a fresh index without an intervening `content_block_stop`.
- **`ThinkStreamParser`** — splits a stream into text vs. thinking chunks by recognising `think`/`/think` markers that some backends inline into `content`. Buffers across chunks so a marker split mid-stream is handled.
- **`SseFormatter`** — stateless helpers for the Anthropic SSE event shape (`event: <name>\ndata: <json>\n\n`). The trailing `data: [DONE]\n\n` is the only frame without an `event:` line.

Native `delta.reasoning_content` from the upstream is honoured before falling back to  think-tag parsing.

### Environment / network

These three knobs make the proxy work on isolated networks:

- `TIKTOKEN_OFFLINE` (default `true`) — stubs `tiktoken.get_encoding` and `encoding_for_model` with a fake encoding that returns 1-token-per-4-chars. Real counts come from upstream's `usage`. Set to `false` to let tiktoken fetch `cl100k_base.tiktoken` from Azure. Resolved standalone (reads `[proxy].tiktoken_offline`, then the env var) so it can run before `import litellm`.
- `LITELLM_LOCAL_MODEL_COST_MAP=True` — set before importing litellm so it doesn't try to refresh the model cost map from GitHub.
- `OPENAI_TLS_VERIFY=false` — disables TLS verification for `OPENAI_BASE_URL` endpoints with self-signed certs.

`MAX_OUTPUT_TOKENS = 16384` is the documented OpenAI cap; the proxy no longer clamps — client `max_tokens` flows through as `max_completion_tokens` unmodified. If a user asks for 24000, upstream gets 24000.

### Configuration (config.toml)

The proxy loads `config.toml` (default `./config.toml`, override via `CONFIG_PATH`) once at import time. Every key is optional and falls back to the equivalent env var — so `.env` is only needed for Docker overrides. The loader uses stdlib `tomllib` (Python ≥ 3.12).

**Schema** (normalised to `CONFIG = {"proxy": ..., "global": ..., "big": ..., "small": ..., "tiers": {...}}`):

- `[proxy]` — `openai_api_key`, `openai_base_url`, `openai_tls_verify`. `tiktoken_offline` is parsed by the standalone resolver above (not part of the main loader schema).
- `[big]` / `[small]` — bucket sections; accept `model` (str) + `extra_body` (dict). haiku → `[small]`, everything else → `[big]`.
- `[global]` — fallback per-tier settings; accepts `model` (catch-all for any model, including unmapped) + `extra_body` (sampling/reasoning/vendor knobs).
- `[haiku]` / `[sonnet]` / `[opus]` / `[fable]` / `[mythos]` — per-tier sections; same shape as `[big]`/`[small]` (optional `model` + `extra_body`).

**Resolver** (`_proxy_value`, `_proxy_bool`) is the single read path for every proxy setting. Lookup order per key: env var → `CONFIG` → built-in default. Env wins so `docker run -e KEY=VAL` and `docker-compose.yml: environment:` override `config.toml` without rebuilding the image.

**Per-tier capture** happens in `MessagesRequest.derive_tier` (a `@model_validator(mode="after")`) which inspects `original_model`, strips any `anthropic/` / `openai/` / `gemini/` prefix, and substring-matches against `TIER_KEYS` (insertion-order priority: `haiku` first). Unknown models end up with `tier=None` and fall back to `[global]`.

**Per-tier injection** at the tail of `convert_anthropic_to_litellm`:

- The tier's merged config is produced by `_resolve_tier_config(request)`: layers `[global]` → `[bucket]` (`[small]` for haiku, `[big]` otherwise) → `[tier]` are deep-merged in order; later wins per leaf. `model` is stripped from the result.
- Sampling fields (`temperature`, `top_p`, `top_k`, `stop_sequences`): Pydantic pass-through (only when client sent a non-null value via `model_fields_set`), then `extra_body` overrides per leaf. Field is **omitted from the upstream call** when neither sets it (no Anthropic defaults auto-applied).
- `seed`: config-only field (no Anthropic counterpart), forwarded as top-level kwarg when set in `[tier].extra_body`.
- `extra_body`: deep-merged via `_deep_merge` from `request.extra_body` (client) then `tier_cfg["extra_body"]` (already global+bucket+tier-merged). Config wins per leaf; unrelated client keys are preserved. Keys are lifted to top-level kwargs on the upstream call (works for any OpenAI-compatible backend: llama-server, ollama, vLLM, llama.cpp). The merged key list is also set as `allowed_openai_params` twice — top-level so our litellm hop extends `supported_params` and lets vendor keys through (`utils.py:3877`), and inside `extra_body` so cascade proxies like OpenWebUI see the whitelist in the wire body and forward verbatim instead of filtering (`llms/openai_like/chat/handler.py:241,254-259`).
- Protected keys (`model`, `messages`, `stream`, `tools`): if present in `extra_body`, the proxy warns and skips — these are owned by the proxy itself.

The discriminator between "client sent the field" and "Pydantic default applied" is `MessagesRequest.model_fields_set` (Pydantic v2) — an O(1) set lookup. `_patched_config` in `tests.py` is the patching helper for unit tests.

### Logging

Logs go to stderr with a single timestamped format. `log_request` prints one line per request with colour-coded model names (ANSI when stderr is a TTY). `uvicorn.access` is silenced because we already log via `log_request`. The lifespan hook `_configure_logging` re-applies the format after uvicorn installs its own handlers, and reads `LOG_LEVEL` (default `INFO`). At `DEBUG` level, the request handler dumps the effective upstream sampling params + `extra_body` (whether sourced from request or `[tier]` config) so operators can verify backend knobs without attaching mitmproxy.

### Comments

Comments are for non-obvious things only. Laconic, direct, concise — one line when possible. No prose explanations of what the code obviously does. No "what I changed and why" history in the diff itself (commit messages carry that).

## Tests

`tests.py` has two modes:

- **Unit tests** are top-level `test_*` functions (no `pytest` required). They cover model mapping, request/response conversion, image conversion, tool choice, dangling tool call handling, SSE formatting.
- **Integration tests** are scenarios in `TEST_SCENARIOS` exercised via `httpx` against a running proxy. They verify streaming and non-streaming behaviour end-to-end and require a valid `OPENAI_API_KEY`.

Follow the convention: add a unit test by writing a `test_foo` function; it'll be picked up automatically. Use `assert` statements — failures are caught and printed with the function name.

## Files to know

- `server.py` — everything (proxy, models, translation, streaming, config loader)
- `tests.py` — everything (unit + integration tests)
- `pyproject.toml` / `uv.lock` — dependency manifest (FastAPI, uvicorn, pydantic, litellm, python-dotenv); `requires-python = ">=3.12"` for stdlib `tomllib`
- `config.toml.example` — reference for the TOML schema (copy to `config.toml` to use)
- `.env.example` — documents every env var the proxy reads (env-var fallback per key)
- `Dockerfile` — `python:latest` + uv; exposes port 8082
- `README.md` — quick start, model mapping table, full config schema
