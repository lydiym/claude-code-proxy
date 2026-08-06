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
- `haiku` uses `SMALL_MODEL`; everything else uses `BIG_MODEL` by default
- Per-tier overrides (`HAIKU_MODEL`, `SONNET_MODEL`, `OPUS_MODEL`, `FABLE_MODEL`, `MYTHOS_MODEL`) take precedence over `BIG_MODEL`/`SMALL_MODEL`
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

- `TIKTOKEN_OFFLINE` (default `true`) — stubs `tiktoken.get_encoding` and `encoding_for_model` with a fake encoding that returns 1-token-per-4-chars. Real counts come from upstream's `usage`. Set to `false` to let tiktoken fetch `cl100k_base.tiktoken` from Azure.
- `LITELLM_LOCAL_MODEL_COST_MAP=True` — set before importing litellm so it doesn't try to refresh the model cost map from GitHub.
- `OPENAI_TLS_VERIFY=false` — disables TLS verification for `OPENAI_BASE_URL` endpoints with self-signed certs.

`MAX_OUTPUT_TOKENS = 16384` is the OpenAI cap; `litellm_request["max_completion_tokens"]` is clamped to it.

### Logging

Logs go to stderr with a single timestamped format. `log_request` prints one line per request with colour-coded model names (ANSI when stderr is a TTY). `uvicorn.access` is silenced because we already log via `log_request`. The lifespan hook `_configure_logging` re-applies the format after uvicorn installs its own handlers.

## Tests

`tests.py` has two modes:

- **Unit tests** are top-level `test_*` functions (no `pytest` required). They cover model mapping, request/response conversion, image conversion, tool choice, dangling tool call handling, SSE formatting.
- **Integration tests** are scenarios in `TEST_SCENARIOS` exercised via `httpx` against a running proxy. They verify streaming and non-streaming behaviour end-to-end and require a valid `OPENAI_API_KEY`.

Follow the convention: add a unit test by writing a `test_foo` function; it'll be picked up automatically. Use `assert` statements — failures are caught and printed with the function name.

## Files to know

- `server.py` — everything (proxy, models, translation, streaming, env config)
- `tests.py` — everything (unit + integration tests)
- `pyproject.toml` / `uv.lock` — dependency manifest (FastAPI, uvicorn, pydantic, litellm, python-dotenv)
- `.env.example` — documents every env var the proxy reads
- `Dockerfile` — `python:latest` + uv; exposes port 8082
- `README.md` — quick start and model mapping table
