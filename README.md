# Anthropic API Proxy for OpenAI 🔄

**Use Anthropic clients (like Claude Code) with any OpenAI-compatible backend.** 🤝

A small proxy that accepts requests in the Anthropic Messages API format, translates them to OpenAI Chat Completions via LiteLLM, and converts the response back. Single backend, single code path.

## Quick Start ⚡

### Prerequisites

- An OpenAI API key, or a key for any OpenAI-compatible endpoint 🔑
- Python ≥ 3.12 🐍
- [uv](https://github.com/astral-sh/uv) installed

### Setup 🛠️

#### From source

1. **Clone and enter the repo:**
   ```bash
   git clone https://github.com/lydiym/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Configure the proxy.** You have two equivalent options — pick whichever fits your workflow:
   - **`config.toml`** (recommended): copy `config.toml.example` to `config.toml`, then edit. See [Configuration](#configuration) below for the full schema.
   - **`.env`** (for Docker / simple setups): copy `.env.example` to `.env`, then edit. Every key in `.env` has a TOML equivalent in `[proxy]` / `[big]` / `[small]`.

   ```bash
   cp config.toml.example config.toml   # or: cp .env.example .env
   ```

3. **Run the server:**
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```

### Using with Claude Code 🎮

```bash
npm install -g @anthropic-ai/claude-code
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it — Claude Code sends Anthropic-format requests; the proxy translates them to OpenAI format and returns the response in Anthropic format.

## Model Mapping 🗺️

Claude Code sends requests naming Claude models (`claude-3-5-sonnet-...`, `claude-3-5-haiku-...`). The proxy remaps them to the OpenAI backend like this:

| Claude Model | Default Mapping | Override |
|--------------|-----------------|----------|
| haiku        | `openai/[small].model` (default `gpt-4.1-mini`) | set `SMALL_MODEL`, `HAIKU_MODEL`, or `[small].model` / `[haiku].model` |
| sonnet       | `openai/[big].model` (default `gpt-4.1`) | set `BIG_MODEL`, `SONNET_MODEL`, or `[big].model` / `[sonnet].model` |
| opus / fable / mythos | `openai/[big].model` (default `gpt-4.1`) | set `BIG_MODEL` or `[tier].model` |
| anything else with `openai/` prefix | passed through | — |
| bare model name in `OPENAI_MODELS` | `openai/<name>` | add to the list in `server.py` |
| anything else | `openai/<name>` (assumes custom OpenAI-compatible endpoint) | — |

A per-tier section (`[haiku]`, `[sonnet]`, `[opus]`, `[fable]`, `[mythos]`) with its own `model` overrides `[big].model` / `[small].model` for that tier only; tiers without an override keep using the bucket default. Use this to point different Claude tiers at different backends — e.g. opus at a strong model while sonnet uses the default.

To target a custom model on a compatible endpoint, set `[big].model` and `[small].model` (or `BIG_MODEL` / `SMALL_MODEL` env vars):

```toml
[big]
model = "your-model-name"

[small]
model = "your-model-name"
```

## Configuration ⚙️

The proxy has a single TOML config file (`config.toml`, defaulting to `./config.toml` next to `server.py`) as the primary source for every setting. Env vars and `.env` work as fallback for each key — useful for Docker overrides. **You don't need `.env` if `config.toml` exists**, but you can mix both.

### Schema

```toml
[proxy]
openai_api_key     = "sk-..."                       # env: OPENAI_API_KEY
openai_base_url    = "http://localhost:8081/v1"     # env: OPENAI_BASE_URL (optional)
openai_tls_verify  = true                           # env: OPENAI_TLS_VERIFY
tiktoken_offline   = true                           # env: TIKTOKEN_OFFLINE

[global]
model       = "gpt-4.1"        # catch-all fallback for any model
extra_body  = { temperature = 0.3 }

[big]
model       = "gpt-4.1"        # env: BIG_MODEL
extra_body  = { ... }          # optional

[small]
model       = "gpt-4.1-mini"   # env: SMALL_MODEL
extra_body  = { ... }          # optional

[sonnet]
model       = "gpt-4.1"
extra_body  = { temperature = 0.5, reasoning_effort = "low" }

[opus]
model       = "gpt-4.1"
extra_body  = { temperature = 0.7, top_p = 0.95 }
```

Merge chain is `[global] → [bucket] → [tier]` (later wins per leaf). Sampling / reasoning / vendor knobs all live in `extra_body` — no per-key whitelist.

Pick the knobs your backend actually understands — don't mix `reasoning_effort` (OpenAI o-series), `chat_template_kwargs` (llama.cpp), or Anthropic-native `thinking` in one section. They belong to different backends.

### Lookup order

For each setting, the proxy uses the first non-empty value from this list:

1. Environment variable (e.g. `OPENAI_API_KEY`, `BIG_MODEL`, `HAIKU_MODEL`)
2. `config.toml` (the matching `[proxy]` / `[big]` / `[small]` / `[global]` / `[tier]` section)
3. Built-in default (e.g. `[big].model` → `gpt-4.1`)

Env wins so `docker run -e KEY=VAL` and `docker-compose.yml: environment:` override `config.toml` without rebuilding the image.

### Per-tier merge semantics

- **Model selection**: per tier, the resolver walks `{TIER}_MODEL` env → `{BIG|SMALL}_MODEL` env → `[tier].model` → `[bucket].model` → `[global].model` → built-in default. First non-empty wins. `[global].model` is the catch-all for any model — including unmapped ones (tier=None).
- **extra_body merge chain**: `[global] → [bucket] → [tier]` (haiku → `small`, others → `big`). Each layer deep-merges; later wins per leaf. Keys are lifted to top-level kwargs on the upstream call. The merged key list is published as `allowed_openai_params` (top-level + inside `extra_body`) so the litellm hop and any cascade proxy forward vendor keys (`chat_template_kwargs`, `cache_prompt`, `n_predict`, `reasoning_effort`, …) instead of dropping them.
- **Sampling / reasoning / vendor fields** all live inside `[tier].extra_body` (and `[global].extra_body` / `[bucket].extra_body`). There is no per-key whitelist — pass any top-level key the upstream OpenAI Chat Completions API (or your compatible backend) accepts: `temperature`, `top_p`, `top_k`, `stop`, `seed`, `max_completion_tokens`, `reasoning_effort`, `chat_template_kwargs`, `cache_prompt`, `n_predict`, …
- **Conflict resolution**: when both a config layer (`[global]` / `[bucket]` / `[tier]`) and the client request set the same key (whether via Pydantic sampling fields or a request-level `extra_body`), **config wins** per leaf.
- **No defaults applied**: when neither config nor the request sets a key, it is **omitted from the upstream call** (we don't auto-apply Anthropic defaults like `temperature=1.0`).
- **No `max_tokens` clamp**: client `max_tokens` flows through unmodified. If a user asks for 24000, upstream gets 24000.

### Targeting llama-server / ollama / vLLM

These backends ignore standard OpenAI sampling params but accept a `chat_template_kwargs` knob to disable thinking-mode artefacts (critical for Qwen3.5+):

```toml
[proxy]
openai_api_key  = "no-key"
openai_base_url = "http://localhost:8081/v1"

[big]
model = "qwen3.5"

[small]
model = "qwen3.5"

[haiku]
extra_body = { temperature = 0.3, cache_prompt = true, n_predict = 4096, chat_template_kwargs = { enable_thinking = false } }
```

Inspect upstream logs (or use `mitmproxy`) to confirm `cache_prompt`, `chat_template_kwargs`, etc. land in the body. For offline checks, set `LOG_LEVEL=DEBUG` — the proxy logs the effective `extra_body` per request (sourced from request or `[tier]` config).

## How It Works 🧩

1. **Receive** the request in Anthropic's Messages API format 📥
2. **Remap** the model (`haiku`/`sonnet`/`opus`/... → `[small].model` / `[big].model` / `[tier].model`) 🗺️
3. **Translate** to OpenAI Chat Completions format via LiteLLM 🔄
4. **Send** to OpenAI (or any `OPENAI_BASE_URL`) 📤
5. **Convert** the response back to Anthropic format 🔄
6. **Return** the formatted response (streaming or non-streaming) ✅

Tool calls round-trip natively: `assistant.tool_calls` and `role="tool"` messages are preserved so tool use works with any OpenAI-compatible backend.

## Development 🛠️

```bash
# Install dev deps (ruff, ty, vulture) — uv manages them via PEP 735 [dependency-groups].
uv sync

# Lint (autofix what's safe, suggest fixes for the rest).
uv run ruff check --fix

# Format.
uv run ruff format

# Type-check (ty config in pyproject.toml; checks server.py and tests.py).
uv run ty check

# Find dead code (vulture config in pyproject.toml; false positives in vulture_whitelist.py).
uv run vulture

# Run the test suite (unit tests, no network needed).
uv run python tests.py
```

The pre-commit hook runs ruff (check + format), ty, and vulture on every commit:

```bash
uv tool install pre-commit
pre-commit install
```

`tests.py` is linted by ruff, ty, and vulture with the same active rule set as `server.py` — per-file ignores in `pyproject.toml` (`assert`, `private-member-access`, `unused-async`, `float-equality-comparison`, `module-import-not-at-top-of-file`) suppress test-context noise (asserts are the test pattern, monkey-patching internals, async generators as fixtures, TOML-roundtrip exact floats, intentional late `import server`). `vulture_whitelist.py` is excluded from ruff: it's a tool config file, not production code.

## Contributing 🤝

Pull requests welcome. 🎁