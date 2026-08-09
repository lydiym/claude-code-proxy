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
   - **`.env`** (for Docker / simple setups): copy `.env.example` to `.env`, then edit. Every key in `.env` has a TOML equivalent in `[proxy]` / `[routing]`.

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
| haiku        | `openai/SMALL_MODEL` (default `gpt-4.1-mini`) | set `SMALL_MODEL` or `HAIKU_MODEL` |
| sonnet       | `openai/BIG_MODEL` (default `gpt-4.1`) | set `BIG_MODEL` or `SONNET_MODEL` |
| opus / fable / mythos | `openai/BIG_MODEL` (default `gpt-4.1`) | set `BIG_MODEL`, or `OPUS_MODEL` / `FABLE_MODEL` / `MYTHOS_MODEL` |
| anything else with `openai/` prefix | passed through | — |
| bare model name in `OPENAI_MODELS` | `openai/<name>` | add to the list in `server.py` |
| anything else | `openai/<name>` (assumes custom OpenAI-compatible endpoint) | — |

A per-tier override (`HAIKU_MODEL`, `SONNET_MODEL`, `OPUS_MODEL`, `FABLE_MODEL`, `MYTHOS_MODEL`) takes precedence over `BIG_MODEL` / `SMALL_MODEL` for that tier only; tiers without an override keep using `BIG_MODEL` / `SMALL_MODEL`. Use this to point different Claude tiers at different backends — e.g. opus at a strong model while sonnet uses the default.

To target a custom model on a compatible endpoint, set both `BIG_MODEL` and `SMALL_MODEL` to that name:

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.your-provider.com/v1"
BIG_MODEL="your-model-name"
SMALL_MODEL="your-model-name"
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

[routing]
big_model      = "gpt-4.1"        # env: BIG_MODEL
small_model    = "gpt-4.1-mini"   # env: SMALL_MODEL
haiku_model    = "qwen3.5"        # env: HAIKU_MODEL   (optional)
sonnet_model   = "qwen3.5"        # env: SONNET_MODEL  (optional)
# opus_model / fable_model / mythos_model all optional

# Per-tier settings. [global] applies to every tier that has no explicit
# section; a tier-specific section deep-merges over [global] at each leaf.

[global]
# extra_body = { cache_prompt = true }

[haiku]
temperature = 0.3
top_p       = 0.9
top_k       = 40

[haiku.extra_body]
cache_prompt = true
n_predict    = 4096

[haiku.extra_body.chat_template_kwargs]
enable_thinking = false

[sonnet]
extra_body = { chat_template_kwargs = { enable_thinking = false } }
# temperature inherits from [global] if unset here
```

### Lookup order

For each setting, the proxy uses the first non-empty value from this list:

1. Environment variable (e.g. `OPENAI_API_KEY`)
2. `config.toml` (the matching `[proxy]` / `[routing]` / `[global]` / `[tier]` section)
3. Built-in default (e.g. `BIG_MODEL` → `gpt-4.1`)

Env wins so `docker run -e KEY=VAL` and `docker-compose.yml: environment:` override `config.toml` without rebuilding the image.

### Per-tier merge semantics

- **Sampling fields** (`temperature`, `top_p`, `top_k`, `max_completion_tokens`, `stop`, `seed`): when both config and the request set the same key, **config wins** at the leaf. When neither config nor the request sets the key, it is **omitted from the upstream call** (we don't auto-apply Anthropic defaults like `temperature=1.0`).
- **`extra_body`**: deep-merged. Config keys override request keys at each leaf; unrelated request keys are preserved. LiteLLM merges this into the upstream OpenAI Chat Completions request body at top level — pass any backend-specific knob (`cache_prompt`, `n_predict`, `chat_template_kwargs`, …).
- **`max_completion_tokens`**: config can only **lower** the ceiling. The OpenAI-side cap (`MAX_OUTPUT_TOKENS = 16384`) remains the absolute maximum.

### Targeting llama-server / ollama / vLLM

These backends ignore standard OpenAI sampling params but accept a `chat_template_kwargs` knob to disable thinking-mode artefacts (critical for Qwen3.5+):

```toml
[proxy]
openai_api_key  = "no-key"
openai_base_url = "http://localhost:8081/v1"

[routing]
big_model   = "qwen3.5"
small_model = "qwen3.5"

[haiku]
temperature = 0.3

[haiku.extra_body]
cache_prompt = true
n_predict    = 4096

[haiku.extra_body.chat_template_kwargs]
enable_thinking = false
```

Inspect upstream logs (or use `mitmproxy`) to confirm `cache_prompt`, `chat_template_kwargs`, etc. land in the body. For offline checks, set `LOG_LEVEL=DEBUG` — the proxy logs the effective sampling fields + `extra_body` per request (sourced from request or `[tier]` config).

## How It Works 🧩

1. **Receive** the request in Anthropic's Messages API format 📥
2. **Remap** the model (`haiku`/`sonnet` → `SMALL_MODEL`/`BIG_MODEL`) 🗺️
3. **Translate** to OpenAI Chat Completions format via LiteLLM 🔄
4. **Send** to OpenAI (or any `OPENAI_BASE_URL`) 📤
5. **Convert** the response back to Anthropic format 🔄
6. **Return** the formatted response (streaming or non-streaming) ✅

Tool calls round-trip natively: `assistant.tool_calls` and `role="tool"` messages are preserved so tool use works with any OpenAI-compatible backend.

## Contributing 🤝

Pull requests welcome. 🎁