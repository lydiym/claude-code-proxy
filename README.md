# Anthropic API Proxy for OpenAI 🔄

**Use Anthropic clients (like Claude Code) with any OpenAI-compatible backend.** 🤝

A small proxy that accepts requests in the Anthropic Messages API format, translates them to OpenAI Chat Completions via LiteLLM, and converts the response back. Single backend, single code path.

## Quick Start ⚡

### Prerequisites

- An OpenAI API key, or a key for any OpenAI-compatible endpoint 🔑
- [uv](https://github.com/astral-sh/uv) installed

### Setup 🛠️

#### From source

1. **Clone and enter the repo:**
   ```bash
   git clone https://github.com/lydiym/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   - `OPENAI_API_KEY`: your OpenAI (or compatible) API key
   - `OPENAI_BASE_URL` (optional): override the endpoint, e.g. `https://api.your-provider.com/v1`
   - `BIG_MODEL` / `SMALL_MODEL` (optional): target models for `sonnet` / `haiku` requests. Defaults: `gpt-4.1` / `gpt-4.1-mini`.

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
| haiku        | `openai/SMALL_MODEL` (default `gpt-4.1-mini`) | set `SMALL_MODEL` |
| sonnet       | `openai/BIG_MODEL` (default `gpt-4.1`)        | set `BIG_MODEL`   |
| anything else with `openai/` prefix | passed through | — |
| bare model name in `OPENAI_MODELS` | `openai/<name>` | add to the list in `server.py` |
| anything else | `openai/<name>` (assumes custom OpenAI-compatible endpoint) | — |

To target a custom model on a compatible endpoint, set both `BIG_MODEL` and `SMALL_MODEL` to that name:

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.your-provider.com/v1"
BIG_MODEL="your-model-name"
SMALL_MODEL="your-model-name"
```

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