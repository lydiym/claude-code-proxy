"""Anthropic → OpenAI Chat Completions proxy.

Translates Anthropic Messages API requests to OpenAI Chat Completions via
LiteLLM and converts the response back. Single FastAPI app, single code path.
"""

import json
import logging
import os
import pathlib
import re
import sys
import time
import tomllib
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast, override

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator, model_validator

load_dotenv()

# Must be set before litellm is imported — otherwise it tries to refresh the
# model cost map from GitHub on every call and spams warnings on isolated nets.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# basicConfig formats import-time logs (litellm, uvicorn, _load_config).
# _configure_logging re-applies it after uvicorn installs its handlers.
_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.WARNING,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-import: tiktoken stub before `import litellm`. Resolver is standalone
# so the main loader sits below.
# ---------------------------------------------------------------------------


def _str_to_bool(value: object, *, default: bool = False) -> bool:
    """Parse an env-style boolean; unrecognised strings fall back to ``default``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if str(value).strip().lower() not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _litellm_debug_http_enabled() -> bool:
    """LITELLM_DEBUG_HTTP=1 turns on verbose litellm + httpx/httpcore logging.

    Surfaces the actual wire payload sent to the upstream OpenAI endpoint.
    """
    return _str_to_bool(os.environ.get("LITELLM_DEBUG_HTTP"), default=False)


def _debug_json_dump(label: str, obj: object) -> None:
    """Best-effort debug-level JSON dump.

    Logs the failure cause instead of raising — debug logging must never crash
    the request.
    """
    try:
        logger.debug("%s: %s", label, json.dumps(obj, default=str, ensure_ascii=False))
    except Exception as e:
        logger.debug("%s dump failed: %s", label, e)


def _resolve_tiktoken_offline() -> bool:
    """[proxy].tiktoken_offline from CONFIG_PATH, TIKTOKEN_OFFLINE env, then True."""
    path = os.environ.get("CONFIG_PATH", "./config.toml")
    if path and pathlib.Path(path).is_file():
        try:
            with pathlib.Path(path).open("rb") as f:
                raw = tomllib.load(f)
            cfg = raw.get("proxy", {})
            if isinstance(cfg, dict) and "tiktoken_offline" in cfg:
                return _str_to_bool(cfg["tiktoken_offline"], default=True)
        except (OSError, tomllib.TOMLDecodeError):
            pass
    env = os.environ.get("TIKTOKEN_OFFLINE")
    if env not in {None, ""}:
        return _str_to_bool(env, default=True)
    return True


TIKTOKEN_OFFLINE = _resolve_tiktoken_offline()

if TIKTOKEN_OFFLINE:
    # Stub tiktoken: skip the Azure blob fetch of cl100k_base.tiktoken.
    # Token counts are approximate; real counts come from upstream usage.
    import tiktoken

    class _OfflineEncoding(tiktoken.Encoding):
        def __init__(self) -> None:
            # Override parent's __init__ to skip the required mergeable_ranks / special_tokens.
            pass

        @override
        def encode(
            self, text: str, *, allowed_special: object = ..., disallowed_special: object = ...,
        ) -> list[int]:
            return [1] * max(1, len(text) // 4)

        @override
        def encode_ordinary(self, text: str) -> list[int]:
            return self.encode(text)

        @override
        def encode_single_token(self, text_or_bytes: str | bytes) -> int:
            return 1

        @override
        def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
            return ""

        @override
        def decode_single_token_bytes(self, token: int) -> bytes:
            return b""

    def _get_encoding(_encoding_name: str) -> tiktoken.Encoding:
        return _OfflineEncoding()

    def _encoding_for_model(_model_name: str) -> tiktoken.Encoding:
        return _OfflineEncoding()

    # ty invalid-assignment: monkey-patching module functions, intentional.
    tiktoken.get_encoding = _get_encoding  # type: ignore[ty:invalid-assignment]
    tiktoken.encoding_for_model = _encoding_for_model  # type: ignore[ty:invalid-assignment]

# These imports must come after the env-var + tiktoken patch above.
import litellm  # ruff: ignore[module-import-not-at-top-of-file]
import uvicorn  # ruff: ignore[module-import-not-at-top-of-file]

# ---------------------------------------------------------------------------
# Config loader (TOML primary source; per-key env-var fallback via _proxy_value)
# ---------------------------------------------------------------------------

# Tier names. Insertion order is the routing priority (haiku is checked first so
# it wins over big tiers in substring matches). Single source of truth — the
# TOML-loader section validator and the per-request tier lookup both read this.
TIER_KEYS = ("haiku", "sonnet", "opus", "fable", "mythos")
_VALID_TIERS = set(TIER_KEYS)
# Top-level OpenAI Chat Completions keys that `extra_body` must never overwrite — the proxy owns these.
_PROTECTED_KEYS = {"model", "messages", "stream", "tools"}
_PROXY_KEYS = {
    "openai_api_key",
    "openai_base_url",
    "openai_tls_verify",
}
# haiku → small; everything else → big. Single source of truth for which bucket a tier inherits from.
_BUCKET_FOR_TIER = {t: ("small" if t == "haiku" else "big") for t in TIER_KEYS}
_VALID_SECTIONS = {"proxy", "global", "big", "small"} | _VALID_TIERS


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge: base + override, where override wins per leaf.

    Used at request time to layer tier config over [global] and to merge
    config extra_body into the upstream body.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_PROVIDER_PREFIXES = ("anthropic/", "openai/", "gemini/")


def _strip_provider_prefix(name: str) -> str:
    """Strip a known provider prefix (case-insensitive). Bare-name case preserved."""
    for prefix in _PROVIDER_PREFIXES:
        if name.lower().startswith(prefix):
            return name[len(prefix) :]
    return name


def _match_tier(name: str) -> str | None:
    """First TIER_KEYS substring that appears in the lower-cased name.

    Returns None when no tier matches.
    """
    lower = name.lower()
    for tier in TIER_KEYS:  # insertion order = routing priority (haiku first)
        if tier in lower:
            return tier
    return None


def _parse_tier_section(body: dict[str, Any], section: str) -> dict[str, Any]:
    """Parse [global]. Accepts `model` (str) + `extra_body` (dict).

    Bad values are warned and skipped so one error doesn't drop the rest.
    """
    out: dict[str, Any] = {}
    for k, v in body.items():
        if k == "model":
            if not isinstance(v, str) or not v:
                logger.warning("[%s].model must be a non-empty string; ignoring", section)
                continue
            out[k] = v
        elif k == "extra_body":
            if not isinstance(v, dict):
                logger.warning("[%s].extra_body must be a table; ignoring", section)
                continue
            out[k] = deepcopy(v)
        else:
            logger.warning("[%s].%s is not a recognised key; put it inside extra_body", section, k)
    return out


def _parse_bucket_section(body: dict[str, Any], section: str) -> dict[str, Any]:
    """Parse [big], [small], or a per-tier section. Accepts `model` (str) and `extra_body` (dict).

    Bad values warned and skipped.
    """
    out: dict[str, Any] = {}
    for k, v in body.items():
        if k == "model":
            if not isinstance(v, str) or not v:
                logger.warning("[%s].model must be a non-empty string; ignoring", section)
                continue
            out[k] = v
        elif k == "extra_body":
            if not isinstance(v, dict):
                logger.warning("[%s].extra_body must be a table; ignoring", section)
                continue
            out[k] = deepcopy(v)
        else:
            logger.warning("[%s].%s is not a recognised key; put it inside extra_body", section, k)
    return out


_COERCE_DROP = object()  # sentinel: coercion failed, key should be dropped
_BOOL_TLS_VERIFY = {"openai_tls_verify"}


def _coerce_proxy_value(key: str, value: object, section: str) -> str | bool | object:
    """Coerce a [proxy] TOML value to the expected Python type.

    Returns the coerced value, or ``_COERCE_DROP`` when the key must be skipped.
    """
    if isinstance(value, str):
        return value  # strings are the canonical type for api_key, base_url, model names
    if key in _BOOL_TLS_VERIFY and isinstance(value, bool):
        return value
    if key in _BOOL_TLS_VERIFY and isinstance(value, int) and not isinstance(value, bool):
        return bool(value)
    logger.warning(
        "[%s].%s=%r has wrong type (%s); ignoring",
        section,
        key,
        value,
        type(value).__name__,
    )
    return _COERCE_DROP


def _load_config(path: str) -> dict[str, Any]:
    """Parse TOML at path; fail-open on every parse failure (logged, not raised)."""
    out: dict[str, Any] = {"proxy": {}, "global": {}, "big": {}, "small": {}, "tiers": {}}
    if not path:
        return out
    if not pathlib.Path(path).is_file():
        logger.info("CONFIG_PATH=%r not found; using env vars only", path)
        return out
    try:
        with pathlib.Path(path).open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError:
        logger.exception("Malformed TOML in %s; falling back to env vars", path)
        return out
    for section, body in raw.items():
        if section not in _VALID_SECTIONS:
            logger.warning(
                "Unknown section [%s] in %s; ignoring (valid: %s)",
                section,
                path,
                sorted(_VALID_SECTIONS),
            )
            continue
        if not isinstance(body, dict):
            logger.warning("[%s] must be a table, got %s; ignoring", section, type(body).__name__)
            continue
        if section == "proxy":
            out["proxy"] = _parse_proxy_section(body)
        elif section in {"big", "small"}:
            out[section] = _parse_bucket_section(body, section)
        elif section == "global":
            out["global"] = _parse_tier_section(body, section)
        else:  # per-tier section
            out["tiers"][section] = _parse_bucket_section(body, section)
    return out


def _parse_proxy_section(body: dict[str, object]) -> dict[str, object]:
    """Pick out the keys we recognise from [proxy]; warn and skip the rest."""
    out: dict[str, object] = {}
    for k, v in body.items():
        if k not in _PROXY_KEYS:
            logger.warning("[proxy].%s is not a recognised key; ignoring", k)
            continue
        coerced = _coerce_proxy_value(k, v, "proxy")
        if coerced is _COERCE_DROP:
            continue
        out[k] = coerced
    return out


CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.toml")
try:
    CONFIG = _load_config(CONFIG_PATH)
except Exception:
    # Infrastructure error only — parse failures are handled inside _load_config.
    logger.exception("Failed to load CONFIG_PATH=%r; using env vars only", CONFIG_PATH)
    CONFIG = {"proxy": {}, "global": {}, "big": {}, "small": {}, "tiers": {}}

# WARNING so the boot summary shows up before uvicorn installs its own handlers.
logger.warning(
    "Loaded config from %r: proxy=%s, big=%s, small=%s, global=%s, tiers=%s",
    CONFIG_PATH,
    list(CONFIG["proxy"]),
    CONFIG["big"],
    CONFIG["small"],
    "yes" if CONFIG["global"] else "no",
    list(CONFIG["tiers"]),
)


def _proxy_value(key: str, env_name: str, default: object = None) -> object:
    """Env var → CONFIG[proxy][key] → default. None / "" fall through."""
    if key not in _PROXY_KEYS:
        raise ValueError(f"_proxy_value: {key!r} is not a recognised proxy key")
    env_val = os.environ.get(env_name)
    if env_val not in {None, ""}:
        return env_val
    val = CONFIG["proxy"].get(key)
    if val not in {None, ""}:
        return val
    return default


def _proxy_bool(key: str, env_name: str, *, default: bool = True) -> bool:
    """Resolve a boolean config value; unrecognised strings keep ``default``."""
    val = _proxy_value(key, env_name, default)
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    return _str_to_bool(val, default=default)


@cache
def _default_model_for_tier(tier: str | None) -> str:
    """Per-tier upstream model.

    Lookup order:
      1. {TIER}_MODEL env (e.g. HAIKU_MODEL; skipped when tier=None)
      2. {BIG|SMALL}_MODEL env — bucket-level (haiku → SMALL_MODEL, others → BIG_MODEL)
      3. [tier].model config (skipped when tier=None)
      4. [bucket].model config (haiku → small, others → big; bucket skipped when tier=None)
      5. [global].model config — fallback for any model
      6. Built-in default (gpt-4.1-mini for haiku bucket, gpt-4.1 otherwise)

    Cached at first call — env or CONFIG.toml edits after import require restart.
    Live edits to [tier].extra_body still take effect via the per-call
    _resolve_tier_config.
    """
    bucket = _bucket_for_tier(tier)
    built_in = "gpt-4.1-mini" if bucket == "small" else "gpt-4.1"

    if tier:
        env_val = _env_nonempty(f"{tier.upper()}_MODEL")
        if env_val is not None:
            return env_val
    bucket_env = "SMALL_MODEL" if bucket == "small" else "BIG_MODEL"
    env_val = _env_nonempty(bucket_env)
    if env_val is not None:
        return env_val

    if tier:
        cfg_model = _config_model((CONFIG.get("tiers") or {}).get(tier) or {})
        if cfg_model is not None:
            return cfg_model
    if bucket:
        cfg_model = _config_model(CONFIG.get(bucket) or {})
        if cfg_model is not None:
            return cfg_model
    cfg_model = _config_model(CONFIG.get("global") or {})
    if cfg_model is not None:
        return cfg_model
    return built_in


def _bucket_for_tier(tier: str | None) -> str | None:
    """Map tier to its bucket section.

    Returns None for tier=None so an unmapped model skips the bucket config
    lookup and falls through to [global].model (the catch-all).
    """
    if tier == "haiku":
        return "small"
    if tier is None:
        return None
    return "big"


def _env_nonempty(name: str) -> str | None:
    val = os.environ.get(name) if name else None
    return val or None


def _config_model(section: dict[str, object]) -> str | None:
    val = section.get("model")
    return val if isinstance(val, str) and val else None


class _Colors:
    """ANSI color codes used to highlight parts of operational log lines."""

    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _color(code: str, text: str | int) -> str:
    """Wrap text in ANSI code when stderr is a TTY, otherwise return plain."""
    try:
        if sys.stderr.isatty():
            return f"{code}{text}{_Colors.RESET}"
    except (ValueError, AttributeError):
        pass
    return str(text)


def _reset_logger(name: str, *, propagate: bool) -> None:
    log = logging.getLogger(name)
    log.handlers.clear()
    log.propagate = propagate


@asynccontextmanager
async def _configure_logging(_app: FastAPI) -> AsyncIterator[None]:
    """Unify the log format and silence uvicorn.access.

    Runs after uvicorn's own configure_logging() so it overrides whatever
    uvicorn set up.
    """
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(handler)

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        logger.setLevel(log_level)
    except ValueError:
        logger.setLevel(logging.INFO)
        logger.warning("Invalid LOG_LEVEL=%r; falling back to INFO", log_level)
    for noisy in ("LiteLLM", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if _litellm_debug_http_enabled():
        # Verbose mode: see exactly what litellm sends/receives and what
        # goes over the wire via httpx. Heavy — only for live debugging.
        litellm.set_verbose = True  # type: ignore[ty:invalid-assignment]
        for noisy in ("httpx", "httpcore", "LiteLLM"):
            logging.getLogger(noisy).setLevel(logging.DEBUG)
        logger.warning("LITELLM_DEBUG_HTTP=1 — verbose litellm + httpx/httpcore DEBUG logs enabled")

    _reset_logger("uvicorn", propagate=True)
    _reset_logger("uvicorn.error", propagate=True)
    _reset_logger("uvicorn.access", propagate=False)

    yield


app = FastAPI(lifespan=_configure_logging)


OPENAI_API_KEY = _proxy_value("openai_api_key", "OPENAI_API_KEY")
OPENAI_BASE_URL = _proxy_value("openai_base_url", "OPENAI_BASE_URL")

# Skip TLS validation when OPENAI_BASE_URL uses a self-signed cert (local LLM).
OPENAI_TLS_VERIFY = _proxy_bool("openai_tls_verify", "OPENAI_TLS_VERIFY", default=True)
litellm.ssl_verify = OPENAI_TLS_VERIFY
if not OPENAI_TLS_VERIFY:
    logger.warning("OPENAI_TLS_VERIFY=false — TLS certificate validation is disabled. Do not use this in production.")

# OpenAI Chat Completions caps max_completion_tokens at this value for most
# current models; over it the API rejects the request.
MAX_OUTPUT_TOKENS = 16384

DEFAULT_PORT = 8082
DEFAULT_HOST = "127.0.0.1"

MSG_ID_HEX_LEN = 24

# Per-streak tolerance for transient chunk failures before surfacing an error.
MAX_CONSECUTIVE_CHUNK_ERRORS = 5

STATUS_OK = 200

# Recognised bare names; anything else is opaque and passed through with openai/.
OPENAI_MODELS = {
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",
    "gpt-4.1-mini",
}


def _to_anthropic_stop_reason(finish_reason: object) -> Literal["end_turn", "max_tokens", "tool_use"]:
    mapping: dict[str, Literal["end_turn", "max_tokens", "tool_use"]] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    key = str(finish_reason) if finish_reason else ""
    return mapping.get(key, "end_turn")


# Maps litellm exception class names to Anthropic error_type enum; unknown classes fall through to "api_error".
_ANTHROPIC_ERROR_TYPES = {
    "RateLimitError": "rate_limit_error",
    "AuthenticationError": "authentication_error",
    "PermissionDeniedError": "permission_error",
    "NotFoundError": "not_found_error",
    "UnprocessableEntityError": "invalid_request_error",
    "BadRequestError": "invalid_request_error",
    "Timeout": "timeout_error",
    "APIConnectionError": "api_error",
    "ContextWindowExceededError": "invalid_request_error",
    "ServiceUnavailableError": "overloaded_error",
    "InternalServerError": "api_error",
}


def _anthropic_error_type(exc: BaseException) -> str:
    cls = type(exc).__name__
    return _ANTHROPIC_ERROR_TYPES.get(cls, "api_error")


# litellm/httpx exception messages echo upstream bodies verbatim and can carry credentials; redact before event:error.
_CREDENTIAL_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"x-api-key=[A-Za-z0-9_-]{16,}"),
    re.compile(r"://[^/\s]+:[^/\s]+@"),
]


def _sanitize_error_message(message: str) -> str:
    for pat in _CREDENTIAL_PATTERNS:
        message = pat.sub("[REDACTED]", message)
    return message


def _get_field(obj: object, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _new_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"


def _short_model(name: str) -> str:
    return name.rsplit("/", maxsplit=1)[-1] if "/" in name else name


class ContentBlockText(BaseModel):
    """Anthropic text block — plain prose returned by the model."""

    type: Literal["text"]
    text: str


class ContentBlockThinking(BaseModel):
    """Anthropic thinking block — extended reasoning, surfaced as a sibling of text."""

    type: Literal["thinking"]
    thinking: str
    # Echoed back in conversation history; we don't generate it locally.
    signature: str | None = None


class ContentBlockImage(BaseModel):
    """Anthropic image block — referenced by source, sent inline in the request."""

    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockToolUse(BaseModel):
    """Anthropic tool_use block — assistant's request to invoke a tool."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    """Anthropic tool_result block — user turn delivering a tool's output."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any] | list[Any] | Any


class SystemContent(BaseModel):
    """Text entry inside the top-level ``system`` array form."""

    type: Literal["text"]
    text: str


class Message(BaseModel):
    """A single turn in the conversation — user, assistant, or system reminder."""

    role: Literal["user", "assistant", "system"]
    content: (
        str
        | list[
            ContentBlockText | ContentBlockThinking | ContentBlockImage | ContentBlockToolUse | ContentBlockToolResult
        ]
    )


class Tool(BaseModel):
    """Tool definition as the client sends it — converted to OpenAI format downstream."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class MessagesRequest(BaseModel):
    """Incoming Anthropic Messages API request.

    Captured into Pydantic for validation; ``model`` is rewritten by
    ``validate_model_field``, ``original_model`` is preserved for logging,
    ``tier`` is populated by ``derive_tier`` so the per-tier CONFIG lookup
    knows which bucket to consult.
    """

    model: str
    max_tokens: int
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    # Pass-through bag for arbitrary OpenAI Chat Completions keys. Merged
    # with [tier].extra_body at convert time; per-leaf config-wins.
    extra_body: dict[str, Any] | None = None
    original_model: str | None = None
    # Populated by `derive_tier` (model_validator below); used by
    # convert_anthropic_to_litellm to look up per-tier CONFIG settings.
    tier: str | None = None

    @model_validator(mode="before")
    @classmethod
    def capture_original_model(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Copy ``model`` into ``original_model`` before the rewrite.

        So the log line can show what the client asked for vs. what we sent upstream.
        """
        if isinstance(data, dict) and "model" in data:
            data = dict(data)
            data["original_model"] = data["model"]
        return data

    @model_validator(mode="after")
    def derive_tier(self) -> "MessagesRequest":  # ruff: ignore[invalid-first-argument-name-for-class-method] — Pydantic model_validator(mode="after") is internally a classmethod that receives the instance
        """Identify the Anthropic tier from the pre-rewrite ``original_model``.

        Unknown models leave ``tier=None`` and fall back to ``[global]`` in the lookup.
        """
        if self.original_model:
            self.tier = _match_tier(self.original_model)
        return self  # tier stays None → falls back to [global] in lookup

    @field_validator("model")
    def validate_model_field(cls, v: str) -> str:
        """Rewrite the inbound model name to the OpenAI Chat Completions form.

        Strips a known provider prefix, matches a tier to pick the per-tier
        upstream model, prefixes known bare OpenAI names, and passes
        already-prefixed or custom-endpoint names through with ``openai/``.
        """
        clean_v = _strip_provider_prefix(v)  # case preserved

        if tier := _match_tier(v):
            chosen = _default_model_for_tier(tier)
            new_model = f"openai/{chosen}"
        elif clean_v.lower() in OPENAI_MODELS and not v.lower().startswith("openai/"):
            new_model = f"openai/{clean_v}"
        elif v.lower().startswith("openai/"):
            new_model = v  # already-prefixed passthrough (case-insensitive)
        else:
            # Custom endpoint: pass the bare name through with the openai/ prefix.
            logger.debug("No mapping rule for model '%s', passing through", v)
            new_model = f"openai/{clean_v}"

        logger.debug("MODEL MAPPING: '%s' -> '%s'", v, new_model)

        return new_model


class Usage(BaseModel):
    """Token accounting returned with every Anthropic response."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Anthropic-shaped response — either non-streaming return value or wrapped by StreamingResponse."""

    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlockText | ContentBlockThinking | ContentBlockToolUse]
    type: Literal["message"] = "message"
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = None
    stop_sequence: str | None = None
    usage: Usage


# Anthropic wants `type: "thinking"` blocks; some backends fold reasoning
# into `<think>...</think>` inside `content` and the parser below splits them out.
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"


class _ThinkStreamParser:
    """Incremental parser that splits a stream into thinking vs. text chunks.

    The OpenAI-compatible model emits <think>...</think> markers inside its
    content stream. Anthropic SSE needs these surfaced as their own content
    blocks, so we buffer just enough to recognise a marker split across chunks
    and yield tagged deltas for the caller to forward.
    """

    def __init__(self) -> None:
        self.in_thinking = False
        self.buffer = ""

    def feed(self, text: object) -> list[tuple[str, str | None]]:
        """Consume a chunk of model output; return a list of (kind, value).

        kind is "text" or "thinking" for a delta, or "open" / "close" for a
        block transition (value is None for those).
        """
        if not isinstance(text, str) or not text:
            return []
        events = []
        self.buffer += text
        while self._drain(events):
            pass
        return events

    def _drain(self, events: list[tuple[str, str | None]]) -> bool:
        if not self.buffer:
            return False
        tag = _THINK_CLOSE_TAG if self.in_thinking else _THINK_OPEN_TAG
        idx = self.buffer.find(tag)
        if idx < 0:
            if self.in_thinking:
                # Hold until close tag: emitting early would commit bytes to
                # "thinking" if the tag never arrives or splits mid-word later.
                return False
            # Text mode: only risk is a stray open tag, so flush up to the last '<'.
            last_lt = self.buffer.rfind("<")
            if last_lt < 0:
                events.append(("text", self.buffer))
                self.buffer = ""
            elif last_lt > 0:
                events.append(("text", self.buffer[:last_lt]))
                self.buffer = self.buffer[last_lt:]
            return False
        if idx > 0:
            kind = "thinking" if self.in_thinking else "text"
            events.append((kind, self.buffer[:idx]))
        events.append(("close" if self.in_thinking else "open", None))
        self.in_thinking = not self.in_thinking
        self.buffer = self.buffer[idx + len(tag) :]
        return True

    def flush(self) -> list[tuple[str, str | None]]:
        if not self.buffer:
            return []
        events = [(("thinking" if self.in_thinking else "text"), self.buffer)]
        self.buffer = ""
        if self.in_thinking:
            events.append(("close", None))
            self.in_thinking = False
        return events


def _parse_tool_result_content(content: object) -> str:
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return _join_tool_result_items(content).strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            return str(content)

    try:
        return str(content)
    except (TypeError, ValueError):
        return "Unparseable content"


def _join_tool_result_items(items: list[object]) -> str:
    result = ""
    for item in items:
        if isinstance(item, dict) and item.get("type") == "text":
            result += item.get("text", "") + "\n"
        elif isinstance(item, str):
            result += item + "\n"
        elif isinstance(item, dict):
            if "text" in item:
                result += item.get("text", "") + "\n"
            else:
                try:
                    result += json.dumps(item) + "\n"
                except (TypeError, ValueError):
                    result += str(item) + "\n"
        else:
            try:
                result += str(item) + "\n"
            except (TypeError, ValueError):
                result += "Unparseable content\n"
    return result


def convert_image_block(source: object) -> dict[str, Any]:
    """Translate an Anthropic image source into OpenAI ``image_url`` content-part.

    Handles ``base64`` and ``url`` source shapes; falls back to a stringified
    ``image_url`` for any other input so a malformed block never crashes the request.
    """
    if isinstance(source, dict):
        if source.get("type") == "base64":
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
        if source.get("type") == "url":
            return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return {"type": "image_url", "image_url": {"url": str(source)}}


def _extract_text(content: object) -> str:
    """Pull plain text out of a content field — None, string, or list of blocks.

    Used for system messages (and any other role) whose text we want to
    concatenate without preserving block structure. Non-text blocks (images,
    tool_use, tool_result, thinking) are skipped.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [cast("str", _get_field(block, "text", "")) for block in content if _get_field(block, "type") == "text"]
    return "\n\n".join(p for p in parts if p)


def _build_system_message(
    system_field: str | list[SystemContent] | None,
    messages: list[Message],
) -> dict[str, str] | None:
    """Combine the top-level system field with any in-band role='system' messages.

    Anthropic's spec only allows system at the top level, but Claude Code
    2.1.154+ has started embedding system reminders inline. We hoist them all
    to the start so OpenAI sees one system message at the top. Order is
    preserved: in-band messages come first, then the top-level field — which
    is the order Claude Code most likely intended when it injected the
    reminders inline.
    """
    parts = [text for text in (_extract_text(m.content) for m in messages if m.role == "system") if text]
    top = _extract_text(system_field)
    if top:
        parts.append(top)
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}


def _collect_tool_ids(messages: list[Message]) -> tuple[set[str], set[str]]:
    call_ids = set()
    result_ids = set()
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            block_type = _get_field(block, "type")
            if block_type == "tool_use":
                call_ids.add(cast("ContentBlockToolUse", block).id)
            elif block_type == "tool_result":
                rid = _get_field(block, "tool_use_id", "") or ""
                if rid:
                    result_ids.add(rid)
    return call_ids, result_ids


def _convert_assistant_message(msg: Message, result_ids: set[str]) -> dict[str, Any]:
    text_parts = []
    tool_calls = []

    for block in msg.content:
        block_type = _get_field(block, "type")
        if block_type == "text":
            text_parts.append(cast("ContentBlockText", block).text)
        elif block_type == "tool_use":
            tool_block = cast("ContentBlockToolUse", block)
            if tool_block.id in result_ids:
                tool_calls.append(
                    {
                        "id": tool_block.id,
                        "type": "function",
                        "function": {
                            "name": tool_block.name,
                            "arguments": json.dumps(tool_block.input),
                        },
                    },
                )
            else:
                # Dangling call (result truncated from history): describe in
                # prose — small models mimic tool-call syntax otherwise.
                text_parts.append(f"(An earlier {tool_block.name} tool call is missing its result in this context.)")

    text = "\n".join(text_parts).strip()
    out = {"role": "assistant"}
    # OpenAI allows null content only when tool_calls are present.
    out["content"] = text or (None if tool_calls else "")
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _convert_user_message(msg: Message, call_ids: set[str]) -> list[dict[str, Any]]:
    tool_messages = []
    user_parts = []

    for block in msg.content:
        block_type = _get_field(block, "type")
        if block_type == "text":
            user_parts.append({"type": "text", "text": cast("ContentBlockText", block).text})
        elif block_type == "image":
            user_parts.append(convert_image_block(cast("ContentBlockImage", block).source))
        elif block_type == "tool_result":
            tool_use_id = _get_field(block, "tool_use_id", "") or ""
            result_text = _parse_tool_result_content(_get_field(block, "content"))
            if tool_use_id in call_ids:
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": result_text,
                    },
                )
            else:
                # Orphaned result (truncated call): fold into user text; the ghost id is meaningless.
                user_parts.append(
                    {
                        "type": "text",
                        "text": f"(Result from an earlier tool call:)\n{result_text}",
                    },
                )

    # Tool results must follow the matching assistant turn, so emit them first.
    out = list(tool_messages)
    if user_parts:
        if all(part.get("type") == "text" for part in user_parts):
            text = "\n".join(part["text"] for part in user_parts).strip()
            out.append({"role": "user", "content": text or "..."})
        else:
            out.append({"role": "user", "content": user_parts})
    return out


def _convert_message(
    msg: Message,
    result_ids: set[str],
    call_ids: set[str],
) -> list[dict[str, Any]]:
    if isinstance(msg.content, str):
        return [{"role": msg.role, "content": msg.content}]
    if msg.role == "assistant":
        return [_convert_assistant_message(msg, result_ids)]
    return _convert_user_message(msg, call_ids)


def _convert_tool_definitions(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _convert_tool_choice(choice: dict[str, Any] | None) -> str | dict[str, Any]:
    choice_type = _get_field(choice, "type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "any"
    if choice_type == "tool":
        name = _get_field(choice, "name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return "auto"


def sanitize_messages_for_openai(messages: list[dict[str, Any]]) -> None:
    """Strip message keys OpenAI doesn't accept and coerce empty content.

    Mutates in place. Keeps ``role``, ``content``, ``name``, ``tool_call_id``,
    and ``tool_calls``; everything else is dropped. Empty/None ``content`` is
    replaced with ``"..."`` when no tool_calls are present — OpenAI rejects
    empty content outright.
    """
    allowed_keys = {"role", "content", "name", "tool_call_id", "tool_calls"}
    for msg in messages:
        for key in list(msg.keys()):
            if key not in allowed_keys:
                logger.debug("Removing unsupported message field: %s", key)
                del msg[key]
        if msg.get("content") in {None, ""} and not msg.get("tool_calls"):
            msg["content"] = "..."


def _resolve_tier_config(request: "MessagesRequest") -> dict[str, Any]:
    """extra_body merge chain: [global] → [bucket] → [tier].

    haiku → small bucket; others → big bucket; tier=None → no bucket or tier.
    `model` is consumed by _default_model_for_tier and stripped from the result.
    Deep-copied so downstream mutations can't corrupt CONFIG's nested dicts.
    """
    layers: list[dict[str, Any]] = []
    global_cfg = CONFIG.get("global")
    if global_cfg is not None:
        layers.append(global_cfg)
    bucket = _BUCKET_FOR_TIER.get(request.tier) if request.tier else None
    if bucket:
        bucket_cfg = CONFIG.get(bucket)
        if bucket_cfg is not None:
            layers.append(bucket_cfg)
    if request.tier:
        tier_cfg = (CONFIG.get("tiers") or {}).get(request.tier)
        if tier_cfg is not None:
            layers.append(tier_cfg)

    merged: dict[str, Any] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer)
    merged.pop("model", None)
    return deepcopy(merged)


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> dict[str, Any]:
    """Translate an Anthropic Messages request into OpenAI Chat Completions kwargs.

    Hoists the system field (including in-band ``role='system'`` messages),
    preserves tool call/result turns as OpenAI expects, applies per-tier
    ``extra_body`` and client sampling fields, and lifts every merged
    ``extra_body`` key to a top-level kwarg on the upstream call.
    """
    call_ids, result_ids = _collect_tool_ids(anthropic_request.messages)

    messages = []
    if system := _build_system_message(anthropic_request.system, anthropic_request.messages):
        messages.append(system)

    # LiteLLM uses assistant.tool_calls + role="tool"; flattening taught small
    # models to emit tool calls as literal text and broke tool use.
    for msg in anthropic_request.messages:
        if msg.role != "system":
            messages.extend(_convert_message(msg, result_ids, call_ids))

    litellm_request: dict[str, Any] = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_completion_tokens": anthropic_request.max_tokens,
        "stream": anthropic_request.stream,
    }
    if anthropic_request.tools:
        litellm_request["tools"] = _convert_tool_definitions(anthropic_request.tools)
    if anthropic_request.tool_choice:
        litellm_request["tool_choice"] = _convert_tool_choice(anthropic_request.tool_choice)

    _apply_sampling_fields(litellm_request, anthropic_request)

    # Tier config applied after client sampling fields → config wins per leaf.
    # Client extra_body merged in before tier extra_body → config wins there too.
    tier_cfg = _resolve_tier_config(anthropic_request)
    if "seed" in tier_cfg:
        litellm_request["seed"] = tier_cfg["seed"]
    _apply_merged_extra_body(litellm_request, anthropic_request, tier_cfg)
    return litellm_request


def _apply_sampling_fields(litellm_request: dict[str, Any], anthropic_request: MessagesRequest) -> None:
    """Forward sampling fields only when the client sent them.

    None means "use the upstream's own default" rather than MessagesRequest's default.
    """
    fields_set = anthropic_request.model_fields_set
    if "temperature" in fields_set and anthropic_request.temperature is not None:
        litellm_request["temperature"] = anthropic_request.temperature
    if "top_p" in fields_set and anthropic_request.top_p is not None:
        litellm_request["top_p"] = anthropic_request.top_p
    if "top_k" in fields_set and anthropic_request.top_k is not None:
        litellm_request["top_k"] = anthropic_request.top_k
    if "stop_sequences" in fields_set and anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences


def _apply_merged_extra_body(
    litellm_request: dict[str, Any],
    anthropic_request: MessagesRequest,
    tier_cfg: dict[str, Any],
) -> None:
    """Merge client extra_body under tier extra_body (config wins per leaf).

    Lift every key to top-level kwargs on the upstream call and publish the
    whitelist twice — top-level so litellm extends supported_params
    (utils.py:3877), and inside extra_body so cascade proxies forward vendor
    keys instead of filtering them (openai_like/chat/handler.py:241,254-259).
    """
    merged_extra: dict[str, Any] = {}
    if anthropic_request.extra_body:
        merged_extra = _deep_merge(merged_extra, anthropic_request.extra_body)
    tier_extra = tier_cfg.get("extra_body")
    if isinstance(tier_extra, dict):
        merged_extra = _deep_merge(merged_extra, tier_extra)

    for k, v in merged_extra.items():
        if k in _PROTECTED_KEYS:
            logger.warning("ignoring protected key in extra_body: %s", k)
            continue
        litellm_request[k] = v

    if merged_extra:
        keys = list(merged_extra.keys())
        litellm_request["allowed_openai_params"] = keys
        litellm_request["extra_body"] = {"allowed_openai_params": keys}


def _first_choice(response: object) -> object:
    choices = _get_field(response, "choices", [])
    if not choices:
        return None
    return choices[0]


def _first_message(response: object) -> dict[str, object]:
    choice = _first_choice(response)
    if choice is None:
        return {}
    return _get_field(choice, "message", {}) or {}


def _extract_tool_calls(message: dict[str, object]) -> list[dict[str, object]]:
    raw = _get_field(message, "tool_calls")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _parse_tool_arguments(raw: str | dict[str, object] | None) -> dict[str, object]:
    if not isinstance(raw, str):
        return raw if raw is not None else {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool arguments as JSON: %s", raw)
        return {"raw": raw}


def _build_content_blocks(
    text: str | None,
    reasoning: str | None,
    tool_calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Turn message text + reasoning + tool calls into Anthropic content blocks.

    For non-streaming responses, litellm has already extracted any
    ``<think>...</think>`` text the backend inlined into ``content`` and surfaced
    it as ``reasoning_content``; we forward that as a ``thinking`` block.
    Streaming responses are built incrementally in ``handle_streaming`` and do
    not go through this helper.
    """
    blocks: list[dict[str, object]] = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if text:
        blocks.append({"type": "text", "text": text})
    for tool_call in tool_calls:
        function = _get_field(tool_call, "function", {}) or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": _get_field(tool_call, "id", f"toolu_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"),
                "name": _get_field(function, "name", ""),
                "input": _parse_tool_arguments(_get_field(function, "arguments", "{}")),
            },
        )
    return blocks or [{"type": "text", "text": ""}]


def _extract_usage(usage: dict[str, object]) -> tuple[int, int]:
    return (
        _get_field(usage, "prompt_tokens", 0) or 0,
        _get_field(usage, "completion_tokens", 0) or 0,
    )


def convert_litellm_to_anthropic(
    litellm_response: object,
    original_request: MessagesRequest,
) -> MessagesResponse:
    """Translate a non-streaming LiteLLM response into an Anthropic Messages response.

    Returns a placeholder error response (logged) on any conversion failure so
    a broken upstream body never causes the proxy to crash.
    """
    try:
        return _build_anthropic_response(litellm_response, original_request)
    except Exception as e:
        logger.exception("Error converting response")
        return _build_error_response(original_request.model, e)


def _build_anthropic_response(litellm_response: object, original_request: MessagesRequest) -> MessagesResponse:
    message = _first_message(litellm_response)
    text = str(_get_field(message, "content") or "")
    reasoning = str(_get_field(message, "reasoning_content") or "")
    tool_calls = _extract_tool_calls(message)
    choice = _first_choice(litellm_response)
    finish_reason = _get_field(choice, "finish_reason", "stop")
    usage = cast("dict[str, object]", _get_field(litellm_response, "usage", {}))
    response_id = str(_get_field(litellm_response, "id", _new_msg_id()))
    prompt_tokens, completion_tokens = _extract_usage(usage)
    return MessagesResponse(
        id=response_id,
        model=original_request.model,
        role="assistant",
        content=_build_content_blocks(text, reasoning, tool_calls),
        stop_reason=_to_anthropic_stop_reason(finish_reason),
        stop_sequence=None,
        usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
    )


def _build_error_response(model: str, error: Exception) -> MessagesResponse:
    return MessagesResponse(
        id=_new_msg_id(),
        model=model,
        role="assistant",
        content=[
            {
                "type": "text",
                "text": f"Error converting response: {error}. Please check server logs.",
            },
        ],
        stop_reason="end_turn",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


@dataclass
class _OpenBlock:
    index: int
    kind: str  # "text" | "thinking" | "tool_use"


class _BlockTracker:
    """Allocates indices and tracks the currently-open Anthropic content block.

    The state machine is caller-driven: ``ensure(kind)`` closes any
    different-kind block that is open, then ``open(kind)`` allocates a fresh
    index. ``open()`` is unconditional and overwrites the prior block — callers
    that need close-before-open behaviour must call ``ensure()`` first, or call
    ``close()`` explicitly when emitting consecutive parallel tool blocks.
    """

    def __init__(self) -> None:
        self._next_index = 0
        self._current: _OpenBlock | None = None

    def is_open(self, kind: str | None = None) -> bool:
        if kind is None:
            return self._current is not None
        return self._current is not None and self._current.kind == kind

    def open(self, kind: str) -> _OpenBlock:
        block = _OpenBlock(index=self._next_index, kind=kind)
        self._next_index += 1
        self._current = block
        return block

    def ensure(self, kind: str) -> list[str]:
        if self._current is not None and self._current.kind != kind:
            return self.close()
        return []

    def delta(self, delta_payload: dict[str, Any]) -> str:
        if self._current is None:
            raise RuntimeError("no block is open; call open() first")
        return _SseFormatter.content_block_delta(self._current.index, delta_payload)

    def close(self) -> list[str]:
        if self._current is None:
            return []
        events = [_SseFormatter.content_block_stop(self._current.index)]
        self._current = None
        return events


class _SseFormatter:
    r"""Stateless formatters for Anthropic SSE events.

    Each event is framed as ``event: <name>\\ndata: <json>\\n\\n``; the trailing
    ``[DONE]`` sentinel is the only frame without an ``event:`` line. Keeping
    these as pure functions makes the streaming loop read as a sequence of
    named events rather than a wall of JSON literals.
    """

    @staticmethod
    def event(event_type: str, payload: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    @staticmethod
    def content_block_start(index: int, block: dict[str, Any]) -> str:
        return _SseFormatter.event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            },
        )

    @staticmethod
    def content_block_delta(index: int, delta: dict[str, Any]) -> str:
        return _SseFormatter.event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            },
        )

    @staticmethod
    def content_block_stop(index: int) -> str:
        return _SseFormatter.event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": index,
            },
        )

    @staticmethod
    def message_delta(stop_reason: str, output_tokens: int) -> str:
        return _SseFormatter.event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )

    @staticmethod
    def message_stop() -> str:
        return _SseFormatter.event("message_stop", {"type": "message_stop"})

    @staticmethod
    def ping() -> str:
        return _SseFormatter.event("ping", {"type": "ping"})

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"

    @staticmethod
    def error(error_type: str, message: str) -> str:
        # Anthropic SDK raises APIStatusError on `event: error`.
        return _SseFormatter.event(
            "error",
            {
                "type": "error",
                "error": {"type": error_type, "message": message},
            },
        )

    @staticmethod
    def finish(stop_reason: str, output_tokens: int) -> list[str]:
        return [
            _SseFormatter.message_delta(stop_reason, output_tokens),
            _SseFormatter.message_stop(),
            _SseFormatter.done(),
        ]


def _open_block(tracker: _BlockTracker, kind: str, block_dict: dict[str, Any]) -> Iterator[str]:
    yield from tracker.ensure(kind)
    if not tracker.is_open(kind):
        block = tracker.open(kind)
        yield _SseFormatter.content_block_start(block.index, block_dict)


def _emit_thinking(tracker: _BlockTracker, text: object) -> Iterator[str]:
    if not isinstance(text, str) or not text:
        return
    yield from _open_block(tracker, "thinking", {"type": "thinking", "thinking": ""})
    yield tracker.delta({"type": "thinking_delta", "thinking": text})


def _translate_parser_events(events: list[tuple[str, str | None]], tracker: _BlockTracker) -> Iterator[str]:
    for kind, value in events:
        if kind == "open":
            yield from _open_block(tracker, "thinking", {"type": "thinking", "thinking": ""})
        elif kind == "close":
            yield from tracker.close()
        elif kind == "thinking" and value:
            yield from _open_block(tracker, "thinking", {"type": "thinking", "thinking": ""})
            yield tracker.delta({"type": "thinking_delta", "thinking": value})
        elif kind == "text" and value:
            yield from _open_block(tracker, "text", {"type": "text", "text": ""})
            yield tracker.delta({"type": "text_delta", "text": value})


def _emit_failure(
    parser: "_ThinkStreamParser",
    tracker: _BlockTracker,
    output_tokens: int,
    exc: BaseException,
    message_prefix: str,
) -> Iterator[str]:
    """Drain, close, message_delta → event:error → done.

    Per-step try-wrap so a broken upstream can't mask the primary error.
    """
    try:
        for event in _translate_parser_events(parser.flush(), tracker):
            yield event
    except Exception:
        logger.exception("draining parser on fail")
    try:
        for event in tracker.close():
            yield event
    except Exception:
        logger.exception("closing tracker on fail")
    yield _SseFormatter.message_delta("end_turn", output_tokens)
    yield _SseFormatter.error(
        _anthropic_error_type(exc),
        _sanitize_error_message(f"{message_prefix}: {exc}"),
    )
    yield _SseFormatter.done()


def _log_request(
    method: str,
    path: str,
    source_model: str,
    target_model: str,
    tier: str | None,
    num_messages: int,
    num_tools: int,
    status_code: int,
) -> None:
    endpoint = path.split("?", 1)[0] if "?" in path else path
    status_color = _Colors.GREEN if status_code == STATUS_OK else _Colors.RED
    tier_str = f" tier={_color(_Colors.YELLOW, tier)}" if tier else ""
    line = (
        f"{_color(_Colors.BOLD, method)} {_color(_Colors.BOLD, endpoint)} "
        f"{_color(status_color, status_code)} "
        f"{_color(_Colors.CYAN, _short_model(source_model))} "
        f"{_color(_Colors.BOLD, '→')} "
        f"{_color(_Colors.GREEN, _short_model(target_model))}"
        f"{tier_str} "
        f"({_color(_Colors.MAGENTA, f'{num_tools} tools')}, "
        f"{_color(_Colors.BLUE, f'{num_messages} messages')})"
    )
    logger.info(line)


def _new_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"


@dataclass
class _StreamState:
    """Mutable per-request state for the SSE stream.

    Lives only as long as ``handle_streaming``'s iteration. ``should_stop``
    is set by chunk processors when they emit a finish_reason so the outer
    loop can break cleanly.
    """

    tool_index: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    has_sent_stop_reason: bool = False
    debug_first_chunk_logged: bool = False
    debug_chunk_count: int = 0
    consecutive_chunk_errors: int = 0
    should_stop: bool = False


def _stream_prologue(original_request: MessagesRequest, tracker: _BlockTracker) -> Iterator[str]:
    yield _SseFormatter.event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": _new_msg_id(),
                "type": "message",
                "role": "assistant",
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        },
    )
    # Always start with a text block; close-and-open transitions handle upstream switches.
    yield from _open_block(tracker, "text", {"type": "text", "text": ""})
    yield _SseFormatter.ping()


def _stream_epilogue(tracker: _BlockTracker, think_parser: _ThinkStreamParser, output_tokens: int) -> Iterator[str]:
    yield from _translate_parser_events(think_parser.flush(), tracker)
    yield from tracker.close()
    yield from _SseFormatter.finish("end_turn", output_tokens)


def _log_stream_finished(state: _StreamState) -> None:
    if not _litellm_debug_http_enabled():
        return
    logger.debug(
        "litellm stream finished: %s chunks, input_tokens=%s, output_tokens=%s",
        state.debug_chunk_count,
        state.input_tokens,
        state.output_tokens,
    )


def _record_chunk_debug(chunk: object, state: _StreamState) -> None:
    if not _litellm_debug_http_enabled():
        return
    state.debug_chunk_count += 1
    # Log the first chunk so we see the upstream's wire shape;
    # for the rest, litellm.set_verbose + httpx DEBUG cover it.
    if not state.debug_first_chunk_logged:
        state.debug_first_chunk_logged = True
        _debug_json_dump("litellm stream chunk #1 (first)", chunk)


def _record_chunk_usage(chunk: object, state: _StreamState) -> None:
    """Overwrite only when upstream sends non-zero.

    Some providers emit ``0`` mid-stream as a no-op and we don't want to
    clobber a real count with 0.
    """
    usage = _get_field(chunk, "usage")
    if usage is None:
        return
    incoming = _get_field(usage, "prompt_tokens")
    if isinstance(incoming, (int, float)) and incoming:
        state.input_tokens = int(incoming)
    incoming = _get_field(usage, "completion_tokens")
    if isinstance(incoming, (int, float)) and incoming:
        state.output_tokens = int(incoming)


def _coerce_delta_field(delta: object, key: str) -> object:
    """Read ``delta[key]`` with a defensive re-fetch.

    If delta has the key but ``_get_field`` returned None (some upstream
    quirk), grab it directly. In practice ``delta.get(key)`` and
    ``delta[key]`` agree, but the belt-and-braces fetch preserves the
    original behaviour.
    """
    value = _get_field(delta, key)
    if value is None and isinstance(delta, dict) and key in delta:
        value = delta[key]
    return value


def _process_chunk(
    chunk: object,
    tracker: _BlockTracker,
    think_parser: _ThinkStreamParser,
    state: _StreamState,
) -> Iterator[str]:
    _record_chunk_debug(chunk, state)
    _record_chunk_usage(chunk, state)

    choices = _get_field(chunk, "choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    delta = _get_field(choice, "delta") or _get_field(choice, "message", {}) or {}
    finish_reason = _get_field(choice, "finish_reason")

    if not isinstance(delta, dict):
        delta = {}

    # litellm exposes native reasoning as delta.reasoning_content on some providers;
    # honour it before falling back to think-tag parsing.
    delta_reasoning = _coerce_delta_field(delta, "reasoning_content")
    if delta_reasoning:
        yield from _emit_thinking(tracker, delta_reasoning)

    # Text after a tool_use closes the tool block first; track
    # "is a tool block open now" rather than "has one ever opened".
    delta_content = _coerce_delta_field(delta, "content")
    if delta_content:
        yield from _translate_parser_events(think_parser.feed(delta_content), tracker)

    delta_tool_calls = _coerce_delta_field(delta, "tool_calls")
    if delta_tool_calls:
        yield from _process_tool_calls(delta_tool_calls, tracker, state)

    if finish_reason and not state.has_sent_stop_reason:
        state.has_sent_stop_reason = True
        yield from _translate_parser_events(think_parser.flush(), tracker)
        yield from tracker.close()
        yield from _SseFormatter.finish(_to_anthropic_stop_reason(finish_reason), state.output_tokens)
        state.should_stop = True


def _process_tool_calls(
    delta_tool_calls: object,
    tracker: _BlockTracker,
    state: _StreamState,
) -> Iterator[str]:
    if state.tool_index is None:
        yield from tracker.close()
    if not isinstance(delta_tool_calls, list):
        delta_tool_calls = [delta_tool_calls]

    for tool_call in delta_tool_calls:
        yield from _process_single_tool_call(tool_call, tracker, state)


def _process_single_tool_call(tool_call: object, tracker: _BlockTracker, state: _StreamState) -> Iterator[str]:
    raw_index = _get_field(tool_call, "index", 0)
    current_index = raw_index if isinstance(raw_index, int) else 0

    if state.tool_index is None or current_index != state.tool_index:
        # Anthropic SSE requires content_block_stop(N) before
        # content_block_start(N+1); close the prior tool block
        # when a parallel call arrives in the same delta.
        if state.tool_index is not None:
            yield from tracker.close()
        state.tool_index = current_index
        function = _get_field(tool_call, "function", {}) or {}
        name = _get_field(function, "name", "")
        tool_id = _get_field(tool_call, "id") or _new_tool_id()
        block = tracker.open("tool_use")
        yield _SseFormatter.content_block_start(
            block.index,
            {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        )

    function = _get_field(tool_call, "function", {}) or {}
    arguments = _get_field(function, "arguments", "")
    if arguments:
        if isinstance(arguments, str):
            yield tracker.delta({"type": "input_json_delta", "partial_json": arguments})
        else:
            yield tracker.delta({"type": "input_json_delta", "partial_json": json.dumps(arguments)})


def _handle_chunk_error(
    exc: Exception,
    tracker: _BlockTracker,
    think_parser: _ThinkStreamParser,
    state: _StreamState,
) -> Iterator[str]:
    logger.warning(
        "Error processing chunk (%d/%d): %s",
        state.consecutive_chunk_errors + 1,
        MAX_CONSECUTIVE_CHUNK_ERRORS + 1,
        exc,
    )
    state.consecutive_chunk_errors += 1
    if state.consecutive_chunk_errors <= MAX_CONSECUTIVE_CHUNK_ERRORS:
        return
    # Threshold tripped — escalate with traceback and surface error.
    logger.error("Chunk error threshold exceeded; surfacing error", exc_info=exc)
    yield from _emit_failure(think_parser, tracker, state.output_tokens, exc, "chunk processing failed")
    state.should_stop = True


async def _stream_chunks(
    response_generator: AsyncGenerator[Any, None],
    tracker: _BlockTracker,
    think_parser: _ThinkStreamParser,
    state: _StreamState,
) -> AsyncIterator[str]:
    """Drive the per-chunk loop with its own try/except.

    ``handle_streaming``'s outer try stays under the PL statements-in-try-clause cap.
    """
    async for chunk in response_generator:
        try:
            for event in _process_chunk(chunk, tracker, think_parser, state):
                yield event
            # Successful chunk — reset streak so tolerance is per-streak.
            state.consecutive_chunk_errors = 0
        except Exception as e:
            for event in _handle_chunk_error(e, tracker, think_parser, state):
                yield event
        if state.should_stop:
            return


async def handle_streaming(
    response_generator: AsyncGenerator[Any, None],
    original_request: MessagesRequest,
) -> AsyncIterator[str]:
    """Yield Anthropic SSE frames for a streaming LiteLLM response.

    Drives prologue (message_start + ping), the per-chunk loop (text, thinking,
    tool calls, finish_reason), and epilogue (final flush + message_stop + done).
    Outer try catches upstream connection failures and surfaces them as a
    canonical error frame; inner tolerance lets isolated bad chunks through
    without breaking the stream.
    """
    state = _StreamState()
    tracker = _BlockTracker()
    think_parser = _ThinkStreamParser()

    try:
        for event in _stream_prologue(original_request, tracker):
            yield event

        async for event in _stream_chunks(response_generator, tracker, think_parser, state):
            yield event

        # Skip epilogue if chunk loop already terminated the stream via _emit_failure
        # — calling _translate_parser_events again here would re-emit the error frame.
        if not state.has_sent_stop_reason and not state.should_stop:
            for event in _stream_epilogue(tracker, think_parser, state.output_tokens):
                yield event
        _log_stream_finished(state)
    except Exception as e:
        logger.exception("Error in streaming")
        for event in _emit_failure(think_parser, tracker, state.output_tokens, e, "upstream streaming failed"):
            yield event
    finally:
        # Close so the internal httpx client releases deterministically —
        # otherwise mid-stream cancellation leaks it until GC.
        await response_generator.aclose()


def _prepare_litellm_request(request: MessagesRequest) -> dict[str, Any]:
    litellm_request = convert_anthropic_to_litellm(request)
    # After validation every model has the openai/ prefix, so this is the only branch we need.
    litellm_request["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        litellm_request["api_base"] = OPENAI_BASE_URL
    sanitize_messages_for_openai(litellm_request["messages"])
    return litellm_request


def _log_upstream_params_debug(litellm_request: dict[str, Any]) -> None:
    # Skip the bulky fields (messages, tools) — they dominate the dump and
    # are visible in litellm.set_verbose anyway.
    if not logger.isEnabledFor(logging.DEBUG):
        return
    debug = {k: v for k, v in litellm_request.items() if k not in {"messages", "tools"}}
    logger.debug("upstream params: %s", debug)
    if _litellm_debug_http_enabled():
        # Verbose: dump the entire kwargs dict going into litellm
        # (messages, tools, tool_choice, …) so we can confirm the
        # exact payload upstream sees, not just the sampling subset.
        _debug_json_dump("litellm.completion kwargs (full)", litellm_request)


def _log_response_debug(litellm_response: object, model: str, start_time: float) -> None:
    logger.debug("Response received: model=%s, time=%.2fs", model, time.time() - start_time)
    if not _litellm_debug_http_enabled():
        return
    payload: object
    response = cast("Any", litellm_response)
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()
    else:
        payload = repr(response)
    _debug_json_dump("litellm.completion response (full)", payload)


async def _handle_request(request: MessagesRequest) -> MessagesResponse | StreamingResponse:
    litellm_request = _prepare_litellm_request(request)
    _log_upstream_params_debug(litellm_request)
    _log_request(
        "POST",
        "/v1/messages",
        request.original_model or "unknown",
        litellm_request.get("model", "unknown"),
        request.tier,
        len(litellm_request["messages"]),
        len(request.tools) if request.tools else 0,
        STATUS_OK,
    )

    if request.stream:
        response_generator = await litellm.acompletion(**litellm_request)
        return StreamingResponse(
            handle_streaming(response_generator, request),
            media_type="text/event-stream",
        )

    start_time = time.time()
    litellm_response = litellm.completion(**litellm_request)
    _log_response_debug(litellm_response, str(litellm_request.get("model") or ""), start_time)
    return convert_litellm_to_anthropic(litellm_response, request)


@app.post("/v1/messages", response_model=None)
async def create_message(request: MessagesRequest) -> MessagesResponse | StreamingResponse:
    """Translate an Anthropic Messages request and dispatch to LiteLLM.

    Returns a ``MessagesResponse`` for non-streaming calls and a
    ``StreamingResponse`` wrapping the SSE generator for streaming calls.
    Wraps every error as an ``HTTPException`` with the upstream status code
    (falling back to 500) so the Anthropic SDK sees a real HTTP error.
    """
    try:
        return await _handle_request(request)
    except Exception as e:
        logger.exception("Error processing request")
        status_code = getattr(e, "status_code", 500)
        message = getattr(e, "message", None) or str(e)
        raise HTTPException(status_code=status_code, detail=message) from e


@app.get("/")
async def root() -> dict[str, str]:
    """Health endpoint — confirms the proxy is up and reports its identity."""
    return {"message": "Anthropic Proxy for LiteLLM"}


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", DEFAULT_HOST), port=DEFAULT_PORT)
