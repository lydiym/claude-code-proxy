from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Literal, Iterator
import logging
import json
import os
import sys
import time
import tomllib
import uuid
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

# Must be set before litellm is imported — otherwise it tries to refresh the
# model cost map from GitHub on every call and spams warnings on isolated nets.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# basicConfig formats import-time logs (litellm, uvicorn, _load_config).
# _configure_logging re-applies it after uvicorn installs its handlers.
_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.WARN,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-import: tiktoken stub before `import litellm`. Resolver is standalone
# so the main loader sits below.
# ---------------------------------------------------------------------------

def _str_to_bool(value, *, default=False):
    """Parse an env-style boolean; unrecognised strings fall back to ``default``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if str(value).strip().lower() not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_tiktoken_offline() -> bool:
    """[proxy].tiktoken_offline from CONFIG_PATH, TIKTOKEN_OFFLINE env, then True."""
    path = os.environ.get("CONFIG_PATH", "./config.toml")
    if path and os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            cfg = raw.get("proxy", {})
            if isinstance(cfg, dict) and "tiktoken_offline" in cfg:
                return _str_to_bool(cfg["tiktoken_offline"], default=True)
        except (OSError, tomllib.TOMLDecodeError):
            pass
    env = os.environ.get("TIKTOKEN_OFFLINE")
    if env not in (None, ""):
        return _str_to_bool(env, default=True)
    return True


TIKTOKEN_OFFLINE = _resolve_tiktoken_offline()

if TIKTOKEN_OFFLINE:
    # Stub tiktoken: skip the Azure blob fetch of cl100k_base.tiktoken.
    # Token counts are approximate; real counts come from upstream usage.
    import tiktoken

    class _OfflineEncoding:
        def encode(self, text, *args, **kwargs):
            return [1] * max(1, len(text) // 4)

        def encode_ordinary(self, text, *args, **kwargs):
            return self.encode(text, *args, **kwargs)

        def encode_single_token(self, token, *args, **kwargs):
            return [1]

        def decode(self, tokens, *args, **kwargs):
            return ""

        def decode_single_token_bytes(self, token):
            return b""

    tiktoken.get_encoding = lambda name: _OfflineEncoding()
    tiktoken.encoding_for_model = lambda model: _OfflineEncoding()

import litellm
import uvicorn


# ---------------------------------------------------------------------------
# Config loader (TOML primary source; per-key env-var fallback via _proxy_value)
# ---------------------------------------------------------------------------

# Tier names. Insertion order is the routing priority (haiku is checked first so
# it wins over big tiers in substring matches). Single source of truth — the
# TOML-loader section validator and the per-request tier lookup both read this.
TIER_KEYS = ("haiku", "sonnet", "opus", "fable", "mythos")
_VALID_TIERS = set(TIER_KEYS)
_ALLOWED_SAMPLING_KEYS = {
    "temperature", "top_p", "top_k",
    "max_completion_tokens", "stop", "seed",
}
_PROXY_KEYS = {
    "openai_api_key", "openai_base_url",
    "openai_tls_verify",
}
_ROUTING_KEYS = {
    "big_model", "small_model",
    "haiku_model", "sonnet_model", "opus_model", "fable_model", "mythos_model",
}
_VALID_SECTIONS = {"proxy", "routing", "global"} | _VALID_TIERS


def _coerce_sampling(key, value):
    """Coerce a TOML value into the expected Python type for a sampling field.
    Raises ValueError on bad type. TOML ints already come through as Python int,
    floats as float, strings as str, lists as list."""
    if key in ("temperature", "top_p"):
        # TOML ints/floats both acceptable; bool is rejected (bool ⊂ int in Python).
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"must be a number, got {type(value).__name__}")
        # OpenAI spec: temperature ∈ [0, 2], top_p ∈ [0, 1]. We enforce the
        # universally-rejected bounds (negatives, top_p > 1) and leave the
        # upper temperature bound permissive (some backends allow > 2).
        if value < 0:
            raise ValueError(f"must be non-negative, got {value}")
        if key == "top_p" and value > 1:
            raise ValueError(f"must be ≤ 1, got {value}")
        return float(value)
    if key in ("top_k", "max_completion_tokens", "seed"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"must be an integer, got {type(value).__name__}")
        # max_completion_tokens=0 is nonsensical (asking for zero tokens);
        # top_k=0 / seed=0 are valid sentinels some backends accept.
        if key == "max_completion_tokens" and value <= 0:
            raise ValueError(f"must be positive, got {value}")
        if value < 0:
            raise ValueError(f"must be non-negative, got {value}")
        return int(value)
    if key == "stop":
        if isinstance(value, str):
            raise ValueError("must be a list of strings, got a single string")
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError("must be a list of strings")
        # backends reject "" as a stop string; warn + drop so an empty entry
        # in [tier].stop = ["foo", ""] doesn't break the upstream call.
        if any(x == "" for x in value):
            raise ValueError("must not contain empty strings")
        return list(value)
    return value  # unknown keys handled by caller


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge: base + override, where override wins per leaf.
    Used at request time to layer tier config over [global] and to merge
    config extra_body into the upstream body."""
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
            return name[len(prefix):]
    return name


def _match_tier(name: str) -> Optional[str]:
    """First TIER_KEYS substring that appears in the lower-cased name.
    Returns None when no tier matches."""
    lower = name.lower()
    for tier in TIER_KEYS:  # insertion order = routing priority (haiku first)
        if tier in lower:
            return tier
    return None


def _parse_tier_section(body: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Parse a per-tier body (or [global]). Per-key type errors are warned and
    skipped so one bad value doesn't drop the rest of the section."""
    out: Dict[str, Any] = {}
    for k, v in body.items():
        if k == "extra_body":
            if not isinstance(v, dict):
                logger.warning(f"[{section}].extra_body must be a table; ignoring")
                continue
            out[k] = deepcopy(v)
        elif k in _ALLOWED_SAMPLING_KEYS:
            try:
                out[k] = _coerce_sampling(k, v)
            except ValueError as e:
                logger.warning(f"[{section}].{k}: {e}; ignoring")
        else:
            logger.warning(f"[{section}].{k} is not a recognised key; ignoring")
    return out


def _load_config(path: str) -> Dict[str, Any]:
    """Parse TOML at path; fail-open on every parse failure (logged, not raised)."""
    out: Dict[str, Any] = {"proxy": {}, "routing": {}, "global": {}, "tiers": {}}
    if not path:
        return out
    if not os.path.isfile(path):
        logger.info(f"CONFIG_PATH={path!r} not found; using env vars only")
        return out
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Malformed TOML in {path}: {e}; falling back to env vars")
        return out
    for section, body in raw.items():
        if section not in _VALID_SECTIONS:
            logger.warning(
                f"Unknown section [{section}] in {path}; ignoring "
                f"(valid: {sorted(_VALID_SECTIONS)})"
            )
            continue
        if not isinstance(body, dict):
            logger.warning(f"[{section}] must be a table, got {type(body).__name__}; ignoring")
            continue
        if section == "proxy":
            allowed = _PROXY_KEYS
            for k, v in body.items():
                if k not in allowed:
                    logger.warning(f"[{section}].{k} is not a recognised key; ignoring")
                    continue
                out["proxy"][k] = v
        elif section == "routing":
            allowed = _ROUTING_KEYS
            for k, v in body.items():
                if k not in allowed:
                    logger.warning(f"[{section}].{k} is not a recognised key; ignoring")
                    continue
                out["routing"][k] = v
        elif section == "global":
            out["global"] = _parse_tier_section(body, section)
        else:  # per-tier section
            out["tiers"][section] = _parse_tier_section(body, section)
    return out


CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.toml")
try:
    CONFIG = _load_config(CONFIG_PATH)
except Exception as e:
    # Infrastructure error only — parse failures are handled inside _load_config.
    logger.error(f"Failed to load CONFIG_PATH={CONFIG_PATH!r}: {e}; using env vars only")
    CONFIG = {"proxy": {}, "routing": {}, "global": {}, "tiers": {}}

# WARNING so the boot summary shows up before uvicorn installs its own handlers.
logger.warning(
    f"Loaded config from {CONFIG_PATH!r}: "
    f"proxy={list(CONFIG['proxy'])}, routing={list(CONFIG['routing'])}, "
    f"global={'yes' if CONFIG['global'] else 'no'}, tiers={list(CONFIG['tiers'])}"
)


def _proxy_value(key: str, env_name: str, default: Any = None) -> Any:
    """CONFIG[proxy|routing][key] → env var → default. None / "" fall through."""
    if key in _PROXY_KEYS:
        section = "proxy"
    elif key in _ROUTING_KEYS:
        section = "routing"
    else:
        raise ValueError(f"_proxy_value: {key!r} is not a recognised proxy/routing key")
    val = CONFIG[section].get(key)
    if val not in (None, ""):
        return val
    env_val = os.environ.get(env_name)
    if env_val not in (None, ""):
        return env_val
    return default


def _proxy_bool(key: str, env_name: str, default: bool = True) -> bool:
    """Resolve a boolean config value; unrecognised strings keep ``default``."""
    val = _proxy_value(key, env_name, default)
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    return _str_to_bool(val, default=default)


def _get_tier_override(tier: str) -> Optional[str]:
    """Per-request lookup of the per-tier routing override (e.g. HAIKU_MODEL).
    Re-read each call so CONFIG edits to [routing] take effect without restart;
    the BIG/SMALL fallback is captured at import (see _default_for_tier)."""
    return _proxy_value(f"{tier}_model", f"{tier.upper()}_MODEL")


def _default_for_tier(tier: str) -> str:
    """Fallback model for a tier when no per-tier override is set.
    Re-reads per call so edits to [routing].big_model / small_model take
    effect without restart."""
    if tier == "haiku":
        return _proxy_value("small_model", "SMALL_MODEL", "gpt-4.1-mini")
    return _proxy_value("big_model", "BIG_MODEL", "gpt-4.1")


class Colors:
    """ANSI color codes used to highlight parts of operational log lines."""
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _color(code, text):
    """Wrap text in ANSI code when stderr is a TTY, otherwise return plain."""
    try:
        if sys.stderr.isatty():
            return f"{code}{text}{Colors.RESET}"
    except (ValueError, AttributeError):
        pass
    return text


def _reset_logger(name, *, propagate):
    log = logging.getLogger(name)
    log.handlers.clear()
    log.propagate = propagate


@asynccontextmanager
async def _configure_logging(app: FastAPI):
    """Unify the log format and silence uvicorn.access. Runs after uvicorn's
    own configure_logging() so it overrides whatever uvicorn set up."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(handler)

    _LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        logger.setLevel(_LOG_LEVEL)
    except ValueError:
        logger.setLevel(logging.INFO)
        logger.warning(f"Invalid LOG_LEVEL={_LOG_LEVEL!r}; falling back to INFO")
    for noisy in ("LiteLLM", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _reset_logger("uvicorn", propagate=True)
    _reset_logger("uvicorn.error", propagate=True)
    _reset_logger("uvicorn.access", propagate=False)

    yield


app = FastAPI(lifespan=_configure_logging)


OPENAI_API_KEY = _proxy_value("openai_api_key", "OPENAI_API_KEY")
OPENAI_BASE_URL = _proxy_value("openai_base_url", "OPENAI_BASE_URL")
BIG_MODEL = _proxy_value("big_model", "BIG_MODEL", "gpt-4.1")
SMALL_MODEL = _proxy_value("small_model", "SMALL_MODEL", "gpt-4.1-mini")

# Per-tier default; the matching TIERNAME_MODEL config (or env var) overrides it.
# TIER_KEYS (defined near the top) is the canonical tier list; _default_for_tier
# maps each tier to SMALL_MODEL (haiku) or BIG_MODEL (everything else).

# Skip TLS validation when OPENAI_BASE_URL uses a self-signed cert (local LLM).
OPENAI_TLS_VERIFY = _proxy_bool("openai_tls_verify", "OPENAI_TLS_VERIFY", True)
litellm.ssl_verify = OPENAI_TLS_VERIFY
if not OPENAI_TLS_VERIFY:
    logger.warning("OPENAI_TLS_VERIFY=false — TLS certificate validation is disabled. Do not use this in production.")

# OpenAI Chat Completions caps max_completion_tokens at this value for most
# current models; over it the API rejects the request.
MAX_OUTPUT_TOKENS = 16384

DEFAULT_PORT = 8082

MSG_ID_HEX_LEN = 24

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


def to_anthropic_stop_reason(finish_reason):
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason or "", "end_turn")


def get_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def new_msg_id():
    return f"msg_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"


def short_model(name):
    return name.split("/")[-1] if "/" in name else name


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    # Echoed back in conversation history; we don't generate it locally.
    signature: Optional[str] = None


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockThinking,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None
    # Populated by `derive_tier` (model_validator below); used by
    # convert_anthropic_to_litellm to look up per-tier CONFIG settings.
    tier: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def capture_original_model(cls, data):
        if isinstance(data, dict) and "model" in data:
            data = dict(data)
            data["original_model"] = data["model"]
        return data

    @model_validator(mode="after")
    def derive_tier(self) -> "MessagesRequest":
        """Identify the Anthropic tier from the pre-rewrite original_model."""
        if self.original_model:
            self.tier = _match_tier(self.original_model)
        return self  # tier stays None → falls back to [global] in lookup

    @field_validator("model")
    def validate_model_field(cls, v):
        clean_v = _strip_provider_prefix(v)  # case preserved

        new_model = None
        if tier := _match_tier(v):
            chosen = _get_tier_override(tier) or _default_for_tier(tier)
            new_model = f"openai/{chosen}"
        elif clean_v.lower() in OPENAI_MODELS and not v.lower().startswith("openai/"):
            new_model = f"openai/{clean_v}"
        elif v.lower().startswith("openai/"):
            new_model = v  # already-prefixed passthrough (case-insensitive)
        else:
            # Custom endpoint: pass the bare name through with the openai/ prefix.
            logger.debug(f"No mapping rule for model '{v}', passing through")
            new_model = f"openai/{clean_v}"

        logger.debug(f"MODEL MAPPING: '{v}' -> '{new_model}'")

        return new_model


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockThinking, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# Anthropic wants `type: "thinking"` blocks; some backends fold reasoning
# into `<think>...</think>` inside `content` and the parser below splits them out.
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


class ThinkStreamParser:
    """Incremental parser that splits a stream into thinking vs. text chunks.

    The OpenAI-compatible model emits <think>...</think> markers inside its
    content stream. Anthropic SSE needs these surfaced as their own content
    blocks, so we buffer just enough to recognise a marker split across chunks
    and yield tagged deltas for the caller to forward.
    """

    def __init__(self):
        self.in_thinking = False
        self.buffer = ""

    def feed(self, text):
        """Consume a chunk of model output; return a list of (kind, value).

        kind is "text" or "thinking" for a delta, or "open" / "close" for a
        block transition (value is None for those).
        """
        if not text:
            return []
        events = []
        self.buffer += text
        while self._drain(events):
            pass
        return events

    def _drain(self, events):
        if not self.buffer:
            return False
        tag = THINK_CLOSE_TAG if self.in_thinking else THINK_OPEN_TAG
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
        self.buffer = self.buffer[idx + len(tag):]
        return True

    def flush(self):
        if not self.buffer:
            return []
        events = [(("thinking" if self.in_thinking else "text"), self.buffer)]
        self.buffer = ""
        if self.in_thinking:
            events.append(("close", None))
            self.in_thinking = False
        return events


def parse_tool_result_content(content):
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
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
        return result.strip()

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


def convert_image_block(source: Any) -> Dict[str, Any]:
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


def _extract_text(content) -> str:
    """Pull plain text out of a content field — None, string, or list of blocks.

    Used for system messages (and any other role) whose text we want to
    concatenate without preserving block structure. Non-text blocks (images,
    tool_use, tool_result, thinking) are skipped.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    parts = [
        get_field(block, "text", "")
        for block in content
        if get_field(block, "type") == "text"
    ]
    return "\n\n".join(p for p in parts if p)


def system_to_message(system):
    text = _extract_text(system) if system else ""
    return {"role": "system", "content": text} if text else None


def _build_system_message(system_field, messages) -> Optional[Dict[str, str]]:
    """Combine the top-level system field with any in-band role='system' messages
    into a single OpenAI system message.

    Anthropic's spec only allows system at the top level, but Claude Code
    2.1.154+ has started embedding system reminders inline. We hoist them all
    to the start so OpenAI sees one system message at the top. Order is
    preserved: in-band messages come first, then the top-level field — which
    is the order Claude Code most likely intended when it injected the
    reminders inline.
    """
    parts = [
        text
        for text in (_extract_text(m.content) for m in messages if m.role == "system")
        if text
    ]
    top = _extract_text(system_field)
    if top:
        parts.append(top)
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}


def collect_tool_ids(messages):
    call_ids = set()
    result_ids = set()
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            block_type = get_field(block, "type")
            if block_type == "tool_use":
                call_ids.add(block.id)
            elif block_type == "tool_result":
                rid = get_field(block, "tool_use_id", "") or ""
                if rid:
                    result_ids.add(rid)
    return call_ids, result_ids


def convert_assistant_message(msg, result_ids):
    text_parts = []
    tool_calls = []

    for block in msg.content:
        block_type = get_field(block, "type")
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            if block.id in result_ids:
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
            else:
                # Dangling call (result truncated from history): describe in
                # prose — small models mimic tool-call syntax otherwise.
                text_parts.append(
                    f"(An earlier {block.name} tool call is missing its "
                    f"result in this context.)"
                )

    text = "\n".join(text_parts).strip()
    out = {"role": "assistant"}
    # OpenAI allows null content only when tool_calls are present.
    out["content"] = text if text else (None if tool_calls else "")
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def convert_user_message(msg, call_ids):
    tool_messages = []
    user_parts = []

    for block in msg.content:
        block_type = get_field(block, "type")
        if block_type == "text":
            user_parts.append({"type": "text", "text": block.text})
        elif block_type == "image":
            user_parts.append(convert_image_block(block.source))
        elif block_type == "tool_result":
            tool_use_id = get_field(block, "tool_use_id", "") or ""
            result_text = parse_tool_result_content(get_field(block, "content"))
            if tool_use_id in call_ids:
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": result_text,
                })
            else:
                # Orphaned result (truncated call): fold into user text; the ghost id is meaningless.
                user_parts.append({
                    "type": "text",
                    "text": f"(Result from an earlier tool call:)\n{result_text}",
                })

    # Tool results must follow the matching assistant turn, so emit them first.
    out = list(tool_messages)
    if user_parts:
        if all(part.get("type") == "text" for part in user_parts):
            text = "\n".join(part["text"] for part in user_parts).strip()
            out.append({"role": "user", "content": text or "..."})
        else:
            out.append({"role": "user", "content": user_parts})
    return out


def _convert_message(msg, result_ids, call_ids) -> List[Dict[str, Any]]:
    if isinstance(msg.content, str):
        return [{"role": msg.role, "content": msg.content}]
    if msg.role == "assistant":
        return [convert_assistant_message(msg, result_ids)]
    return convert_user_message(msg, call_ids)


def convert_tool_definitions(tools):
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


def convert_tool_choice(choice):
    choice_type = get_field(choice, "type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "any"
    if choice_type == "tool":
        name = get_field(choice, "name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return "auto"


def sanitize_messages_for_openai(messages):
    allowed_keys = {"role", "content", "name", "tool_call_id", "tool_calls"}
    for msg in messages:
        for key in list(msg.keys()):
            if key not in allowed_keys:
                logger.debug(f"Removing unsupported message field: {key}")
                del msg[key]
        if msg.get("content") in (None, "") and not msg.get("tool_calls"):
            msg["content"] = "..."


def _resolve_tier_config(request: "MessagesRequest") -> Dict[str, Any]:
    """Tier-specific config deep-merged over [global]; tier=None → just [global].
    Deep-copied so downstream mutations can't corrupt CONFIG's nested dicts.
    None guards on global/tiers/tier itself so a partially-patched CONFIG doesn't crash."""
    base = CONFIG.get("global") or {}
    tiers = CONFIG.get("tiers") or {}
    if request.tier:
        tier_cfg = tiers.get(request.tier)
        if tier_cfg is not None:
            return deepcopy(_deep_merge(base, tier_cfg))
    return deepcopy(base)


def _is_skip_sampling(key: str, value: Any) -> bool:
    """True when a sampling value should be treated as absent (None or empty stop)."""
    if value is None:
        return True
    if key == "stop" and isinstance(value, list) and (not value or any(s == "" for s in value)):
        return True
    return False


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    call_ids, result_ids = collect_tool_ids(anthropic_request.messages)

    messages = []
    if system := _build_system_message(anthropic_request.system, anthropic_request.messages):
        messages.append(system)

    # LiteLLM uses assistant.tool_calls + role="tool"; flattening taught small
    # models to emit tool calls as literal text and broke tool use.
    for msg in anthropic_request.messages:
        if msg.role != "system":
            messages.extend(_convert_message(msg, result_ids, call_ids))

    litellm_request: Dict[str, Any] = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_completion_tokens": min(anthropic_request.max_tokens, MAX_OUTPUT_TOKENS),
        "stream": anthropic_request.stream,
    }
    if anthropic_request.tools:
        litellm_request["tools"] = convert_tool_definitions(anthropic_request.tools)
    if anthropic_request.tool_choice:
        litellm_request["tool_choice"] = convert_tool_choice(anthropic_request.tool_choice)

    # Per-tier CONFIG: config wins per key; request value preserved only when
    # the client explicitly sent it (model_fields_set); otherwise omit so the
    # upstream API uses its own default rather than MessagesRequest's Pydantic
    # defaults like temperature=1.0.
    tier_cfg = _resolve_tier_config(anthropic_request)
    fields_set = anthropic_request.model_fields_set

    for kw, attr in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop", "stop_sequences"),
    ):
        if kw in tier_cfg and not _is_skip_sampling(kw, tier_cfg[kw]):
            litellm_request[kw] = tier_cfg[kw]
        elif attr in fields_set:
            value = getattr(anthropic_request, attr)
            if _is_skip_sampling(kw, value):
                continue
            litellm_request[kw] = value

    # Anthropic always sends max_tokens, so config can only lower the ceiling;
    # MAX_OUTPUT_TOKENS remains the absolute OpenAI cap.
    if "max_completion_tokens" in tier_cfg:
        litellm_request["max_completion_tokens"] = min(
            litellm_request["max_completion_tokens"],
            int(tier_cfg["max_completion_tokens"]),
        )

    if "seed" in tier_cfg:
        litellm_request["seed"] = tier_cfg["seed"]

    # deep-merge: config keys win per leaf; unrelated request keys preserved.
    # isinstance guard handles runtime-patched CONFIG that bypassed the loader.
    cfg_extra = tier_cfg.get("extra_body")
    if isinstance(cfg_extra, dict) and cfg_extra:
        litellm_request["extra_body"] = _deep_merge(
            litellm_request.get("extra_body", {}), cfg_extra
        )

    return litellm_request


def _first_choice(response):
    choices = get_field(response, "choices", [])
    if not choices:
        return None
    return choices[0]


def _first_message(response):
    choice = _first_choice(response)
    if choice is None:
        return {}
    return get_field(choice, "message", {}) or {}


def _extract_tool_calls(message):
    raw = get_field(message, "tool_calls")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _parse_tool_arguments(raw):
    if not isinstance(raw, str):
        return raw if raw is not None else {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool arguments as JSON: {raw}")
        return {"raw": raw}


def _build_content_blocks(text, reasoning, tool_calls):
    """Turn message text + reasoning + tool calls into Anthropic content blocks.

    For non-streaming responses, litellm has already extracted any
    ``<think>...</think>`` text the backend inlined into ``content`` and surfaced
    it as ``reasoning_content``; we forward that as a ``thinking`` block.
    Streaming responses are built incrementally in ``handle_streaming`` and do
    not go through this helper.
    """
    blocks = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if text:
        blocks.append({"type": "text", "text": text})
    for tool_call in tool_calls:
        function = get_field(tool_call, "function", {}) or {}
        blocks.append({
            "type": "tool_use",
            "id": get_field(tool_call, "id", f"toolu_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"),
            "name": get_field(function, "name", ""),
            "input": _parse_tool_arguments(get_field(function, "arguments", "{}")),
        })
    return blocks or [{"type": "text", "text": ""}]


def _extract_usage(usage):
    return (
        get_field(usage, "prompt_tokens", 0) or 0,
        get_field(usage, "completion_tokens", 0) or 0,
    )


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    try:
        message = _first_message(litellm_response)
        text = get_field(message, "content") or ""
        reasoning = get_field(message, "reasoning_content") or ""
        tool_calls = _extract_tool_calls(message)
        choice = _first_choice(litellm_response)
        finish_reason = get_field(choice, "finish_reason", "stop")
        usage = get_field(litellm_response, "usage", {})
        response_id = get_field(litellm_response, "id", new_msg_id())
        prompt_tokens, completion_tokens = _extract_usage(usage)

        return MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=_build_content_blocks(text, reasoning, tool_calls),
            stop_reason=to_anthropic_stop_reason(finish_reason),
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )
    except Exception as e:
        logger.error(f"Error converting response: {e}", exc_info=True)
        return MessagesResponse(
            id=new_msg_id(),
            model=original_request.model,
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {e}. Please check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


@dataclass
class OpenBlock:
    index: int
    kind: str  # "text" | "thinking" | "tool_use"


class BlockTracker:
    """Allocates indices and tracks the currently-open Anthropic content block.

    The state machine is caller-driven: ``ensure(kind)`` closes any
    different-kind block that is open, then ``open(kind)`` allocates a fresh
    index. ``open()`` is unconditional and overwrites the prior block — callers
    that need close-before-open behaviour must call ``ensure()`` first. The
    tool_use path relies on this so consecutive parallel tool calls each get
    their own index without an intermediate ``content_block_stop``.
    """

    def __init__(self) -> None:
        self._next_index = 0
        self._current: Optional[OpenBlock] = None

    @property
    def current(self) -> Optional[OpenBlock]:
        return self._current

    def is_open(self, kind: Optional[str] = None) -> bool:
        if kind is None:
            return self._current is not None
        return self._current is not None and self._current.kind == kind

    def open(self, kind: str) -> OpenBlock:
        block = OpenBlock(index=self._next_index, kind=kind)
        self._next_index += 1
        self._current = block
        return block

    def ensure(self, kind: str) -> List[str]:
        if self._current is not None and self._current.kind != kind:
            return self.close()
        return []

    def delta(self, delta_payload: Dict[str, Any]) -> str:
        if self._current is None:
            raise RuntimeError("no block is open; call open() first")
        return SseFormatter.content_block_delta(self._current.index, delta_payload)

    def close(self) -> List[str]:
        if self._current is None:
            return []
        events = [SseFormatter.content_block_stop(self._current.index)]
        self._current = None
        return events


class SseFormatter:
    """Stateless formatters for Anthropic SSE events.

    Each event is framed as ``event: <name>\\ndata: <json>\\n\\n``; the trailing
    ``[DONE]`` sentinel is the only frame without an ``event:`` line. Keeping
    these as pure functions makes the streaming loop read as a sequence of
    named events rather than a wall of JSON literals.
    """

    @staticmethod
    def event(event_type: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    @staticmethod
    def content_block_start(index: int, block: Dict[str, Any]) -> str:
        return SseFormatter.event("content_block_start", {
            "type": "content_block_start", "index": index, "content_block": block,
        })

    @staticmethod
    def content_block_delta(index: int, delta: Dict[str, Any]) -> str:
        return SseFormatter.event("content_block_delta", {
            "type": "content_block_delta", "index": index, "delta": delta,
        })

    @staticmethod
    def content_block_stop(index: int) -> str:
        return SseFormatter.event("content_block_stop", {
            "type": "content_block_stop", "index": index,
        })

    @staticmethod
    def message_delta(stop_reason: str, output_tokens: int) -> str:
        return SseFormatter.event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

    @staticmethod
    def message_stop() -> str:
        return SseFormatter.event("message_stop", {"type": "message_stop"})

    @staticmethod
    def ping() -> str:
        return SseFormatter.event("ping", {"type": "ping"})

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"

    @staticmethod
    def finish(stop_reason: str, output_tokens: int) -> List[str]:
        return [
            SseFormatter.message_delta(stop_reason, output_tokens),
            SseFormatter.message_stop(),
            SseFormatter.done(),
        ]


def _open_block(
    tracker: BlockTracker, kind: str, block_dict: Dict[str, Any]
) -> Iterator[str]:
    for event in tracker.ensure(kind):
        yield event
    if not tracker.is_open(kind):
        block = tracker.open(kind)
        yield SseFormatter.content_block_start(block.index, block_dict)


def _emit_thinking(tracker: BlockTracker, text: str) -> Iterator[str]:
    if not text:
        return
    yield from _open_block(tracker, "thinking", {"type": "thinking", "thinking": ""})
    yield tracker.delta({"type": "thinking_delta", "thinking": text})


def _translate_parser_events(
    events: List[tuple], tracker: BlockTracker
) -> Iterator[str]:
    for kind, value in events:
        if kind == "open":
            yield from _open_block(tracker, "thinking", {"type": "thinking", "thinking": ""})
        elif kind == "close":
            for event in tracker.close():
                yield event
        elif kind == "thinking" and value:
            yield from _open_block(tracker, "thinking", {"type": "thinking", "thinking": ""})
            yield tracker.delta({"type": "thinking_delta", "thinking": value})
        elif kind == "text" and value:
            yield from _open_block(tracker, "text", {"type": "text", "text": ""})
            yield tracker.delta({"type": "text_delta", "text": value})


def log_request(method, path, source_model, target_model, tier, num_messages, num_tools, status_code):
    endpoint = path.split("?", 1)[0] if "?" in path else path
    status_color = Colors.GREEN if status_code == 200 else Colors.RED
    tier_str = f" tier={_color(Colors.YELLOW, tier)}" if tier else ""
    line = (
        f"{_color(Colors.BOLD, method)} {_color(Colors.BOLD, endpoint)} "
        f"{_color(status_color, status_code)} "
        f"{_color(Colors.CYAN, short_model(source_model))} "
        f"{_color(Colors.BOLD, '→')} "
        f"{_color(Colors.GREEN, short_model(target_model))}"
        f"{tier_str} "
        f"({_color(Colors.MAGENTA, f'{num_tools} tools')}, "
        f"{_color(Colors.BLUE, f'{num_messages} messages')})"
    )
    logger.info(line)


async def handle_streaming(response_generator, original_request: MessagesRequest):
    tracker = BlockTracker()
    think_parser = ThinkStreamParser()
    tool_index: Optional[int] = None
    input_tokens = 0
    output_tokens = 0
    has_sent_stop_reason = False

    def new_tool_id() -> str:
        return f"toolu_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"

    try:
        yield SseFormatter.event("message_start", {
            "type": "message_start",
            "message": {
                "id": new_msg_id(),
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
        })

        # Always start with a text block; close-and-open transitions handle upstream switches.
        for event in _open_block(tracker, "text", {"type": "text", "text": ""}):
            yield event
        yield SseFormatter.ping()

        async for chunk in response_generator:
            try:
                usage = get_field(chunk, "usage")
                if usage is not None:
                    input_tokens = get_field(usage, "prompt_tokens", input_tokens) or 0
                    output_tokens = get_field(usage, "completion_tokens", output_tokens) or 0

                choices = get_field(chunk, "choices")
                if not choices:
                    continue
                choice = choices[0]
                delta = get_field(choice, "delta") or get_field(choice, "message", {}) or {}
                finish_reason = get_field(choice, "finish_reason")

                # litellm exposes native reasoning as delta.reasoning_content on some providers;
                # honour it before falling back to `` parsing.
                delta_reasoning = get_field(delta, "reasoning_content")

                delta_content = get_field(delta, "content")
                if isinstance(delta, dict) and "content" in delta and delta_content is None:
                    delta_content = delta["content"]

                if delta_reasoning:
                    for event in _emit_thinking(tracker, delta_reasoning):
                        yield event

                if delta_content and tool_index is None:
                    for event in _translate_parser_events(think_parser.feed(delta_content), tracker):
                        yield event

                delta_tool_calls = get_field(delta, "tool_calls")
                if isinstance(delta, dict) and "tool_calls" in delta and delta_tool_calls is None:
                    delta_tool_calls = delta["tool_calls"]

                if delta_tool_calls:
                    if tool_index is None:
                        for event in tracker.close():
                            yield event
                    if not isinstance(delta_tool_calls, list):
                        delta_tool_calls = [delta_tool_calls]

                    for tool_call in delta_tool_calls:
                        current_index = get_field(tool_call, "index", 0)

                        if tool_index is None or current_index != tool_index:
                            tool_index = current_index
                            function = get_field(tool_call, "function", {}) or {}
                            name = get_field(function, "name", "")
                            tool_id = get_field(tool_call, "id") or new_tool_id()
                            block = tracker.open("tool_use")
                            yield SseFormatter.content_block_start(block.index, {
                                "type": "tool_use", "id": tool_id, "name": name, "input": {},
                            })

                        function = get_field(tool_call, "function", {}) or {}
                        arguments = get_field(function, "arguments", "")
                        if arguments:
                            if isinstance(arguments, str):
                                yield tracker.delta({"type": "input_json_delta", "partial_json": arguments})
                            else:
                                yield tracker.delta({"type": "input_json_delta", "partial_json": json.dumps(arguments)})

                if finish_reason and not has_sent_stop_reason:
                    has_sent_stop_reason = True
                    for event in _translate_parser_events(think_parser.flush(), tracker):
                        yield event
                    for event in tracker.close():
                        yield event
                    for event in SseFormatter.finish(to_anthropic_stop_reason(finish_reason), output_tokens):
                        yield event
                    return
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue

        if not has_sent_stop_reason:
            for event in _translate_parser_events(think_parser.flush(), tracker):
                yield event
            for event in tracker.close():
                yield event
            for event in SseFormatter.finish("end_turn", output_tokens):
                yield event

    except Exception as e:
        logger.error(f"Error in streaming: {e}", exc_info=True)
        yield SseFormatter.message_delta("error", 0)
        yield SseFormatter.message_stop()
        yield SseFormatter.done()
    finally:
        # Close so the internal httpx client releases deterministically —
        # otherwise mid-stream cancellation leaks it until GC.
        await response_generator.aclose()


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    try:
        litellm_request = convert_anthropic_to_litellm(request)

        # After validation every model has the openai/ prefix, so this is the only branch we need.
        litellm_request["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            litellm_request["api_base"] = OPENAI_BASE_URL

        sanitize_messages_for_openai(litellm_request["messages"])

        # DEBUG: dump effective upstream params (request or config source).
        if logger.isEnabledFor(logging.DEBUG):
            debug = {"model": litellm_request.get("model")}
            for k in _ALLOWED_SAMPLING_KEYS:
                if k in litellm_request:
                    debug[k] = litellm_request[k]
            if eb := litellm_request.get("extra_body"):
                debug["extra_body"] = eb
            logger.debug(f"upstream params: {debug}")

        log_request(
            "POST",
            "/v1/messages",
            request.original_model or "unknown",
            litellm_request.get("model"),
            request.tier,
            len(litellm_request["messages"]),
            len(request.tools) if request.tools else 0,
            200,
        )

        if request.stream:
            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )

        start_time = time.time()
        litellm_response = litellm.completion(**litellm_request)
        logger.debug(
            f"Response received: model={litellm_request.get('model')}, time={time.time() - start_time:.2f}s"
        )
        return convert_litellm_to_anthropic(litellm_response, request)

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        status_code = getattr(e, "status_code", 500)
        message = getattr(e, "message", None) or str(e)
        raise HTTPException(status_code=status_code, detail=message)


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)
