from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Literal
import logging
import json
import os
import sys
import time
import uuid
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import litellm
import uvicorn

load_dotenv()

# Single source of truth for log format. Used by the root handler below and
# applied again via the lifespan hook so it sticks even when uvicorn is
# started directly via `uvicorn server:app` (which would otherwise install
# its own formatters and handlers).
_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.WARN,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

for noisy in ("LiteLLM", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


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


@asynccontextmanager
async def _configure_logging(app: FastAPI):
    """Runs after uvicorn has installed its log handlers. Replace them with
    our unified format and silence uvicorn.access, which would otherwise
    duplicate log_request on every call."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]

    for name in ("uvicorn", "uvicorn.error"):
        log = logging.getLogger(name)
        log.handlers = []
        log.propagate = True

    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False

    yield


app = FastAPI(lifespan=_configure_logging)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# OpenAI Chat Completions caps max_completion_tokens at this value for most
# current models; over it the API rejects the request.
MAX_OUTPUT_TOKENS = 16384

DEFAULT_PORT = 8082

MSG_ID_HEX_LEN = 24

# OpenAI models recognised without an explicit `openai/` prefix. Anything not
# here is treated as opaque — pass through, prefixed with `openai/` — so users
# can target custom OpenAI-compatible endpoints that use arbitrary names.
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
    """Map an OpenAI/LiteLLM finish_reason to an Anthropic stop_reason."""
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason or "", "end_turn")


def get_field(obj, key, default=None):
    """Read a field from a dict or object uniformly."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def new_msg_id():
    return f"msg_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"


def short_model(name):
    return name.split("/")[-1] if "/" in name else name


# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


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
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
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

    @model_validator(mode="before")
    @classmethod
    def capture_original_model(cls, data):
        if isinstance(data, dict) and "model" in data:
            data = dict(data)
            data["original_model"] = data["model"]
        return data

    @field_validator("model")
    def validate_model_field(cls, v):
        # Strip any prefix the client might have added so we match on the bare
        # name (Claude Code sends `claude-3-5-sonnet-20241022`).
        clean_v = v
        for prefix in ("anthropic/", "openai/", "gemini/"):
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix):]
                break

        lower = clean_v.lower()
        if "haiku" in lower:
            new_model = f"openai/{SMALL_MODEL}"
        elif "sonnet" in lower:
            new_model = f"openai/{BIG_MODEL}"
        elif clean_v in OPENAI_MODELS and not v.startswith("openai/"):
            new_model = f"openai/{clean_v}"
        elif v.startswith("openai/"):
            new_model = v
        else:
            # Custom OpenAI-compatible endpoint (e.g. local models). Pass the
            # bare name through with the openai/ prefix so LiteLLM routes it.
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
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
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
    """Convert an Anthropic image source block to OpenAI image_url format."""
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


def system_to_message(system):
    """Build a single OpenAI system message from the Anthropic system field."""
    if not system:
        return None
    if isinstance(system, str):
        return {"role": "system", "content": system}
    parts = []
    for block in system:
        if get_field(block, "type") != "text":
            continue
        text = get_field(block, "text")
        if text:
            parts.append(text)
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}


def collect_tool_ids(messages):
    """Return (call_ids, result_ids) sets across all messages."""
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
    """Convert an Anthropic assistant message to OpenAI format."""
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
                # Dangling call whose result was truncated from history.
                # Describe it in prose rather than a tool-call-like syntax
                # (which small models tend to mimic) so the backend does
                # not demand a response for an unanswerable call.
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
    """Convert an Anthropic user message into one or more OpenAI messages."""
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
                # Orphaned result whose call was truncated from history. Fold
                # into user text so we never emit an unpaired tool message;
                # the ghost id would be meaningless to the model, so drop it.
                user_parts.append({
                    "type": "text",
                    "text": f"(Result from an earlier tool call:)\n{result_text}",
                })

    # Tool results must immediately follow the assistant turn that produced
    # the matching tool_calls, so emit them before any trailing user text.
    out = list(tool_messages)
    if user_parts:
        if all(part.get("type") == "text" for part in user_parts):
            text = "\n".join(part["text"] for part in user_parts).strip()
            out.append({"role": "user", "content": text or "..."})
        else:
            out.append({"role": "user", "content": user_parts})
    return out


def convert_tool_definitions(tools):
    """Convert Anthropic tool definitions to OpenAI function tool definitions."""
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
    """Convert an Anthropic tool_choice to OpenAI tool_choice."""
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
    """Strip message fields foreign to OpenAI Chat and fill empty content."""
    allowed_keys = {"role", "content", "name", "tool_call_id", "tool_calls"}
    for msg in messages:
        for key in list(msg.keys()):
            if key not in allowed_keys:
                logger.debug(f"Removing unsupported message field: {key}")
                del msg[key]
        if msg.get("content") in (None, "") and not msg.get("tool_calls"):
            msg["content"] = "..."


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert an Anthropic Messages request to a LiteLLM (OpenAI Chat) request."""
    messages = []
    if system := system_to_message(anthropic_request.system):
        messages.append(system)

    call_ids, result_ids = collect_tool_ids(anthropic_request.messages)

    # LiteLLM's canonical input is OpenAI Chat format for every provider, so
    # tool calls travel as assistant.tool_calls and their results as role="tool"
    # messages. Flattening them into plain text (the previous behaviour) taught
    # small models to emit tool calls as literal text, which broke tool use.
    for msg in anthropic_request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        elif msg.role == "assistant":
            messages.append(convert_assistant_message(msg, result_ids))
        else:
            messages.extend(convert_user_message(msg, call_ids))

    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_completion_tokens": min(anthropic_request.max_tokens, MAX_OUTPUT_TOKENS),
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    if anthropic_request.tools:
        litellm_request["tools"] = convert_tool_definitions(anthropic_request.tools)
    if anthropic_request.tool_choice:
        litellm_request["tool_choice"] = convert_tool_choice(anthropic_request.tool_choice)

    return litellm_request


def _first_choice(response):
    """Return the first choice of a LiteLLM response, or None."""
    choices = get_field(response, "choices", [])
    if not choices:
        return None
    return choices[0]


def _first_message(response):
    """Return the first choice's message dict/object, or empty dict."""
    choice = _first_choice(response)
    if choice is None:
        return {}
    return get_field(choice, "message", {}) or {}


def _extract_tool_calls(message):
    """Return a list of tool_call dicts/objects from a message, or []."""
    raw = get_field(message, "tool_calls")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _parse_tool_arguments(raw):
    """Parse a tool call arguments string into a dict; return raw on failure."""
    if not isinstance(raw, str):
        return raw if raw is not None else {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool arguments as JSON: {raw}")
        return {"raw": raw}


def _build_content_blocks(text, tool_calls):
    """Turn message text + tool calls into Anthropic content blocks."""
    blocks = []
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
    """Return (prompt_tokens, completion_tokens) from a usage object or dict."""
    return (
        get_field(usage, "prompt_tokens", 0) or 0,
        get_field(usage, "completion_tokens", 0) or 0,
    )


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    """Convert a LiteLLM (OpenAI Chat) response to an Anthropic Messages response."""
    try:
        message = _first_message(litellm_response)
        text = get_field(message, "content") or ""
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
            content=_build_content_blocks(text, tool_calls),
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


def log_request(method, path, source_model, target_model, num_messages, num_tools, status_code):
    """Log a one-line request summary highlighting the source/target model mapping."""
    endpoint = path.split("?", 1)[0] if "?" in path else path
    status_color = Colors.GREEN if status_code == 200 else Colors.RED
    line = (
        f"{_color(Colors.BOLD, method)} {_color(Colors.BOLD, endpoint)} "
        f"{_color(status_color, status_code)} "
        f"{_color(Colors.CYAN, short_model(source_model))} "
        f"{_color(Colors.BOLD, '→')} "
        f"{_color(Colors.GREEN, short_model(target_model))} "
        f"({_color(Colors.MAGENTA, f'{num_tools} tools')}, "
        f"{_color(Colors.BLUE, f'{num_messages} messages')})"
    )
    logger.info(line)


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Convert a LiteLLM streaming response into Anthropic SSE events."""

    def emit(event: str, data: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def text_block_open() -> str:
        return emit(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )

    def text_delta(text: str) -> str:
        return emit(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        )

    def text_block_close() -> str:
        return emit("content_block_stop", {"type": "content_block_stop", "index": 0})

    def tool_block_open(idx: int, tool_id: str, name: str) -> str:
        return emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            },
        )

    def tool_delta(idx: int, partial_json: str) -> str:
        return emit(
            "content_block_delta",
            {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": partial_json}},
        )

    def tool_block_close(idx: int) -> str:
        return emit("content_block_stop", {"type": "content_block_stop", "index": idx})

    def finish_stream(stop_reason: str, output_tokens: int) -> List[str]:
        return [
            emit(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            ),
            emit("message_stop", {"type": "message_stop"}),
            "data: [DONE]\n\n",
        ]

    def new_tool_id():
        return f"toolu_{uuid.uuid4().hex[:MSG_ID_HEX_LEN]}"

    try:
        yield emit(
            "message_start",
            {
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
            },
        )
        yield text_block_open()
        yield emit("ping", {"type": "ping"})

        tool_index: Optional[int] = None
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

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

                delta_content = get_field(delta, "content")
                if isinstance(delta, dict) and "content" in delta and delta_content is None:
                    delta_content = delta["content"]
                if delta_content:
                    accumulated_text += delta_content
                    if tool_index is None and not text_block_closed:
                        text_sent = True
                        yield text_delta(delta_content)

                delta_tool_calls = get_field(delta, "tool_calls")
                if isinstance(delta, dict) and "tool_calls" in delta and delta_tool_calls is None:
                    delta_tool_calls = delta["tool_calls"]

                if delta_tool_calls:
                    if tool_index is None:
                        # Tool calls must come after the text block is closed.
                        # If we've been streaming text, flush the trailing
                        # delta (if any) and close the block first.
                        if text_sent:
                            if not text_block_closed:
                                text_block_closed = True
                                yield text_block_close()
                        elif accumulated_text and not text_block_closed:
                            text_sent = True
                            yield text_delta(accumulated_text)
                            text_block_closed = True
                            yield text_block_close()
                        elif not text_block_closed:
                            text_block_closed = True
                            yield text_block_close()

                    if not isinstance(delta_tool_calls, list):
                        delta_tool_calls = [delta_tool_calls]

                    for tool_call in delta_tool_calls:
                        current_index = get_field(tool_call, "index", 0)

                        if tool_index is None or current_index != tool_index:
                            tool_index = current_index
                            last_tool_index += 1
                            function = get_field(tool_call, "function", {}) or {}
                            name = get_field(function, "name", "")
                            tool_id = get_field(tool_call, "id") or new_tool_id()
                            yield tool_block_open(last_tool_index, tool_id, name)

                        function = get_field(tool_call, "function", {}) or {}
                        arguments = get_field(function, "arguments", "")
                        if arguments:
                            if isinstance(arguments, str):
                                yield tool_delta(last_tool_index, arguments)
                            else:
                                yield tool_delta(last_tool_index, json.dumps(arguments))

                if finish_reason and not has_sent_stop_reason:
                    has_sent_stop_reason = True
                    if tool_index is not None:
                        for i in range(1, last_tool_index + 1):
                            yield tool_block_close(i)

                    if not text_block_closed:
                        if accumulated_text and not text_sent:
                            yield text_delta(accumulated_text)
                        yield text_block_close()

                    for chunk in finish_stream(to_anthropic_stop_reason(finish_reason), output_tokens):
                        yield chunk
                    return
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue

        if not has_sent_stop_reason:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield tool_block_close(i)
            if not text_block_closed:
                yield text_block_close()
            for chunk in finish_stream("end_turn", output_tokens):
                yield chunk

    except Exception as e:
        logger.error(f"Error in streaming: {e}", exc_info=True)
        yield emit(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "error", "stop_sequence": None}, "usage": {"output_tokens": 0}},
        )
        yield emit("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    try:
        litellm_request = convert_anthropic_to_litellm(request)

        # OpenAI (or any OpenAI-compatible endpoint). After validation every
        # request.model has the openai/ prefix, so this is the only branch.
        litellm_request["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            litellm_request["api_base"] = OPENAI_BASE_URL

        sanitize_messages_for_openai(litellm_request["messages"])

        log_request(
            "POST",
            "/v1/messages",
            request.original_model or "unknown",
            litellm_request.get("model"),
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
