from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, field_validator
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

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


class MessageFilter(logging.Filter):
    """Suppress noisy LiteLLM/HTTP chatter from the root logger."""

    blocked_phrases = [
        "LiteLLM completion()",
        "HTTP Request:",
        "selected model name for cost calculation",
        "utils.py",
        "cost_calculator",
    ]

    def filter(self, record):
        if isinstance(getattr(record, "msg", None), str):
            for phrase in self.blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


logging.getLogger().addFilter(MessageFilter())

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

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


STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


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

    @field_validator("model")
    def validate_model_field(cls, v):
        original_model = v

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
            logger.debug(f"No mapping rule for model '{original_model}', passing through")
            new_model = f"openai/{clean_v}"

        logger.debug(f"MODEL MAPPING: '{original_model}' -> '{new_model}'")

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
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except:
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


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format

    messages = []

    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, "type") and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"

            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    # Context compaction in the client can truncate history and leave tool
    # calls and their results unpaired. Strict OpenAI backends reject both an
    # assistant tool_call with no following tool message and a role="tool"
    # message with no preceding call, so pre-scan the ids of each side to know
    # which pairs actually survive.
    call_ids = set()
    result_ids = set()
    for msg in anthropic_request.messages:
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                call_ids.add(block.id)
            elif block_type == "tool_result":
                rid = getattr(block, "tool_use_id", "") or ""
                if rid:
                    result_ids.add(rid)

    # Add conversation messages, converting to OpenAI/LiteLLM format.
    # LiteLLM's canonical input is OpenAI Chat format for every provider, so
    # tool calls travel as assistant.tool_calls and their results as role="tool"
    # messages. Flattening them into plain text (the previous behaviour) taught
    # small models to emit tool calls as literal text, which broke tool use.
    for msg in anthropic_request.messages:
        content = msg.content

        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
            continue

        if msg.role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_parts.append(block.text)
                elif block_type == "tool_use":
                    if block.id in result_ids:
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )
                    else:
                        # Dangling call whose result was truncated from history.
                        # Describe it in prose rather than a tool-call-like syntax
                        # (which small models tend to mimic) so the backend does
                        # not demand a response for an unanswerable call.
                        text_parts.append(
                            f"(An earlier {block.name} tool call is missing its "
                            f"result in this context.)"
                        )

            assistant_msg = {"role": "assistant"}
            text = "\n".join(text_parts).strip()
            # OpenAI allows null content only when tool_calls are present.
            assistant_msg["content"] = text if text else (None if tool_calls else "")
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            continue

        # user role with structured content: split tool results from the rest.
        tool_messages = []
        user_parts = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                user_parts.append({"type": "text", "text": block.text})
            elif block_type == "image":
                user_parts.append(convert_image_block(block.source))
            elif block_type == "tool_result":
                tool_use_id = getattr(block, "tool_use_id", "") or ""
                result_text = parse_tool_result_content(
                    getattr(block, "content", None)
                )
                if tool_use_id in call_ids:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": result_text,
                        }
                    )
                else:
                    # Orphaned result whose call was truncated from history. Fold
                    # into user text so we never emit an unpaired tool message;
                    # the ghost id would be meaningless to the model, so drop it.
                    user_parts.append(
                        {
                            "type": "text",
                            "text": f"(Result from an earlier tool call:)\n{result_text}",
                        }
                    )

        # Tool results must immediately follow the assistant turn that produced
        # the matching tool_calls, so emit them before any trailing user text.
        messages.extend(tool_messages)

        if user_parts:
            if all(part.get("type") == "text" for part in user_parts):
                messages.append(
                    {
                        "role": "user",
                        "content": "\n".join(
                            part["text"] for part in user_parts
                        ).strip()
                        or "...",
                    }
                )
            else:
                messages.append({"role": "user", "content": user_parts})

    # OpenAI Chat Completions caps max_completion_tokens at 16384 for most
    # current models; over that and the API rejects the request outright.
    max_tokens = min(anthropic_request.max_tokens, 16384)

    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
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
        openai_tools = []
        for tool in anthropic_request.tools:
            tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_dict["name"],
                        "description": tool_dict.get("description", ""),
                        "parameters": tool_dict.get("input_schema", {}),
                    },
                }
            )
        litellm_request["tools"] = openai_tools

    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, "dict"):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice

        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]},
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"

    return litellm_request


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    try:
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            finish_reason = (
                choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            )
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            # Fall back to dict-style extraction for non-standard responses.
            try:
                if isinstance(litellm_response, dict):
                    response_dict = litellm_response
                elif hasattr(litellm_response, "model_dump"):
                    response_dict = litellm_response.model_dump()
                else:
                    response_dict = litellm_response.__dict__
            except AttributeError:
                response_dict = {
                    "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                    "choices": getattr(litellm_response, "choices", [{}]),
                    "usage": getattr(litellm_response, "usage", {}),
                }

            choices = response_dict.get("choices", [{}])
            message = (
                choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            )
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = (
                choices[0].get("finish_reason", "stop")
                if choices and len(choices) > 0
                else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Create content list for Anthropic format
        content = []

        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Add tool calls if present (tool_use in Anthropic format)
        # For ALL models, not just Claude models - convert tool_calls to tool_use blocks
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments as JSON: {arguments}"
                        )
                        arguments = {"raw": arguments}

                logger.debug(
                    f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}"
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": arguments,
                    }
                )

        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        return MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

    except Exception as e:
        logger.error(f"Error converting response: {e}", exc_info=True)
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
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

    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
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
                if getattr(chunk, "usage", None) is not None:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", input_tokens) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", output_tokens) or 0

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None) or getattr(choice, "message", {}) or {}
                finish_reason = getattr(choice, "finish_reason", None)

                delta_content = getattr(delta, "content", None)
                if isinstance(delta, dict) and "content" in delta and delta_content is None:
                    delta_content = delta["content"]
                if delta_content:
                    accumulated_text += delta_content
                    if tool_index is None and not text_block_closed:
                        text_sent = True
                        yield text_delta(delta_content)

                delta_tool_calls = getattr(delta, "tool_calls", None)
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
                        current_index = (
                            tool_call["index"] if isinstance(tool_call, dict) and "index" in tool_call
                            else getattr(tool_call, "index", 0)
                        )

                        if tool_index is None or current_index != tool_index:
                            tool_index = current_index
                            last_tool_index += 1
                            anthropic_tool_index = last_tool_index

                            if isinstance(tool_call, dict):
                                function = tool_call.get("function", {}) or {}
                                name = function.get("name", "") if isinstance(function, dict) else ""
                                tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                            else:
                                function = getattr(tool_call, "function", None)
                                name = getattr(function, "name", "") if function else ""
                                tool_id = getattr(tool_call, "id", None) or f"toolu_{uuid.uuid4().hex[:24]}"

                            yield tool_block_open(anthropic_tool_index, tool_id, name)

                        if isinstance(tool_call, dict):
                            function = tool_call.get("function", {}) or {}
                            arguments = function.get("arguments", "") if isinstance(function, dict) else ""
                        else:
                            function = getattr(tool_call, "function", None)
                            arguments = getattr(function, "arguments", "") if function else ""

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

                    stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")
                    for chunk in finish_stream(stop_reason, output_tokens):
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
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        body_json = json.loads((await raw_request.body()).decode("utf-8"))
        original_model = body_json.get("model", "unknown")
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        litellm_request = convert_anthropic_to_litellm(request)

        # OpenAI (or any OpenAI-compatible endpoint). After validation every
        # request.model has the openai/ prefix, so this is the only branch.
        litellm_request["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            litellm_request["api_base"] = OPENAI_BASE_URL

        # OpenAI Chat Completions rejects unknown message fields. Strip to the
        # canonical set, and guarantee non-empty content where the API
        # requires it (assistant.tool_calls may omit content; nothing else).
        allowed_keys = {"role", "content", "name", "tool_call_id", "tool_calls"}
        for msg in litellm_request["messages"]:
            for key in list(msg.keys()):
                if key not in allowed_keys:
                    logger.debug(f"Removing unsupported message field: {key}")
                    del msg[key]
            if msg.get("content") in (None, "") and not msg.get("tool_calls"):
                msg["content"] = "..."

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            litellm_request.get("model"),
            len(litellm_request["messages"]),
            num_tools,
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


# ANSI color codes for terminal output.
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def log_request_beautifully(method, path, source_model, target_model, num_messages, num_tools, status_code):
    """Print a one-line request summary with the source/target model mapping."""
    endpoint = path.split("?", 1)[0] if "?" in path else path
    if "/" in source_model:
        source_model = source_model.split("/")[-1]
    if "/" in target_model:
        target_model = target_model.split("/")[-1]

    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = (
        f"{Colors.CYAN}{source_model}{Colors.RESET} → "
        f"{Colors.GREEN}{target_model}{Colors.RESET} "
        f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET} "
        f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    )

    print(log_line)
    print(model_line)
    sys.stdout.flush()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
