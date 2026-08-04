#!/usr/bin/env python3
"""
Tests for the Anthropic -> OpenAI proxy.

Includes integration smoke tests (require a running server on PROXY_URL)
and unit tests for request/response translation (no network needed).

Usage:
  python tests.py                  # run unit tests only (default)
  python tests.py --integration    # run integration smoke tests
  python tests.py --all            # run both
  python tests.py --simple         # (integration) skip tool scenarios
  python tests.py --tools          # (integration) only tool scenarios
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

PROXY_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

HEADERS = {
    "x-api-key": "dummy",
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Evaluate a math expression and return the result.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}

TEST_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "simple": {
        "model": DEFAULT_MODEL,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Reply with the word OK and nothing else."}],
    },
    "multi_turn": {
        "model": DEFAULT_MODEL,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "My favourite colour is blue. Acknowledge in one sentence."},
            {"role": "assistant", "content": "Got it — your favourite colour is blue."},
            {"role": "user", "content": "What is my favourite colour? Answer in one short sentence."},
        ],
    },
    "tools": {
        "model": DEFAULT_MODEL,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "What is 12 * 7?"}],
        "tools": [CALCULATOR_TOOL],
        "tool_choice": {"type": "auto"},
    },
    "streaming": {
        "model": DEFAULT_MODEL,
        "max_tokens": 100,
        "stream": True,
        "messages": [{"role": "user", "content": "Count 1, 2, 3, each on its own line."}],
    },
    "streaming_tools": {
        "model": DEFAULT_MODEL,
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "user", "content": "Compute (45 + 15) / 4 with the calculator tool."}],
        "tools": [CALCULATOR_TOOL],
        "tool_choice": {"type": "auto"},
    },
}

REQUIRED_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}


# ---------------------------------------------------------------------------
# Imports and helpers
# ---------------------------------------------------------------------------

import server as srv


def _make_request(payload: Dict[str, Any]) -> srv.MessagesRequest:
    return srv.MessagesRequest(**payload)


def _base_request(**overrides) -> srv.MessagesRequest:
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    payload.update(overrides)
    return _make_request(payload)


async def _aiter(items: List[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _parse_sse(raw_chunks: List[str]) -> List[Dict[str, Any]]:
    """Parse SSE blocks from a list of yielded strings into a list of events.

    Skips ``[DONE]`` sentinels; returns them as a dict with type="[DONE]"
    so tests can assert on their presence.
    """
    events: List[Dict[str, Any]] = []
    for raw in raw_chunks:
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            event_type: Optional[str] = None
            data_lines: List[str] = []
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_type = line[len("event: "):]
                elif line.startswith("data: "):
                    data_lines.append(line[len("data: "):])
            if not data_lines:
                continue
            payload = "".join(data_lines)
            if payload == "[DONE]":
                events.append({"type": "[DONE]"})
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONError:
                continue
            if event_type and "type" not in parsed:
                parsed = {"type": event_type, **parsed}
            events.append(parsed)
    return events


async def _run_stream(chunks: List[Any], req: srv.MessagesRequest) -> List[Dict[str, Any]]:
    """Drive handle_streaming over a fake upstream and return parsed events."""
    raw: List[str] = []
    async for piece in srv.handle_streaming(_aiter(chunks), req):
        raw.append(piece)
    return _parse_sse(raw)


def _text_chunk(text: str, **extra) -> Dict[str, Any]:
    chunk = {
        "choices": [{
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }
    chunk["choices"][0].update(extra.get("choice_extra", {}))
    chunk.update({k: v for k, v in extra.items() if k != "choice_extra"})
    return chunk


def _tool_delta_chunk(
    index: int,
    *,
    id: Optional[str] = None,
    name: Optional[str] = None,
    arguments: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    function: Dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    tool_call: Dict[str, Any] = {"index": index, "type": "function"}
    if id is not None:
        tool_call["id"] = id
    if function:
        tool_call["function"] = function
    return {
        "choices": [{
            "delta": {"tool_calls": [tool_call]},
            "finish_reason": finish_reason,
        }],
    }


def _finish_chunk(reason: str, *, output_tokens: int = 5) -> Dict[str, Any]:
    return {
        "choices": [{"delta": {}, "finish_reason": reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": output_tokens},
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


# --- Model mapping ---

def test_capture_original_model_copies_model_field() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.original_model == "claude-3-5-sonnet-20241022"
    assert req.model == f"openai/{srv.BIG_MODEL}"


def test_capture_original_model_preserves_explicit_override() -> None:
    req = _make_request({
        "model": "openai/gpt-4.1",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.original_model == "openai/gpt-4.1"
    assert req.model == "openai/gpt-4.1"


def test_validate_model_field_haiku_mapping() -> None:
    req = _make_request({
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.SMALL_MODEL}"


def test_validate_model_field_sonnet_mapping() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.BIG_MODEL}"


def test_validate_model_field_opus_maps_to_big_model() -> None:
    req = _make_request({
        "model": "claude-opus-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.BIG_MODEL}"


def test_validate_model_field_opus_with_dated_id() -> None:
    req = _make_request({
        "model": "claude-opus-5-20251215",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.BIG_MODEL}"


def test_validate_model_field_fable_maps_to_big_model() -> None:
    req = _make_request({
        "model": "claude-fable-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.BIG_MODEL}"


def test_validate_model_field_mythos_maps_to_big_model() -> None:
    req = _make_request({
        "model": "claude-mythos-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.BIG_MODEL}"


def test_validate_model_field_sonnet_override_takes_precedence() -> None:
    original = srv.TIER_OVERRIDE["sonnet"]
    srv.TIER_OVERRIDE["sonnet"] = "custom-sonnet-model"
    try:
        req = _make_request({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert req.model == "openai/custom-sonnet-model"
    finally:
        srv.TIER_OVERRIDE["sonnet"] = original


def test_validate_model_field_opus_override_is_independent() -> None:
    original_opus = srv.TIER_OVERRIDE["opus"]
    srv.TIER_OVERRIDE["opus"] = "custom-opus-model"
    try:
        opus_req = _make_request({
            "model": "claude-opus-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert opus_req.model == "openai/custom-opus-model"
        sonnet_req = _make_request({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert sonnet_req.model == f"openai/{srv.BIG_MODEL}"
    finally:
        srv.TIER_OVERRIDE["opus"] = original_opus


def test_validate_model_field_haiku_override() -> None:
    original = srv.TIER_OVERRIDE["haiku"]
    srv.TIER_OVERRIDE["haiku"] = "custom-haiku-model"
    try:
        req = _make_request({
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert req.model == "openai/custom-haiku-model"
    finally:
        srv.TIER_OVERRIDE["haiku"] = original


def test_validate_model_field_known_openai_model_gets_prefix() -> None:
    req = _make_request({
        "model": "gpt-4.1",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == "openai/gpt-4.1"


def test_validate_model_field_existing_openai_prefix_passthrough() -> None:
    req = _make_request({
        "model": "openai/custom-model",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == "openai/custom-model"


def test_validate_model_field_unknown_name_gets_prefix() -> None:
    req = _make_request({
        "model": "my-local-llama",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == "openai/my-local-llama"


def test_validate_model_field_strips_anthropic_prefix() -> None:
    req = _make_request({
        "model": "anthropic/claude-3-5-haiku-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.SMALL_MODEL}"
    assert req.original_model == "anthropic/claude-3-5-haiku-20241022"


def test_validate_model_field_strips_gemini_prefix() -> None:
    req = _make_request({
        "model": "gemini/claude-3-5-haiku-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == f"openai/{srv.SMALL_MODEL}"
    assert req.original_model == "gemini/claude-3-5-haiku-20241022"


def test_tls_verify_wiring_matches_module_setting() -> None:
    """OPENAI_TLS_VERIFY is read at import time and propagated to litellm."""
    assert srv.OPENAI_TLS_VERIFY == srv.litellm.ssl_verify


# --- Message sanitization ---

def test_sanitize_messages_for_openai_removes_foreign_keys() -> None:
    messages = [
        {"role": "user", "content": "hi", "stop_reason": "end_turn", "type": "message"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "plain"},
    ]
    srv.sanitize_messages_for_openai(messages)
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1] == {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}
    assert messages[2] == {"role": "assistant", "content": "..."}
    assert messages[3] == {"role": "user", "content": "plain"}


def test_sanitize_messages_for_openai_keeps_allowed_keys() -> None:
    messages = [
        {
            "role": "tool",
            "name": "calculator",
            "tool_call_id": "abc",
            "content": "42",
            "foreign_field": "x",
        }
    ]
    srv.sanitize_messages_for_openai(messages)
    assert "foreign_field" not in messages[0]
    assert messages[0]["name"] == "calculator"
    assert messages[0]["tool_call_id"] == "abc"
    assert messages[0]["content"] == "42"


# --- Request conversion ---

def test_convert_anthropic_to_litellm_minimal_request() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["model"] == f"openai/{srv.BIG_MODEL}"
    assert out["max_completion_tokens"] == 200
    assert out["temperature"] == 1.0
    assert out["stream"] is False
    assert out["messages"] == [{"role": "user", "content": "Hello"}]
    assert "system" not in out["messages"][0]
    assert "tools" not in out
    assert "tool_choice" not in out


def test_convert_anthropic_to_litellm_clamps_max_tokens() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": srv.MAX_OUTPUT_TOKENS + 1000,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["max_completion_tokens"] == srv.MAX_OUTPUT_TOKENS


def test_convert_anthropic_to_litellm_with_string_system() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["messages"][0] == {"role": "system", "content": "You are helpful."}


def test_convert_anthropic_to_litellm_with_list_system_joins_text_blocks() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "Be concise."},
            {"type": "text", "text": "Answer in English."},
        ],
        "messages": [{"role": "user", "content": "Hi"}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["messages"][0]["role"] == "system"
    assert "Be concise." in out["messages"][0]["content"]
    assert "Answer in English." in out["messages"][0]["content"]


def test_convert_anthropic_to_litellm_with_tools_and_choice() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [{
            "name": "calc",
            "description": "calculator",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }],
        "tool_choice": {"type": "tool", "name": "calc"},
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["tools"] == [{
        "type": "function",
        "function": {
            "name": "calc",
            "description": "calculator",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        },
    }]
    assert out["tool_choice"] == {"type": "function", "function": {"name": "calc"}}


def test_convert_anthropic_to_litellm_passes_optional_sampling() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
        "stop_sequences": ["END"],
        "top_p": 0.9,
        "top_k": 40,
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["stop"] == ["END"]
    assert out["top_p"] == 0.9
    assert out["top_k"] == 40


def test_convert_anthropic_to_litellm_pairs_tool_call_with_tool_result() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "calc", "input": {"q": "2+2"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "4"},
            ]},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["messages"][1]["role"] == "assistant"
    assert out["messages"][1]["tool_calls"] == [{
        "id": "t1", "type": "function",
        "function": {"name": "calc", "arguments": '{"q": "2+2"}'},
    }]
    assert out["messages"][2] == {"role": "tool", "tool_call_id": "t1", "content": "4"}


def test_user_content_list_with_single_text_block() -> None:
    """A user message as a list of content blocks must be flattened to string content."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Hello there"},
        ]}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["messages"] == [{"role": "user", "content": "Hello there"}]


def test_user_content_list_with_text_and_image() -> None:
    """Text + image must produce a structured OpenAI content array (not flattened)."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo=",
            }},
        ]}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    msg = out["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0] == {"type": "text", "text": "What is this?"}
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_convert_image_block_url_source() -> None:
    src = {"type": "url", "url": "https://example.com/cat.jpg"}
    out = srv.convert_image_block(src)
    assert out == {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}


def test_convert_image_block_unknown_source_falls_back_gracefully() -> None:
    """An unrecognized source shape must not crash; produces a best-effort string."""
    out = srv.convert_image_block({"weird": "thing"})
    assert out["type"] == "image_url"
    assert "url" in out["image_url"]


# --- Tool edge cases ---

def test_dangling_tool_use_folded_into_text() -> None:
    """A tool_use with no matching tool_result (truncated history) must be turned into prose,
    not emitted as a tool_call — otherwise the model would have to answer for an unanswerable call."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "SF"}},
            ]},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assistant = out["messages"][1]
    assert assistant["role"] == "assistant"
    assert "tool_calls" not in assistant, "orphan tool_use must not be emitted as tool_call"
    assert "get_weather" in assistant["content"]
    assert "missing its result" in assistant["content"].lower()


def test_orphaned_tool_result_folded_into_user_text() -> None:
    """A tool_result with no matching tool_use must be folded into user text rather
    than emitted as a role='tool' message (which would dangle)."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ghost", "content": "old data"},
            ]},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    user_msgs = [m for m in out["messages"] if m["role"] == "user"]
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    assert tool_msgs == [], "orphan tool_result must not be emitted as role=tool"
    assert len(user_msgs) == 1
    assert "old data" in user_msgs[0]["content"]


def test_tool_use_and_tool_result_ordering() -> None:
    """When a user turn contains tool_results followed by user text, the tool message(s)
    must come BEFORE the user text — OpenAI requires tool messages to immediately follow
    the assistant tool_call turn."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "calc", "input": {"x": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                {"type": "text", "text": "Thanks. Now also do Y."},
            ]},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "user"], f"got {roles}"


def test_tool_choice_any_passes_through() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "t", "description": "t", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["tool_choice"] == "any"


def test_tool_choice_unknown_type_falls_back_to_auto() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "t", "description": "t", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "bogus_type"},
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["tool_choice"] == "auto"


def test_tool_choice_tool_with_missing_name_falls_back_to_auto() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "t", "description": "t", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool"},
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["tool_choice"] == "auto"


# --- Response conversion ---

def test_convert_litellm_to_anthropic_text_response() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    response = {
        "id": "resp-1",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    assert out.id == "resp-1"
    assert out.model == f"openai/{srv.BIG_MODEL}"
    assert out.role == "assistant"
    assert out.stop_reason == "end_turn"
    assert [b.model_dump() for b in out.content] == [{"type": "text", "text": "Hello there"}]
    assert out.usage.input_tokens == 11
    assert out.usage.output_tokens == 5


def test_convert_litellm_to_anthropic_tool_use_response() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"q": "2+2"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    assert out.stop_reason == "tool_use"
    assert len(out.content) == 1
    block = out.content[0]
    assert block.model_dump() == {
        "type": "tool_use", "id": "call_1", "name": "calc", "input": {"q": "2+2"},
    }


def test_convert_litellm_to_anthropic_generates_id_when_missing() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    response = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    assert out.id.startswith("msg_")
    assert out.usage.input_tokens == 0
    assert out.usage.output_tokens == 0


def test_convert_litellm_to_anthropic_maps_length_stop_reason() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Tell me a long story"}],
    })
    response = {
        "choices": [{"message": {"role": "assistant", "content": "Once upon..."}, "finish_reason": "length"}],
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    assert out.stop_reason == "max_tokens"


def test_convert_litellm_to_anthropic_handles_empty_choices() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    response = {"choices": []}
    out = srv.convert_litellm_to_anthropic(response, req)
    assert [b.model_dump() for b in out.content] == [{"type": "text", "text": ""}]
    assert out.stop_reason == "end_turn"


def test_convert_litellm_to_anthropic_uses_keyword_usage_args() -> None:
    """Regression: Usage(...) must be built with keyword args, not positional."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    response = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    assert out.usage.input_tokens == 4
    assert out.usage.output_tokens == 2


def test_convert_litellm_to_anthropic_recovers_from_broken_usage() -> None:
    """Regression: even if usage is malformed, we still return a usable response."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    response = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": "not-a-dict",
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    assert out.usage.input_tokens == 0
    assert out.usage.output_tokens == 0
    assert out.content[0].model_dump() == {"type": "text", "text": "ok"}


def test_convert_litellm_to_anthropic_handles_string_arguments_gracefully() -> None:
    """Tool arguments are typically strings (JSON-encoded); a non-string must not crash."""
    req = _base_request()
    response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "calc", "arguments": {"already": "a dict"}},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    block = out.content[0]
    dumped = block.model_dump()
    assert dumped["type"] == "tool_use"
    assert dumped["input"] == {"already": "a dict"}


def test_convert_litellm_to_anthropic_recovers_from_invalid_json_arguments() -> None:
    """If tool arguments are not valid JSON, we must not crash — fall back to a raw wrapper."""
    req = _base_request()
    response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "calc", "arguments": "{this is not json"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    block = out.content[0]
    dumped = block.model_dump()
    assert dumped["input"] == {"raw": "{this is not json"}


# --- System messages ---

def test_system_message_list_with_only_text_blocks() -> None:
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "You are concise."},
            {"type": "text", "text": "Answer in English."},
        ],
        "messages": [{"role": "user", "content": "Hi"}],
    })
    out = srv.convert_anthropic_to_litellm(req)
    sys_msg = out["messages"][0]
    assert sys_msg["role"] == "system"
    assert "concise" in sys_msg["content"]
    assert "English" in sys_msg["content"]


def test_system_role_message_in_messages_array_is_hoisted() -> None:
    """Claude Code 2.1.154+ embeds system reminders as messages with role='system'.
    The proxy must accept them (Pydantic Literal must include 'system') and
    hoist the content into a single system message at the start of the OpenAI
    request — never inline as a 'system' message in the middle.
    """
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "system": "You are concise.",
        "messages": [
            {"role": "system", "content": "[skill: foo] description"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "system", "content": [{"type": "text", "text": "[skill: baz] more"}]},
            {"role": "user", "content": "and now?"},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant", "user"], f"got {roles}"
    sys_content = out["messages"][0]["content"]
    assert sys_content.index("[skill: foo]") < sys_content.index("[skill: baz]") < sys_content.index("You are concise.")


def test_system_role_message_with_string_content_is_hoisted() -> None:
    """A system message with string content is treated identically to a list."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "top-of-stream reminder"},
            {"role": "user", "content": "Hi"},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["messages"][0] == {"role": "system", "content": "top-of-stream reminder"}
    assert out["messages"][1]["role"] == "user"


# --- Content block assembly ---

def test_build_content_blocks_text_only() -> None:
    """No reasoning -> just a text block."""
    blocks = srv._build_content_blocks("hi", "", [])
    assert blocks == [{"type": "text", "text": "hi"}]


def test_build_content_blocks_reasoning_becomes_thinking_block() -> None:
    """reasoning_content must surface as a `type: thinking` block, ahead of text."""
    blocks = srv._build_content_blocks("answer", "my reasoning", [])
    assert blocks == [
        {"type": "thinking", "thinking": "my reasoning"},
        {"type": "text", "text": "answer"},
    ]


def test_build_content_blocks_reasoning_plus_tool_call() -> None:
    """reasoning -> text -> tool_use in that order."""
    blocks = srv._build_content_blocks(
        "answer",
        "thought",
        [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}],
    )
    assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]
    assert blocks[2]["name"] == "f"


def test_convert_litellm_to_anthropic_uses_reasoning_content() -> None:
    """When litellm has separated reasoning into reasoning_content, we emit a
    thinking block before the text block — matching the Anthropic spec."""
    req = _base_request()
    response = {
        "choices": [{
            "message": {
                "content": "final answer",
                "reasoning_content": "step by step",
            },
            "finish_reason": "stop",
        }],
    }
    out = srv.convert_litellm_to_anthropic(response, req)
    dumped = [b.model_dump(exclude_none=True) for b in out.content]
    assert dumped == [
        {"type": "thinking", "thinking": "step by step"},
        {"type": "text", "text": "final answer"},
    ]


def test_request_accepts_thinking_block_in_history() -> None:
    """Claude Code echoes the assistant's previous thinking block back as part
    of the next turn. The proxy must accept it (signature optional) so the
    request validates, even though the block is dropped before the upstream
    OpenAI call (OpenAI has no equivalent concept)."""
    req = _make_request({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "I should say hi back.", "signature": ""},
            ]},
            {"role": "user", "content": "and now?"},
        ],
    })
    out = srv.convert_anthropic_to_litellm(req)
    assistant = out["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant.get("content") == "hello"
    assert "tool_calls" not in assistant


# --- Think stream parser ---

def test_think_stream_parser_text_only() -> None:
    """Plain text with no tags passes through verbatim."""
    p = srv.ThinkStreamParser()
    assert p.feed("hello ") == [("text", "hello ")]
    assert p.feed("world") == [("text", "world")]
    assert p.flush() == []


def test_think_stream_parser_single_think_block() -> None:
    """A single ``<think>...</think>`` splits cleanly.

    Thinking content is held back until the close tag is observed — this is
    what prevents the parser from prematurely committing answer text to a
    thinking block when chunks split a word.
    """
    p = srv.ThinkStreamParser()
    events = p.feed("<think>reasoning here</think>answer") + p.flush()
    assert events == [
        ("open", None),
        ("thinking", "reasoning here"),
        ("close", None),
        ("text", "answer"),
    ]


def test_think_stream_parser_text_around_block() -> None:
    p = srv.ThinkStreamParser()
    events = p.feed("before<think>x</think>after") + p.flush()
    assert events == [
        ("text", "before"),
        ("open", None),
        ("thinking", "x"),
        ("close", None),
        ("text", "after"),
    ]


def test_think_stream_parser_think_split_across_chunks() -> None:
    """Tags that straddle chunk boundaries must not be emitted until complete.

    Crucially, thinking content is held until `` appears — even if the
    word before it could be emitted safely. Otherwise, splitting a word
    like "Hello" across chunks would misclassify "Hel" as thinking.
    """
    p = srv.ThinkStreamParser()
    assert p.feed("hello <") == [("text", "hello ")]
    assert p.feed("think>Hel") == [("open", None)]
    assert p.feed("lo!</thin") == []
    assert p.feed("k>World") + p.flush() == [
        ("thinking", "Hello!"),
        ("close", None),
        ("text", "World"),
    ]


def test_think_stream_parser_multiple_think_blocks() -> None:
    """Two separate think blocks round-trip cleanly."""
    p = srv.ThinkStreamParser()
    events = p.feed("a<think>1</think>b<think>2</think>c") + p.flush()
    assert events == [
        ("text", "a"),
        ("open", None),
        ("thinking", "1"),
        ("close", None),
        ("text", "b"),
        ("open", None),
        ("thinking", "2"),
        ("close", None),
        ("text", "c"),
    ]


def test_think_stream_parser_holds_until_close() -> None:
    """Thinking content must NOT be emitted chunk-by-chunk before ``."""
    p = srv.ThinkStreamParser()
    assert p.feed("<think>a") == [("open", None)]
    assert p.feed("b") == []
    assert p.feed("c") == []
    assert p.feed("</think>x") + p.flush() == [
        ("thinking", "abc"),
        ("close", None),
        ("text", "x"),
    ]


def test_think_stream_parser_empty_think_block() -> None:
    p = srv.ThinkStreamParser()
    events = p.feed("a<think></think>b") + p.flush()
    assert events == [
        ("text", "a"),
        ("open", None),
        ("close", None),
        ("text", "b"),
    ]


def test_think_stream_parser_unclosed_at_flush() -> None:
    """If the stream ends inside ``<think>`` with content still buffered, flush
    must release everything as a thinking delta and emit close."""
    p = srv.ThinkStreamParser()
    p.feed("<think>hello ")
    assert p.feed("<") == []
    assert p.flush() == [("thinking", "hello <"), ("close", None)]


# --- Streaming ---

async def test_streaming_text_only_emits_required_events() -> None:
    """A pure-text stream must emit message_start, ping, deltas, message_delta, message_stop, [DONE]."""
    req = _base_request(stream=True)
    chunks = [
        _text_chunk("Hello"),
        _text_chunk(" world"),
        _finish_chunk("stop"),
    ]
    events = await _run_stream(chunks, req)
    types = [e["type"] for e in events]

    for required in ("message_start", "content_block_start", "content_block_delta",
                     "content_block_stop", "message_delta", "message_stop"):
        assert required in types, f"missing {required}; got {types}"

    assert events[-1] == {"type": "[DONE]"}, "stream must end with [DONE]"


async def test_streaming_text_only_accumulates_text() -> None:
    req = _base_request(stream=True)
    chunks = [_text_chunk("foo"), _text_chunk("bar"), _finish_chunk("stop")]
    events = await _run_stream(chunks, req)
    text = "".join(
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    )
    assert text == "foobar"


async def test_streaming_text_then_tool_call_closes_text_block_first() -> None:
    """When the model emits text and then a tool call, the text block must be closed
    before the tool_use block starts — Anthropic's SSE protocol requires this ordering."""
    req = _base_request(stream=True, tools=[{
        "name": "calc", "description": "calc", "input_schema": {"type": "object"},
    }])
    chunks = [
        _text_chunk("Let me calculate"),
        _tool_delta_chunk(0, id="call_1", name="calc", arguments='{"x":1}'),
        _finish_chunk("tool_calls"),
    ]
    events = await _run_stream(chunks, req)

    def find(predicate):
        return next(i for i, e in enumerate(events) if predicate(e))

    text_close_idx = find(lambda e: e["type"] == "content_block_stop" and e.get("index") == 0)
    tool_start_idx = find(
        lambda e: e["type"] == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "tool_use"
    )
    assert text_close_idx < tool_start_idx, (
        f"text block (idx {text_close_idx}) must close before tool block opens (idx {tool_start_idx})"
    )


async def test_streaming_tool_call_then_text_block_stays_closed() -> None:
    """After tool calls, no new text block should appear."""
    req = _base_request(stream=True, tools=[{
        "name": "calc", "description": "calc", "input_schema": {"type": "object"},
    }])
    chunks = [
        _tool_delta_chunk(0, id="call_1", name="calc", arguments='{"x":1}'),
        _text_chunk("after tool"),
        _finish_chunk("tool_calls"),
    ]
    events = await _run_stream(chunks, req)
    text_starts = [
        e for e in events
        if e["type"] == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "text"
    ]
    assert len(text_starts) == 1, f"expected exactly one text block, got {len(text_starts)}"


async def test_streaming_tool_only_no_text() -> None:
    """Tool-only response: no text deltas, just tool_use block and finish."""
    req = _base_request(stream=True, tools=[{
        "name": "calc", "description": "calc", "input_schema": {"type": "object"},
    }])
    chunks = [
        _tool_delta_chunk(0, id="call_1", name="calc", arguments='{"x":1}'),
        _finish_chunk("tool_calls"),
    ]
    events = await _run_stream(chunks, req)
    text_deltas = [
        e for e in events
        if e["type"] == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    assert text_deltas == []
    stop = next(e for e in events if e["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "tool_use"


async def test_streaming_no_finish_reason_falls_back_to_end_turn() -> None:
    """If the upstream never sends a finish_reason, we must still close cleanly."""
    req = _base_request(stream=True)
    chunks = [_text_chunk("partial...")]
    events = await _run_stream(chunks, req)
    types = [e["type"] for e in events]
    assert "message_stop" in types, "stream must terminate with message_stop"
    assert events[-1] == {"type": "[DONE]"}
    stop = next(e for e in events if e["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"


async def test_streaming_multiple_tool_calls_use_distinct_indices() -> None:
    """Parallel tool calls must each get their own SSE block index."""
    req = _base_request(stream=True, tools=[{
        "name": "calc", "description": "calc", "input_schema": {"type": "object"},
    }])
    chunks = [
        _tool_delta_chunk(0, id="call_1", name="calc", arguments='{"a":1}'),
        _tool_delta_chunk(1, id="call_2", name="calc", arguments='{"b":2}'),
        _finish_chunk("tool_calls"),
    ]
    events = await _run_stream(chunks, req)
    tool_starts = [
        e for e in events
        if e["type"] == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 2
    indices = {t["index"] for t in tool_starts}
    assert len(indices) == 2, f"tool blocks must have distinct indices, got {indices}"


async def test_streaming_tool_arguments_streamed_as_partial_json() -> None:
    """Tool argument fragments must be wrapped in input_json_delta deltas."""
    req = _base_request(stream=True, tools=[{
        "name": "calc", "description": "calc", "input_schema": {"type": "object"},
    }])
    chunks = [
        _tool_delta_chunk(0, id="call_1", name="calc", arguments='{"x":'),
        _tool_delta_chunk(0, arguments='1}'),
        _finish_chunk("tool_calls"),
    ]
    events = await _run_stream(chunks, req)
    arg_deltas = [
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert "".join(arg_deltas) == '{"x":1}'


async def test_streaming_maps_finish_reason_length_to_max_tokens() -> None:
    req = _base_request(stream=True)
    chunks = [_text_chunk("too long..."), _finish_chunk("length", output_tokens=99)]
    events = await _run_stream(chunks, req)
    stop = next(e for e in events if e["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "max_tokens"
    assert stop["usage"]["output_tokens"] == 99


async def test_streaming_message_id_format() -> None:
    """message_id must be `msg_` + 24 hex chars, matching Anthropic's format."""
    req = _base_request(stream=True)
    chunks = [_text_chunk("x"), _finish_chunk("stop")]
    events = await _run_stream(chunks, req)
    start = next(e for e in events if e["type"] == "message_start")
    msg_id = start["message"]["id"]
    assert msg_id.startswith("msg_")
    assert len(msg_id) == len("msg_") + 24
    assert all(c in "0123456789abcdef" for c in msg_id[len("msg_"):])


async def test_streaming_emits_thinking_block_for_think_tags() -> None:
    """When the upstream stream contains ``<think>...</think>`` markers inside
    its content deltas, the proxy must surface them as a separate thinking
    content block (not as plain text)."""
    req = _base_request(stream=True)
    chunks = [
        _text_chunk("<think>step 1; step 2;</think>final answer"),
        _finish_chunk("stop"),
    ]
    events = await _run_stream(chunks, req)
    thinking_starts = [
        e for e in events
        if e["type"] == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "thinking"
    ]
    assert len(thinking_starts) == 1, "exactly one thinking block expected"
    thinking_deltas = [
        e["delta"]["thinking"]
        for e in events
        if e["type"] == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
    ]
    assert "".join(thinking_deltas) == "step 1; step 2;"
    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    assert "".join(text_deltas) == "final answer"


async def test_streaming_emits_thinking_for_native_reasoning_content() -> None:
    """When the upstream delta has a structured reasoning_content field (some
    providers expose it), route it to a thinking block too."""
    req = _base_request(stream=True)
    chunk = {
        "choices": [{
            "delta": {"content": "answer", "reasoning_content": "thinking"},
            "finish_reason": "stop",
        }],
    }
    events = await _run_stream([chunk], req)
    thinking_starts = [
        e for e in events
        if e["type"] == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "thinking"
    ]
    assert len(thinking_starts) == 1


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------


def _check_non_streaming(payload: Dict[str, Any], *, expect_tools: bool) -> None:
    assert payload.get("role") == "assistant", f"role={payload.get('role')!r}"
    assert payload.get("type") == "message", f"type={payload.get('type')!r}"
    assert payload.get("stop_reason") in {"end_turn", "max_tokens", "tool_use", "stop_sequence", None}
    content = payload.get("content") or []
    assert isinstance(content, list) and content, "content must be a non-empty list"

    has_tool_use = any(block.get("type") == "tool_use" for block in content)
    has_text = any(block.get("type") == "text" for block in content)

    if expect_tools:
        assert has_tool_use, "expected a tool_use block when tools are provided"
    else:
        assert has_text, "expected a text block"


async def run_non_streaming(name: str, payload: Dict[str, Any]) -> bool:
    print(f"\n--- {name} (non-streaming) ---")
    start = time.time()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(PROXY_URL, headers=HEADERS, json=payload)
    elapsed = time.time() - start
    print(f"HTTP {response.status_code} in {elapsed:.2f}s")

    if response.status_code != 200:
        print(f"FAIL: {response.text[:300]}")
        return False

    body = response.json()
    _check_non_streaming(body, expect_tools="tools" in payload)
    print(f"OK  stop_reason={body.get('stop_reason')!r}, content_blocks={len(body.get('content', []))}")
    return True


async def run_streaming(name: str, payload: Dict[str, Any]) -> bool:
    print(f"\n--- {name} (streaming) ---")
    payload = {**payload, "stream": True}
    event_types: set = set()
    text_content = ""
    saw_tool_use = False
    saw_done = False

    start = time.time()
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", PROXY_URL, headers=HEADERS, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(f"FAIL: HTTP {response.status_code} {body.decode('utf-8', 'replace')[:300]}")
                return False

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    if not event.strip():
                        continue
                    data_lines = [
                        line[len("data: "):]
                        for line in event.splitlines()
                        if line.startswith("data: ")
                    ]
                    if not data_lines:
                        continue
                    data_str = "".join(data_lines)
                    if data_str == "[DONE]":
                        saw_done = True
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    event_type = data.get("type")
                    if event_type:
                        event_types.add(event_type)
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_content += delta.get("text", "")
                    if event_type == "content_block_start":
                        if (data.get("content_block") or {}).get("type") == "tool_use":
                            saw_tool_use = True

    elapsed = time.time() - start
    print(f"stream finished in {elapsed:.2f}s, events={sorted(event_types)}, text_len={len(text_content)}")

    missing = REQUIRED_EVENT_TYPES - event_types
    if missing:
        print(f"FAIL: missing event types: {missing}")
        return False
    if not saw_done:
        print("FAIL: no [DONE] sentinel")
        return False
    if "tools" in payload and not saw_tool_use:
        print("FAIL: expected a tool_use block in streaming response")
        return False
    if "tools" not in payload and not text_content:
        print("FAIL: expected text content in streaming response")
        return False

    print("OK")
    return True


async def run_one(name: str, payload: Dict[str, Any]) -> bool:
    if payload.get("stream"):
        return await run_streaming(name, payload)
    return await run_non_streaming(name, payload)


async def _run_integration_scenarios(scenarios: Dict[str, Dict[str, Any]]) -> List[bool]:
    return [await run_one(name, payload) for name, payload in scenarios.items()]


def filter_scenarios(scenarios: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    if args.simple:
        return {k: v for k, v in scenarios.items() if "tools" not in v}
    if args.tools:
        return {k: v for k, v in scenarios.items() if "tools" in v}
    return scenarios


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def discover_unit_tests() -> List[str]:
    """Collect every top-level test_* function defined in this module."""
    import inspect
    return [name for name, _ in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
            if name.startswith("test_")]


def run_unit_tests(names: List[str]) -> List[bool]:
    """Invoke each test_* function; await coroutines via a fresh event loop per test."""
    results: List[bool] = []
    for name in names:
        try:
            fn = getattr(sys.modules[__name__], name)
            result = fn()
            if asyncio.iscoroutine(result):
                asyncio.run(result)
            print(f"OK   {name}")
            results.append(True)
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            results.append(False)
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            results.append(False)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Tests for the Anthropic → OpenAI proxy")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--unit", action="store_true", help="run unit tests only (default)")
    mode.add_argument("--integration", action="store_true", help="run integration smoke tests")
    mode.add_argument("--all", action="store_true", help="run unit + integration tests")
    parser.add_argument("--simple", action="store_true", help="(integration) skip tool scenarios")
    parser.add_argument("--tools", action="store_true", help="(integration) only tool scenarios")
    args = parser.parse_args()

    run_units = args.all or (not args.integration)
    run_integration = args.all or args.integration

    unit_results: List[bool] = []
    if run_units:
        names = discover_unit_tests()
        print(f"--- unit tests ({len(names)}) ---")
        unit_results = run_unit_tests(names)

    integration_results: List[bool] = []
    if run_integration:
        scenarios = filter_scenarios(TEST_SCENARIOS, args)
        print(f"\n--- integration tests ({len(scenarios)}) ---")
        integration_results = asyncio.run(_run_integration_scenarios(scenarios))

    passed = sum(unit_results) + sum(integration_results)
    total = len(unit_results) + len(integration_results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())