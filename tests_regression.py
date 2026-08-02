#!/usr/bin/env python3
"""
Regression tests for the Anthropic -> OpenAI translation logic.

These tests target code paths NOT covered by tests.py — streaming, list-form
content blocks, image conversion, orphan tool calls, edge cases of model
mapping, and helper functions.

They are pure unit tests (no network, no live OpenAI calls). To run against
the feature branch: `python tests_regression.py`. To run against main,
checkout main, drop this file in, and adapt the import (the helper module
name and several function names differ).

Usage:
    python tests_regression.py
"""

import asyncio
import json
import sys
from typing import Any, AsyncIterator, Dict, List, Optional

import server as srv


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

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
            except json.JSONDecodeError:
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


# ===========================================================================
# 1. STREAMING — most critical gap
# ===========================================================================

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

    # Find indices of key events
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
        _text_chunk("after tool"),  # should be ignored — block already in tool mode
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
    chunks = [_text_chunk("partial...")]  # no finish chunk
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


# ===========================================================================
# 2. CONTENT BLOCKS (list-form user messages) + IMAGES
# ===========================================================================

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
    # implementation-specific, but must not raise and must return image_url-shaped dict
    assert out["type"] == "image_url"
    assert "url" in out["image_url"]


# ===========================================================================
# 3. ORPHAN TOOL CALL / TOOL RESULT handling
# ===========================================================================

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
            # NO tool_result — context was truncated
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
    # Expected order: user, assistant, tool, user
    assert roles == ["user", "assistant", "tool", "user"], f"got {roles}"


# ===========================================================================
# 4. MODEL MAPPING — edge cases
# ===========================================================================

def test_validate_model_field_strips_gemini_prefix() -> None:
    """`gemini/foo` was stripped on main too; must continue to work (treated as bare foo)."""
    req = _make_request({
        "model": "gemini/claude-3-5-haiku-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    # haiku detection should still fire after prefix strip
    assert req.model == f"openai/{srv.SMALL_MODEL}"
    assert req.original_model == "gemini/claude-3-5-haiku-20241022"


def test_validate_model_field_claude_opus_passes_through_with_openai_prefix() -> None:
    """A bare `claude-3-opus-...` has no haiku/sonnet substring and is not in OPENAI_MODELS.
    Per the simplified code, it becomes `openai/claude-3-opus-...` (custom-endpoint pass-through).
    This is a behavior change vs main (which would have left it unchanged); document it."""
    req = _make_request({
        "model": "claude-3-opus-20240229",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == "openai/claude-3-opus-20240229"


def test_validate_model_field_unknown_model_gets_openai_prefix() -> None:
    """Any unknown model name gets `openai/` prefix — enables custom OpenAI-compatible endpoints."""
    req = _make_request({
        "model": "my-custom-llm-7b",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == "openai/my-custom-llm-7b"


# ===========================================================================
# 5. TOOL CHOICE variants
# ===========================================================================

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
        "tool_choice": {"type": "tool"},  # no "name"
    })
    out = srv.convert_anthropic_to_litellm(req)
    assert out["tool_choice"] == "auto"


# ===========================================================================
# 6. STOP REASON mapping
# ===========================================================================

def test_to_anthropic_stop_reason_table() -> None:
    """Direct unit test for the mapping function — covers ALL paths at once."""
    assert srv.to_anthropic_stop_reason("stop") == "end_turn"
    assert srv.to_anthropic_stop_reason("length") == "max_tokens"
    assert srv.to_anthropic_stop_reason("tool_calls") == "tool_use"
    assert srv.to_anthropic_stop_reason("unknown_thing") == "end_turn"  # default
    assert srv.to_anthropic_stop_reason(None) == "end_turn"  # None -> default
    assert srv.to_anthropic_stop_reason("") == "end_turn"  # empty -> default


# ===========================================================================
# 7. HELPER FUNCTIONS
# ===========================================================================

def test_get_field_reads_from_dict() -> None:
    assert srv.get_field({"a": 1, "b": 2}, "a") == 1
    assert srv.get_field({"a": 1}, "missing") is None
    assert srv.get_field({"a": 1}, "missing", "fallback") == "fallback"


def test_get_field_reads_from_object() -> None:
    class Obj:
        a = 5
        b = None
    obj = Obj()
    assert srv.get_field(obj, "a") == 5
    assert srv.get_field(obj, "b") is None
    assert srv.get_field(obj, "missing", "x") == "x"


def test_new_msg_id_format() -> None:
    mid = srv.new_msg_id()
    assert mid.startswith("msg_")
    assert len(mid) == len("msg_") + 24
    hexpart = mid[len("msg_"):]
    assert all(c in "0123456789abcdef" for c in hexpart)


def test_short_model_strips_prefix() -> None:
    assert srv.short_model("openai/gpt-4.1") == "gpt-4.1"
    assert srv.short_model("anthropic/claude-3") == "claude-3"


def test_short_model_unchanged_when_no_prefix() -> None:
    assert srv.short_model("gpt-4.1") == "gpt-4.1"
    assert srv.short_model("claude-3-opus") == "claude-3-opus"


# ===========================================================================
# 8. MALFORMED INPUT handling
# ===========================================================================

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


# ===========================================================================
# 9. SYSTEM message variants
# ===========================================================================

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


def test_no_system_message_does_not_add_one() -> None:
    req = _base_request()
    out = srv.convert_anthropic_to_litellm(req)
    assert all(m["role"] != "system" for m in out["messages"])


# ===========================================================================
# 10. REASONING CONTENT — `<think>...</think>` parsing
# ===========================================================================

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
    """Text before and after the think block becomes text deltas."""
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
    assert p.feed("hello <") == [("text", "hello ")]  # trailing `<` held back
    assert p.feed("think>Hel") == [("open", None)]
    # "Hel" above is the prefix of "Hello". The parser must NOT emit it
    # yet — doing so would commit it to "thinking" prematurely.
    assert p.feed("lo!</thin") == []  # `</thin` is a partial close
    assert p.feed("k>World") + p.flush() == [
        ("thinking", "Hello!"),  # released only on ``
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
    """Thinking content must NOT be emitted chunk-by-chunk before ``.

    Earlier behaviour emitted every text-only chunk as a thinking delta,
    which misclassified answer text whenever the close tag arrived later
    than the buffer flush.
    """
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
    """Empty ``<think></think>`` produces open/close with no thinking delta."""
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
    # Trailing `<` is held back as a potential close-tag prefix — not flushed
    # yet. At end of stream, flush() must release the whole buffered chunk
    # as thinking and close.
    assert p.feed("<") == []
    assert p.flush() == [("thinking", "hello <"), ("close", None)]


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
    # Thinking block must NOT survive the conversion to OpenAI format.
    assert assistant.get("content") == "hello"
    assert "tool_calls" not in assistant


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


# ===========================================================================
# Runner
# ===========================================================================

def _discover() -> List[str]:
    import inspect
    return sorted(
        name for name, _ in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("test_")
    )


async def _run_all() -> int:
    names = _discover()
    print(f"--- regression tests ({len(names)}) ---")
    passed = 0
    for name in names:
        fn = getattr(sys.modules[__name__], name)
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            print(f"OK   {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(names)} regression tests passed")
    return 0 if passed == len(names) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
