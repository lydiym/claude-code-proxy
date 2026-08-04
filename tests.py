#!/usr/bin/env python3
"""
Tests for the Anthropic → OpenAI proxy.

Includes both integration smoke tests (require a running server on PROXY_URL)
and unit tests for the request/response translation (no network needed).

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
from typing import Any, Dict, List

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


# ---------------------------------------------------------------------------
# Unit tests — translation logic only, no network.
# ---------------------------------------------------------------------------

import server as srv


def _make_request(payload: Dict[str, Any]) -> srv.MessagesRequest:
    """Build a MessagesRequest from a dict (validates + runs model_validator)."""
    return srv.MessagesRequest(**payload)


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


def test_sanitize_messages_for_openai_removes_foreign_keys() -> None:
    messages = [
        {"role": "user", "content": "hi", "stop_reason": "end_turn", "type": "message"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "plain"},
    ]
    srv.sanitize_messages_for_openai(messages)
    assert messages[0] == {"role": "user", "content": "hi"}
    # When tool_calls is present OpenAI allows content=None, so we leave it.
    assert messages[1] == {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}
    # Empty content with no tool_calls gets filled with the ellipsis sentinel.
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


def test_str_to_bool_truthy_values() -> None:
    for v in ("1", "true", "TRUE", "True", "yes", "YES", "on", "On"):
        assert srv._str_to_bool(v) is True, v


def test_str_to_bool_falsy_values() -> None:
    for v in ("0", "false", "False", "no", "off", "", "garbage"):
        assert srv._str_to_bool(v) is False, v


def test_str_to_bool_none_uses_default() -> None:
    assert srv._str_to_bool(None) is False
    assert srv._str_to_bool(None, default=True) is True


def test_str_to_bool_strips_whitespace() -> None:
    assert srv._str_to_bool("  true  ") is True
    assert srv._str_to_bool(" false ") is False


def test_tls_verify_wiring_matches_module_setting() -> None:
    """OPENAI_TLS_VERIFY is read at import time and propagated to litellm."""
    assert srv.OPENAI_TLS_VERIFY == srv.litellm.ssl_verify


REQUIRED_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}


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


def filter_scenarios(scenarios: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    if args.simple:
        return {k: v for k, v in scenarios.items() if "tools" not in v}
    if args.tools:
        return {k: v for k, v in scenarios.items() if "tools" in v}
    return scenarios


def discover_unit_tests() -> List[str]:
    """Collect every top-level test_* function defined in this module."""
    import inspect
    return [name for name, _ in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
            if name.startswith("test_")]


def run_unit_tests(names: List[str]) -> List[bool]:
    """Invoke each test_* function; return one bool per test."""
    results: List[bool] = []
    for name in names:
        try:
            getattr(sys.modules[__name__], name)()
            print(f"OK   {name}")
            results.append(True)
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            results.append(False)
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            results.append(False)
    return results


async def main() -> int:
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
        for name, payload in scenarios.items():
            integration_results.append(await run_one(name, payload))

    passed = sum(unit_results) + sum(integration_results)
    total = len(unit_results) + len(integration_results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))