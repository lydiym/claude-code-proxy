#!/usr/bin/env python3
"""
Smoke tests for the Anthropic → OpenAI proxy.

Each test sends a real request to a locally running proxy on PROXY_URL and
checks the response shape (no live Anthropic API calls). Useful for catching
regressions in the request/response translation.

Usage:
  python tests.py                  # run all tests
  python tests.py --simple         # skip the tool scenarios
  python tests.py --tools          # only tool scenarios
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


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for the Anthropic → OpenAI proxy")
    parser.add_argument("--simple", action="store_true", help="skip tool scenarios")
    parser.add_argument("--tools", action="store_true", help="only run tool scenarios")
    args = parser.parse_args()

    scenarios = filter_scenarios(TEST_SCENARIOS, args)
    results: List[bool] = []
    for name, payload in scenarios.items():
        results.append(await run_one(name, payload))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))