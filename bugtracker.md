# Bug tracker

Known latent issues in `server.py` / `tests.py`. Items are grouped by area;
new findings belong at the top of the relevant section. Delete an entry
when fixed — git history preserves it.

Line numbers in headers are approximate anchors; they drift as the
surrounding code changes.

## Active

### Streaming resilience

#### `end_turn` hardcoded when upstream omits finish_reason — `server.py:1635`

- **Severity**: high — silently drops tool_use when upstream closes the
  stream before sending `finish_reason`.
- **Where**: `_stream_epilogue` always calls `_SseFormatter.finish("end_turn", …)`,
  even when a `tool_use` block is mid-emission. The Anthropic SDK treats
  `end_turn` as "no pending work" and never asks the user for tool results.
- **Repro**: tool_use stream where upstream emits two valid tool_use blocks,
  then closes without `finish_reason`.
- **Suggested fix**: track whether any `tool_use` block was emitted in
  `_StreamState`; pick `_to_anthropic_stop_reason("tool_use")` when true,
  `end_turn` otherwise. Requires the in-flight tool_use block to remain
  open through the finish event (Anthropic SSE expects
  `content_block_stop` before `message_delta`).
- **Source**: pre-existing in `main`.

#### `_log_request` emits STATUS_OK before upstream call — `server.py:1907`

- **Severity**: medium — failed requests logged as green 200, no failure
  line emitted. `uvicorn.access` is silenced, so the framework's own log
  is also suppressed; only the green 200 surfaces.
- **Where**: `_handle_request` writes the per-request log line before
  invoking `litellm.completion` / `litellm.acompletion`. When the upstream
  call raises (auth error, rate limit, schema mismatch, timeout),
  `create_message` re-raises as `HTTPException` with no replacement log
  line.
- **Repro**: send a request that passes validation, but `litellm` raises
  `AuthenticationError`. The log shows green 200; the actual 401 is not
  visible to log scrapers.
- **Suggested fix**: wrap the `_handle_request` body in `try/except` that
  re-emits `_log_request` with the failure status code, or move the log
  line into a finally-style emission that captures the actual outcome.
- **Source**: pre-existing in `main`.

### Code quality / DRY

#### `_parse_tier_section` and `_parse_bucket_section` byte-identical — `server.py:207-248`

- **Severity**: low — DRY violation; future key additions risk asymmetric
  behaviour between `[global]` and the per-tier/bucket sections.
- **Where**: both functions have identical bodies; only the docstrings
  differ. Dispatch routes `[global]` to one and `[big]/[small]/[tier]` to
  the other.
- **Repro**: maintainer adds a new recognised key (say `provider_params`)
  to `_parse_bucket_section` only — `[global]` silently omits the key.
  Per-tier configs pick up the new key while `[global]` drops it;
  asymmetric behaviour invisible until a user complains.
- **Suggested fix**: collapse to a single `_parse_section(body, section,
  accepts_model: bool)` and call it from both dispatch paths. Or keep
  separate functions but factor the body into `_validate_section_key`.
- **Source**: pre-existing in `main`.

### Design / docs

#### `if bucket:` truthy gate for unmapped models — `server.py:1122-1123`

- **Severity**: low — by design per `CLAUDE.md` ("tier=None → [global]"),
  but operators who skim the schema naturally expect `[big]` to be the
  catch-all for anything non-haiku.
- **Where**: `_resolve_tier_config` skips the bucket layer when
  `request.tier` is `None`. An operator who sets `[big] model = "X"`
  expecting it to apply to unmapped models like `my-custom-llama` is
  silently ignored; only `[global].model` is consulted.
- **Suggested fix**: either (a) tighten the resolver to always include
  `[big]` for non-haiku unmapped tiers (changes documented behaviour), or
  (b) keep current behaviour and add an explicit warning at startup when
  `[big].model` is set but the resolver never reads it for tier=None.
- **Source**: pre-existing in `main`.

#### HOST default `127.0.0.1` — `server.py:1958`

- **Severity**: low — by design (security: don't bind LAN by default), but
  a behaviour break for existing `python server.py` invocations without
  the `HOST` env var.
- **Where**: `__main__` previously hardcoded `0.0.0.0`; now reads
  `os.environ.get("HOST", DEFAULT_HOST)` with `DEFAULT_HOST =
  "127.0.0.1"`. Bare `python server.py` no longer LAN-bindable without
  `HOST=0.0.0.0`.
- **Suggested fix**: documented in `README.md`, `.env.example`, and
  `CLAUDE.md`. No code change needed; re-evaluate if users report
  migration friction.
- **Source**: intentional behaviour change.

### Format drift

Not run through `ruff format` because the diff would have been unrelated
to the work that surfaced these items. Re-run `uv run ruff format` and
verify the test suite still passes after reformatting.

- `server.py:659` — long type alias wraps awkwardly.
- `tests.py` — 4 places of accumulated format drift.

### Code robustness

Latent bugs in `_first_choice`, `_extract_tool_calls`, `_parse_tool_arguments`,
`_extract_usage`, `_join_tool_result_items`. Each site carries an explicit
`# ty: ignore[unsound-return-statement]` so `ty` exits clean; the runtime
hazards below remain until the actual bug is fixed.

#### `_join_tool_result_items` crashes on non-string `text` — `server.py:886`

- **Severity**: medium — uncaught `TypeError` escapes to the request handler as a 500.
- **Where**: `_join_tool_result_items` (extracted from `parse_tool_result_content`)
  does `item.get("text", "") + "\n"` without coercing. Pydantic accepts
  `content: [{"type": "text", "text": null}]` because
  `ContentBlockToolResult.content` is typed loosely (`str | list[...] | ... | Any`).
- **Repro**: `tool_result.content = [{"type": "text", "text": null}]` →
  `result += None + "\n"` → `TypeError`. Same for `text: 42`, `text: [1,2]`,
  `text: {...}`. Sibling `_parse_tool_result_dict` already does
  `str(content.get("text", ""))` safely.
- **Suggested fix**: wrap with `str(...)` like the sibling does, or push the
  type narrowing up into the caller by tightening `ContentBlockToolResult.content`.
- **Source**: pre-existing in `main`.

#### `_first_choice` indexes without `isinstance(choices, list)` — `server.py:1233`

- **Severity**: low — malformed upstream returns 500 via broad exception handler.
- **Where**: `_first_choice` does `if not choices: return None` then
  `return choices[0]`. A dict-keyed or string-valued `choices` passes the
  falsy check but raises on `[0]`.
- **Repro**: cascade proxy returns `{"choices": {"0": {...}}}` or `{"choices": "error"}`.
- **Suggested fix**: `if not isinstance(choices, list) or not choices: return None`.
  Same `isinstance` guard would also clear the `unsound-return` inference.
- **Source**: pre-existing in `main`.

#### `_extract_tool_calls` wraps non-list into `[raw]` — `server.py:1248`

- **Severity**: low — fabricates an empty `tool_use` block the client rejects.
- **Where**: `_extract_tool_calls` does `if not raw: return []` then
  `if isinstance(raw, list): return raw; return [raw]`. A non-list, non-falsy
  value (string `"call_1"`, integer `42`, tuple) wraps into `[raw]` and
  proceeds to `_build_content_blocks`, which builds
  `{"type": "tool_use", "name": "", "input": {}}` — passes Pydantic validation
  but is semantically broken.
- **Repro**: `message = {"tool_calls": "call_1"}` → empty `name` + `input`.
- **Suggested fix**: drop the `return [raw]` fallback, or guard with
  `isinstance(raw, dict)` to wrap a single dict-call (not a string/int/tuple).
- **Source**: pre-existing in `main`.

#### `_parse_tool_arguments` returns non-dict from `json.loads` — `server.py:1257`

- **Severity**: medium — Pydantic rejects the resulting `tool_use.input`,
  broad handler swallows and emits an error placeholder; the real tool call
  is silently discarded.
- **Where**: `_parse_tool_arguments` does `return json.loads(raw)` on success.
  The `JSONDecodeError` branch wraps in `{"raw": raw}`; the success branch
  has no equivalent `isinstance(..., dict)` guard.
- **Repro**: upstream returns `tool_calls[0].function.arguments == "5"` (or
  `"null"`, `"[1,2]"`, `'"hi"'`) → `input: 5` → ValidationError.
- **Suggested fix**: `result = json.loads(raw); return result if isinstance(result, dict) else {"raw": raw}`.
- **Source**: pre-existing in `main`.

#### `_extract_usage` doesn't coerce non-int token counts — `server.py:1295`

- **Severity**: low — float `usage.prompt_tokens` slips through and breaks
  Pydantic validation; negative counts ship a contract-violating Usage block.
- **Where**: `_extract_usage` does `_get_field(usage, "prompt_tokens", 0) or 0`.
  `or 0` only collapses falsy; a float (`1500.5 or 0 → 1500.5`) and negative
  count (`-1 or 0 → -1`) pass through.
- **Repro**: upstream `usage = {"prompt_tokens": 1500.5, "completion_tokens": 100}`
  → ValidationError → broad handler → error placeholder.
- **Suggested fix**: mirror streaming path `_record_chunk_usage` (~line 1672):
  `isinstance(incoming, (int, float)) and int(incoming)` else `0`.
- **Source**: pre-existing in `main`.
