# Bug tracker

Known issues and review findings deferred from the linter branch (`chore/linters`).

Items in **Active** are still latent in `main`; items in **Resolved** were
fixed in this branch (see commit `8eb227d`). New findings go to **Active**
and migrate down as work lands.

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
- **Source**: code review, `server.py:1635` — pre-existing in `main`,
  preserved through our refactor; not introduced by the linter branch.

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
- **Source**: code review, `server.py:1916` — pre-existing in `main`. We
  refactored the call shape to a `_LogContext` dataclass in `cd0baca` but
  kept the same call position. No regression vs. `main`.

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
- **Source**: code review, `server.py:207,229` — both functions created in
  `1f179439` (2026-08-11), pre-dates the linter branch.

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
- **Source**: code review, `server.py:1122` — pre-existing in `main`.

#### HOST default `127.0.0.1` — `server.py:1958`

- **Severity**: low — by design (security: don't bind LAN by default), but
  a behaviour break for existing `python server.py` invocations without
  the `HOST` env var.
- **Where**: `__main__` previously hardcoded `0.0.0.0`; now reads
  `os.environ.get("HOST", DEFAULT_HOST)` with `DEFAULT_HOST =
  "127.0.0.1"`. Bare `python server.py` no longer LAN-bindable without
  `HOST=0.0.0.0`.
- **Suggested fix**: documented in `README.md`, `.env.example`, and
  `CLAUDE.md` (commits `1363364` + `2ce4cf7`). No code change needed;
  re-evaluate if users report migration friction.
- **Source**: code review, `server.py:1958` — intentional change in
  `1363364`.

### Pre-existing format drift

Side-finding from the linter branch. Not run through `ruff format`
because the diff would be unrelated to the linter work and pollute the
history.

- `server.py:659` — long type alias wraps awkwardly.
- `tests.py` — 4 places of accumulated format drift (`uv run ruff format`
  would normalise; verify the test suite still passes after).

### Code robustness

Latent bugs in `_first_choice`, `_extract_tool_calls`, `_parse_tool_arguments`,
`_extract_usage`, `_join_tool_result_items`. Pre-existing in `main`; the linter
branch refactored bodies but preserved the unsound patterns. Each site now
carries an explicit `# ty: ignore[unsound-return-statement]` so `ty` exits
clean, but the runtime hazards remain until fixed.

#### `_join_tool_result_items` crashes on non-string `text` — `server.py:886`

- **Severity**: medium — uncaught `TypeError` escapes to the request handler as a 500.
- **Where**: `_join_tool_result_items` (extracted by `dd8516ea` from main's
  `parse_tool_result_content` at `~server.py:717`) does `item.get("text", "") + "\n"`
  without coercing. Pydantic accepts `content: [{"type": "text", "text": null}]`
  because `ContentBlockToolResult.content` is typed loosely (`str | list[...] | ... | Any`).
- **Repro**: `tool_result.content = [{"type": "text", "text": null}]` →
  `result += None + "\n"` → `TypeError`. Same for `text: 42`, `text: [1,2]`,
  `text: {...}`. Sibling `_parse_tool_result_dict` already does `str(content.get("text", ""))`
  safely.
- **Suggested fix**: wrap with `str(...)` like the sibling does, or push the
  type narrowing up into the caller by tightening `ContentBlockToolResult.content`.
- **Source**: code review pass 2 (post-`54c37ac`). Pre-existing in main's
  `parse_tool_result_content`; refactored into a helper by `dd8516ea`.

#### `_first_choice` indexes without `isinstance(choices, list)` — `server.py:1233`

- **Severity**: low — malformed upstream returns 500 via broad exception handler.
- **Where**: `_first_choice` does `if not choices: return None` then `return choices[0]`.
  A dict-keyed or string-valued `choices` passes the falsy check but raises
  on `[0]`.
- **Repro**: cascade proxy returns `{"choices": {"0": {...}}}` or `{"choices": "error"}`.
- **Suggested fix**: `if not isinstance(choices, list) or not choices: return None`.
  Same `isinstance` guard would also clear the `unsound-return` inference.
- **Source**: code review pass 2. Pre-existing in main (same code at `~server.py:1056`).

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
- **Source**: code review pass 2. Pre-existing in main (same code).

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
- **Source**: code review pass 2. Pre-existing in main.

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
- **Source**: code review pass 2. Pre-existing in main (`~server.py:1108`).

## Resolved

Items fixed in `8eb227d` "fix: revert 5 regressions introduced by linter branch".

| # | File | What | Why |
|---|------|------|-----|
| 3 | `server.py:1707` | Drop `isinstance(delta, dict)` coercion in `_process_chunk` | Downstream `_get_field` / `_coerce_delta_field` already handle non-dict via duck-typed getattr fallbacks. |
| 4 | `server.py:1796` | Move `state.should_stop = True` BEFORE `yield from _emit_failure(...)` in `_handle_chunk_error` | Main used `return` immediately after the yield-from; we replaced with `should_stop` but kept post-yield ordering, leaving the guard unset if `_emit_failure` raises. |
| 5 | `server.py:792` | Drop `isinstance(text, str)` guard in `_ThinkStreamParser.feed` | Restore main's truthiness-only guard; truthy non-strings now crash on `buffer += text` and surface via chunk tolerance. |
| 6 | `server.py:1499` | Drop `isinstance(text, str)` guard in `_emit_thinking` | Same rationale as #5. |
| 7 | `server.py:107` | Drop `tiktoken.Encoding` subclass, revert to plain duck-typed class for `_OfflineEncoding` | Subclass made `isinstance(enc, Encoding)` True and dispatched into non-overridden parent methods that dereference uninitialised `_mergeable_ranks` / `_special_tokens`. Duck-typed class only exercises methods we explicitly define. |

## Source

Code-review findings from the `chore/linters` review pass (10 findings).
5 fixed in `8eb227d`; 5 listed in **Active** above.

Code-review findings from pass 2 (post-`54c37ac`, 10 findings):

| # | Finding | Origin | Resolution |
|---|---------|--------|------------|
| 1 | `unsound-return-statement = "warn"` regresses `ty check` exit 1 | NEW (this branch) | Mitigated: 6 sites now carry `# ty: ignore[unsound-return-statement]`, ty exits 0 |
| 2 | `_join_tool_result_items` TypeError on non-string `text` | PRE-EXISTING | Logged in Active → Code robustness |
| 3 | `_parse_tool_arguments` returns non-dict from `json.loads` | PRE-EXISTING | Logged in Active → Code robustness |
| 4 | `_extract_usage` doesn't coerce float/negative | PRE-EXISTING | Logged in Active → Code robustness |
| 5 | 6 trailing comments document unsoundness instead of suppressing | NEW (this branch) | Replaced with `# ty: ignore[unsound-return-statement]` at each site |
| 6 | Comments violate CLAUDE.md (prose apologia, false claims) | NEW (this branch) | Same — replaced with structured ignore comments |
| 7 | `_extract_tool_calls` wraps non-list as `[raw]` | PRE-EXISTING | Logged in Active → Code robustness |
| 8 | `_first_choice` indexes without `isinstance(choices, list)` | PRE-EXISTING | Logged in Active → Code robustness |
| 9 | pyproject.toml header comment miscategorises 4 rules | NEW (this branch) | Rewritten to "Promote every default-ignore rule to warn" |
| 10 | README documents clean ty gate; baseline was 6 warnings | NEW (this branch) | ty now exits 0 after fix #5; README note added |