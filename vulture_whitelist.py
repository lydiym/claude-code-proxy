"""False positives for vulture — names used by frameworks, introspection, or
Pydantic that vulture can't trace statically.

Re-generate the bare list with:

    uv run vulture server.py --make-whitelist

then prune to the FP lines that are real uses.

Each entry is either a real Python reference (so vulture sees the name as
used) or a `name  # unused <kind> (file.py:line)` location-tagged comment that
whitelists a specific finding. Excluded from ruff (it's a tool config file,
not production code).
"""

# Monkey-patches from server.py onto third-party module attributes.
# vulture can't trace cross-module attribute assignment, so the LHS attrs
# look unused to it. The references below teach vulture they're set.
import tiktoken
import litellm

tiktoken.get_encoding  # server.py:130 sets this to server._get_encoding
tiktoken.encoding_for_model  # server.py:131 sets this to server._encoding_for_model
litellm.set_verbose  # server.py:452 sets this to True
litellm.ssl_verify  # server.py:472 sets this to OPENAI_TLS_VERIFY

# Documented OpenAI cap; the proxy no longer enforces it (see CLAUDE.md —
# "MAX_OUTPUT_TOKENS = 16384 is the documented OpenAI cap; the proxy no
# longer clamps"). Kept for clarity; will go live again if we ever clamp.
from server import MAX_OUTPUT_TOKENS

MAX_OUTPUT_TOKENS

# Pydantic BaseModel fields — present in the wire schema and used by
# Pydantic's model machinery. vulture can't see class-attribute use through
# __init__ / model_dump.
from server import ContentBlockThinking, Usage, MessagesResponse

ContentBlockThinking.thinking
ContentBlockThinking.signature
Usage.cache_creation_input_tokens
Usage.cache_read_input_tokens
MessagesResponse.stop_sequence

# Mirror signature params on `_OfflineEncoding` overrides — names required to
# match the parent `tiktoken.Encoding` signature. ty's LSP check forbids
# underscore-prefixing them, so vulture sees them as unused. Whitelist by
# exact file:line.
token  # unused variable (server.py:112)
tokens  # unused variable (server.py:116)
token  # unused variable (server.py:120)

# Test fixture lambda replaces `srv._translate_parser_events`; the
# upstream signature is unknown to vulture and the params are positional/
# keyword placeholders.
a  # unused variable (tests.py:1394)
kw  # unused variable (tests.py:1394)