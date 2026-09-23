"""Render states, instructions, and criteria into plain text for the engines.

Mirrors the official flexibility: ``state`` and ``instructions`` may each be a
string, a JSON object, or an array. Objects render as ``key: value`` lines so
that backtick path references in instructions (e.g. `` `ticket.messages[0]` ``)
line up with what the model sees.
"""

from __future__ import annotations

import json
from typing import Any


def _inline(value: Any) -> str:
    """Render a leaf value: strings as-is, structured values as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_text(value: Any) -> str:
    """Render a str | object | array | null field into readable text.

    ``None`` renders as an empty string: the official schema marks
    ``instructions`` optional, and a missing instruction contributes no text.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_inline(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {_inline(v)}" for v in value)
    return _inline(value)


def render_state(state: Any) -> str:
    """Render the request state. Strings pass through; structure becomes JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when no tokenizer is available."""
    return max(1, (len(text) + 3) // 4)
