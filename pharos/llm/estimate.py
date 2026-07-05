"""Lightweight token estimation.

We do NOT bundle tiktoken (heavy dep). This estimator is approximate but
good enough for budget planning. Providers report exact counts in Usage;
this is for pre-flight checks and dry-runs.

Rule of thumb (Anthropic & OpenAI roughly agree):
  ~4 chars per token for English
  ~1.5 chars per token for CJK
"""

from __future__ import annotations

import re

# CJK Unified Ideographs + Hiragana + Katakana + Hangul ranges
_CJK_PATTERN = re.compile(
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af]"
)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string. Returns ≥1 for non-empty text.

    Uses a hybrid rule: CJK chars count as ~1.5 chars/token, ASCII as ~4.
    For mixed text we take a weighted average.
    """
    if not text:
        return 0
    cjk = len(_CJK_PATTERN.findall(text))
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens across a list of pharos messages.

    Accepts any message-like objects with a `content` attribute. Adds a
    small per-message framing overhead (role markers, separators).
    """
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += estimate_tokens(content) + 4  # role + delimiters
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "text", "") or getattr(block, "thinking", "")
                if text:
                    total += estimate_tokens(text) + 4
    return total


__all__ = ["estimate_messages_tokens", "estimate_tokens"]
