"""Best-effort secret redaction for trace attributes and events."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "cookie",
    "password",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
    "x_api_key",
}

_SECRET_PATTERNS = [
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
]


def redact(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact known credential keys and token-like strings."""
    normalized = (key or "").lower().replace("-", "_")
    if normalized in _SENSITIVE_KEYS:
        return REDACTED
    if isinstance(value, str):
        out = value
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(REDACTED, out)
        return out
    if isinstance(value, dict):
        return {name: redact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


__all__ = ["REDACTED", "redact"]
