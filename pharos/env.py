"""Minimal .env loader — keeps shell env vars authoritative.

Reads `~/.pharos/.env` (and any path in `PHAROS_DOTENV`) and sets
variables into `os.environ` ONLY if not already set. This means:

  - Shell env always wins
  - The .env file provides defaults for missing keys

Why no `python-dotenv` dep: this is 30 lines, and we don't need
variable expansion, multiline values, or command substitution.
Plain `KEY=VALUE` lines are enough for our use case.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_PATHS = [
    Path.home() / ".pharos" / ".env",
]


def _load_file(path: Path) -> int:
    """Apply a single .env file. Returns count of vars loaded."""
    if not path.exists():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Skip comments and blank lines
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes if present
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        # Only set if not already in os.environ (shell wins)
        if key and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def load_dotenv() -> int:
    """Load .env files. Honors `PHAROS_DOTENV` (colon-separated paths).

    Returns total number of variables loaded.
    """
    paths: list[Path] = []
    extra = os.environ.get("PHAROS_DOTENV")
    if extra:
        paths.extend(Path(p).expanduser() for p in extra.split(":"))
    paths.extend(_DEFAULT_PATHS)

    total = 0
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        total += _load_file(p)
    return total


__all__ = ["load_dotenv"]