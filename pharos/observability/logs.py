"""Structured JSON logger.

Every log line carries trace_id and span_id (if a span is active),
so logs can be cross-referenced with traces.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from pharos.observability.trace import current_trace_id


@dataclass
class StructuredLogger:
    """A logger that writes JSON lines to a sink (default: stdout).

    The sink is pluggable so tests can capture and assert on output.
    """

    name: str
    sink: Callable[[str], None] = lambda line: print(line, file=sys.stderr)
    min_level: str = "info"  # "debug" | "info" | "warning" | "error"

    _LEVELS: ClassVar[dict[str, int]] = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
    }

    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        if self._LEVELS[level] < self._LEVELS[self.min_level]:
            return
        record = {
            "ts": time.time(),
            "level": level,
            "logger": self.name,
            "msg": msg,
            "trace_id": current_trace_id(),
            **fields,
        }
        # Always emit a complete line; callers' sinks should not need
        # to add their own newline.
        self.sink(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("debug", msg, **fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("info", msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit("warning", msg, **fields)

    def error(
        self, msg: str, exc: BaseException | None = None, **fields: Any
    ) -> None:
        if exc is not None:
            fields["error"] = f"{type(exc).__name__}: {exc}"
        self._emit("error", msg, **fields)


__all__ = ["StructuredLogger"]
