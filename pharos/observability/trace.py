"""Trace — the spine of pharos observability.

A `Span` represents a unit of work. Spans form a tree via parent_span_id.
Each LLM call, tool call, or significant entity fire should produce a span.

Design notes:
- Compatible with OpenTelemetry semantic conventions (but pharos has
  no OTel dependency; the schema is open).
- `tracer` is a Protocol — the runtime can use any backend (in-memory,
  SQLite, OTel exporter, console).
- `current_trace_id` returns the trace_id of the active span (or None)
  so logs/metrics can auto-link without explicit threading.
"""

from __future__ import annotations

import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Thread-/task-local current span. ContextVar works across asyncio tasks.
_active_span: ContextVar[Span | None] = ContextVar("pharos_active_span", default=None)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Bounds that keep a single run's trace from growing without limit (the
# in-memory span list + the persisted JSON). Long agent runs with large
# tool outputs would otherwise OOM or produce multi-GB run files.
#   PHAROS_TRACE_MAX_EVENTS — max events kept per span (default 2000)
#   PHAROS_TRACE_MAX_ATTR   — max chars per string attribute (default 16384)
MAX_EVENTS_PER_SPAN = _int_env("PHAROS_TRACE_MAX_EVENTS", 2000)
MAX_ATTR_CHARS = _int_env("PHAROS_TRACE_MAX_ATTR", 16384)


def _truncate_value(value: Any, max_chars: int) -> Any:
    """Recursively cap long strings inside an attribute value."""
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + f"...(truncated {len(value) - max_chars} chars)"
        return value
    if isinstance(value, dict):
        return {k: _truncate_value(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(v, max_chars) for v in value]
    return value


def _truncate_attrs(attrs: dict[str, Any], max_chars: int) -> dict[str, Any]:
    return {k: _truncate_value(v, max_chars) for k, v in attrs.items()}


def current_trace_id() -> str | None:
    """Return the trace_id of the currently active span, or None."""
    span = _active_span.get()
    return span.trace_id if span is not None else None


def current_span() -> Span | None:
    """Return the currently active span, or None."""
    return _active_span.get()


@dataclass
class SpanEvent:
    """A timed annotation within a span (e.g. 'llm.first_token')."""

    name: str
    ts: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single unit of traced work.

    Lifecycle (managed by Tracer, not the user):
        span = tracer.start_span("llm_call", **attrs)
        ... do work ...
        span.set_attribute("tokens.input", 100)
        span.add_event("first_token", {"delta": "hello"})
        span.end()
    """

    id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    started_at: float
    ended_at: float | None = None
    status: str = "ok"  # "ok" | "error" | "unset"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error: str | None = None
    # Per-span bounds (copied from the module defaults at creation so a
    # single span can be tuned independently in tests).
    max_events: int = MAX_EVENTS_PER_SPAN
    max_attr_chars: int = MAX_ATTR_CHARS
    dropped_events: int = 0

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return (time.time() - self.started_at) * 1000
        return (self.ended_at - self.started_at) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def record_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        *,
        ts: float | None = None,
    ) -> None:
        """Append an event, enforcing the per-span cap and truncating long
        string attributes. Once the cap is hit, further events are counted
        in `dropped_events` but not stored (keeps memory/JSON bounded).
        """
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(
            SpanEvent(
                name=name,
                ts=ts if ts is not None else time.time(),
                attributes=_truncate_attrs(attributes or {}, self.max_attr_chars),
            )
        )

    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> None:
        self.record_event(name, attributes)

    def record_exception(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = f"{type(exc).__name__}: {exc}"
        self.add_event("exception", {"error": self.error})

    def end(self, status: str | None = None) -> None:
        if self.ended_at is None:
            self.ended_at = time.time()
        if status is not None:
            self.status = status


@runtime_checkable
class Tracer(Protocol):
    """All Tracer backends implement this Protocol."""

    def start_span(
        self,
        name: str,
        *,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Open a new span. If `parent` is given, it becomes the parent."""
        ...

    def finish_span(self, span: Span) -> None:
        """Called when a span ends. Backends persist it."""
        ...


class InMemoryTracer:
    """Reference Tracer: holds all spans in memory.

    Useful for tests, hello world, and small dev runs. For real
    production usage, use a SQLite or OTel backend.
    """

    name = "memory"

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._trace_counter = 0

    def start_span(
        self,
        name: str,
        *,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        if parent is None:
            self._trace_counter += 1
            trace_id = f"trace_{self._trace_counter:04d}_{uuid.uuid4().hex[:6]}"
        else:
            trace_id = parent.trace_id
        span = Span(
            id=f"span_{uuid.uuid4().hex[:8]}",
            trace_id=trace_id,
            parent_span_id=parent.id if parent is not None else None,
            name=name,
            started_at=time.time(),
            attributes=dict(attributes or {}),
        )
        _active_span.set(span)
        return span

    def finish_span(self, span: Span) -> None:
        span.end()
        self.spans.append(span)
        # Pop the active span context (best effort — assumes proper nesting)
        # If a different span is active, leave it (caller bug).
        if _active_span.get() is span:
            _active_span.set(None)

    # ----- convenience -----

    def get_trace(self, trace_id: str) -> list[Span]:
        """Return all spans for a given trace_id, in start order."""
        return [s for s in self.spans if s.trace_id == trace_id]

    def clear(self) -> None:
        self.spans.clear()


__all__ = [
    "MAX_ATTR_CHARS",
    "MAX_EVENTS_PER_SPAN",
    "InMemoryTracer",
    "Span",
    "SpanEvent",
    "Tracer",
    "current_span",
    "current_trace_id",
]
