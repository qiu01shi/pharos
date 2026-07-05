"""Observability: trace, metrics, logs, events.

The pharos observability layer is built around four primitives:

    Trace    — span tree (parent / child), one per LLM call / tool call
    Metrics  — counters, histograms, gauges (per-step latency, tokens, cost)
    Logs     — structured JSON, automatically linked to trace_id
    Events   — publish/subscribe for high-level business events

This module provides:
- `Span` / `Tracer` Protocol (compatible with OTel semantic conventions)
- `EventBus` for in-process pub/sub
- A `StructuredLogger` (writes JSON lines to stdout for now; pluggable)

Backends (SQLite, console) live in `pharos.observability.backend`.
"""

from pharos.observability.events import EventBus
from pharos.observability.logs import StructuredLogger
from pharos.observability.metrics import Metrics
from pharos.observability.trace import (
    InMemoryTracer,
    Span,
    SpanEvent,
    Tracer,
    current_span,
    current_trace_id,
)

__all__ = [
    "EventBus",
    "InMemoryTracer",
    "Metrics",
    "Span",
    "SpanEvent",
    "StructuredLogger",
    "Tracer",
    "current_span",
    "current_trace_id",
]
