"""Lightweight metrics: counters, histograms, gauges.

Not Prometheus-format (yet) — we expose a simple in-process registry.
A future backend can export to OTel or Prometheus.

Usage:
    m = Metrics()
    m.counter("llm.calls", 1, model="gpt-4o")
    m.histogram("llm.latency_ms", 234.5, model="gpt-4o")
    m.gauge("tokens_in_flight", 12)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _CounterEntry:
    value: float = 0.0


@dataclass
class _GaugeEntry:
    value: float = 0.0


@dataclass
class _HistogramEntry:
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.samples.append(value)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"count": 0}
        s = sorted(self.samples)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(n - 1, int(p * n))
            return s[idx]

        return {
            "count": n,
            "min": s[0],
            "max": s[-1],
            "mean": statistics.fmean(s),
            "p50": pct(0.5),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }


def _key(name: str, labels: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
    return (name, tuple(sorted(labels.items())))


class Metrics:
    """Process-local metrics registry."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, Any], ...]], _CounterEntry] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, Any], ...]], _GaugeEntry] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, Any], ...]], _HistogramEntry
        ] = defaultdict(_HistogramEntry)

    # ----- writers -----

    def counter(
        self, name: str, value: int | float = 1, **labels: Any
    ) -> None:
        key = _key(name, labels)
        entry = self._counters.setdefault(key, _CounterEntry())
        entry.value += value

    def histogram(
        self, name: str, value: float, **labels: Any
    ) -> None:
        key = _key(name, labels)
        self._histograms[key].observe(value)

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        key = _key(name, labels)
        self._gauges[key] = _GaugeEntry(value)

    # ----- readers -----

    def counter_value(self, name: str, **labels: Any) -> float:
        return self._counters[_key(name, labels)].value

    def gauge_value(self, name: str, **labels: Any) -> float:
        return self._gauges[_key(name, labels)].value

    def histogram_summary(
        self, name: str, **labels: Any
    ) -> dict[str, float]:
        return self._histograms[_key(name, labels)].summary()

    def snapshot(self) -> dict[str, Any]:
        """Export all metrics as a dict (for /metrics endpoints etc)."""
        return {
            "counters": {
                f"{k[0]}{dict(k[1])}": v.value
                for k, v in self._counters.items()
            },
            "gauges": {
                f"{k[0]}{dict(k[1])}": v.value for k, v in self._gauges.items()
            },
            "histograms": {
                f"{k[0]}{dict(k[1])}": v.summary()
                for k, v in self._histograms.items()
            },
        }


__all__ = ["Metrics"]
