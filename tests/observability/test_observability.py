"""Tests for pharos.observability."""
from __future__ import annotations

import json
from io import StringIO

from pharos.observability import (
    EventBus,
    Metrics,
    Span,
    StructuredLogger,
    current_span,
)
from pharos.observability.backend.console import ConsoleTraceBackend
from pharos.observability.trace import InMemoryTracer


class TestSpan:
    def test_basic_creation(self):
        s = Span(
            id="s1", trace_id="t1", parent_span_id=None,
            name="test", started_at=0.0,
        )
        assert s.duration_ms > 0  # not yet ended; uses time.time()
        s.end()
        assert s.duration_ms >= 0

    def test_set_attribute(self):
        s = Span(id="s1", trace_id="t1", parent_span_id=None, name="x", started_at=0.0)
        s.set_attribute("tokens", 100)
        s.set_attributes({"a": 1, "b": 2})
        assert s.attributes == {"tokens": 100, "a": 1, "b": 2}

    def test_add_event(self):
        s = Span(id="s1", trace_id="t1", parent_span_id=None, name="x", started_at=0.0)
        s.add_event("first_token", {"delta": "hi"})
        assert len(s.events) == 1
        assert s.events[0].name == "first_token"

    def test_record_exception(self):
        s = Span(id="s1", trace_id="t1", parent_span_id=None, name="x", started_at=0.0)
        s.record_exception(ValueError("bad"))
        assert s.status == "error"
        assert "ValueError" in (s.error or "")

    def test_status_override_on_end(self):
        s = Span(id="s1", trace_id="t1", parent_span_id=None, name="x", started_at=0.0)
        s.end(status="error")
        assert s.status == "error"
        assert s.ended_at is not None


class TestInMemoryTracer:
    def test_root_span_creates_new_trace_id(self):
        t = InMemoryTracer()
        s = t.start_span("llm_call")
        assert s.trace_id.startswith("trace_")
        assert s.parent_span_id is None
        t.finish_span(s)
        assert len(t.spans) == 1

    def test_child_span_inherits_trace_id(self):
        t = InMemoryTracer()
        parent = t.start_span("root")
        child = t.start_span("child", parent=parent)
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.id
        t.finish_span(child)
        t.finish_span(parent)

    def test_current_span_context(self):
        t = InMemoryTracer()
        s = t.start_span("hello")
        # After start_span, the active span should be s
        active = current_span()
        assert active is s
        t.finish_span(s)
        # After finish, context is reset
        assert current_span() is None

    def test_get_trace(self):
        t = InMemoryTracer()
        a = t.start_span("a")
        b = t.start_span("b", parent=a)
        t.finish_span(b)
        t.finish_span(a)
        c = t.start_span("c")
        t.finish_span(c)
        # a's trace has 2 spans; c's has 1
        assert len(t.get_trace(a.trace_id)) == 2
        assert len(t.get_trace(c.trace_id)) == 1

    def test_clear(self):
        t = InMemoryTracer()
        s = t.start_span("x")
        t.finish_span(s)
        assert len(t.spans) == 1
        t.clear()
        assert len(t.spans) == 0


class TestEventBus:
    async def test_publish_subscribe(self):
        bus = EventBus()
        received: list[dict] = []

        @bus.subscribe("test.event")
        async def handler(payload):
            received.append(payload)

        await bus.publish("test.event", {"a": 1})
        assert received == [{"a": 1}]

    async def test_sync_handler_works(self):
        bus = EventBus()
        received: list[dict] = []

        @bus.subscribe("e")
        def handler(payload):
            received.append(payload)

        await bus.publish("e", {"v": "x"})
        assert received == [{"v": "x"}]

    async def test_handler_error_does_not_block(self):
        bus = EventBus()
        received: list[dict] = []

        @bus.subscribe("e")
        async def bad(_payload):
            raise RuntimeError("boom")

        @bus.subscribe("e")
        async def good(payload):
            received.append(payload)

        await bus.publish("e", {"ok": True})
        assert received == [{"ok": True}]

    async def test_no_handlers(self):
        bus = EventBus()
        await bus.publish("nothing")  # no error


class TestStructuredLogger:
    def test_writes_json_line(self):
        out = StringIO()
        log = StructuredLogger(name="t", sink=out.write, min_level="info")
        log.info("hello", model="gpt-4o")
        line = out.getvalue().strip()
        rec = json.loads(line)
        assert rec["msg"] == "hello"
        assert rec["level"] == "info"
        assert rec["logger"] == "t"
        assert rec["model"] == "gpt-4o"

    def test_min_level_filtering(self):
        out = StringIO()
        log = StructuredLogger(name="t", sink=out.write, min_level="warning")
        log.debug("d")
        log.info("i")
        log.warning("w")
        log.error("e")
        lines = out.getvalue().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["level"] == "warning"
        assert json.loads(lines[1])["level"] == "error"

    def test_error_includes_exception(self):
        out = StringIO()
        log = StructuredLogger(name="t", sink=out.write, min_level="info")
        try:
            raise ValueError("nope")
        except ValueError as e:
            log.error("bad", exc=e)
        rec = json.loads(out.getvalue().strip())
        assert "ValueError" in rec["error"]

    def test_links_to_active_trace(self):
        out = StringIO()
        log = StructuredLogger(name="t", sink=out.write)
        t = InMemoryTracer()
        s = t.start_span("op")
        log.info("inside")
        rec = json.loads(out.getvalue().strip())
        assert rec["trace_id"] == s.trace_id
        t.finish_span(s)


class TestMetrics:
    def test_counter(self):
        m = Metrics()
        m.counter("calls", 1, model="gpt-4o")
        m.counter("calls", 1, model="gpt-4o")
        m.counter("calls", 1, model="gpt-4o-mini")
        assert m.counter_value("calls", model="gpt-4o") == 2
        assert m.counter_value("calls", model="gpt-4o-mini") == 1

    def test_histogram_summary(self):
        m = Metrics()
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            m.histogram("latency_ms", v, op="x")
        s = m.histogram_summary("latency_ms", op="x")
        assert s["count"] == 10
        assert s["min"] == 10
        assert s["max"] == 100
        assert s["mean"] == 55.0
        assert s["p50"] == 60  # sorted: 10..100; idx=int(0.5*10)=5 -> 60

    def test_gauge(self):
        m = Metrics()
        m.gauge("inflight", 5)
        m.gauge("inflight", 10)  # overwrites
        assert m.gauge_value("inflight") == 10

    def test_snapshot(self):
        m = Metrics()
        m.counter("c", 1, k="v")
        m.gauge("g", 7)
        m.histogram("h", 1.0)
        snap = m.snapshot()
        assert "c{'k': 'v'}" in snap["counters"]
        assert "g{}" in snap["gauges"]
        assert "h{}" in snap["histograms"]


class TestConsoleBackend:
    def test_renders_trace_tree(self):
        backend = ConsoleTraceBackend()
        tracer = InMemoryTracer()
        root = tracer.start_span("root", attributes={"model": "x"})
        c1 = tracer.start_span("child1", parent=root)
        c2 = tracer.start_span("child2", parent=root)
        c1.end()
        c2.end()
        root.end()
        for s in [c1, c2, root]:
            backend.write(s)
        out = backend.render()
        assert "root" in out
        assert "child1" in out
        assert "child2" in out
        assert "ms" in out

    def test_renders_no_spans(self):
        backend = ConsoleTraceBackend()
        assert backend.render() == "(no spans)"
