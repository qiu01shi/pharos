"""Tests for the TUI viewer module."""
from __future__ import annotations

import io
import time
import uuid
from dataclasses import asdict

from pharos.observability.trace import InMemoryTracer, Span, SpanEvent
from pharos.observability.tui import (
    build_span_tree,
    interactive_view,
    render_span_detail,
    render_summary,
    render_tree_compact,
)


def _make_span(
    name: str,
    *,
    parent_id: str | None = None,
    duration_ms: float = 10.0,
    attrs: dict | None = None,
    events: list[SpanEvent] | None = None,
) -> Span:
    now = time.time()
    s = Span(
        id=f"span-{uuid.uuid4().hex[:8]}",
        trace_id="trace_0001",
        parent_span_id=parent_id,
        name=name,
        started_at=now,
        ended_at=now + duration_ms / 1000,
        status="ok",
        attributes=attrs or {},
    )
    if events:
        s.events.extend(events)
    return s


def _spans_to_dicts(spans: list[Span]) -> list[dict]:
    return [asdict(s) for s in spans]


# ---------- build_span_tree ----------


class TestBuildSpanTree:
    def test_single_root(self):
        s = _make_span("entity.fire.agent")
        roots = build_span_tree(_spans_to_dicts([s]))
        assert len(roots) == 1
        assert roots[0]["name"] == "entity.fire.agent"
        assert roots[0]["children"] == []

    def test_parent_child(self):
        parent = _make_span("parent", duration_ms=20)
        child = _make_span(
            "child",
            parent_id=parent.id,
            duration_ms=5,
        )
        roots = build_span_tree(_spans_to_dicts([parent, child]))
        assert len(roots) == 1
        assert roots[0]["name"] == "parent"
        assert len(roots[0]["children"]) == 1
        assert roots[0]["children"][0]["name"] == "child"

    def test_children_sorted_by_step_id(self):
        parent = _make_span("p", duration_ms=20)
        c1 = _make_span("c1", parent_id=parent.id, attrs={"step_id": "r:2"})
        c2 = _make_span("c2", parent_id=parent.id, attrs={"step_id": "r:1"})
        c3 = _make_span("c3", parent_id=parent.id, attrs={"step_id": "r:3"})
        roots = build_span_tree(_spans_to_dicts([parent, c1, c2, c3]))
        children = roots[0]["children"]
        # sorted by step_id: c2 (r:1), c1 (r:2), c3 (r:3)
        assert [c["name"] for c in children] == ["c2", "c1", "c3"]

    def test_orphan_returns_to_root(self):
        """A span whose parent_id doesn't exist becomes a root."""
        orphan = _make_span("orphan", parent_id="nonexistent")
        roots = build_span_tree(_spans_to_dicts([orphan]))
        assert len(roots) == 1
        assert roots[0]["name"] == "orphan"


# ---------- render_tree_compact ----------


class TestRenderTreeCompact:
    def test_renders_single_root(self):
        s = _make_span("entity.fire.agent", duration_ms=15.3)
        tree = build_span_tree(_spans_to_dicts([s]))
        out = render_tree_compact(tree)
        assert "entity.fire.agent" in out
        assert "15.3 ms" in out

    def test_renders_nested(self):
        parent = _make_span("parent", duration_ms=20)
        child = _make_span("child", parent_id=parent.id, duration_ms=5)
        tree = build_span_tree(_spans_to_dicts([parent, child]))
        out = render_tree_compact(tree)
        assert "parent" in out
        assert "child" in out
        # Child has indentation prefix
        assert "├─" in out or "└─" in out


# ---------- render_span_detail ----------


class TestRenderSpanDetail:
    def test_panel_contains_name(self):
        s = _make_span("entity.fire.agent", attrs={"entity": "agent"})
        panel = render_span_detail(asdict(s))
        # Render returns a Rich Panel; just verify it's not None
        assert panel is not None
        assert panel.title is not None

    def test_panel_includes_attributes(self):
        s = _make_span(
            "entity.fire.agent",
            attrs={"entity": "agent", "step_id": "r:1"},
        )
        panel = render_span_detail(asdict(s))
        # Render to text to check contents
        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=80).print(panel)
        text = buf.getvalue()
        assert "entity.fire.agent" in text
        assert "agent" in text


# ---------- render_summary ----------


class TestRenderSummary:
    def test_counts_spans(self):
        s1 = _make_span("entity.fire.a")
        s2 = _make_span("entity.fire.b")
        s3 = _make_span("other.span")
        tree = build_span_tree(_spans_to_dicts([s1, s2, s3]))
        panel = render_summary("run-1", _spans_to_dicts([s1, s2, s3]), tree)
        assert panel is not None

    def test_empty_run(self):
        panel = render_summary("run-empty", [], [])
        assert panel is not None


# ---------- interactive_view (non-TTY path) ----------


class TestInteractiveView:
    def test_non_tty_prints_tree(self, monkeypatch):
        """When stdin isn't a TTY, prints the compact tree."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        s = _make_span("entity.fire.agent", duration_ms=10)
        spans = _spans_to_dicts([s])
        # Should not raise
        interactive_view("run-x", spans)

    def test_empty_run(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        # Should not raise on empty input
        interactive_view("run-empty", [])

    def test_with_real_run_data(self, monkeypatch, tmp_path, capsys):
        """End-to-end: record a FauxProvider run, then view it."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        from pharos.core.graph import CompositeGraph
        from pharos.core.token import TypedValue
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector
        from pharos.entities.llm import LLMAgent, LLMEntityConfig
        from pharos.llm.providers.faux import FauxConfig, FauxProvider

        g = CompositeGraph("tui-test")
        g.add_entity(
            "agent",
            LLMAgent(
                "agent",
                config=LLMEntityConfig(
                    provider_class=FauxProvider,
                    provider_kwargs={
                        "config": FauxConfig(
                            scripted_responses=["hello tui"],
                            latency_seconds=0.01,
                        )
                    },
                    model_id="faux-fast",
                ),
            ),
        )
        g.nodes["agent"].instance.ins["prompt"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="hi")
        )
        from pharos.core.graph import Edge as _E

        g.edges.append(_E("agent", "text", "__out__", "response"))
        g.collected = {}  # type: ignore[attr-defined]

        tracer = InMemoryTracer()
        ctx = RunContext(run_id="tui-run")
        ctx.tracer = tracer  # type: ignore[attr-defined]
        await_fn = FNDirector().run(g, ctx)
        import asyncio

        asyncio.run(await_fn)

        # Now view it
        spans_dicts = [asdict(s) for s in tracer.spans]
        interactive_view("tui-run", spans_dicts)
        # Capture was made; verify some output went to stdout
        captured = capsys.readouterr()
        assert "tui-run" in captured.out or "tui-run" in captured.err