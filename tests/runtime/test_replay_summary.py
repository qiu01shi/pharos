"""Tests for the P3 deterministic replay summary."""
from __future__ import annotations

import uuid

import pytest

from pharos.observability.trace import Span, SpanEvent
from pharos.runtime import (
    _extract_entity_output_text,
    record_run,
    replay_run_summary,
)


def _make_span(
    name: str,
    events: list[SpanEvent],
    attrs: dict | None = None,
    duration: float = 10.0,
) -> Span:
    import time

    now = time.time()
    s = Span(
        id=f"span-{uuid.uuid4().hex[:8]}",
        trace_id="trace_0001",
        parent_span_id=None,
        name=name,
        started_at=now,
        ended_at=now + duration / 1000,
        status="ok",
        attributes=attrs or {},
    )
    s.events.extend(events)
    return s


class TestExtractEntityOutputText:
    def test_text_deltas(self):
        s = _make_span(
            "entity.fire.a",
            [
                SpanEvent(name="text_delta", ts=0.0, attributes={"delta": "hello "}),
                SpanEvent(name="text_delta", ts=0.0, attributes={"delta": "world"}),
            ],
        )
        # asdict it as record_run would
        from dataclasses import asdict
        serialized = asdict(s)
        text = _extract_entity_output_text(serialized)
        assert text == "hello world"

    def test_done_event_with_message(self):
        from dataclasses import asdict
        s = _make_span(
            "entity.fire.a",
            [
                SpanEvent(
                    name="done",
                    ts=0.0,
                    attributes={
                        "message": {
                            "content": [{"type": "text", "text": "final"}]
                        }
                    },
                ),
            ],
        )
        text = _extract_entity_output_text(asdict(s))
        assert text == "final"

    def test_no_events(self):
        from dataclasses import asdict
        s = _make_span("entity.fire.a", [])
        assert _extract_entity_output_text(asdict(s)) == ""

    def test_prefers_done_message_over_deltas(self):
        from dataclasses import asdict
        s = _make_span(
            "entity.fire.a",
            [
                SpanEvent(name="text_delta", ts=0.0, attributes={"delta": "partial"}),
                SpanEvent(
                    name="done",
                    ts=0.0,
                    attributes={
                        "message": {
                            "content": [{"type": "text", "text": "FINAL"}]
                        }
                    },
                ),
            ],
        )
        # done_event short-circuits and returns "FINAL"
        text = _extract_entity_output_text(asdict(s))
        assert text == "FINAL"


class TestReplayRunSummary:
    def test_empty_run(self, tmp_path, monkeypatch):
        # No file → empty
        summary = replay_run_summary("does-not-exist")
        assert summary["entity_count"] == 0
        assert summary["entities"] == []

    def test_real_run(self, tmp_runs_dir_setup):
        # Simulate recording: build a run with 1 entity that emitted text

        rid = str(uuid.uuid4())
        span = _make_span(
            "entity.fire.reviewer",
            [
                SpanEvent(name="text_delta", ts=0.0, attributes={"delta": "looks good"}),
            ],
            attrs={"entity": "reviewer", "step_id": f"{rid}:1"},
        )
        record_run(rid, [span])
        summary = replay_run_summary(rid)
        assert summary["entity_count"] == 1
        ent = summary["entities"][0]
        assert ent["node_id"] == "reviewer"
        assert ent["step_id"] == f"{rid}:1"
        assert ent["output_text"] == "looks good"


@pytest.fixture
def tmp_runs_dir_setup(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("PHAROS_RUNS_DIR", str(runs))
    import pharos.runtime as rt

    monkeypatch.setattr(rt, "_RUNS_DIR", runs)
    return runs