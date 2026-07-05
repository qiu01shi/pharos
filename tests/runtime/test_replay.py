"""Tests for the runtime / Replay module."""
from __future__ import annotations

import json
import time

import pytest

from pharos.observability.trace import InMemoryTracer, Span
from pharos.runtime import (
    _MAX_RUNS,
    export_run_json,
    get_run,
    list_runs,
    record_run,
)


@pytest.fixture
def tmp_runs_dir(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("PHAROS_RUNS_DIR", str(runs))
    # Re-import to pick up the new env var (the module reads it at
    # import time for the default _RUNS_DIR). For safety, also
    # monkeypatch the module-level constant.
    import pharos.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "_RUNS_DIR", runs)
    return runs


def _make_run(name: str = "test_run") -> tuple[str, list[Span]]:
    """Helper: build a fake run with 2 spans."""
    rid = f"run-{name}"
    tracer = InMemoryTracer()
    s1 = tracer.start_span("root", attributes={"k": "v"})
    s1.end()
    s2 = tracer.start_span("child", parent=s1)
    s2.end()
    tracer.finish_span(s1)
    tracer.finish_span(s2)
    return rid, tracer.spans


class TestRecordRun:
    def test_writes_file(self, tmp_runs_dir):
        rid, spans = _make_run("a")
        record_run(rid, spans)
        path = tmp_runs_dir / f"{rid}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == rid
        assert len(data["spans"]) == 2
        assert data["spans"][0]["name"] == "root"

    def test_overwrites_existing(self, tmp_runs_dir):
        rid, spans1 = _make_run("x")
        record_run(rid, spans1)
        rid2, spans2 = _make_run("x")
        record_run(rid2, spans2)
        # Only one file
        files = list(tmp_runs_dir.glob("*.json"))
        assert len(files) == 1

    def test_cap_evicts_oldest(self, tmp_runs_dir):
        # Write more than _MAX_RUNS
        for i in range(_MAX_RUNS + 5):
            rid, spans = _make_run(f"r{i}")
            record_run(rid, spans)
        # Now only _MAX_RUNS should remain
        remaining = list(tmp_runs_dir.glob("*.json"))
        assert len(remaining) == _MAX_RUNS


class TestListRuns:
    def test_empty(self, tmp_runs_dir):
        assert list_runs() == []

    def test_one_run(self, tmp_runs_dir):
        rid, spans = _make_run("only")
        record_run(rid, spans)
        runs = list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == rid
        assert runs[0]["span_count"] == 2

    def test_most_recent_first(self, tmp_runs_dir):
        for i in range(3):
            rid, spans = _make_run(f"r{i}")
            record_run(rid, spans)
            time.sleep(0.01)  # ensure distinct mtimes
        runs = list_runs()
        # r2 (last) should be first
        assert runs[0]["run_id"].endswith("r2")


class TestGetRun:
    def test_existing(self, tmp_runs_dir):
        rid, spans = _make_run()
        record_run(rid, spans)
        got = get_run(rid)
        assert got is not None
        assert len(got) == 2
        assert got[0]["name"] == "root"

    def test_missing(self, tmp_runs_dir):
        assert get_run("does-not-exist") is None


class TestExportRunJson:
    def test_export(self, tmp_runs_dir):
        rid, spans = _make_run("export")
        record_run(rid, spans)
        out = tmp_runs_dir / "exported.json"
        export_run_json(rid, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["run_id"] == rid

    def test_export_missing_raises(self, tmp_runs_dir):
        with pytest.raises(KeyError):
            export_run_json("nope", tmp_runs_dir / "x.json")
