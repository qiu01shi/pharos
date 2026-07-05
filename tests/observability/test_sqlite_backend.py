"""Tests for the SQLite trace backend (cross-session run index)."""
from __future__ import annotations

from pharos.observability.backend.sqlite import SQLiteTraceBackend


def _span(sid: str, name: str, entity: str | None, dur: float) -> dict:
    attrs = {"entity": entity, "entity_class": "LLMAgent"} if entity else {}
    return {
        "id": sid,
        "name": name,
        "started_at": 100.0,
        "ended_at": 100.0 + dur / 1000,
        "duration_ms": dur,
        "status": "ok",
        "attributes": attrs,
    }


class TestSQLiteBackend:
    async def test_index_and_query_roundtrip(self, tmp_path):
        db = tmp_path / "trace.db"
        backend = SQLiteTraceBackend(db)
        await backend.index_run(
            "run-a",
            [_span("s1", "entity.fire.coder", "coder", 12.0)],
            director="fn",
            total_tokens=120,
            total_cost=0.0034,
            status="ok",
            recorded_at="2026-07-05T10:00:00",
        )
        rows = await backend.query_runs()
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-a"
        assert rows[0]["total_tokens"] == 120
        assert rows[0]["director"] == "fn"

    async def test_filter_by_entity(self, tmp_path):
        backend = SQLiteTraceBackend(tmp_path / "t.db")
        await backend.index_run(
            "run-coder",
            [_span("s1", "entity.fire.coder", "coder", 5.0)],
            recorded_at="2026-07-05T10:00:00",
        )
        await backend.index_run(
            "run-shell",
            [_span("s2", "entity.fire.runner", "runner", 5.0)],
            recorded_at="2026-07-05T11:00:00",
        )
        coder = await backend.query_runs(entity="coder")
        assert [r["run_id"] for r in coder] == ["run-coder"]

    async def test_filter_by_since_and_min_cost(self, tmp_path):
        backend = SQLiteTraceBackend(tmp_path / "t.db")
        await backend.index_run(
            "cheap-old",
            [_span("s1", "n", "a", 1.0)],
            total_cost=0.001,
            recorded_at="2026-07-01T00:00:00",
        )
        await backend.index_run(
            "pricey-new",
            [_span("s2", "n", "a", 1.0)],
            total_cost=0.50,
            recorded_at="2026-07-05T00:00:00",
        )
        since = await backend.query_runs(since="2026-07-03T00:00:00")
        assert [r["run_id"] for r in since] == ["pricey-new"]
        pricey = await backend.query_runs(min_cost=0.10)
        assert [r["run_id"] for r in pricey] == ["pricey-new"]

    async def test_reindex_is_idempotent(self, tmp_path):
        backend = SQLiteTraceBackend(tmp_path / "t.db")
        for _ in range(3):
            await backend.index_run(
                "same",
                [_span("s1", "entity.fire.a", "a", 2.0)],
                recorded_at="2026-07-05T00:00:00",
            )
        rows = await backend.query_runs()
        assert len(rows) == 1  # INSERT OR REPLACE, not duplicated
        by_entity = await backend.query_runs(entity="a")
        assert len(by_entity) == 1  # spans re-indexed, not duplicated
