"""SQLite trace backend — cross-session, queryable run history.

Per-run JSON files (``~/.pharos/runs/<id>.json``) are great for inspecting a
single run but poor for questions across runs ("which runs used the coder
agent?", "what did I spend today?"). This backend indexes each run into a
small SQLite database so ``pharos trace query`` can filter by entity, time,
and cost without scanning every JSON file.

It uses ``aiosqlite`` (already a project dependency). The indexing hook lives
in the async run path; the CLI query wraps the async API in ``asyncio.run``.
Indexing is best-effort: a trace-store failure must never fail a run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiosqlite

_DEFAULT_DB = Path(
    os.environ.get(
        "PHAROS_TRACE_DB", str(Path.home() / ".pharos" / "trace.db")
    )
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    recorded_at    TEXT,
    director       TEXT,
    span_count     INTEGER,
    started_at     REAL,
    ended_at       REAL,
    duration_ms    REAL,
    total_tokens   INTEGER,
    total_cost_usd REAL,
    status         TEXT,
    error          TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    span_id      TEXT,
    run_id       TEXT,
    name         TEXT,
    entity       TEXT,
    entity_class TEXT,
    duration_ms  REAL,
    status       TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
CREATE INDEX IF NOT EXISTS idx_spans_entity ON spans(entity);
"""


def _get(span: Any, key: str, default: Any = None) -> Any:
    """Read a field from a Span object or an equivalent dict."""
    if isinstance(span, dict):
        return span.get(key, default)
    return getattr(span, key, default)


class SQLiteTraceBackend:
    """Indexes runs into SQLite and answers cross-run queries."""

    name = "sqlite"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_DB

    async def _connect(self) -> aiosqlite.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(self.path))
        await db.executescript(_SCHEMA)
        return db

    async def index_run(
        self,
        run_id: str,
        spans: list[Any],
        *,
        director: str = "",
        total_tokens: int = 0,
        total_cost: float = 0.0,
        status: str = "ok",
        error: str | None = None,
        recorded_at: str = "",
    ) -> None:
        """Upsert one run's summary and its entity spans."""
        started = min((_get(s, "started_at", 0.0) for s in spans), default=0.0)
        ended = max((_get(s, "ended_at", 0.0) for s in spans), default=0.0)
        duration_ms = (ended - started) * 1000 if ended and started else 0.0

        db = await self._connect()
        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, recorded_at, director, span_count, started_at,
                    ended_at, duration_ms, total_tokens, total_cost_usd,
                    status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    recorded_at,
                    director,
                    len(spans),
                    started,
                    ended,
                    duration_ms,
                    total_tokens,
                    total_cost,
                    status,
                    error,
                ),
            )
            # Re-index this run's spans from scratch (idempotent upsert).
            await db.execute("DELETE FROM spans WHERE run_id = ?", (run_id,))
            for s in spans:
                attrs = _get(s, "attributes", {}) or {}
                await db.execute(
                    """
                    INSERT INTO spans (
                        span_id, run_id, name, entity, entity_class,
                        duration_ms, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _get(s, "id"),
                        run_id,
                        _get(s, "name"),
                        attrs.get("entity"),
                        attrs.get("entity_class"),
                        _get(s, "duration_ms", 0.0),
                        _get(s, "status", "ok"),
                    ),
                )
            await db.commit()
        finally:
            await db.close()

    async def query_runs(
        self,
        entity: str | None = None,
        since: str | None = None,
        min_cost: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return run summaries matching the filters, newest first."""
        db = await self._connect()
        try:
            db.row_factory = aiosqlite.Row
            sql = "SELECT DISTINCT r.* FROM runs r"
            clauses: list[str] = []
            params: list[Any] = []
            if entity:
                sql += " JOIN spans s ON s.run_id = r.run_id"
                clauses.append("s.entity = ?")
                params.append(entity)
            if since:
                clauses.append("r.recorded_at >= ?")
                params.append(since)
            if min_cost is not None:
                clauses.append("r.total_cost_usd >= ?")
                params.append(min_cost)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY r.recorded_at DESC LIMIT ?"
            params.append(limit)
            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            await db.close()


__all__ = ["SQLiteTraceBackend"]
