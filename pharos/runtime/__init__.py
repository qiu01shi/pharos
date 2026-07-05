"""Replay — re-execute a recorded run from its trace.

P1: in-memory only (single process). A recorded run is lost when
the CLI exits.

P2: persist runs to `~/.pharos/runs/<run_id>.json` so `pharos trace
list` and `pharos trace <run_id>` work across CLI invocations.

Usage:
    pharos trace list
    pharos trace <run_id>
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

_RUNS_DIR = Path(
    os.environ.get("PHAROS_RUNS_DIR", str(Path.home() / ".pharos" / "runs"))
)
# Cap on number of recent runs to keep on disk
_MAX_RUNS = 100


def _ensure_dir() -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _run_path(run_id: str) -> Path:
    # Sanitize run_id (uuid4 is safe, but be defensive)
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_")
    return _RUNS_DIR / f"{safe}.json"


def record_run(run_id: str, spans: list[Any]) -> None:
    """Persist a run's spans to disk.

    A run is a JSON file under `~/.pharos/runs/<run_id>.json` containing
    the full span tree. We keep at most `_MAX_RUNS` recent runs; the
    oldest is deleted on each new record.
    """
    _ensure_dir()
    payload = {
        "run_id": run_id,
        "recorded_at": datetime.now().isoformat(),
        "spans": [
            {
                "id": s.id,
                "trace_id": s.trace_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "attributes": s.attributes,
                "events": [asdict(e) for e in s.events],
                "error": s.error,
            }
            for s in spans
        ],
    }
    _run_path(run_id).write_text(json.dumps(payload, indent=2))
    _enforce_cap()


def _enforce_cap() -> None:
    """Delete oldest runs if we have more than _MAX_RUNS."""
    _ensure_dir()
    runs = sorted(
        _RUNS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    import contextlib
    with contextlib.suppress(OSError):
        for p in runs[_MAX_RUNS:]:
            p.unlink()


def list_runs() -> list[dict[str, Any]]:
    """Return all recorded runs (most recent first)."""
    _ensure_dir()
    out: list[dict[str, Any]] = []
    for p in _RUNS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        spans = data.get("spans", [])
        if not spans:
            continue
        out.append(
            {
                "run_id": data["run_id"],
                "recorded_at": data.get("recorded_at", ""),
                "span_count": len(spans),
                "started_at": min(s.get("started_at", 0) for s in spans),
                "ended_at": max(
                    s.get("ended_at") or s.get("started_at", 0) for s in spans
                ),
            }
        )
    return sorted(out, key=lambda r: r["started_at"], reverse=True)


def get_run(run_id: str) -> list[dict[str, Any]] | None:
    p = _run_path(run_id)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("spans", [])


def export_run_json(run_id: str, path: Path) -> None:
    """Copy a run's file to `path` (caller-chosen location)."""
    src = _run_path(run_id)
    if not src.exists():
        raise KeyError(f"no run with id {run_id!r}")
    path.write_text(src.read_text())


__all__ = [
    "export_run_json",
    "get_run",
    "list_runs",
    "record_run",
]
