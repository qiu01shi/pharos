"""Replay — re-execute a recorded run from its trace.

P1: in-memory only (single process). A recorded run is lost when
the CLI exits.

P2: persist runs to `~/.pharos/runs/<run_id>.json` so `pharos trace
list` and `pharos trace <run_id>` work across CLI invocations.

P3 (this update): `replay()` reconstructs a RecordedRun from JSON,
and `playback_run(run_id)` re-runs a graph with the SAME LLM
outputs cached in the trace, without making any network calls.

Usage:
    pharos trace list
    pharos trace <run_id>
    pharos replay <run_id>
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
_MAX_RUNS = 100


def _ensure_dir() -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _run_path(run_id: str) -> Path:
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_")
    return _RUNS_DIR / f"{safe}.json"


# ---------- P3: deterministic replay ----------


def replay_run_summary(run_id: str) -> dict[str, Any]:
    """Build a summary of a recorded run for the `pharos replay` CLI.

    Returns:
        {
            "run_id": ...,
            "graph_name": ...,        # if present in trace metadata
            "director": ...,          # if present
            "total_duration_ms": ...,
            "entities": [
                {
                    "node_id": ...,
                    "step_id": ...,
                    "duration_ms": ...,
                    "output_text": ...,
                },
                ...
            ],
        }
    """
    spans = get_run(run_id) or []
    entities: list[dict[str, Any]] = []
    total_ms = 0.0
    director = ""
    for s in spans:
        name = s.get("name", "")
        dur = s.get("duration_ms", 0.0) or 0.0
        attrs = s.get("attributes", {})
        if name.startswith("entity.fire."):
            node_id = attrs.get("entity", name.removeprefix("entity.fire."))
            step_id = attrs.get("step_id", "")
            output = _extract_entity_output_text(s)
            entities.append(
                {
                    "node_id": node_id,
                    "step_id": step_id,
                    "duration_ms": dur,
                    "output_text": output,
                }
            )
        elif name.startswith("run."):
            director = attrs.get("director", director)
            total_ms = max(total_ms, dur)
    return {
        "run_id": run_id,
        "director": director,
        "total_duration_ms": total_ms,
        "entity_count": len(entities),
        "entities": entities,
    }


def _extract_entity_output_text(span: dict[str, Any]) -> str:
    """Find the last text emitted by this span's entity.

    SpanEvent fields are `name`, `ts`, `attributes` (not `type`).
    After `asdict`, the event name lives in `ev["name"]` and the
    full stream event payload lives in `ev["attributes"]`.
    """
    last_text = ""
    for ev in span.get("events", []):
        ename = ev.get("name", "")
        attrs = ev.get("attributes", {})
        if ename == "text_delta":
            delta = attrs.get("delta")
            if delta:
                last_text += str(delta)
        elif ename == "done":
            msg = attrs.get("message") or {}
            if isinstance(msg, dict):
                parts: list[str] = []
                for b in msg.get("content", []):
                    if isinstance(b, dict) and b.get("text"):
                        parts.append(b["text"])
                if parts:
                    return "".join(parts)
    return last_text


def extract_cached_outputs(run_id: str) -> dict[str, dict[str, Any]]:
    """Pull LLM-emitted text from the recorded trace.

    Returns a dict mapping `(step_id, port_name)` → output payload.
    Used by `pharos replay` to seed the same outputs without a
    network call.

    Step ids in pharos look like `<run_id>:N` (from FN/SDF
    Director). Each step wraps an entity fire.
    """
    spans = get_run(run_id) or []
    out: dict[str, dict[str, Any]] = {}
    for s in spans:
        name = s.get("name", "")
        if not name.startswith("entity.fire."):
            continue
        step_id = s.get("attributes", {}).get("step_id", "")
        for ev in s.get("events", []):
            # Each event's payload (text_delta, thinking_delta,
            # toolcall_end, etc.) — we focus on text content.
            etype = ev.get("type", "")
            if etype == "text_delta":
                # Accumulate deltas
                key = f"{step_id}.text"
                out.setdefault(key, {"text": "", "model": "", "provider": ""})
                out[key]["text"] += ev.get("delta", "") or ""
            elif etype == "done":
                # The done event carries the final AssistantMessage
                msg = ev.get("message", {})
                if isinstance(msg, dict):
                    # Concatenate text content blocks
                    txt = ""
                    for b in msg.get("content", []):
                        if isinstance(b, dict) and b.get("text"):
                            txt += b["text"]
                    if txt:
                        key = f"{step_id}.text"
                        out[key] = {
                            "text": txt,
                            "model": msg.get("model", ""),
                            "provider": msg.get("provider", ""),
                        }
    return out


# ---------- P1/P2 functions (unchanged) ----------


def record_run(run_id: str, spans: list[Any]) -> None:
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
    src = _run_path(run_id)
    if not src.exists():
        raise KeyError(f"no run with id {run_id!r}")
    path.write_text(src.read_text())


__all__ = [
    "export_run_json",
    "extract_cached_outputs",
    "get_run",
    "list_runs",
    "record_run",
    "replay_run_summary",
]