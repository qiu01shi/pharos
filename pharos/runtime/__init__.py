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

import contextlib
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

    Returns a dict mapping `(node_id, step_index)` → dict of the
    full recorded StreamEvent sequence + final message + usage.
    Used by `pharos replay --re-run` to seed the same outputs
    without a network call.

    Each LLM step in the trace produces a sequence of StreamEvents
    (text_delta, thinking_delta, toolcall_end, done). We capture
    all of them so the ReplayProvider can replay the exact event
    sequence.

    The returned dict has shape:
        {
            "agent:1": {
                "node_id": "agent",
                "step_index": 1,
                "events": [StreamEvent_dict, ...],   # full event list
                "final_text": "...",                  # convenience
                "usage": {...},                       # final usage dict
                "model": "glm-4.5-air",
                "provider": "glm",
            },
            ...
        }
    """
    spans = get_run(run_id) or []
    # Group by (node_id, step_index); a node can fire multiple
    # times in a single run (SDF iterations).
    by_key: dict[str, dict[str, Any]] = {}
    for s in spans:
        name = s.get("name", "")
        if not name.startswith("entity.fire."):
            continue
        attrs = s.get("attributes", {})
        node_id = attrs.get("entity", name.removeprefix("entity.fire."))
        step_id = attrs.get("step_id", "")
        # Step index: the trailing `:N` after `<run_id>:N`
        step_index = 0
        if ":" in step_id:
            with contextlib.suppress(ValueError):
                step_index = int(step_id.rsplit(":", 1)[-1])
        # Class filter: only LLM-class entities are replay-able
        cls = attrs.get("entity_class", "")
        if cls != "LLMAgent":
            continue

        events: list[dict[str, Any]] = []
        final_text = ""
        usage_dict: dict[str, Any] = {}
        model = ""
        provider = ""
        for ev in s.get("events", []):
            ename = ev.get("name", "")
            ev_attrs = ev.get("attributes", {})
            events.append({"type": ename, **ev_attrs})
            if ename == "done":
                msg = ev_attrs.get("message") or {}
                if isinstance(msg, dict):
                    parts = []
                    for b in msg.get("content", []):
                        if isinstance(b, dict) and b.get("text"):
                            parts.append(b["text"])
                    final_text = "".join(parts)
                    model = msg.get("model", "")
                    provider = msg.get("provider", "")
                    u = msg.get("usage") or {}
                    if isinstance(u, dict):
                        usage_dict = u
            elif ename == "text_delta":
                final_text += str(ev_attrs.get("delta") or "")

        key = f"{node_id}:{step_index}"
        by_key[key] = {
            "node_id": node_id,
            "step_index": step_index,
            "events": events,
            "final_text": final_text,
            "usage": usage_dict,
            "model": model,
            "provider": provider,
        }
    return by_key


# ---------- P1/P2 functions (unchanged) ----------


def record_run(
    run_id: str,
    spans: list[Any],
    *,
    outputs: dict[str, list[dict[str, Any]]] | None = None,
    director: str | None = None,
) -> None:
    _ensure_dir()
    payload = {
        "run_id": run_id,
        "recorded_at": datetime.now().isoformat(),
        "director": director or "",
        # General entity-output cache (node_id:fire_index -> [emitted values]).
        # Enables byte-equal replay of ANY entity (LLM / shell / python /
        # tool), not just LLM providers.
        "outputs": outputs or {},
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
    _run_path(run_id).write_text(json.dumps(payload, indent=2, default=str))
    _enforce_cap()


def get_run_outputs(run_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return the recorded entity-output cache for a run (empty if none)."""
    p = _run_path(run_id)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    outputs = data.get("outputs", {})
    return outputs if isinstance(outputs, dict) else {}


def get_run_director(run_id: str) -> str:
    """Return the director name recorded for a run (empty if unknown)."""
    p = _run_path(run_id)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("director", ""))


class RunRecorder:
    """Captures each entity's emitted output tokens during a run.

    Keyed by ``"<node_id>:<fire_index>"`` so a node that fires many times
    (SDF/DE iterations) records each fire separately. Directors call
    ``capture()`` from ``safe_fire`` right after ``fire()`` returns, while
    the tokens still sit in the entity's output ports (delivery happens
    later). The result is a provider-agnostic replay cache.
    """

    def __init__(
        self,
        prefix: str = "",
        data: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        # `prefix` namespaces keys for nested (subgraph) runs, e.g.
        # "reviewer/" -> "reviewer/coder:0". `data` lets a child recorder
        # write into the parent's shared dict.
        self.prefix = prefix
        self._data: dict[str, list[dict[str, Any]]] = (
            data if data is not None else {}
        )

    def child(self, prefix: str) -> RunRecorder:
        """A recorder that writes into the same dict under a nested prefix."""
        return RunRecorder(prefix=f"{self.prefix}{prefix}", data=self._data)

    def capture(self, entity: Any, node_id: str, fire_index: int) -> None:
        emitted: list[dict[str, Any]] = []
        for port_name, port in entity.outs.items():
            for tok in port.peek_all():
                emitted.append(
                    {
                        "port": port_name,
                        "type": tok.value.type,
                        "payload": tok.value.payload,
                        # Lineage travels with the recording so a fixture
                        # carries the hash chain (used by chain_digest / diff).
                        "self_hash": tok.self_hash,
                        "prev_hash": tok.prev_hash,
                        "origin": tok.origin,
                    }
                )
        self._data[f"{self.prefix}{node_id}:{fire_index}"] = emitted

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return dict(self._data)


class RunReplayer:
    """Replays recorded entity outputs onto ports, skipping execution.

    ``safe_fire`` consults ``has()`` and, when a fire was recorded, calls
    ``apply()`` to re-emit the cached tokens instead of running the entity.
    Because it works at the entity output boundary, it replays shell
    commands, python entities, and tools too — not only LLM calls.
    """

    def __init__(
        self, data: dict[str, list[dict[str, Any]]], resume: bool = False
    ) -> None:
        self._data = data
        # Resume mode: fires WITHOUT a recorded output run live instead of
        # emitting nothing. This turns a full-run replayer into a "continue
        # from where recording stopped" checkpoint — completed fires replay
        # cheaply, remaining fires execute for real.
        self.resume = resume

    @classmethod
    def load(
        cls, run_id: str, resume: bool = False
    ) -> RunReplayer | None:
        outputs = get_run_outputs(run_id)
        if not outputs:
            return None
        return cls(outputs, resume=resume)

    def has(self, node_id: str, fire_index: int) -> bool:
        return f"{node_id}:{fire_index}" in self._data

    def apply(self, entity: Any, node_id: str, fire_index: int) -> None:
        from pharos.core.token import TypedValue

        for rec in self._data.get(f"{node_id}:{fire_index}", []):
            port = entity.outs.get(rec["port"])
            if port is not None:
                port.emit(
                    TypedValue(type=rec["type"], payload=rec["payload"])
                )


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
    spans: list[dict[str, Any]] = data.get("spans", [])
    return spans


def export_run_json(run_id: str, path: Path) -> None:
    src = _run_path(run_id)
    if not src.exists():
        raise KeyError(f"no run with id {run_id!r}")
    path.write_text(src.read_text())


__all__ = [
    "RunRecorder",
    "RunReplayer",
    "export_run_json",
    "extract_cached_outputs",
    "get_run",
    "get_run_director",
    "get_run_outputs",
    "list_runs",
    "record_run",
    "replay_run_summary",
]