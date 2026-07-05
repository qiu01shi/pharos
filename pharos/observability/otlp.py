"""OTLP export — turn pharos spans into OpenTelemetry OTLP/JSON.

pharos keeps its trace schema OTel-shaped but dependency-free. This module
is the bridge to the wider ecosystem: it serialises spans into the OTLP
``ExportTraceServiceRequest`` JSON shape so a run can be inspected in Jaeger /
Tempo / any OTLP-compatible collector, either by writing a file or POSTing to
an OTLP/HTTP endpoint.

It accepts both live ``Span`` objects and the plain dicts persisted in
``~/.pharos/runs/<id>.json`` (so `pharos trace <id> --otlp out.json` works on
a recorded run without re-executing it).

Notes on the mapping:
  * OTLP ids are hex (16-byte trace, 8-byte span). pharos ids are opaque
    strings, so we hash them to stable hex — good enough to preserve the
    parent/child tree, which is what viewers need.
  * Timestamps are converted to Unix nanoseconds.
  * Status maps ok->OK(1), error->ERROR(2), else UNSET(0).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SCHEMA_URL = "https://opentelemetry.io/schemas/1.20.0"
_SCOPE_NAME = "pharos"
_STATUS_CODE = {"unset": 0, "ok": 1, "error": 2}


def _hex_id(value: str, n_bytes: int) -> str:
    """Deterministic `n_bytes`-byte hex id derived from an opaque string."""
    return hashlib.sha256(value.encode()).hexdigest()[: n_bytes * 2]


def _unix_nano(ts: float | None) -> str:
    """OTLP wants Unix nanoseconds as a string; None -> "0"."""
    if ts is None:
        return "0"
    return str(int(ts * 1_000_000_000))


def _any_value(v: Any) -> dict[str, Any]:
    """Wrap a Python value in an OTLP AnyValue."""
    # bool is a subclass of int — check it first.
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    return {"stringValue": json.dumps(v, default=str, ensure_ascii=False)}


def _attributes(attrs: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [{"key": k, "value": _any_value(v)} for k, v in (attrs or {}).items()]


def _as_dict(span: Any) -> dict[str, Any]:
    """Normalise a live Span object or a persisted span dict to one shape."""
    if isinstance(span, dict):
        return span
    events = []
    for e in getattr(span, "events", []) or []:
        if isinstance(e, dict):
            events.append(e)
        else:
            events.append(
                {"name": e.name, "ts": e.ts, "attributes": e.attributes}
            )
    return {
        "id": span.id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "status": span.status,
        "attributes": span.attributes,
        "events": events,
        "error": span.error,
    }


def span_to_otlp(span: Any) -> dict[str, Any]:
    """Convert one span (object or dict) to an OTLP span dict."""
    d = _as_dict(span)
    out: dict[str, Any] = {
        "traceId": _hex_id(str(d.get("trace_id", "")), 16),
        "spanId": _hex_id(str(d.get("id", "")), 8),
        "name": d.get("name", ""),
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": _unix_nano(d.get("started_at")),
        "endTimeUnixNano": _unix_nano(d.get("ended_at") or d.get("started_at")),
        "attributes": _attributes(d.get("attributes")),
        "events": [
            {
                "name": e.get("name", ""),
                "timeUnixNano": _unix_nano(e.get("ts")),
                "attributes": _attributes(e.get("attributes")),
            }
            for e in (d.get("events") or [])
        ],
    }
    parent = d.get("parent_span_id")
    if parent:
        out["parentSpanId"] = _hex_id(str(parent), 8)
    status: dict[str, Any] = {"code": _STATUS_CODE.get(d.get("status", "unset"), 0)}
    if d.get("error"):
        status["message"] = str(d["error"])
    out["status"] = status
    return out


def to_otlp(spans: list[Any], service_name: str = "pharos") -> dict[str, Any]:
    """Build a full OTLP ExportTraceServiceRequest for a list of spans."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attributes({"service.name": service_name})
                },
                "scopeSpans": [
                    {
                        "scope": {"name": _SCOPE_NAME},
                        "spans": [span_to_otlp(s) for s in spans],
                        "schemaUrl": _SCHEMA_URL,
                    }
                ],
                "schemaUrl": _SCHEMA_URL,
            }
        ]
    }


def write_otlp_json(
    spans: list[Any], path: str | Path, service_name: str = "pharos"
) -> Path:
    """Serialise spans to an OTLP/JSON file. Returns the written path."""
    p = Path(path)
    p.write_text(
        json.dumps(to_otlp(spans, service_name), indent=2, ensure_ascii=False)
    )
    return p


def post_otlp(
    spans: list[Any],
    endpoint: str,
    *,
    service_name: str = "pharos",
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> int:
    """POST spans to an OTLP/HTTP traces endpoint (e.g. a collector at
    ``http://localhost:4318/v1/traces``). Returns the HTTP status code.

    Requires network access to a running collector, so it is not exercised
    by the offline test suite.
    """
    import httpx

    payload = to_otlp(spans, service_name)
    resp = httpx.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json", **(headers or {})},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.status_code


__all__ = [
    "post_otlp",
    "span_to_otlp",
    "to_otlp",
    "write_otlp_json",
]
