"""Tests for OTLP/JSON export of pharos spans."""
from __future__ import annotations

import json

from pharos.observability.otlp import (
    span_to_otlp,
    to_otlp,
    write_otlp_json,
)
from pharos.observability.trace import Span, SpanEvent


def _span(**kw) -> Span:
    base = dict(
        id="span_abc",
        trace_id="trace_0001",
        parent_span_id=None,
        name="fire",
        started_at=1000.0,
        ended_at=1000.5,
        status="ok",
    )
    base.update(kw)
    return Span(**base)


class TestSpanToOtlp:
    def test_ids_are_hex_of_correct_length(self):
        o = span_to_otlp(_span())
        assert len(o["traceId"]) == 32  # 16 bytes
        assert len(o["spanId"]) == 16   # 8 bytes
        int(o["traceId"], 16)  # valid hex
        int(o["spanId"], 16)

    def test_ids_are_deterministic(self):
        a = span_to_otlp(_span())
        b = span_to_otlp(_span())
        assert a["traceId"] == b["traceId"]
        assert a["spanId"] == b["spanId"]

    def test_parent_span_id_only_when_present(self):
        assert "parentSpanId" not in span_to_otlp(_span())
        child = span_to_otlp(_span(id="span_child", parent_span_id="span_abc"))
        assert child["parentSpanId"] == span_to_otlp(_span())["spanId"]

    def test_timestamps_in_nanoseconds(self):
        o = span_to_otlp(_span())
        assert o["startTimeUnixNano"] == str(int(1000.0 * 1e9))
        assert o["endTimeUnixNano"] == str(int(1000.5 * 1e9))

    def test_status_mapping(self):
        assert span_to_otlp(_span(status="ok"))["status"]["code"] == 1
        err = span_to_otlp(_span(status="error", error="boom"))
        assert err["status"]["code"] == 2
        assert err["status"]["message"] == "boom"

    def test_attribute_typing(self):
        s = _span()
        s.set_attributes(
            {"s": "text", "i": 7, "f": 1.5, "b": True, "obj": {"k": "v"}}
        )
        attrs = {a["key"]: a["value"] for a in span_to_otlp(s)["attributes"]}
        assert attrs["s"] == {"stringValue": "text"}
        assert attrs["i"] == {"intValue": "7"}
        assert attrs["f"] == {"doubleValue": 1.5}
        assert attrs["b"] == {"boolValue": True}
        assert "stringValue" in attrs["obj"]  # complex -> JSON string

    def test_events_converted(self):
        s = _span()
        s.events.append(SpanEvent(name="first_token", ts=1000.1, attributes={"d": "hi"}))
        ev = span_to_otlp(s)["events"][0]
        assert ev["name"] == "first_token"
        assert ev["timeUnixNano"] == str(int(1000.1 * 1e9))

    def test_accepts_persisted_dict(self):
        d = {
            "id": "span_abc",
            "trace_id": "trace_0001",
            "parent_span_id": None,
            "name": "fire",
            "started_at": 1000.0,
            "ended_at": 1000.5,
            "status": "ok",
            "attributes": {"x": 1},
            "events": [{"name": "e", "ts": 1000.2, "attributes": {}}],
            "error": None,
        }
        o = span_to_otlp(d)
        assert o["spanId"] == span_to_otlp(_span())["spanId"]
        assert o["events"][0]["name"] == "e"


class TestToOtlpEnvelope:
    def test_resource_and_scope_shape(self):
        env = to_otlp([_span()], service_name="pharos:run1")
        rs = env["resourceSpans"][0]
        svc = {a["key"]: a["value"] for a in rs["resource"]["attributes"]}
        assert svc["service.name"] == {"stringValue": "pharos:run1"}
        scope = rs["scopeSpans"][0]
        assert scope["scope"]["name"] == "pharos"
        assert len(scope["spans"]) == 1

    def test_multiple_spans_grouped(self):
        env = to_otlp([_span(), _span(id="s2", parent_span_id="span_abc")])
        assert len(env["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2


class TestWriteOtlpJson:
    def test_writes_valid_json(self, tmp_path):
        p = write_otlp_json([_span()], tmp_path / "trace.json")
        data = json.loads(p.read_text())
        assert data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] == "fire"
