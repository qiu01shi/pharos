"""Trace storage must not retain common credential forms."""

from __future__ import annotations

import json
import stat

from pharos.observability.redaction import REDACTED, redact
from pharos.observability.trace import Span


def test_recursive_secret_key_and_bearer_redaction():
    value = {
        "headers": {
            "Authorization": "Bearer abcdefghijklmnop",
            "x-api-key": "secret-value",
        },
        "prompt": "do not reveal sk-abcdefgh12345678",
        "tokens": 42,
    }
    result = redact(value)
    assert result["headers"]["Authorization"] == REDACTED
    assert result["headers"]["x-api-key"] == REDACTED
    assert REDACTED in result["prompt"]
    assert result["tokens"] == 42


def test_span_setters_redact_after_creation():
    span = Span(
        id="span",
        trace_id="trace",
        parent_span_id=None,
        name="test",
        started_at=0.0,
    )
    span.set_attribute("password", "hunter2")
    span.set_attributes({"nested": {"client_secret": "value"}})
    span.record_event("request", {"Authorization": "Bearer abcdefghijkl"})
    assert span.attributes["password"] == REDACTED
    assert span.attributes["nested"]["client_secret"] == REDACTED
    assert span.events[0].attributes["Authorization"] == REDACTED


def test_run_record_file_is_private(tmp_path, monkeypatch):
    import pharos.runtime as runtime

    monkeypatch.setattr(runtime, "_RUNS_DIR", tmp_path / "runs")
    runtime.record_run("private", [], outputs={})
    path = runtime._run_path("private")
    assert json.loads(path.read_text())["version"] == 2
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
