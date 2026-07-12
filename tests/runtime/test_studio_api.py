"""Local Runtime API integration tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from pharos.studio_api import RunManager, create_app

GRAPH = """
apiVersion: pharos.ai/v1
kind: Graph
name: studio-api-demo
director: fn
nodes:
  - id: agent
    type: llm
    provider: faux
    model: faux-echo
edges:
  - { src: __in__.prompt, dst: agent.prompt }
  - { src: agent.text, dst: __out__.response }
"""

SHELL_GRAPH = """
apiVersion: pharos.ai/v1
kind: Graph
name: denied-shell
director: fn
nodes:
  - id: runner
    type: shell
edges:
  - { src: __in__.command, dst: runner.command }
  - { src: runner.stdout, dst: __out__.response }
"""


@pytest.fixture
def api_app(tmp_path, monkeypatch):
    import pharos.observability.backend.sqlite as sqlite_backend
    import pharos.runtime as runtime

    runs = tmp_path / "runs"
    monkeypatch.setattr(runtime, "_RUNS_DIR", runs)
    monkeypatch.setattr(sqlite_backend, "_DEFAULT_DB", tmp_path / "trace.db")
    return create_app(RunManager())


@pytest.mark.asyncio
async def test_health_and_canonical_validation(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        valid = await client.post(
            "/api/graphs/validate", json={"graph": GRAPH}
        )
        assert valid.status_code == 200
        assert valid.json() == {
            "valid": True,
            "name": "studio-api-demo",
            "director": "fn",
            "node_count": 1,
            "edge_count": 2,
        }


@pytest.mark.asyncio
async def test_validation_rejects_invalid_port(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/graphs/validate",
            json={"graph": GRAPH.replace("agent.prompt", "agent.missing")},
        )
    assert response.status_code == 422
    assert "missing" in str(response.json()["detail"])


@pytest.mark.asyncio
async def test_create_run_streams_fire_events_and_outputs(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/runs",
            json={"graph": GRAPH, "input": "hello from studio"},
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        for _ in range(100):
            snapshot = (await client.get(f"/api/runs/{run_id}")).json()
            if snapshot["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)

        assert snapshot["status"] == "completed"
        assert snapshot["outputs"]["__out__"]["response"] == [
            "echo: hello from studio"
        ]
        event_types = [event["type"] for event in snapshot["events"]]
        assert event_types == [
            "run.queued",
            "run.started",
            "fire.started",
            "fire.completed",
            "run.completed",
        ]
        assert snapshot["spans"][0]["attributes"]["entity"] == "agent"

        streamed = await client.get(f"/api/runs/{run_id}/events")
        assert streamed.status_code == 200
        assert "fire.completed" in streamed.text
        assert "run.completed" in streamed.text

        resumed = await client.get(
            f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "3"}
        )
        assert "run.queued" not in resumed.text
        assert "fire.completed" in resumed.text


@pytest.mark.asyncio
async def test_permission_failure_is_visible_as_a_fire_event(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/runs",
            json={"graph": SHELL_GRAPH, "inputs": {"command": "echo safe"}},
        )
        run_id = created.json()["run_id"]
        for _ in range(100):
            snapshot = (await client.get(f"/api/runs/{run_id}")).json()
            if snapshot["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)

    assert snapshot["status"] == "failed"
    event_types = [event["type"] for event in snapshot["events"]]
    assert event_types == [
        "run.queued",
        "run.started",
        "fire.started",
        "fire.failed",
        "run.failed",
    ]
    assert "shell:execute" in snapshot["events"][3]["error"]
