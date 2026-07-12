"""Heterogeneous worker protocol adapters."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.container import ContainerEntity
from pharos.entities.remote import RemoteEntity
from pharos.ir import load_graph_from_dict
from pharos.worker import WORKER_PROTOCOL_VERSION


class TestRemoteEntity:
    async def test_http_worker_protocol_and_idempotency(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            assert request.headers["Idempotency-Key"]
            return httpx.Response(
                200,
                json={
                    "protocol": WORKER_PROTOCOL_VERSION,
                    "outputs": [
                        {"port": "output", "type": "json", "payload": {"ok": True}}
                    ],
                    "metadata": {"worker": "typescript"},
                },
            )

        entity = RemoteEntity(
            "remote",
            "https://worker.test/fire",
            transport=httpx.MockTransport(handler),
        )
        graph = CompositeGraph("remote")
        graph.add_entity("remote", entity)
        entity.ins["input"].emit(TypedValue(type="json", payload={"task": "x"}))

        result = await FNDirector().run(
            graph,
            RunContext(run_id="run-1", granted_permissions={"net:connect"}),
        )

        assert result.converged
        assert captured["protocol"] == WORKER_PROTOCOL_VERSION
        assert captured["node_id"] == "remote"
        assert captured["inputs"][0]["payload"] == {"task": "x"}
        assert entity.outs["output"].peek().value.payload == {"ok": True}
        assert entity.outs["metadata"].peek().value.payload["worker"] == "typescript"

    async def test_remote_requires_network_capability(self):
        entity = RemoteEntity("remote", "https://worker.test/fire")
        graph = CompositeGraph("remote")
        graph.add_entity("remote", entity)
        entity.ins["input"].emit(TypedValue(type="json", payload={}))
        result = await FNDirector().run(graph, RunContext(run_id="denied"))
        assert not result.converged
        assert "net:connect" in (result.error or "")


class _FakeProcess:
    returncode = 0

    def __init__(self, response: dict) -> None:
        self.response = response
        self.stdin_payload: dict | None = None

    async def communicate(self, payload: bytes):
        self.stdin_payload = json.loads(payload)
        return json.dumps(self.response).encode(), b""

    def kill(self) -> None:
        return None

    async def wait(self) -> int:
        return self.returncode


class TestContainerEntity:
    def test_image_digest_required_by_default(self):
        with pytest.raises(ValueError, match="pinned by digest"):
            ContainerEntity("worker", "example/worker:latest")

    async def test_container_uses_read_only_no_network_protocol(self, monkeypatch):
        fake = _FakeProcess(
            {
                "protocol": WORKER_PROTOCOL_VERSION,
                "outputs": [
                    {"port": "output", "type": "text", "payload": "done"}
                ],
            }
        )
        command: list[str] = []

        async def create(*args, **kwargs):
            command.extend(args)
            assert kwargs["stdin"] == asyncio.subprocess.PIPE
            return fake

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        entity = ContainerEntity(
            "worker",
            "example/worker@sha256:" + "a" * 64,
            command=["serve", "--stdio"],
        )
        graph = CompositeGraph("container")
        graph.add_entity("worker", entity)
        entity.ins["input"].emit(TypedValue(type="json", payload={"task": 1}))
        result = await FNDirector().run(
            graph,
            RunContext(
                run_id="container-run",
                granted_permissions={"container:execute"},
            ),
        )

        assert result.converged
        assert command[:7] == [
            "docker",
            "run",
            "--rm",
            "-i",
            "--read-only",
            "--network",
            "none",
        ]
        assert fake.stdin_payload["protocol"] == WORKER_PROTOCOL_VERSION
        assert entity.outs["output"].peek().value.payload == "done"

    def test_ir_builds_remote_and_container_nodes(self):
        graph, _ = load_graph_from_dict(
            {
                "name": "heterogeneous",
                "nodes": [
                    {
                        "id": "remote",
                        "type": "remote",
                        "endpoint": "https://worker.test/fire",
                    },
                    {
                        "id": "container",
                        "type": "container",
                        "image": "example/worker@sha256:" + "b" * 64,
                    },
                ],
                "edges": [
                    {"src": "remote.output", "dst": "container.input"}
                ],
            }
        )
        assert isinstance(graph.node("remote").instance, RemoteEntity)
        assert isinstance(graph.node("container").instance, ContainerEntity)
