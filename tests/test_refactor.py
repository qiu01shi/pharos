"""Regression tests for the architecture refactor:

  #1 general entity-output record/replay (not just LLM)
  #2 tools as first-class graph nodes (ToolEntity) + graph-level RBAC
  #4 unified PermissionPolicy (with capability aliases)
  #5 bounded trace (event cap + attribute truncation)
"""
from __future__ import annotations

import uuid

import pytest

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.permissions import PermissionPolicy, canonical
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.tool_node import ToolEntity
from pharos.entities.tools import ToolRegistry
from pharos.entities.tools_builtins import register_builtins
from pharos.entities.tools_coding import register_coding_tools
from pharos.observability.trace import Span
from pharos.runtime import RunRecorder, RunReplayer


@entity
class _Counter(Entity):
    """Side-effecting entity: counts fire() calls and echoes its input."""

    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id=node_id)
        self.calls = 0

    async def fire(self, ctx):  # type: ignore[override]
        self.calls += 1
        for t in self.ins["x"].consume():
            self.outs["y"].emit(
                TypedValue(type="text", payload=f"out:{t.value.payload}")
            )


# ---------- #4 PermissionPolicy ----------


class TestPermissionPolicy:
    def test_alias_grant_authorizes_both_names(self):
        # Granting shell:execute authorizes the bash:execute alias too.
        policy = PermissionPolicy.from_grants({"shell:execute"})
        assert policy.allows("shell:execute")
        assert policy.allows("bash:execute")
        assert canonical("bash:execute") == "shell:execute"

    def test_missing_reports_original_spelling(self):
        policy = PermissionPolicy.from_grants(set())
        assert policy.missing({"fs:read"}) == {"fs:read"}

    def test_check_raises_with_subject(self):
        policy = PermissionPolicy.from_grants(set())
        with pytest.raises(PermissionError, match="my-tool"):
            policy.check({"fs:write"}, subject="my-tool")

    async def test_tool_registry_uses_alias(self):
        # A tool needing bash:execute is allowed when shell:execute granted.
        reg = ToolRegistry()
        register_coding_tools(reg)
        result = await reg.execute(
            "bash",
            {"command": "echo hi"},
            granted_permissions={"shell:execute"},
        )
        assert not result.is_error


# ---------- #2 ToolEntity as a graph node ----------


class TestToolEntity:
    async def test_tool_node_runs_with_grant(self):
        reg = ToolRegistry()
        register_builtins(reg)
        g = CompositeGraph("g")
        g.add_entity("echo", ToolEntity("echo", "echo", reg))
        g.node("echo").instance.ins["input"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hello")
        )
        result = await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        out = [t.value.payload for t in g.node("echo").instance.outs["output"].peek_all()]  # type: ignore[union-attr]
        assert out == ["hello"]

    async def test_tool_node_enforces_permission_at_graph_level(self):
        # `read` needs fs:read; the Director must deny it without a grant.
        reg = ToolRegistry()
        register_coding_tools(reg)
        node = ToolEntity("reader", "read", reg)
        assert node.required_permissions == {"fs:read"}
        g = CompositeGraph("g")
        g.add_entity("reader", node)
        g.node("reader").instance.ins["input"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="/nonexistent")
        )
        result = await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is False
        assert "fs:read" in (result.error or "")

    def test_tool_node_from_ir(self):
        from pharos.ir import load_graph_from_dict

        g, _ = load_graph_from_dict(
            {
                "name": "g",
                "nodes": [
                    {"id": "r", "type": "tool", "tool": "read", "preset": "coding"}
                ],
                "edges": [],
            }
        )
        inst = g.node("r").instance
        assert isinstance(inst, ToolEntity)
        assert inst.required_permissions == {"fs:read"}

    def test_tool_node_unknown_tool_raises(self):
        from pharos.ir import load_graph_from_dict

        with pytest.raises(ValueError, match="unknown tool"):
            load_graph_from_dict(
                {
                    "name": "g",
                    "nodes": [
                        {"id": "x", "type": "tool", "tool": "nope", "preset": "coding"}
                    ],
                    "edges": [],
                }
            )


# ---------- #1 general record/replay ----------


class TestGeneralReplay:
    async def _record(self) -> RunRecorder:
        g = CompositeGraph("g")
        c = _Counter("c")
        g.add_entity("c", c)
        g.node("c").instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        rec = RunRecorder()
        await FNDirector().run(
            g, RunContext(run_id=str(uuid.uuid4()), recorder=rec)
        )
        assert c.calls == 1
        return rec

    async def test_replay_skips_execution_but_reproduces_output(self):
        rec = await self._record()
        data = rec.to_dict()
        assert data["c:0"][0]["payload"] == "out:hi"

        # Replay on a fresh graph — the entity must NOT execute.
        g2 = CompositeGraph("g")
        c2 = _Counter("c")
        g2.add_entity("c", c2)
        g2.node("c").instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        replayer = RunReplayer(data)
        await FNDirector().run(
            g2, RunContext(run_id=str(uuid.uuid4()), replayer=replayer)
        )
        assert c2.calls == 0  # never executed
        out = [t.value.payload for t in c2.outs["y"].peek_all()]
        assert out == ["out:hi"]  # output reproduced

    async def test_replay_needs_no_permission_grant(self):
        # A permission-gated entity records WITH a grant, then replays
        # WITHOUT one — replay must not re-check permissions or setup(),
        # since nothing actually executes.
        @entity
        class _Gated(Entity):
            required_permissions = {"shell:execute"}
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                for t in self.ins["x"].consume():
                    self.outs["y"].emit(
                        TypedValue(type="text", payload=f"ran:{t.value.payload}")
                    )

        g = CompositeGraph("g")
        g.add_entity("s", _Gated("s"))
        g.node("s").instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="cmd")
        )
        rec = RunRecorder()
        r1 = await FNDirector().run(
            g,
            RunContext(
                run_id=str(uuid.uuid4()),
                granted_permissions={"shell:execute"},
                recorder=rec,
            ),
        )
        assert r1.converged is True

        # Replay with NO grant — must succeed and reproduce the output.
        g2 = CompositeGraph("g")
        g2.add_entity("s", _Gated("s"))
        g2.node("s").instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="cmd")
        )
        r2 = await FNDirector().run(
            g2,
            RunContext(run_id=str(uuid.uuid4()), replayer=RunReplayer(rec.to_dict())),
        )
        assert r2.converged is True
        out = [t.value.payload for t in g2.node("s").instance.outs["y"].peek_all()]  # type: ignore[union-attr]
        assert out == ["ran:cmd"]


# ---------- #5 bounded trace ----------


class TestBoundedTrace:
    def test_event_cap(self):
        s = Span(
            id="s",
            trace_id="t",
            parent_span_id=None,
            name="n",
            started_at=0.0,
            max_events=3,
        )
        for i in range(10):
            s.record_event("ev", {"i": i})
        assert len(s.events) == 3
        assert s.dropped_events == 7

    def test_attribute_truncation(self):
        s = Span(
            id="s",
            trace_id="t",
            parent_span_id=None,
            name="n",
            started_at=0.0,
            max_attr_chars=10,
        )
        s.record_event("ev", {"big": "x" * 100})
        stored = s.events[0].attributes["big"]
        assert stored.startswith("x" * 10)
        assert "truncated" in stored
