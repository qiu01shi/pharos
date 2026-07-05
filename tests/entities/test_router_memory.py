"""Tests for Router and Memory entities."""
from __future__ import annotations

import uuid

from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.memory import Memory, _MemoryState
from pharos.entities.router import Router

# ---------- Router ----------


class TestRouter:
    async def test_simple_routing(self):
        g = CompositeGraph("g")
        r = Router(
            node_id="r",
            guards={
                "errors": lambda v: "error" in v.lower(),
                "questions": lambda v: v.startswith("?"),
            },
            default="info",
        )
        g.add_entity("r", r)
        g.nodes["r"].instance.ins["in"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="got an ERROR here")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        errors = [t.value.payload for t in r.outs["errors"].peek_all()]
        questions = [t.value.payload for t in r.outs["questions"].peek_all()]
        info = [t.value.payload for t in r.outs["info"].peek_all()]
        assert errors == ["got an ERROR here"]
        assert questions == []
        assert info == []

    async def test_default_branch(self):
        g = CompositeGraph("g")
        r = Router(
            node_id="r",
            guards={"a": lambda v: v == "x"},
            default="other",
        )
        g.add_entity("r", r)
        g.nodes["r"].instance.ins["in"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="not x")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        assert [t.value.payload for t in r.outs["other"].peek_all()] == ["not x"]

    async def test_first_match_wins(self):
        g = CompositeGraph("g")
        r = Router(
            node_id="r",
            guards={
                "a": lambda v: "x" in v,  # matches first
                "b": lambda v: "x" in v,  # would also match
            },
            default="c",
        )
        g.add_entity("r", r)
        g.nodes["r"].instance.ins["in"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="xxx")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        a = [t.value.payload for t in r.outs["a"].peek_all()]
        b = [t.value.payload for t in r.outs["b"].peek_all()]
        assert a == ["xxx"]
        assert b == []

    async def test_empty_guards(self):
        """No guards → all inputs go to default."""
        g = CompositeGraph("g")
        r = Router(node_id="r", guards={}, default="d")
        g.add_entity("r", r)
        g.nodes["r"].instance.ins["in"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="x")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        assert [t.value.payload for t in r.outs["d"].peek_all()] == ["x"]


# ---------- Memory ----------


class TestMemory:
    async def test_write_then_read(self):
        g = CompositeGraph("g")
        store = _MemoryState()
        m = Memory(node_id="m", store=store)
        g.add_entity("m", m)
        # Write: write_key=K, write_val=V
        m.ins["write_key"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="k1")
        )
        m.ins["write_val"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="v1")
        )
        # Read: read_key=K
        m.ins["read_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="k1")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        values = [t.value.payload for t in m.outs["value"].peek_all()]
        missing = [t.value.payload for t in m.outs["missing"].peek_all()]
        assert values == ["v1"]
        assert missing == []

    async def test_missing_key(self):
        g = CompositeGraph("g")
        m = Memory(node_id="m", store=_MemoryState())
        g.add_entity("m", m)
        m.ins["read_key"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="nope")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        assert [t.value.payload for t in m.outs["missing"].peek_all()] == ["nope"]
        assert [t.value.payload for t in m.outs["value"].peek_all()] == []

    async def test_clear(self):
        g = CompositeGraph("g")
        m = Memory(node_id="m", store=_MemoryState())
        g.add_entity("m", m)
        m.ins["write_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="k")
        )
        m.ins["write_val"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="v")
        )
        m.ins["clear"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="all")
        )
        m.ins["read_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="k")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        # After clear, read should emit missing
        assert [t.value.payload for t in m.outs["missing"].peek_all()] == ["k"]

    async def test_shared_state_across_instances(self):
        """Two Memory entities without explicit store share _DEFAULT_STORE."""
        from pharos.entities.memory import _DEFAULT_STORE

        # Reset to a known state
        _DEFAULT_STORE.clear()

        g = CompositeGraph("g")
        m1 = Memory(node_id="m1")  # uses default store
        m2 = Memory(node_id="m2")  # uses default store
        g.add_entity("m1", m1)
        g.add_entity("m2", m2)
        m1.ins["write_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="shared")
        )
        m1.ins["write_val"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="value")
        )
        m2.ins["read_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="shared")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        assert [t.value.payload for t in m2.outs["value"].peek_all()] == ["value"]
        _DEFAULT_STORE.clear()  # cleanup

    async def test_json_value(self):
        g = CompositeGraph("g")
        m = Memory(node_id="m", store=_MemoryState())
        g.add_entity("m", m)
        m.ins["write_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="data")
        )
        m.ins["write_val"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload='{"a": 1, "b": 2}')
        )
        m.ins["read_key"].emit(  # type: ignore[attr-defined]
            TypedValue(type="text", payload="data")
        )
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        # Value should be the parsed JSON, but since we serialize to text
        # for the port, it gets re-stringified
        values = [t.value.payload for t in m.outs["value"].peek_all()]
        assert values == ['{"a": 1, "b": 2}']
