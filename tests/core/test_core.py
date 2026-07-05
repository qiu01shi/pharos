"""Tests for pharos.core (token, port, entity, graph, type_system)."""
from __future__ import annotations

import pytest

from pharos.core import (
    CompositeGraph,
    Edge,
    Entity,
    InputPort,
    OutputPort,
    PortContractViolation,
    Token,
    TypedValue,
    entity,
)
from pharos.core.type_system import accepts_schema, model_dump_safe, schema_name


class TestToken:
    def test_basic(self):
        t = Token(value=TypedValue(type="text", payload="hi"))
        assert t.value.type == "text"
        assert t.value.payload == "hi"
        # self_hash is computed automatically
        assert len(t.self_hash) == 16

    def test_hash_deterministic(self):
        # Same value + same origin + same ts = same hash
        t1 = Token(
            value=TypedValue(type="text", payload="hi"),
            origin="n.p",
            ts=1000.0,
            run_id="r1",
            iter=0,
        )
        t2 = Token(
            value=TypedValue(type="text", payload="hi"),
            origin="n.p",
            ts=1000.0,
            run_id="r1",
            iter=0,
        )
        assert t1.self_hash == t2.self_hash

    def test_hash_changes_with_payload(self):
        t1 = Token(value=TypedValue(type="text", payload="hi"))
        t2 = Token(value=TypedValue(type="text", payload="bye"))
        assert t1.self_hash != t2.self_hash

    def test_hash_chain(self):
        t1 = Token(value=TypedValue(type="text", payload="a"))
        t2 = t1.with_prev(t1)
        assert t2.prev_hash == t1.self_hash
        # Different prev → different self_hash (even if value same)
        t3 = Token(value=TypedValue(type="text", payload="a"))
        assert t3.self_hash != t2.self_hash

    def test_frozen(self):
        t = Token(value=TypedValue(type="text", payload="x"))
        with pytest.raises(Exception):  # FrozenInstanceError
            t.value = TypedValue(type="text", payload="y")  # type: ignore[misc]

    def test_partial_flag(self):
        t = Token(value=TypedValue(type="text", payload="..."), is_partial=True)
        assert t.is_partial is True

    def test_cost_recorded(self):
        t = Token(value=TypedValue(type="text", payload="x"), cost_usd=0.0123)
        assert t.cost_usd == pytest.approx(0.0123)


class TestPort:
    def test_input_accepts_matching_type(self):
        p = InputPort(name="q", accepted_types=["text"])
        p.receive(Token(value=TypedValue(type="text", payload="hi")))
        assert len(p) == 1

    def test_input_rejects_unknown_type(self):
        p = InputPort(name="q", accepted_types=["text"])
        with pytest.raises(PortContractViolation):
            p.receive(Token(value=TypedValue(type="image", payload=b"\x00")))

    def test_permissive_when_no_types(self):
        p = InputPort(name="q", accepted_types=[])
        p.receive(Token(value=TypedValue(type="anything", payload=42)))
        assert len(p) == 1

    def test_emit_returns_token(self):
        p = OutputPort(name="p", accepted_types=["text"])
        tok = p.emit(TypedValue(type="text", payload="x"))
        assert isinstance(tok, Token)
        assert tok.value.payload == "x"

    def test_capacity_backpressure(self):
        from pharos.core.port import PortCapacityExceeded

        p = InputPort(name="q", capacity=2, overflow="backpressure")
        p.receive(Token(value=TypedValue(type="text", payload="1")))
        p.receive(Token(value=TypedValue(type="text", payload="2")))
        with pytest.raises(PortCapacityExceeded):
            p.receive(Token(value=TypedValue(type="text", payload="3")))

    def test_capacity_drop_oldest(self):
        p = InputPort(name="q", capacity=2, overflow="drop_oldest")
        p.receive(Token(value=TypedValue(type="text", payload="1")))
        p.receive(Token(value=TypedValue(type="text", payload="2")))
        p.receive(Token(value=TypedValue(type="text", payload="3")))
        assert len(p) == 2
        # First one was dropped; buffer holds 2 and 3
        payloads = [t.value.payload for t in p.peek_all()]
        assert payloads == ["2", "3"]
        assert p.total_dropped == 1

    def test_capacity_drop_newest(self):
        p = InputPort(name="q", capacity=2, overflow="drop_newest")
        p.receive(Token(value=TypedValue(type="text", payload="1")))
        p.receive(Token(value=TypedValue(type="text", payload="2")))
        p.receive(Token(value=TypedValue(type="text", payload="3")))
        assert len(p) == 2
        assert p.total_dropped == 1

    def test_consume_n(self):
        p = OutputPort(name="p", accepted_types=["text"])
        for i in range(5):
            p.emit(TypedValue(type="text", payload=str(i)))
        out = p.consume(3)
        assert len(out) == 3
        assert [t.value.payload for t in out] == ["0", "1", "2"]
        assert len(p) == 2

    def test_consume_all(self):
        p = OutputPort(name="p", accepted_types=["text"])
        for i in range(3):
            p.emit(TypedValue(type="text", payload=str(i)))
        out = p.consume()
        assert len(out) == 3
        assert len(p) == 0

    def test_peek(self):
        p = OutputPort(name="p", accepted_types=["text"])
        p.emit(TypedValue(type="text", payload="head"))
        p.emit(TypedValue(type="text", payload="tail"))
        assert p.peek().value.payload == "head"  # type: ignore[union-attr]
        assert len(p) == 2  # peek doesn't remove

    def test_metrics_count(self):
        p = OutputPort(name="p", accepted_types=["text"])
        for i in range(3):
            p.emit(TypedValue(type="text", payload=str(i)))
        p.consume(1)
        p.consume(1)
        assert p.total_emitted == 3
        assert len(p) == 1


@entity
class _Echo(Entity):
    """Test entity: copies input to output."""

    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        toks = self.ins["x"].consume()
        for t in toks:
            self.outs["y"].emit(t.value)


@entity
class _Doubler(Entity):
    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        t = self.ins["x"].consume()[0]
        self.outs["y"].emit(TypedValue(type="text", payload=t.value.payload * 2))


class TestEntity:
    def test_decorator_collects_ports(self):
        assert "x" in _Echo.spec.inputs
        assert "y" in _Echo.spec.outputs
        assert _Echo.spec.name == "_Echo"

    def test_per_instance_buffers(self):
        a = _Echo(node_id="a")
        b = _Echo(node_id="b")
        a.ins["x"].emit(TypedValue(type="text", payload="hi"))
        assert len(a.ins["x"]) == 1
        assert len(b.ins["x"]) == 0  # independent buffer

    def test_fire_must_be_async(self):
        with pytest.raises(TypeError, match="must be defined as"):

            @entity
            class Bad(Entity):  # type: ignore[misc]
                ins = {"x": InputPort(name="x")}
                outs = {}

                def fire(self, ctx):  # type: ignore[override]
                    pass


class TestGraph:
    def test_minimal(self):
        g = CompositeGraph(name="g")
        assert "g" not in g.nodes  # IO nodes are __in__ / __out__
        assert g.INPUT_NODE_ID in g.nodes
        assert g.OUTPUT_NODE_ID in g.nodes

    def test_add_and_connect(self):
        g = CompositeGraph(name="g")
        a = _Echo(node_id="a")
        b = _Doubler(node_id="b")
        g.add_entity("a", a)
        g.add_entity("b", b)
        g.connect("__in__.prompt", "a.x")
        g.connect("a.y", "b.x")
        g.connect("b.y", "__out__.text")
        assert len(g.edges) == 3

    def test_duplicate_node_raises(self):
        g = CompositeGraph(name="g")
        g.add_entity("a", _Echo())
        with pytest.raises(ValueError, match="duplicate node"):
            g.add_entity("a", _Echo())

    def test_unknown_node_in_edge(self):
        g = CompositeGraph(name="g")
        g.add_entity("a", _Echo())
        with pytest.raises(ValueError, match="not in graph"):
            g.connect("a.y", "ghost.x")

    def test_topo_order_dag(self):
        g = CompositeGraph(name="g")
        a = _Echo(node_id="a")
        b = _Doubler(node_id="b")
        g.add_entity("a", a)
        g.add_entity("b", b)
        g.connect("a.y", "b.x")
        order = g.topo_order()
        # a must come before b
        assert order.index("a") < order.index("b")

    def test_cycle_detection(self):
        g = CompositeGraph(name="g")
        a = _Echo(node_id="a")
        b = _Echo(node_id="b")
        g.add_entity("a", a)
        g.add_entity("b", b)
        g.connect("a.y", "b.x")
        g.connect("b.y", "a.x")
        assert g.has_cycle()
        with pytest.raises(Exception):  # NetworkXUnfeasible
            g.topo_order()

    def test_entry_exit_nodes(self):
        g = CompositeGraph(name="g")
        a = _Echo(node_id="a")
        b = _Echo(node_id="b")
        g.add_entity("a", a)
        g.add_entity("b", b)
        g.connect("a.y", "b.x")
        entries = g.entry_nodes()
        exits = g.exit_nodes()
        assert "__in__" in entries
        assert "__out__" in exits

    def test_validate_clean(self):
        g = CompositeGraph(name="g")
        a = _Echo(node_id="a")
        g.add_entity("a", a)
        g.connect("__in__.prompt", "a.x")
        g.connect("a.y", "__out__.text")
        assert g.validate() == []

    def test_validate_with_cycle(self):
        g = CompositeGraph(name="g")
        a = _Echo(node_id="a")
        g.add_entity("a", a)
        g.connect("a.y", "a.x")  # self-loop
        errors = g.validate()
        assert any("cycle" in e for e in errors)

    async def test_subgraph_executes_and_exports(self):
        # A subgraph runs as a composite node: seed parent __in__, the child
        # runs, and its __out__ flows back out. Also asserts exports are
        # recorded for introspection.
        from pharos.core.entity import Entity, entity
        from pharos.core.port import InputPort, OutputPort
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector

        @entity
        class Stub(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                for t in self.ins["x"].consume():
                    self.outs["y"].emit(
                        TypedValue(type="text", payload=t.value.payload + "!")
                    )

        inner = CompositeGraph(name="inner")
        inner.add_entity("stub", Stub("stub"))
        inner.connect("__in__.x", "stub.x")
        inner.connect("stub.y", "__out__.y")

        g = CompositeGraph(name="outer")
        g.add_subgraph("inner", inner)
        # exports records the subgraph's public output ports.
        assert "y" in g.exports
        g.connect("__in__.msg", "inner.x")
        g.connect("inner.y", "__out__.done")

        for e in g.edges:
            if e.src_node == "__in__" and e.src_port == "msg":
                g.node(e.dst_node).instance.ins[e.dst_port].emit(
                    TypedValue(type="text", payload="hi")
                )
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is True
        out = getattr(g, "collected", {}).get("__out__", {}).get("done", [])
        assert [t.value.payload for t in out] == ["hi!"]


class TestTypeSystem:
    def test_accepts_when_empty(self):
        assert accepts_schema([], "anything")

    def test_accepts_when_match(self):
        assert accepts_schema(["text", "json"], "text")

    def test_rejects_when_no_match(self):
        assert not accepts_schema(["text"], "image")

    def test_model_dump_safe_dict(self):
        assert model_dump_safe({"a": 1}) == {"a": 1}

    def test_model_dump_safe_none(self):
        assert model_dump_safe(None) is None

    def test_schema_name(self):
        from pydantic import BaseModel

        class Foo(BaseModel):
            x: int

        assert schema_name(Foo) == "Foo"
        assert schema_name(None) == ""


class TestEdge:
    def test_reverse(self):
        e = Edge(src_node="a", src_port="x", dst_node="b", dst_port="y")
        r = e.reverse()
        assert r.src_node == "b"
        assert r.dst_node == "a"
