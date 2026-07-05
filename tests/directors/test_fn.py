"""Tests for pharos.directors (FN + base helpers)."""
from __future__ import annotations

import uuid

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import (
    RunContext,
    deliver_upstream,
    topo_layers,
)
from pharos.directors.fn import FNDirector

# ----- test entities -----

@entity
class _Source(Entity):
    outs = {"prompt": OutputPort(name="prompt", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        self.outs["prompt"].emit(TypedValue(type="text", payload="hi"))


@entity
class _Sink(Entity):
    ins = {"text": InputPort(name="text", accepted_types=["text"])}
    received: list[str] = []

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["text"].consume():
            self.received.append(t.value.payload)


@entity
class _PassThrough(Entity):
    ins = {"p": InputPort(name="p", accepted_types=["text"])}
    outs = {"o": OutputPort(name="o", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["p"].consume():
            self.outs["o"].emit(t.value)


# ----- topo_layers -----

class TestTopoLayers:
    def test_linear_chain(self):
        g = CompositeGraph("g")
        g.add_entity("a", _PassThrough(node_id="a"))
        g.add_entity("b", _PassThrough(node_id="b"))
        g.add_entity("c", _PassThrough(node_id="c"))
        g.connect("a.o", "b.p")
        g.connect("b.o", "c.p")
        layers = topo_layers(g)
        # __in__ + a in layer 0, b in layer 1, c in layer 2, __out__ ... no, no __out__ edges
        # Just check a < b < c
        flat = [n for layer in layers for n in layer]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    def test_fan_out_one_layer(self):
        g = CompositeGraph("g")
        g.add_entity("s", _Source(node_id="s"))
        for i in range(3):
            g.add_entity(f"w{i}", _Sink(node_id=f"w{i}"))
            g.connect("s.prompt", f"w{i}.text")
        layers = topo_layers(g)
        flat = {n for layer in layers for n in layer}
        assert {"w0", "w1", "w2"}.issubset(flat)


# ----- deliver_upstream -----

class TestDeliverUpstream:
    async def test_fan_out_replicates_token(self):
        """A single source token must reach every downstream edge."""
        g = CompositeGraph("g")
        g.add_entity("s", _Source(node_id="s"))
        for i in range(3):
            sink = _Sink(node_id=f"w{i}")
            g.add_entity(f"w{i}", sink)
            g.connect("s.prompt", f"w{i}.text")
        # Seed source buffer
        g.nodes["s"].instance.outs["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        await deliver_upstream(g, {})
        for i in range(3):
            buf = [t.value.payload for t in g.nodes[f"w{i}"].instance.ins["text"]]  # type: ignore[union-attr]
            assert buf == ["hi"], f"sink {i} got {buf}"

    async def test_chain_propagation(self):
        g = CompositeGraph("g")
        g.add_entity("a", _PassThrough(node_id="a"))
        g.add_entity("b", _PassThrough(node_id="b"))
        g.add_entity("c", _Sink(node_id="c"))
        g.connect("a.o", "b.p")
        g.connect("b.o", "c.text")
        # Seed
        g.nodes["a"].instance.outs["o"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="chain")
        )
        await deliver_upstream(g, {})
        # a's token -> b's p
        assert len(g.nodes["b"].instance.ins["p"]) == 1  # type: ignore[union-attr]
        # b fires, emits o
        g.nodes["b"].instance.outs["o"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="chain")
        )
        await deliver_upstream(g, {})
        assert len(g.nodes["c"].instance.ins["text"]) == 1  # type: ignore[union-attr]

    async def test_empty_buffer_no_op(self):
        g = CompositeGraph("g")
        g.add_entity("s", _Source(node_id="s"))
        g.add_entity("w", _Sink(node_id="w"))
        g.connect("s.prompt", "w.text")
        # No seed; deliver should be a no-op
        await deliver_upstream(g, {})
        assert len(g.nodes["w"].instance.ins["text"]) == 0  # type: ignore[union-attr]


# ----- FNDirector -----

class TestFNDirector:
    async def test_simple_chain(self):
        g = CompositeGraph("g")
        g.add_entity("a", _PassThrough(node_id="a"))
        g.add_entity("b", _PassThrough(node_id="b"))
        g.add_entity("c", _Sink(node_id="c"))
        g.connect("a.o", "b.p")
        g.connect("b.o", "c.text")
        g.nodes["a"].instance.outs["o"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="x")
        )
        d = FNDirector()
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        assert result.error is None
        sink = g.nodes["c"].instance  # type: ignore[union-attr]
        assert "x" in sink.received  # type: ignore[attr-defined]

    async def test_fan_out_three_sinks(self):
        g = CompositeGraph("g")
        g.add_entity("s", _Source(node_id="s"))
        for i in range(3):
            g.add_entity(f"w{i}", _Sink(node_id=f"w{i}"))
            g.connect("s.prompt", f"w{i}.text")
        # Seed
        g.nodes["s"].instance.outs["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="broadcast")
        )
        d = FNDirector()
        await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        for i in range(3):
            sink = g.nodes[f"w{i}"].instance  # type: ignore[union-attr]
            assert "broadcast" in sink.received  # type: ignore[attr-defined]

    async def test_empty_graph(self):
        g = CompositeGraph("g")
        d = FNDirector()
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        # The graph always has __in__ and __out__ virtual nodes; the
        # result includes at least one iteration if those were counted.
        assert result.tokens_emitted == 0
        assert result.error is None

    async def test_error_in_fire_propagates(self):
        @entity
        class Boom(Entity):
            async def fire(self, ctx):  # type: ignore[override]
                raise RuntimeError("kaboom")

        g = CompositeGraph("g")
        g.add_entity("b", Boom(node_id="b"))
        d = FNDirector()
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is False
        assert "kaboom" in (result.error or "")
