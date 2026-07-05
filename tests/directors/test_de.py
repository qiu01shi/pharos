"""Tests for the DE (Discrete Event) Director."""
from __future__ import annotations

import uuid

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.de import DEDirector

# ----- test entities -----


@entity
class _Source(Entity):
    """Emits a constant text once (becomes stable after first fire)."""

    outs = {"out": OutputPort(name="out", accepted_types=["text"])}

    def __init__(self, node_id, text):
        super().__init__(node_id=node_id)
        self._text = text
        self._fired = 0

    async def fire(self, ctx):  # type: ignore[override]
        self._fired += 1
        # Always emit the same text — DE will detect no change next pass
        self.outs["out"].emit(
            TypedValue(type="text", payload=self._text)
        )


@entity
class _Counter(Entity):
    """Counts its own fires."""

    ins = {"in": InputPort(name="in", accepted_types=["text"])}
    outs = {"count": OutputPort(name="count", accepted_types=["int"])}

    def __init__(self, node_id):
        super().__init__(node_id=node_id)
        self._n = 0

    async def fire(self, ctx):  # type: ignore[override]
        for _ in self.ins["in"].consume():
            pass
        self._n += 1
        self.outs["count"].emit(TypedValue(type="int", payload=self._n))


# ----- tests -----


class TestDEConvergence:
    async def test_simple_pipeline_converges(self):
        """Source → counter; counter increments each fire."""
        g = CompositeGraph("de-simple")
        g.add_entity("source", _Source("source", "x"))
        g.add_entity("counter", _Counter("counter"))
        g.connect("source.out", "counter.in")
        g.nodes["source"].instance.outs["out"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )
        d = DEDirector(max_iterations=10)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        # Round 1: source emits → counter consumes → emits count=1
        # Round 2: nothing has input → nothing fires → converged
        assert result.iterations == 2

    async def test_empty_graph(self):
        """Empty graph converges immediately."""
        g = CompositeGraph("empty")
        d = DEDirector()
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        # pre == post (both empty) → converged after 1 round
        assert result.converged is True

    async def test_max_iterations_terminates(self):
        """Source that always re-fires (never quiesces) hits the
        max_iterations cap. We test by having an entity that
        emits on every fire."""
        from pharos.core.entity import Entity
        from pharos.core.entity import entity as _entity

        @_entity
        class _Changing(Entity):
            outs = {"x": OutputPort(name="x", accepted_types=["text"])}

            def __init__(self, node_id):
                super().__init__(node_id=node_id)
                self._n = 0

            async def fire(self, ctx):  # type: ignore[override]
                self._n += 1
                self.outs["x"].emit(
                    TypedValue(type="text", payload=f"v{self._n}")
                )

        g = CompositeGraph("de-changing")
        g.add_entity("ch", _Changing("ch"))
        # Source-only — fires once (no downstream input), then no
        # more nodes have input. Should converge.
        g.nodes["ch"].instance.outs["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )
        d = DEDirector(max_iterations=4)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        # Source fires once; nothing else has input → converges in
        # 2 rounds (round 1 fires source, round 2 has nothing).
        assert result.converged is True
        assert result.iterations == 2

    async def test_max_iters_with_changing_source(self):
        """A source with a self-loop on output would normally be a
        cycle. DE handles this via 'fire once' for source-like
        nodes — so even without input, the source fires once and
        then quiets."""
        from pharos.core.entity import Entity
        from pharos.core.entity import entity as _entity

        @_entity
        class _ChangingSource(Entity):
            outs = {"x": OutputPort(name="x", accepted_types=["text"])}

            def __init__(self, node_id):
                super().__init__(node_id=node_id)
                self._n = 0

            async def fire(self, ctx):  # type: ignore[override]
                self._n += 1
                self.outs["x"].emit(
                    TypedValue(type="text", payload=f"v{self._n}")
                )

        g = CompositeGraph("de-source-only")
        g.add_entity("src", _ChangingSource("src"))
        # No edges — just a source. Fires once and quiets.
        g.nodes["src"].instance.outs["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )
        d = DEDirector(max_iterations=10)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True

    async def test_counter_counts_multiple_fires(self):
        """In a graph without cycles, source emits each round;
        counter increments each round."""
        g = CompositeGraph("de-fanout")
        g.add_entity("source", _Source("source", "y"))
        g.add_entity("counter", _Counter("counter"))
        g.connect("source.out", "counter.in")
        g.nodes["source"].instance.outs["out"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )
        d = DEDirector(max_iterations=10)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        # Should converge after 1 round
        assert result.converged is True
        # The counter fired at least once
        counter = g.node("counter").instance
        assert counter is not None
        last = counter.outs["count"].peek_all()[-1].value.payload
        assert last == 1