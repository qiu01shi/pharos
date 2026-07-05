"""Tests for SDF Director (feedback loops with convergence)."""
from __future__ import annotations

import uuid

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.sdf import SDFDirector

# --------- test entities ---------


@entity
class _Source(Entity):
    outs = {"prompt": OutputPort(name="prompt", accepted_types=["text"])}

    def __init__(self, node_id, text: str):
        super().__init__(node_id=node_id)
        self._text = text

    async def fire(self, ctx):  # type: ignore[override]
        # Always emit the same text so the upstream "converges"
        self.outs["prompt"].emit(
            TypedValue(type="text", payload=self._text)
        )


@entity
class _Reviewer(Entity):
    """Counts how many times it's been fired. Emits the count as
    output, which we then route to the next round or out."""

    ins = {"in": InputPort(name="in", accepted_types=["text"])}
    outs = {"verdict": OutputPort(name="verdict", accepted_types=["int"])}

    def __init__(self, node_id, max_approve: int = 3):
        super().__init__(node_id=node_id)
        self._n = 0
        self._max = max_approve

    async def fire(self, ctx):  # type: ignore[override]
        # Consume to drain
        for _ in self.ins["in"].consume():
            pass
        self._n += 1
        # Approve (verdict=0) after max_approve rounds
        verdict = 0 if self._n >= self._max else 1
        self.outs["verdict"].emit(
            TypedValue(type="int", payload=verdict)
        )


# --------- tests ---------


class TestSDFConvergence:
    async def test_converges_on_stable_output(self):
        """A source that always emits the same value + a reviewer that
        approves after N rounds should converge in exactly N rounds."""
        g = CompositeGraph(name="sdf-stable")
        g.add_entity("source", _Source(node_id="source", text="same"))
        g.add_entity("reviewer", _Reviewer(node_id="reviewer", max_approve=2))
        g.connect("source.prompt", "reviewer.in")
        g.nodes["source"].instance.outs["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )

        d = SDFDirector(max_iterations=20, convergence_k=2)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        # Reviewer approves at round 2; from round 3 onward it's stable
        assert result.iterations <= 5

    async def test_max_iterations_terminates_runaway(self):
        """An entity whose output changes every round prevents
        convergence; the run caps at max_iterations.

        A source that always emits a fresh value (different every
        round) — and gets a chance to fire every round via a
        self-loop — is the simplest example.
        """
        from pharos.core.entity import Entity
        from pharos.core.entity import entity as _entity
        from pharos.core.port import InputPort as _IP
        from pharos.core.port import OutputPort as _OP

        @_entity
        class _ChaoticSource(Entity):
            """Emits a fresh value on every fire. Has a self-loop
            so it gets re-fired every round (SDF always fires every
            node each round)."""

            ins = {"prev": _IP(name="prev", accepted_types=["text"])}
            outs = {"x": _OP(name="x", accepted_types=["text"])}

            def __init__(self, node_id):
                super().__init__(node_id=node_id)
                self._n = 0

            async def fire(self, ctx):  # type: ignore[override]
                # Consume prior value (drain)
                for _ in self.ins["prev"].consume():
                    pass
                self._n += 1
                self.outs["x"].emit(
                    TypedValue(type="text", payload=f"v{self._n}")
                )

        g = CompositeGraph("sdf-runaway")
        g.add_entity("src", _ChaoticSource(node_id="src"))
        # Self-loop: src.x → src.prev (so SDF re-fires it next round)
        g.connect("src.x", "src.prev")
        g.nodes["src"].instance.outs["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )
        d = SDFDirector(max_iterations=5, convergence_k=2)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is False
        assert result.iterations == 5

    async def test_convergence_k_requires_multiple_stable_rounds(self):
        """K=3 means a node must be stable for 3 consecutive rounds
        before the run is considered converged."""
        g = CompositeGraph("sdf-k")
        g.add_entity("source", _Source(node_id="source", text="same"))
        g.add_entity("reviewer", _Reviewer(node_id="reviewer", max_approve=1))
        g.connect("source.prompt", "reviewer.in")
        g.nodes["source"].instance.outs["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="seed")
        )
        d = SDFDirector(max_iterations=20, convergence_k=3)
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        # Reviewer approves at round 1; rounds 2, 3, 4 stable
        # K=3 means stable for 3 rounds → converges on round 4
        assert result.iterations == 4
