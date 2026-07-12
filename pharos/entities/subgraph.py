"""SubgraphEntity — run a nested CompositeGraph as a single composite node.

This is the mechanism behind pharos's composability: a whole graph can be
embedded as one node in a parent graph. Rather than teaching every Director
about nested graphs, we wrap the child graph in an Entity. The parent
Director then schedules it like any other node, and all the cross-cutting
machinery we already have — permission checks, tracing, record/replay,
metric collection in ``safe_fire`` — applies for free.

Execution model (A1: recursive):
  * The child graph keeps its own ``director`` (declared in its YAML), so a
    parent ``fn`` graph can embed an ``sdf`` feedback subgraph — heterogeneous
    nested scheduling, the way Ptolemy II intended.
  * Each ``fire()`` seeds the child's ``__in__`` ports from this node's input
    ports, runs the child to completion with its own Director, then copies the
    child's ``__out__`` results to this node's output ports.

Ports are derived automatically from the child's ``__in__`` / ``__out__``
edges (public name == internal port name), unless an explicit rename map is
given.
"""

from __future__ import annotations

from typing import Any

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort


@entity
class SubgraphEntity(Entity):
    """Wrap a nested ``CompositeGraph`` as one schedulable node."""

    ins = {}
    outs = {}

    def __init__(
        self,
        node_id: str,
        subgraph: CompositeGraph,
        director_name: str = "fn",
        input_map: dict[str, str] | None = None,
        output_map: dict[str, str] | None = None,
        max_iters: int = 20,
        converge_k: int = 2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.subgraph = subgraph
        self.director_name = director_name
        self.max_iters = max_iters
        self.converge_k = converge_k
        self.total_tokens = 0
        self.total_cost = 0.0

        # Derive public->internal port maps. Explicit maps take precedence;
        # otherwise auto-derive from the child's __in__ / __out__ edges with
        # public name == internal port name.
        auto_in = {
            e.src_port
            for e in subgraph.edges
            if e.src_node == subgraph.INPUT_NODE_ID
        }
        auto_out = {
            e.dst_port
            for e in subgraph.edges
            if e.dst_node == subgraph.OUTPUT_NODE_ID
        }
        self._in_public_to_internal: dict[str, str] = (
            dict(input_map) if input_map else {p: p for p in auto_in}
        )
        self._out_public_to_internal: dict[str, str] = (
            dict(output_map) if output_map else {p: p for p in auto_out}
        )

        # Build per-instance pass-through ports (empty accepted_types = any).
        self.ins = {
            p: InputPort(name=p, accepted_types=[])
            for p in self._in_public_to_internal
        }
        self.outs = {
            p: OutputPort(name=p, accepted_types=[])
            for p in self._out_public_to_internal
        }

        # Surface the union of the child entities' permission requirements so
        # the parent Director can fail-fast (and users see one clear picture).
        required: set[str] = set()
        for node in subgraph.nodes.values():
            if node.instance is not None:
                required |= (
                    getattr(node.instance, "required_permissions", set())
                    or set()
                )
        self.required_permissions = required

    async def fire(self, ctx) -> None:
        from pharos.directors import make_director
        from pharos.directors.base import RunContext

        sub = self.subgraph
        # Reset any collected outputs from a previous fire (SDF/DE loops).
        sub.collected = {}

        # 1. Collect this node's inputs, keyed by the child's internal port.
        seeds: dict[str, list[Any]] = {}
        for public, internal in self._in_public_to_internal.items():
            tokens = self.ins[public].consume()
            if tokens:
                seeds[internal] = [t.value for t in tokens]

        # If we declare inputs but none arrived, we're not ready to fire.
        if self.ins and not seeds:
            return

        # 2. Seed the child's __in__ edges.
        for e in sub.edges:
            if e.src_node != sub.INPUT_NODE_ID or e.src_port not in seeds:
                continue
            dst = sub.node(e.dst_node)
            if dst.instance is None or e.dst_port not in dst.instance.ins:
                continue
            for value in seeds[e.src_port]:
                dst.instance.ins[e.dst_port].emit(value)

        # 3. Recursively run the child with its own Director, sharing the
        #    parent's permissions / tracer / recorder (namespaced).
        recorder = getattr(ctx, "recorder", None)
        child_recorder = (
            recorder.child(f"{self.node_id}/") if recorder is not None else None
        )
        sub_ctx = RunContext(
            run_id=f"{ctx.run_id}:{self.node_id}",
            granted_permissions=getattr(ctx, "granted_permissions", set()),
            tracer=getattr(ctx, "tracer", None),
            recorder=child_recorder,
            # Replay is handled at this node's output boundary by the parent
            # safe_fire, so the child sub-run always executes live.
            replayer=None,
        )
        director = make_director(
            self.director_name,
            max_iters=self.max_iters,
            converge_k=self.converge_k,
        )
        result = await director.run(sub, sub_ctx)

        # 4. Copy the child's __out__ results to this node's outputs.
        collected = getattr(sub, "collected", {}) or {}
        out_coll = collected.get(sub.OUTPUT_NODE_ID, {})
        for public, internal in self._out_public_to_internal.items():
            for tok in out_coll.get(internal, []):
                self.outs[public].emit(tok.value)

        # 5. Roll up child metrics; propagate failure to the parent run.
        self.total_tokens += result.tokens_emitted
        self.total_cost += result.cost_usd
        if result.error:
            raise RuntimeError(
                f"subgraph {self.node_id!r} failed: {result.error}"
            )

    async def teardown(self) -> None:
        from pharos.directors.base import teardown_all

        await teardown_all(self.subgraph)

    def reset_run_state(self) -> None:
        super().reset_run_state()
        self.subgraph.reset_run_state()
        self.total_tokens = 0
        self.total_cost = 0.0


__all__ = ["SubgraphEntity"]
