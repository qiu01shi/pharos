"""FN (Function) Director — one-shot topological execution.

Behavior:
- Build layers via topological sort
- Each layer's entities fire concurrently via asyncio.gather
- Between layers, deliver all output port tokens to downstream input ports
- Stop after the final layer

Use case: batch workflows (one-shot generation, batch ETL, deterministic
pipelines). For feedback loops use SDF; for streams use DE.
"""

from __future__ import annotations

import asyncio

from pharos.core.graph import CompositeGraph
from pharos.directors.base import (
    FireContext,
    RunContext,
    RunResult,
    build_edge_index,
    deliver_upstream,
    safe_fire,
    teardown_all,
    topo_layers,
)


class FNDirector:
    """One-shot topological execution with same-layer concurrency."""

    name = "fn"

    def __init__(self) -> None:
        # Edge index: src_node -> [(dst_node, dst_port), ...]
        self._edge_index: dict[str, list[tuple[str, str]]] = {}

    async def run(
        self, graph: CompositeGraph, ctx: RunContext
    ) -> RunResult:
        # Build edge index once per run
        self._edge_index = build_edge_index(graph)

        layers = topo_layers(graph)
        if not layers:
            return RunResult(
                converged=True, iterations=0, tokens_emitted=0, cost_usd=0.0
            )

        # Pre-deliver __in__ tokens (if any) — they live in graph nodes
        # but the FN model treats them as already "delivered" to entry
        # nodes via the loader, so no-op here.

        total_tokens = 0
        total_cost = 0.0
        step_counter = 0

        try:
            for layer_idx, layer in enumerate(layers):
                # Build (entity, fire_ctx) pairs for this layer
                tasks: list[asyncio.Task] = []
                for node_id in layer:
                    node = graph.node(node_id)
                    if node.instance is None:
                        continue
                    step_counter += 1
                    fire_ctx = FireContext(
                        run_id=ctx.run_id,
                        step_id=f"{ctx.run_id}:{step_counter}",
                        iter=layer_idx,
                        granted_permissions=getattr(
                            ctx, "granted_permissions", set()
                        ),
                    )
                    tasks.append(
                        asyncio.create_task(
                            safe_fire(node.instance, fire_ctx, ctx)
                        )
                    )
                if tasks:
                    results = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    # Aggregate
                    for r in results:
                        if isinstance(r, BaseException):
                            return RunResult(
                                converged=False,
                                iterations=layer_idx + 1,
                                tokens_emitted=total_tokens,
                                cost_usd=total_cost,
                                error=f"{type(r).__name__}: {r}",
                            )
                        if r is not None:
                            total_tokens += r[0]
                            total_cost += r[1]

                # Deliver upstream after layer completes
                await deliver_upstream(graph, self._edge_index)
            return RunResult(
                converged=True,
                iterations=len(layers),
                tokens_emitted=total_tokens,
                cost_usd=total_cost,
            )
        finally:
            await teardown_all(graph)


__all__ = ["FNDirector"]
