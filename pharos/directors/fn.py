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
    deliver_upstream,
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
        self._edge_index = {}
        for e in graph.edges:
            self._edge_index.setdefault(e.src_node, []).append(
                (e.dst_node, e.dst_port)
            )

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
                )
                tasks.append(
                    asyncio.create_task(
                        _safe_fire(node.instance, fire_ctx, ctx)
                    )
                )
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
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


async def _safe_fire(
    entity, fire_ctx: FireContext, run_ctx: RunContext
) -> tuple[int, float] | None:
    """Run entity.setup/fire/teardown safely. Returns (token_count, cost).

    Captures exceptions and re-raises so the gather() can fail the run.
    Every fire is wrapped in a span (if a tracer is registered on the
    run context) so post-hoc analysis and replay have a record.

    Permission check:
        Before setup(), we compare the entity's `required_permissions`
        against the run context's `granted_permissions`. If any
        required permission is missing, raise PermissionError
        immediately — better than failing partway through a run.
    """
    from pharos.observability.trace import current_span

    # Permission check (BEFORE setup)
    required = getattr(entity, "required_permissions", set()) or set()
    granted = getattr(run_ctx, "granted_permissions", set()) or set()
    missing = required - granted
    if missing:
        raise PermissionError(
            f"entity {entity.node_id!r} ({type(entity).__name__}) "
            f"requires {sorted(missing)} but run only grants "
            f"{sorted(granted) if granted else 'no permissions'}"
        )

    if not getattr(entity, "_initialized", False):
        await entity.setup(run_ctx)
        entity._initialized = True  # type: ignore[attr-defined]

    tracer = getattr(run_ctx, "tracer", None)
    parent = current_span()
    span = None
    if tracer is not None:
        span = tracer.start_span(
            f"entity.fire.{entity.node_id}",
            parent=parent,
            attributes={
                "entity": entity.node_id,
                "entity_class": type(entity).__name__,
                "step_id": fire_ctx.step_id,
            },
        )
    try:
        await entity.fire(fire_ctx)
    except BaseException as e:
        if span is not None:
            span.record_exception(e)
        raise
    finally:
        if span is not None:
            tracer.finish_span(span)

    token_count = sum(len(p) for p in entity.outs.values())
    cost = sum(t.cost_usd for p in entity.outs.values() for t in p.peek_all())
    return (token_count, cost)


__all__ = ["FNDirector"]
