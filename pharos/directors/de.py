"""DE (Discrete Event) Director — fire whenever input arrives.

Behavior:
- A node fires only when:
    * Its `ins` buffer has pending tokens, OR
    * It is a source-like node (no `ins` declared, has `outs`) that
      has not yet fired (sources fire exactly once).
- After each round, deliver tokens downstream.
- The run converges when a full pass leaves no node eligible to
  fire (the system is quiescent).
- Hard cap on `max_iterations`.

Use case: incremental workflows where events trickle in from
upstream and downstream entities react only when there's new work.
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


class DEDirector:
    """Discrete-event director.

    Args:
        max_iterations: Hard cap on rounds (default 100).
    """

    name = "de"

    def __init__(self, max_iterations: int = 100) -> None:
        self.max_iterations = max_iterations
        self._edge_index: dict[str, list[tuple[str, str]]] = {}

    async def run(
        self, graph: CompositeGraph, ctx: RunContext
    ) -> RunResult:
        self._edge_index = {}
        for e in graph.edges:
            self._edge_index.setdefault(e.src_node, []).append(
                (e.dst_node, e.dst_port)
            )

        layers = topo_layers(graph)
        # Track which source-like nodes have already fired
        fired_sources: set[str] = set()

        total_tokens = 0
        total_cost = 0.0
        converged = False
        final_iter = 0

        for it in range(self.max_iterations):
            final_iter = it + 1
            step_counter = it * 1000
            fired_this_round = False

            for layer in layers:
                tasks = []
                for nid in layer:
                    node = graph.node(nid)
                    if node.instance is None:
                        continue
                    if not _should_fire(node, fired_sources):
                        continue
                    fired_sources.add(nid)
                    fire_ctx = FireContext(
                        run_id=ctx.run_id,
                        step_id=f"{ctx.run_id}:{step_counter + 1}",
                        iter=it,
                    )
                    tasks.append(
                        asyncio.create_task(
                            _safe_fire(node.instance, fire_ctx, ctx)
                        )
                    )
                if tasks:
                    fired_this_round = True
                    results = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    for r in results:
                        if isinstance(r, BaseException):
                            return RunResult(
                                converged=False,
                                iterations=final_iter,
                                tokens_emitted=total_tokens,
                                cost_usd=total_cost,
                                error=f"{type(r).__name__}: {r}",
                            )
                        if r is not None:
                            total_tokens += r[0]
                            total_cost += r[1]

                await deliver_upstream(graph, self._edge_index)

            if not fired_this_round:
                converged = True
                break

        return RunResult(
            converged=converged,
            iterations=final_iter,
            tokens_emitted=total_tokens,
            cost_usd=total_cost,
        )


def _should_fire(node, fired_sources: set[str]) -> bool:
    """DE firing rule: fire if input is non-empty, or if this is a
    source-like node that hasn't fired yet.
    """
    inst = node.instance
    has_input = any(len(p) > 0 for p in inst.ins.values())
    if has_input:
        return True
    # Source-like: no inputs declared, has outputs, hasn't fired yet
    return bool(
        not inst.ins
        and inst.outs
        and node.id not in fired_sources
    )


async def _safe_fire(
    entity, fire_ctx: FireContext, run_ctx: RunContext
) -> tuple[int, float] | None:
    """Fire with optional trace wrapping."""
    from pharos.observability.trace import current_span

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
                "iter": fire_ctx.iter,
            },
        )
    try:
        await entity.fire(fire_ctx)
    except BaseException as e:
        if span is not None:
            span.record_exception(e)
        raise
    finally:
        if span is not None and tracer is not None:
            tracer.finish_span(span)

    token_count = sum(len(p) for p in entity.outs.values())
    cost = sum(t.cost_usd for p in entity.outs.values() for t in p.peek_all())
    return (token_count, cost)


__all__ = ["DEDirector"]