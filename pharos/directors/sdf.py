"""SDF (Synchronous Data Flow) Director — feedback loops with convergence.

Behavior:
- Runs the graph in rounds (iterations).
- After each round, fingerprint exactly the tokens emitted by each fire.
- A node is "stable" if its per-fire output digest didn't change. The run
  converges when ALL nodes have been stable for K consecutive
  rounds (K-of-N strategy).
- Hard cap on `max_iterations` to prevent runaway.

Use case: reviewer rejects → coder rewrites → reviewer accepts.
Each "round" the graph fires once, and we wait for fixed-point.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from pharos.core.graph import CompositeGraph
from pharos.directors.base import (
    FireContext,
    RunContext,
    RunResult,
    build_edge_index,
    deliver_upstream,
    safe_fire,
    teardown_all,
)


class SDFDirector:
    """Feedback loop with K-of-N token-hash convergence.

    Args:
        max_iterations: Hard cap on rounds (default 20).
        convergence_k: Number of consecutive stable fire outputs before
            the run is considered converged (default 2).
    """

    name = "sdf"

    def __init__(self, max_iterations: int = 20, convergence_k: int = 2) -> None:
        self.max_iterations = max_iterations
        self.convergence_k = convergence_k
        self._edge_index: dict[str, list[tuple[str, str]]] = {}

    async def run(
        self, graph: CompositeGraph, ctx: RunContext
    ) -> RunResult:
        self._edge_index = build_edge_index(graph)

        # Per-node fingerprint of exactly what its most recent fire emitted.
        prev_digests: dict[str, str] = {}
        stable_count: dict[str, int] = defaultdict(int)

        total_tokens = 0
        total_cost = 0.0
        converged = False
        final_iter = 0

        try:
            for it in range(self.max_iterations):
                final_iter = it + 1
                step_counter = it * 1000

                # Fire one round: every node with an instance fires exactly
                # once (in id-sorted order for determinism). We avoid
                # topo_layers here because feedback cycles are common.
                all_nodes = sorted(
                    (
                        nid
                        for nid, n in graph.nodes.items()
                        if n.instance is not None
                    )
                )
                tasks = [
                    asyncio.create_task(
                        safe_fire(
                            graph.node(nid).instance,
                            FireContext(
                                run_id=ctx.run_id,
                                step_id=f"{ctx.run_id}:{step_counter + index + 1}",
                                iter=it,
                                granted_permissions=getattr(
                                    ctx, "granted_permissions", set()
                                ),
                            ),
                            ctx,
                        )
                    )
                    for index, nid in enumerate(all_nodes)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
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

                # Snapshot per-fire digests BEFORE delivery.  safe_fire stores
                # them independently from the mutable output buffers, so this
                # remains correct for connected and unconnected ports alike.
                current_digests = {
                    nid: str(
                        getattr(
                            graph.node(nid).instance,
                            "_last_emission_digest",
                            "",
                        )
                    )
                    for nid in all_nodes
                }

                await deliver_upstream(graph, self._edge_index)

                all_stable = True
                for node_id, new_digest in current_digests.items():
                    if new_digest == prev_digests.get(node_id):
                        stable_count[node_id] += 1
                    else:
                        stable_count[node_id] = 0
                        all_stable = False

                if all_stable and stable_count and all(
                    c >= self.convergence_k for c in stable_count.values()
                ):
                    converged = True
                    break
                prev_digests = current_digests

            return RunResult(
                converged=converged,
                iterations=final_iter,
                tokens_emitted=total_tokens,
                cost_usd=total_cost,
            )
        finally:
            await teardown_all(graph)


__all__ = ["SDFDirector"]
