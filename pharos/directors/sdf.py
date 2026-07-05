"""SDF (Synchronous Data Flow) Director — feedback loops with convergence.

Behavior:
- Runs the graph in rounds (iterations).
- After each round, compare the head token of every output port to
  the previous round's head token (via `self_hash`).
- A node is "stable" if its head hash didn't change. The run
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
        convergence_k: Number of consecutive stable rounds before
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

        # Per (node, port) head hash from the previous round
        prev_hashes: dict[tuple[str, str], str] = {}
        # Per (node, port) how many consecutive rounds it has been stable
        stable_count: dict[tuple[str, str], int] = defaultdict(int)

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
                fire_ctx = FireContext(
                    run_id=ctx.run_id,
                    step_id=f"{ctx.run_id}:{step_counter + 1}",
                    iter=it,
                    granted_permissions=getattr(
                        ctx, "granted_permissions", set()
                    ),
                )
                tasks = [
                    asyncio.create_task(
                        safe_fire(graph.node(nid).instance, fire_ctx, ctx)
                    )
                    for nid in all_nodes
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

                await deliver_upstream(graph, self._edge_index)

                # AFTER fire+deliver: snapshot head hashes and compare
                current_hashes: dict[tuple[str, str], str] = {}
                for node_id, node in graph.nodes.items():
                    if node.instance is None:
                        continue
                    for port_name, port in node.instance.outs.items():
                        head = port.peek()
                        if head is not None:
                            current_hashes[(node_id, port_name)] = head.self_hash

                # Compare new head hashes to previous
                all_stable = True
                for key, new_hash in current_hashes.items():
                    old_hash = prev_hashes.get(key)
                    if new_hash == old_hash:
                        stable_count[key] += 1
                    else:
                        stable_count[key] = 0
                        all_stable = False
                # If a key vanished, reset its count
                for key in prev_hashes:
                    if key not in current_hashes:
                        stable_count[key] = 0
                        all_stable = False

                if all_stable and stable_count and all(
                    c >= self.convergence_k for c in stable_count.values()
                ):
                    converged = True
                    break
                prev_hashes = current_hashes

            return RunResult(
                converged=converged,
                iterations=final_iter,
                tokens_emitted=total_tokens,
                cost_usd=total_cost,
            )
        finally:
            await teardown_all(graph)


__all__ = ["SDFDirector"]
