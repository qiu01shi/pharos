"""Directors drive the execution of a CompositeGraph.

A Director is the "scheduler" in pharos — it decides when each Entity
fires, how data flows between ports, and when a run terminates.

Five director semantics, one Protocol:
- FN  (Function)    : fire each entity once in topological order;
                      same layer runs concurrently via asyncio.gather.
- DE  (Discrete Event): fire an entity whenever an upstream port
                      receives a new token; run until all buffers drain.
- PN  (Process Network): each entity is a long-lived process; messages
                      flow over a bus; only stops on explicit signal.
- SDF (Synchronous Data Flow): fire on data availability with
                      production/consumption rates; support feedback
                      loops; converge via K-of-N token-hash check.
- CT  (Continuous Time): fire on external events (WebSocket, file watch,
                      cron); never stops on its own.

The Protocol is intentionally minimal so each director can specialize.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pharos.core.graph import CompositeGraph


@dataclass
class RunContext:
    """Per-run context set by a Director before setup().

    Concrete runs may add fields (tracer, config); this base carries
    just what every Director needs.

    Permissions:
        `granted_permissions` is a set of permission strings the run
        is authorized to use. The Director checks each Entity's
        `required_permissions` against this set during setup(); if
        any required permission is missing, the run fails with
        PermissionError before any entity is fired.

        An empty set means "no permissions granted" — entities
        requiring any permission will be denied. A run that
        intends to run without constraints should pass
        `granted_permissions=set()` (the default) and ensure no
        entity requires permissions.
    """

    run_id: str
    started_at: float = field(default_factory=time.time)
    config: dict[str, Any] = field(default_factory=dict)
    granted_permissions: set[str] = field(default_factory=set)


@dataclass
class FireContext:
    """Per-fire context passed to Entity.fire()."""

    run_id: str
    step_id: str
    iter: int = 0
    started_at: float = field(default_factory=time.time)


@dataclass
class RunResult:
    """Result of a Director.run() invocation."""

    converged: bool
    iterations: int
    tokens_emitted: int
    cost_usd: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Director(Protocol):
    """All directors implement this Protocol.

    `run` is the entry point. It must return a RunResult, never raise
    on business errors (those become error in RunResult).
    """

    name: str

    async def run(
        self, graph: CompositeGraph, ctx: RunContext
    ) -> RunResult: ...


# ---------- helpers shared by FN/DE/SDF ----------

async def deliver_upstream(
    graph: CompositeGraph,
    edge_index: dict[str, list[tuple[str, str]]],
) -> None:
    """Push every output port's tokens to downstream input ports.

    `edge_index` maps src_node -> [(dst_node, dst_port), ...] for O(1)
    lookups. For each (src_node, src_port) we drain its tokens once
    and fan-out a copy to every downstream edge — so a single source
    can feed multiple sinks without losing tokens.

    Single-connection fan-out: a token emitted once is replicated
    once per downstream edge. After delivery the source port's buffer
    is empty; downstream buffers contain one token per (edge, source-token).

    For virtual nodes (__in__ / __out__) that have no instance, we
    collect received tokens onto `graph.collected[<port_name>]` so the
    CLI can display them after the run.
    """
    # Group: (src_node, src_port) -> [(dst_node, dst_port), ...]
    by_source: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for edge in graph.edges:
        by_source.setdefault((edge.src_node, edge.src_port), []).append(
            (edge.dst_node, edge.dst_port)
        )

    for (src_node_id, src_port_name), destinations in by_source.items():
        src_node = graph.node(src_node_id)
        if src_node.instance is None:
            continue
        src_port = src_node.instance.outs.get(src_port_name)
        if src_port is None or not src_port.buffer:
            continue
        # Snapshot the tokens (so each downstream gets a copy)
        tokens = list(src_port.buffer)
        src_port.buffer.clear()
        # Deliver a copy to each downstream
        for dst_node_id, dst_port_name in destinations:
            dst_node = graph.node(dst_node_id)
            if dst_node.instance is None:
                # Virtual node (e.g. __out__): collect onto the graph
                # so the CLI / Replay can read the result.
                if not hasattr(graph, "collected"):
                    graph.collected = {}  # type: ignore[attr-defined]
                coll = graph.collected.setdefault(  # type: ignore[attr-defined]
                    dst_node_id, {}
                )
                coll.setdefault(dst_port_name, []).extend(tokens)
                continue
            dst_port = dst_node.instance.ins.get(dst_port_name)
            if dst_port is None:
                continue
            for tok in tokens:
                dst_port.receive(tok)


def topo_layers(graph: CompositeGraph) -> list[list[str]]:
    """Group nodes into parallel-execution layers (Kahn's algorithm).

    A layer is a set of nodes whose predecessors are all in earlier
    layers. Each layer can run concurrently via asyncio.gather.
    """
    # Work on a copy so we don't mutate the graph
    in_degree: dict[str, int] = {
        n: graph._nx.in_degree(n)  # type: ignore[attr-defined]
        for n in graph._nx.nodes  # type: ignore[attr-defined]
    }
    successors: dict[str, list[str]] = {n: [] for n in in_degree}
    for src, dst in graph._nx.edges:  # type: ignore[attr-defined]
        successors[src].append(dst)

    layers: list[list[str]] = []
    current_layer = [n for n, d in in_degree.items() if d == 0]
    visited: set[str] = set()
    while current_layer:
        layers.append(sorted(current_layer))  # deterministic order
        visited.update(current_layer)
        next_layer: list[str] = []
        for n in current_layer:
            for s in successors[n]:
                in_degree[s] -= 1
                if in_degree[s] == 0 and s not in visited:
                    next_layer.append(s)
        current_layer = next_layer
    return layers


__all__ = [
    "Director",
    "FireContext",
    "RunContext",
    "RunResult",
    "deliver_upstream",
    "topo_layers",
]
