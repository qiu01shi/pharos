"""CompositeGraph — the static structure of a pharos workflow.

Wraps a NetworkX DiGraph for topology queries (cycle detection, topo sort,
reachability) and stores node metadata for runtime use.

Node kinds:
    ENTITY   — a leaf entity (has ins/outs)
    SUBGRAPH — a nested graph (exposes transparent ports)
    INPUT    — virtual __in__ node, feeds the graph from outside
    OUTPUT   — virtual __out__ node, collects the graph's outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import networkx as nx

from pharos.core.entity import Entity
from pharos.core.relation import Edge
from pharos.core.schema import compatibility_errors


class NodeKind(StrEnum):
    """Discriminator for graph nodes."""

    ENTITY = "entity"
    SUBGRAPH = "subgraph"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class GraphNode:
    """A node in the graph. For ENTITY/SUBGRAPH, `instance` is set.

    For INPUT/OUTPUT, `instance` is None and `port_name` is used.
    """

    id: str
    kind: NodeKind
    instance: Entity | None = None
    subgraph: CompositeGraph | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CompositeGraph:
    """A typed dataflow graph: nodes + edges + export ports.

    Topological analysis uses NetworkX. Runtime iteration is performed
    by a Director that consults `topo_order()` and `entry_nodes()`.

    Examples:
        g = CompositeGraph(name="hello")
        g.add_entity("agent", agent_instance)
        g.connect("__in__.prompt", "agent.prompt")
        g.connect("agent.text", "__out__.text")
    """

    INPUT_NODE_ID = "__in__"
    OUTPUT_NODE_ID = "__out__"

    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[Edge] = []
        # NetworkX mirror for analysis
        self._nx: nx.DiGraph[str] = nx.DiGraph()
        # Virtual IO nodes
        self._add_io_nodes()
        # Exports: public_name -> (internal_node, internal_port)
        self.exports: dict[str, tuple[str, str]] = {}
        # Runtime output collection.  Declaring it here makes the lifecycle
        # explicit and lets reset_run_state() reliably prepare graph reuse.
        self.collected: dict[str, dict[str, list[Any]]] = {}

    # ---------- node management ----------

    def _add_io_nodes(self) -> None:
        self.nodes[self.INPUT_NODE_ID] = GraphNode(
            id=self.INPUT_NODE_ID, kind=NodeKind.INPUT
        )
        self.nodes[self.OUTPUT_NODE_ID] = GraphNode(
            id=self.OUTPUT_NODE_ID, kind=NodeKind.OUTPUT
        )
        self._nx.add_node(self.INPUT_NODE_ID)
        self._nx.add_node(self.OUTPUT_NODE_ID)

    def add_entity(self, node_id: str, entity: Entity) -> None:
        """Add a leaf entity. node_id must be unique within the graph."""
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id!r}")
        if entity.node_id != node_id:
            # Sync the entity's id with what the user named it in the graph
            entity.node_id = node_id
        self.nodes[node_id] = GraphNode(
            id=node_id, kind=NodeKind.ENTITY, instance=entity
        )
        self._nx.add_node(node_id)

    def add_subgraph(
        self,
        node_id: str,
        subgraph: CompositeGraph,
        director_name: str = "fn",
        input_map: dict[str, str] | None = None,
        output_map: dict[str, str] | None = None,
        max_iters: int = 20,
        converge_k: int = 2,
    ) -> None:
        """Embed a nested graph as one composite node.

        The subgraph is wrapped in a `SubgraphEntity` (stored as the node's
        `instance`) so every Director schedules it like any other node. Its
        public ports are derived from the child's `__in__` / `__out__` edges
        unless `input_map` / `output_map` (public_name -> internal_port)
        override them. `director_name` selects the child's own scheduler,
        enabling heterogeneous nested scheduling.
        """
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id!r}")
        # Lazy import: core must not import entities at module load time.
        from pharos.entities.subgraph import SubgraphEntity

        inst = SubgraphEntity(
            node_id=node_id,
            subgraph=subgraph,
            director_name=director_name,
            input_map=input_map,
            output_map=output_map,
            max_iters=max_iters,
            converge_k=converge_k,
        )
        self.nodes[node_id] = GraphNode(
            id=node_id,
            kind=NodeKind.SUBGRAPH,
            instance=inst,
            subgraph=subgraph,
        )
        self._nx.add_node(node_id)
        # Record public output ports as exports for introspection.
        for public, internal in inst._out_public_to_internal.items():
            self.exports[public] = (node_id, internal)

    # ---------- edge management ----------

    def connect(self, src: str, dst: str) -> None:
        """Connect src="node.port" to dst="node.port"."""
        src_node, _, src_port = src.partition(".")
        dst_node, _, dst_port = dst.partition(".")
        self._check_node(src_node, "src")
        self._check_node(dst_node, "dst")
        self._check_port(self.nodes[src_node], src_port, "src")
        self._check_port(self.nodes[dst_node], dst_port, "dst")
        self._check_contract(src_node, src_port, dst_node, dst_port)
        edge = Edge(
            src_node=src_node, src_port=src_port,
            dst_node=dst_node, dst_port=dst_port,
        )
        self.edges.append(edge)
        self._nx.add_edge(src_node, dst_node)

    def _check_node(self, node_id: str, role: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"{role} node {node_id!r} not in graph")

    def _check_port(
        self, node: GraphNode, port_name: str, role: str
    ) -> None:
        # Best-effort: we don't have per-port metadata for subgraphs,
        # so we just check the internal entity's ports.
        if (
            node.kind in (NodeKind.ENTITY, NodeKind.SUBGRAPH)
            and node.instance is not None
        ):
            ports = node.instance.ins if role == "dst" else node.instance.outs
            if port_name not in ports:
                raise ValueError(
                    f"{role} port {port_name!r} not declared on "
                    f"entity {node.id!r} ({sorted(ports)})"
                )

    def _check_contract(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str
    ) -> None:
        """Reject port contracts that are provably incompatible."""
        source = self.nodes[src_node].instance
        target = self.nodes[dst_node].instance
        if source is None or target is None:
            return
        source_port = source.outs[src_port]
        target_port = target.ins[dst_port]
        source_types = set(source_port.accepted_types)
        target_types = set(target_port.accepted_types)
        if source_types and target_types and source_types.isdisjoint(target_types):
            raise ValueError(
                f"type contract mismatch: {src_node}.{src_port} emits "
                f"{sorted(source_types)} but {dst_node}.{dst_port} accepts "
                f"{sorted(target_types)}"
            )
        if source_port.schema is not None and target_port.schema is not None:
            errors = compatibility_errors(source_port.schema, target_port.schema)
            if errors:
                raise ValueError(
                    f"schema contract mismatch: {src_node}.{src_port} -> "
                    f"{dst_node}.{dst_port}: {'; '.join(errors)}"
                )

    # ---------- graph analysis ----------

    def topo_order(self) -> list[str]:
        """Return nodes in topological order (entry → exit).

        Raises NetworkXUnfeasible if the graph has a cycle.
        """
        return list(nx.topological_sort(self._nx))

    def has_cycle(self) -> bool:
        return not nx.is_directed_acyclic_graph(self._nx)

    def cycles(self) -> list[list[str]]:
        return list(nx.simple_cycles(self._nx))

    def entry_nodes(self) -> list[str]:
        """Nodes with no incoming edges (sources)."""
        return [n for n in self._nx.nodes if self._nx.in_degree(n) == 0]

    def exit_nodes(self) -> list[str]:
        """Nodes with no outgoing edges (sinks)."""
        return [n for n in self._nx.nodes if self._nx.out_degree(n) == 0]

    def downstream(self, node_id: str) -> list[str]:
        """All nodes reachable from `node_id`."""
        return list(nx.descendants(self._nx, node_id))

    def upstream(self, node_id: str) -> list[str]:
        """All nodes that can reach `node_id`."""
        return list(nx.ancestors(self._nx, node_id))

    def node(self, node_id: str) -> GraphNode:
        """Look up a node by id. Raises KeyError if not present."""
        if node_id not in self.nodes:
            raise KeyError(f"node {node_id!r} not in graph {self.name!r}")
        return self.nodes[node_id]

    def reset_run_state(self) -> None:
        """Prepare this graph instance for a new, independent run.

        Call this *before* seeding the next run's inputs.  Graph topology and
        entity configuration are preserved; port buffers, collected outputs,
        Director bookkeeping, and entity-specific transient counters are
        cleared.  SubgraphEntity propagates the reset into its child graph.
        """
        self.collected.clear()
        for node in self.nodes.values():
            if node.instance is not None:
                node.instance.reset_run_state()

    def validate(self) -> list[str]:
        """Static checks. Returns a list of error messages (empty if valid)."""
        errors: list[str] = []
        if self.has_cycle():
            errors.append(f"graph has cycles: {self.cycles()}")
        for node_id, node in self.nodes.items():
            if node.kind == NodeKind.ENTITY and node.instance is None:
                errors.append(f"entity node {node_id!r} has no instance")
        return errors

    def __repr__(self) -> str:
        return (
            f"CompositeGraph(name={self.name!r}, "
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )


__all__ = ["CompositeGraph", "GraphNode", "NodeKind"]
