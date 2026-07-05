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
        self, node_id: str, subgraph: CompositeGraph, exports: dict[str, str]
    ) -> None:
        """Add a nested subgraph. `exports` maps public_name -> "node.port"."""
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id!r}")
        for public_name, internal in exports.items():
            internal_node, _, internal_port = internal.partition(".")
            if internal_node not in subgraph.nodes:
                raise ValueError(
                    f"export {public_name!r} references unknown node "
                    f"{internal_node!r} in subgraph {subgraph.name!r}"
                )
            self.exports[public_name] = (node_id, internal_port)
        self.nodes[node_id] = GraphNode(
            id=node_id, kind=NodeKind.SUBGRAPH, subgraph=subgraph
        )
        self._nx.add_node(node_id)

    # ---------- edge management ----------

    def connect(self, src: str, dst: str) -> None:
        """Connect src="node.port" to dst="node.port"."""
        src_node, _, src_port = src.partition(".")
        dst_node, _, dst_port = dst.partition(".")
        self._check_node(src_node, "src")
        self._check_node(dst_node, "dst")
        self._check_port(self.nodes[src_node], src_port, "src")
        self._check_port(self.nodes[dst_node], dst_port, "dst")
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
