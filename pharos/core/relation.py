"""A directed edge between two ports.

Edges are explicit data-flow connections. They are stored in
CompositeGraph.edges and consulted by Directors for topological order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """A connection: src_node.src_port -> dst_node.dst_port.

    Both endpoints reference nodes by id. Look up the actual Entity
    objects via CompositeGraph.node(id).
    """

    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    def reverse(self) -> Edge:
        """Return the reverse edge (used for back-edges in SDF)."""
        return Edge(
            src_node=self.dst_node,
            src_port=self.dst_port,
            dst_node=self.src_node,
            dst_port=self.src_port,
        )


__all__ = ["Edge"]
