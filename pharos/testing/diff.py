"""Structured diff between two recorded runs — the "git diff" for agents.

Two runs' ``outputs`` maps are aligned by ``(node_id, fire_index, port)``
rather than compared as opaque blobs, so a diff can say exactly *which node's
which output port changed*, down to a JSON field path or a text line. When a
graph is supplied, changes are traced forward along ``graph.edges`` so you can
see how one node's drift propagates downstream.

This is the primitive behind ``pharos diff`` and the verdict engine behind
``pharos test --live``.
"""

from __future__ import annotations

import difflib
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pharos.testing.digest import Outputs, canonical_payload

if TYPE_CHECKING:
    from pharos.core.graph import CompositeGraph


@dataclass
class FieldChange:
    """One field-level change inside a JSON payload."""

    path: str  # e.g. "$.line" or "$.items[2].name"
    kind: str  # "added" | "removed" | "changed"
    before: Any = None
    after: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class PortDiff:
    """The difference in one node/port between two runs."""

    node_id: str
    fire_index: int
    port: str
    status: str  # "added" | "removed" | "changed"
    kind: str  # "json" | "text" | "value"
    before: Any = None
    after: Any = None
    field_changes: list[FieldChange] = field(default_factory=list)
    line_diff: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "fire_index": self.fire_index,
            "port": self.port,
            "status": self.status,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "field_changes": [c.to_dict() for c in self.field_changes],
            "line_diff": self.line_diff,
        }


@dataclass
class RunDiff:
    """The full structured difference between two runs."""

    port_diffs: list[PortDiff] = field(default_factory=list)
    changed_nodes: set[str] = field(default_factory=set)
    # changed node -> downstream nodes reachable from it (via graph edges)
    propagation: dict[str, list[str]] = field(default_factory=dict)

    def has_changes(self) -> bool:
        return bool(self.port_diffs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_changes": self.has_changes(),
            "changed_nodes": sorted(self.changed_nodes),
            "port_diffs": [d.to_dict() for d in self.port_diffs],
            "propagation": {k: sorted(v) for k, v in self.propagation.items()},
        }


def _parse_key(key: str) -> tuple[str, int]:
    """`"<node_id>:<fire_index>"` -> `(node_id, fire_index)`.

    ``node_id`` may carry a subgraph prefix like ``reviewer/coder``; only the
    trailing ``:<int>`` is the fire index.
    """
    node, _, fire = key.rpartition(":")
    if not node:
        return key, 0
    try:
        return node, int(fire)
    except ValueError:
        return key, 0


def _payloads_by_port(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Group a fire's emitted-token records into ``port -> [payload, ...]``."""
    out: dict[str, list[Any]] = {}
    for rec in records:
        out.setdefault(rec.get("port", ""), []).append(rec.get("payload"))
    return out


def diff_json(before: Any, after: Any, path: str = "$") -> list[FieldChange]:
    """Recursive field-level diff of two JSON-ish values."""
    changes: list[FieldChange] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for k in sorted(set(before) | set(after), key=str):
            child = f"{path}.{k}"
            if k not in after:
                changes.append(FieldChange(child, "removed", before[k], None))
            elif k not in before:
                changes.append(FieldChange(child, "added", None, after[k]))
            else:
                changes.extend(diff_json(before[k], after[k], child))
    elif isinstance(before, list) and isinstance(after, list):
        for i in range(max(len(before), len(after))):
            child = f"{path}[{i}]"
            if i >= len(after):
                changes.append(FieldChange(child, "removed", before[i], None))
            elif i >= len(before):
                changes.append(FieldChange(child, "added", None, after[i]))
            else:
                changes.extend(diff_json(before[i], after[i], child))
    elif before != after:
        changes.append(FieldChange(path, "changed", before, after))
    return changes


def _diff_text(before: Any, after: Any) -> list[str]:
    """Unified line diff between two text payloads."""
    return list(
        difflib.unified_diff(
            str(before).splitlines(),
            str(after).splitlines(),
            fromfile="a",
            tofile="b",
            lineterm="",
            n=1,
        )
    )


def _compare_port(
    node_id: str,
    fire_index: int,
    port: str,
    before: list[Any] | None,
    after: list[Any] | None,
) -> PortDiff | None:
    """Compare one port's payload list across two runs; None if unchanged."""
    if before is None:
        return PortDiff(node_id, fire_index, port, "added", "value", after=after)
    if after is None:
        return PortDiff(node_id, fire_index, port, "removed", "value", before=before)
    if canonical_payload(before) == canonical_payload(after):
        return None

    # Changed. Use the richest diff available for the common single-value case.
    if len(before) == 1 and len(after) == 1:
        b, a = before[0], after[0]
        if isinstance(b, (dict, list)) and isinstance(a, (dict, list)):
            return PortDiff(
                node_id, fire_index, port, "changed", "json",
                before=b, after=a, field_changes=diff_json(b, a),
            )
        if isinstance(b, str) and isinstance(a, str):
            return PortDiff(
                node_id, fire_index, port, "changed", "text",
                before=b, after=a, line_diff=_diff_text(b, a),
            )
    return PortDiff(
        node_id, fire_index, port, "changed", "value", before=before, after=after
    )


def _downstream(graph: CompositeGraph, start: str) -> list[str]:
    """All nodes transitively reachable downstream of ``start`` via edges."""
    adj: dict[str, set[str]] = {}
    for e in graph.edges:
        adj.setdefault(e.src_node, set()).add(e.dst_node)
    seen: set[str] = set()
    q: deque[str] = deque(adj.get(start, set()))
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        q.extend(adj.get(n, set()))
    return sorted(seen)


def diff_runs(
    before: Outputs,
    after: Outputs,
    graph: CompositeGraph | None = None,
) -> RunDiff:
    """Structured diff of two runs' outputs, aligned by node/fire/port.

    When ``graph`` is given, ``propagation`` maps each changed node to the
    downstream nodes reachable from it, so a reviewer can see the blast radius
    of a change rather than just the change itself.
    """
    result = RunDiff()
    for key in sorted(set(before) | set(after)):
        node_id, fire_index = _parse_key(key)
        b_ports = _payloads_by_port(before.get(key, []))
        a_ports = _payloads_by_port(after.get(key, []))
        for port in sorted(set(b_ports) | set(a_ports)):
            pd = _compare_port(
                node_id, fire_index, port,
                b_ports.get(port), a_ports.get(port),
            )
            if pd is not None:
                result.port_diffs.append(pd)
                result.changed_nodes.add(node_id)

    if graph is not None:
        for node_id in result.changed_nodes:
            downstream = _downstream(graph, node_id)
            if downstream:
                result.propagation[node_id] = downstream
    return result


__all__ = [
    "FieldChange",
    "PortDiff",
    "RunDiff",
    "diff_json",
    "diff_runs",
]
