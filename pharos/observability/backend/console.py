"""Console trace backend — pretty-prints spans as a tree to stderr.

Useful for development; not intended for production.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pharos.observability.trace import Span


class ConsoleTraceBackend:
    """Renders finished spans as a tree.

    Example output:
        trace_0001_a1b2c3
        ├─ llm.call (gpt-4o)        1234.5 ms
        │  ├─ llm.first_token         345.2 ms
        │  └─ llm.done                888.3 ms
        └─ tool.bash                  50.0 ms
    """

    name = "console"

    def __init__(self, sink: Any = None) -> None:
        # Default sink writes to stderr via a tiny helper to avoid
        # a hard print at import time.
        if sink is None:
            def _stderr(line: str) -> None:
                import sys
                print(line, file=sys.stderr)
            sink = _stderr
        self.sink = sink
        self.spans: list[Span] = []

    def write(self, span: Span) -> None:
        """Called by Tracer.finish_span."""
        self.spans.append(span)

    def render(self) -> str:
        """Render the full tree as a string."""
        if not self.spans:
            return "(no spans)"

        # Group by trace
        traces: dict[str, list[Span]] = defaultdict(list)
        for s in self.spans:
            traces[s.trace_id].append(s)

        out: list[str] = []
        for trace_id, span_list in traces.items():
            out.append(trace_id)
            # Build child map
            children: dict[str | None, list[Span]] = defaultdict(list)
            for s in span_list:
                children[s.parent_span_id].append(s)
            # Render starting from roots
            roots = children.get(None)
            if not roots:
                # Defensive: if parent_span_id points to a span we lost,
                # treat the orphan as a root.
                roots = [
                    s
                    for s in span_list
                    if all(s.parent_span_id != other.id for other in span_list)
                ]
            for root in sorted(roots, key=lambda s: s.started_at):
                self._render_node(root, children, out, prefix="├─ ", is_last=False)
        return "\n".join(out)

    def _render_node(
        self,
        span: Span,
        children: dict[str | None, list[Span]],
        out: list[str],
        prefix: str,
        is_last: bool,
    ) -> None:
        # Top-level spans get the trace_id header; children get indentation
        indent = "" if prefix.startswith("├─") and not span.parent_span_id else "│  "
        attrs = " ".join(f"{k}={v}" for k, v in span.attributes.items())
        status = f" [{span.status}]" if span.status != "ok" else ""
        out.append(
            f"{prefix}{span.name}  {span.duration_ms:.1f} ms  {attrs}{status}"
        )
        kids = children.get(span.id, [])
        for i, child in enumerate(sorted(kids, key=lambda s: s.started_at)):
            is_child_last = i == len(kids) - 1
            child_prefix = indent + ("└─ " if is_child_last else "├─ ")
            self._render_node(child, children, out, child_prefix, is_child_last)


__all__ = ["ConsoleTraceBackend"]
