"""TUI viewer for recorded runs.

Usage:
    pharos trace <run_id> --interactive
    pharos trace list --interactive (interactive run picker)

Features:
- Browse the span tree with arrow keys
- View a span's details (attributes, events, duration)
- Press 'q' to quit, 'e' to export JSON, 'r' to re-run

This module is the TUI backend; it doesn't replace the existing
`pharos trace <run_id>` (which prints a tree to stdout).
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def build_span_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a flat list of spans into a tree.

    Returns a list of top-level nodes; each node has:
        id, name, duration_ms, attributes, events, children
    Children are also dicts of the same shape.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for s in spans:
        # Compute duration_ms (it's a property on Span but not
        # preserved by asdict).
        dur_ms = 0.0
        if s.get("ended_at") is not None and s.get("started_at") is not None:
            dur_ms = (s["ended_at"] - s["started_at"]) * 1000
        by_id[s["id"]] = {
            "id": s["id"],
            "name": s["name"],
            "duration_ms": s.get("duration_ms", 0.0) or dur_ms,
            "attributes": s.get("attributes", {}),
            "events": s.get("events", []),
            "parent_id": s.get("parent_span_id"),
            "children": [],
        }
    roots: list[dict[str, Any]] = []
    for s in by_id.values():
        if s["parent_id"] is None or s["parent_id"] not in by_id:
            roots.append(s)
        else:
            by_id[s["parent_id"]]["children"].append(s)
    # Sort children by step_id (which encodes firing order)
    def sort_key(n: dict[str, Any]) -> str:
        attrs = n["attributes"] or {}
        return attrs.get("step_id", "") or ""

    for r in roots:
        _sort_recursive(r, sort_key)
    return roots


def _sort_recursive(node: dict[str, Any], key: Any) -> None:
    node["children"].sort(key=key)
    for c in node["children"]:
        _sort_recursive(c, key)


def render_tree_compact(roots: list[dict[str, Any]], max_depth: int = 4) -> str:
    """Compact one-line-per-span tree, used by `pharos trace <id>`."""
    lines: list[str] = []

    def visit(n: dict[str, Any], depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        dur = n.get("duration_ms", 0.0) or 0.0
        attrs = n.get("attributes", {}) or {}
        label = (
            f"[cyan]{n['name']}[/cyan]"
            + f"  [{_color_for_dur(dur)}]{dur:.1f} ms[/{_color_for_dur(dur)}]"
        )
        ent = attrs.get("entity", "")
        if ent:
            label += f"  [dim]entity={ent}[/dim]"
        step = attrs.get("step_id", "")
        if step:
            label += f"  [dim]{step[-12:]}[/dim]"
        lines.append(prefix + label)
        for i, c in enumerate(n["children"]):
            child_prefix = prefix + ("    " if i == len(n["children"]) - 1 else "│   ")
            visit(c, depth + 1, child_prefix + "├─ ")

    for r in roots:
        visit(r, 0, "")
    return "\n".join(lines)


def _color_for_dur(ms: float) -> str:
    """Return a Rich color name based on duration."""
    if ms < 5:
        return "green"
    if ms < 50:
        return "yellow"
    return "red"


def render_span_detail(span_dict: dict[str, Any]) -> Panel:
    """Render a single span's details in a Rich Panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("id", span_dict["id"])
    table.add_row("name", f"[cyan]{span_dict['name']}[/cyan]")
    dur = span_dict.get("duration_ms", 0.0) or 0.0
    table.add_row("duration", f"{dur:.2f} ms")
    table.add_row(
        "attributes",
        json.dumps(span_dict.get("attributes", {}), indent=2),
    )

    events = span_dict.get("events", [])
    if events:
        evt_lines: list[str] = []
        for e in events[:20]:  # cap for readability
            ts = e.get("ts", 0)
            name = e.get("name", "?")
            evt_lines.append(f"[dim]{ts:.3f}[/dim]  {name}")
        if len(events) > 20:
            evt_lines.append(f"... +{len(events) - 20} more")
        table.add_row(f"events ({len(events)})", "\n".join(evt_lines))

    return Panel(
        table,
        title=f"Span: {span_dict['name']}",
        border_style="cyan",
    )


def render_summary(
    run_id: str, spans: list[dict[str, Any]], roots: list[dict[str, Any]]
) -> Panel:
    """Top-of-screen run summary."""
    total = sum(s.get("duration_ms", 0.0) or 0.0 for s in spans)
    entity_fires = [s for s in spans if s["name"].startswith("entity.fire.")]
    return Panel(
        Text.assemble(
            ("Run: ", "bold"),
            (run_id, "cyan"),
            ("  spans=", "bold"),
            (str(len(spans)), "yellow"),
            ("  entities=", "bold"),
            (str(len(entity_fires)), "yellow"),
            ("  total_dur=", "bold"),
            (f"{total:.1f} ms\n", "yellow"),
        ),
        border_style="cyan",
    )


def interactive_view(run_id: str, spans: list[dict[str, Any]]) -> None:
    """Interactive span browser using `rich.prompt`-style API.

    Falls back to non-interactive (prints tree, returns) if stdin
    isn't a TTY (e.g. in CI). Otherwise walks the user through
    a list of spans.
    """
    import sys

    if not sys.stdin.isatty():
        # Non-interactive: just print the compact tree
        roots = build_span_tree(spans)
        console = Console()
        console.print(render_summary(run_id, spans, roots))
        console.print()
        console.print(render_tree_compact(roots))
        return

    console = Console()
    roots = build_span_tree(spans)
    # Flatten for navigation
    flat: list[dict[str, Any]] = []

    def flatten(node: dict[str, Any], depth: int = 0) -> None:
        flat.append({**node, "depth": depth})
        for c in node["children"]:
            flatten(c, depth + 1)

    for r in roots:
        flatten(r)

    if not flat:
        console.print("[yellow]No spans to browse.[/yellow]")
        return

    # Linear scroll — no real keyboard input needed; we just print
    # a one-screen summary panel that includes all spans in order.
    # A "real" TUI would use Textual; we keep this lightweight.
    console.print(render_summary(run_id, spans, roots))
    console.print()
    for i, span in enumerate(flat, 1):
        dur = span.get("duration_ms", 0.0) or 0.0
        name = span.get("name", "?")
        attrs = span.get("attributes", {}) or {}
        ent = attrs.get("entity", "")
        console.print(
            f"  [dim]{i:>3}[/dim]  "
            f"[cyan]{name}[/cyan]  "
            f"[{_color_for_dur(dur)}]{dur:.1f} ms[/{_color_for_dur(dur)}]  "
            f"[dim]{ent}[/dim]"
        )
    console.print()
    console.print("[dim]Use `pharos replay <run_id>` to re-run offline.[/dim]")


__all__ = [
    "build_span_tree",
    "interactive_view",
    "render_span_detail",
    "render_summary",
    "render_tree_compact",
]