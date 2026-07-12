"""pharos CLI — run / validate / inspect / trace commands."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from pharos import __version__
from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunBudget
from pharos.ir import load_graph, load_graph_from_text
from pharos.observability.backend.console import ConsoleTraceBackend

app = typer.Typer(
    name="pharos",
    help="Typed dataflow runtime for LLM workflows.",
    no_args_is_help=False,
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pharos {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True
    ),
) -> None:
    """pharos — run LLM workflows defined in YAML."""
    if ctx.invoked_subcommand is None and not version:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def run(
    graph: Path = typer.Argument(..., help="Path to YAML graph"),
    input: str = typer.Option("", "--input", "-i", help="Initial prompt text"),
    input_extra: list[str] = typer.Option(
        [],
        "--input-extra",
        help=(
            "Extra inputs as port=value (repeatable). "
            "E.g. --input-extra text=hello --input-extra msg=world. "
            "Seeds `__in__.<port>` for any port not in --input."
        ),
    ),
    grant: list[str] = typer.Option(
        [],
        "--grant",
        help=(
            "Grant a permission to this run (repeatable). "
            "Entities declaring `required_permissions` will only run "
            "if every required permission has been granted. "
            "E.g. --grant shell:execute"
        ),
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        help=(
            "Variable substitution as key=value (repeatable). "
            "Replaces `${var_name}` and `$var_name` in the graph YAML "
            "before loading."
        ),
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    trace: bool = typer.Option(False, "--trace", help="Show trace after run"),
    max_iters: int | None = typer.Option(
        None, "--max-iters", help="Override graph execution.max_iterations"
    ),
    converge_k: int | None = typer.Option(
        None, "--converge-k", help="Override graph execution.convergence_k"
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Abort the run above this USD cost"
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Abort the run above this token count"
    ),
    budget_mode: str = typer.Option(
        "hard", "--budget-mode", help="Budget enforcement: 'hard' or 'soft'"
    ),
    answer: list[str] = typer.Option(
        [],
        "--answer",
        help="Preset a human node's answer as node=value (repeatable)",
    ),
    record_fixture: Path | None = typer.Option(
        None,
        "--record-fixture",
        help="Also write this run as an Agent-CI golden fixture (JSON path)",
    ),
    fixture_name: str | None = typer.Option(
        None,
        "--fixture-name",
        help="Name for the recorded fixture (default: graph name)",
    ),
) -> None:
    """Load a graph and run it once with --input as the initial prompt."""
    raw_text = _substitute_vars(graph, var)
    # Pass the graph's directory so `type: subgraph` refs resolve relative
    # to the main graph file.
    g, raw = load_graph_from_text(raw_text, base_dir=Path(graph).parent)
    seed_map: dict[str, str] = {}
    if input:
        seed_map["prompt"] = input
    for item in input_extra:
        if "=" not in item:
            console.print(
                f"[red]Invalid --input-extra (expected port=value): {item!r}[/red]"
            )
            raise typer.Exit(code=1)
        k, _, v = item.partition("=")
        seed_map[k.strip()] = v
    _seed_inputs(g, seed_map)
    _apply_answers(g, answer)
    director_name = raw.get("director", "fn")
    execution = raw.get("execution", {}) or {}
    effective_max_iters = max_iters or int(execution.get("max_iterations", 20))
    effective_converge_k = converge_k or int(execution.get("convergence_k", 2))
    budget = _build_budget(raw.get("budget"), max_cost, max_tokens, budget_mode)
    captured: dict[str, Any] = {}
    result, _backend = asyncio.run(
        _run_with_trace(
            g,
            director_name,
            trace,
            effective_max_iters,
            effective_converge_k,
            granted_permissions=set(grant),
            budget=budget,
            outputs_out=captured if record_fixture else None,
        )
    )
    if record_fixture:
        from pharos.testing.runner import fixture_from_outputs

        fx = fixture_from_outputs(
            g,
            graph,
            director_name,
            captured,
            name=fixture_name or g.name,
            seed=seed_map,
            grant=list(grant),
            var=_parse_kv(var),
        )
        fx.save(record_fixture)
        console.print(
            f"[green]Recorded fixture[/green] {record_fixture} "
            f"([cyan]{len(fx.outputs)} node-fires[/cyan], "
            f"digest={fx.chain_digest[:12]}…)"
        )
    if json_out:
        typer.echo(json.dumps(_result_to_dict(result), indent=2, default=str))
    else:
        _print_summary(g, result, director_name)


@app.command()
def validate(
    graph: Path = typer.Argument(..., help="Path to YAML graph"),
) -> None:
    """Validate a graph without running it."""
    try:
        g, _ = load_graph(graph)
        errors = g.validate()
        if errors:
            console.print(f"[red]INVALID[/red] {graph}: {errors}")
            raise typer.Exit(code=1)
        console.print(f"[green]OK[/green] {graph}")
    except Exception as e:
        console.print(f"[red]ERROR[/red] {graph}: {e}")
        raise typer.Exit(code=1) from None


@app.command()
def list_providers() -> None:
    """List all registered LLM providers."""
    from pharos.llm.registry import list_providers

    table = Table(title="Registered providers")
    table.add_column("name", style="cyan")
    for name in list_providers():
        table.add_row(name)
    console.print(table)


@app.command()
def doctor() -> None:
    """Check the environment for common issues."""
    import os

    console.print("[bold]pharos doctor[/bold]\n")
    rows = [
        ("Python", sys.version.split()[0]),
        ("uv", _which_version("uv") or "(not found)"),
        ("OPENAI_API_KEY", "set" if os.environ.get("OPENAI_API_KEY") else "[yellow]not set[/yellow]"),
        ("GLM_API_KEY", "set" if os.environ.get("GLM_API_KEY") or os.environ.get("ARK_API_KEY") else "[yellow]not set[/yellow]"),
    ]
    for k, v in rows:
        console.print(f"  {k}: {v}")


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Bind address (loopback recommended)"
    ),
    port: int = typer.Option(8765, "--port", help="Local Runtime API port"),
) -> None:
    """Start the local Runtime API used by Pharos Studio."""
    from pharos.studio_api import run_server

    console.print(
        f"[green]Pharos Runtime API[/green] listening on "
        f"[cyan]http://{host}:{port}[/cyan]"
    )
    run_server(host=host, port=port)


def _which_version(cmd: str) -> str | None:
    import shutil
    import subprocess

    p = shutil.which(cmd)
    if not p:
        return None
    try:
        out = subprocess.run(
            [cmd, "--version"], capture_output=True, text=True, timeout=3
        )
        return out.stdout.strip().split("\n")[0]
    except Exception:
        return p


# ---------- helpers ----------


def _build_budget(
    raw_budget: Any,
    max_cost: float | None,
    max_tokens: int | None,
    mode: str,
) -> RunBudget | None:
    """Merge a graph `budget:` block with CLI overrides (CLI wins)."""
    rb = raw_budget if isinstance(raw_budget, dict) else {}
    cost = max_cost if max_cost is not None else rb.get("max_cost_usd")
    toks = max_tokens if max_tokens is not None else rb.get("max_tokens")
    final_mode = mode or rb.get("mode", "hard")
    if cost is None and toks is None:
        return None
    return RunBudget(
        max_tokens=toks, max_cost_usd=cost, mode=final_mode
    )


def _apply_answers(g: CompositeGraph, answers: list[str]) -> None:
    """Apply `--answer node=value` to HumanEntity instances by node id."""
    for item in answers:
        if "=" not in item:
            console.print(
                f"[red]Invalid --answer (expected node=value): {item!r}[/red]"
            )
            raise typer.Exit(code=1)
        node_id, _, value = item.partition("=")
        node = g.nodes.get(node_id.strip())
        inst = node.instance if node is not None else None
        if inst is not None and hasattr(inst, "answer"):
            inst.answer = value


def _seed_input(g: CompositeGraph, text: str) -> None:
    """Inject `text` into `__in__.prompt` for any node that pulls from it.

    For multi-port graphs, use `_seed_inputs(g, {"port": "value", ...})`.
    """
    if not text:
        return
    _seed_inputs(g, {"prompt": text})


def _seed_inputs(g: CompositeGraph, port_to_value: dict[str, str]) -> None:
    """Emit TypedValue strings into multiple `__in__.<port>` destinations.

    Each entry in `port_to_value` is written to every edge that
    originates from `__in__.<port>`. Same port can fan out to
    multiple entity input ports.
    """
    if not port_to_value:
        return
    for edge in g.edges:
        if edge.src_node != "__in__":
            continue
        if edge.src_port not in port_to_value:
            continue
        value = port_to_value[edge.src_port]
        tgt = g.node(edge.dst_node)
        if tgt.instance is None:
            continue
        if edge.dst_port in tgt.instance.ins:
            tgt.instance.ins[edge.dst_port].emit(
                TypedValue(type="text", payload=value)
            )


def _substitute_vars(graph: Path, vars_list: list[str]) -> str:
    """Read a graph YAML and substitute `${var}` / `$var` placeholders.

    Each entry in `vars_list` is `key=value`. Replacement happens
    on the raw text (before YAML parse) so values can be quoted,
    multi-line, etc.

    Unknown vars (referenced but not provided) raise an error at
    load time — better to fail fast than to silently use ''.
    """
    text = Path(graph).read_text(encoding="utf-8")
    var_map: dict[str, str] = {}
    for item in vars_list:
        if "=" not in item:
            raise ValueError(f"Invalid --var (expected key=value): {item!r}")
        k, _, v = item.partition("=")
        var_map[k.strip()] = v
    if not var_map:
        return text

    import re

    # Match `${NAME}` only inside double-quoted YAML strings.
    # This avoids interfering with:
    #   - YAML comments (# something ${foo})
    #   - Unquoted keys/values that happen to contain $
    #   - Shell expansion in CI scripts
    # Pattern: " ... ${NAME} ... " — the closing quote is on the
    # same line. Multi-line strings are not currently supported
    # for substitution (uncommon in our YAML).
    pattern = re.compile(r'"([^"\n]*?)\$\{([A-Za-z_][A-Za-z0-9_]*)\}([^"\n]*?)"')

    def repl(m: re.Match[str]) -> str:
        before, name, after = m.group(1), m.group(2), m.group(3)
        if name not in var_map:
            raise ValueError(
                f"graph references ${{{name}}} but --var {name}=... was not provided"
            )
        return f'"{before}{var_map[name]}{after}"'

    return pattern.sub(repl, text)


async def _run_with_trace(
    g: CompositeGraph,
    director_name: str,
    want_trace: bool,
    max_iters: int = 20,
    converge_k: int = 2,
    granted_permissions: set[str] | None = None,
    replayer: Any = None,
    budget: RunBudget | None = None,
    outputs_out: dict[str, Any] | None = None,
) -> tuple[Any, ConsoleTraceBackend | None]:
    from pharos.runtime.executor import execute_graph

    backend = ConsoleTraceBackend() if want_trace else None
    outcome = await execute_graph(
        g,
        director_name,
        max_iters=max_iters,
        converge_k=converge_k,
        granted_permissions=granted_permissions,
        replayer=replayer,
        budget=budget,
    )
    if backend is not None:
        for s in outcome.spans:
            backend.write(s)
    # Expose the captured entity outputs so `run --record-fixture` can build a
    # fixture from the same live run instead of executing the graph twice.
    if outputs_out is not None:
        outputs_out.update(outcome.outputs)
    return outcome.result, backend


def _trace_query(
    entity: str | None,
    since: str | None,
    min_cost: float | None,
    limit: int,
) -> None:
    """Query the SQLite run index (`pharos trace query ...`)."""
    from pharos.observability.backend.sqlite import SQLiteTraceBackend

    rows = asyncio.run(
        SQLiteTraceBackend().query_runs(
            entity=entity, since=since, min_cost=min_cost, limit=limit
        )
    )
    if not rows:
        console.print("[yellow]No runs match this query.[/yellow]")
        console.print(
            "Runs are indexed as they complete; run something first."
        )
        return
    table = Table(title="Run history (SQLite index)")
    table.add_column("run_id", style="cyan")
    table.add_column("recorded_at", style="dim")
    table.add_column("director")
    table.add_column("status")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")
    for r in rows:
        status = r.get("status", "")
        colour = "red" if status == "error" else "green"
        table.add_row(
            str(r.get("run_id", ""))[:32],
            str(r.get("recorded_at", ""))[:19],
            str(r.get("director", "")),
            f"[{colour}]{status}[/{colour}]",
            str(r.get("total_tokens", 0)),
            f"${r.get('total_cost_usd', 0.0):.4f}",
        )
    console.print(table)


@app.command()
def trace(
    run_id: str = typer.Argument(..., help="Run id to inspect"),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Open the TUI viewer (falls back to text in non-TTY)",
    ),
    otlp: str | None = typer.Option(
        None,
        "--otlp",
        help="Export the run's spans as OTLP/JSON to this path instead of printing",
    ),
    export: Path | None = typer.Option(
        None,
        "--export",
        help="Export the native run JSON for Pharos Studio",
    ),
    entity: str | None = typer.Option(
        None, "--entity", help="[query] filter to runs containing this entity"
    ),
    since: str | None = typer.Option(
        None, "--since", help="[query] ISO timestamp lower bound (recorded_at)"
    ),
    min_cost: float | None = typer.Option(
        None, "--min-cost", help="[query] only runs at/above this USD cost"
    ),
    limit: int = typer.Option(
        50, "--limit", help="[query] max rows to return"
    ),
) -> None:
    """Show a recorded run's trace tree (or `pharos trace query ...`)."""
    from pharos.runtime import get_run, list_runs

    if run_id == "query":
        _trace_query(entity, since, min_cost, limit)
        return

    if run_id == "list":
        # Convenience: `pharos trace list`
        runs = list_runs()
        if not runs:
            console.print("[yellow]No runs recorded yet.[/yellow]")
            return
        table = Table(title="Recorded runs")
        table.add_column("run_id", style="cyan")
        table.add_column("spans", justify="right")
        table.add_column("duration", justify="right")
        for r in runs:
            dur_ms = (r["ended_at"] - r["started_at"]) * 1000
            table.add_row(
                r["run_id"][:32],
                str(r["span_count"]),
                f"{dur_ms:.1f} ms",
            )
        console.print(table)
        return

    spans = get_run(run_id)
    if not spans:
        console.print(f"[red]No run with id {run_id!r}[/red]")
        console.print("Try [cyan]pharos trace list[/cyan] to see available runs.")
        raise typer.Exit(code=1)

    if export is not None:
        from pharos.runtime import export_run_json

        export_run_json(run_id, export)
        console.print(f"[green]Exported native run to {export}[/green]")
        return

    if otlp is not None:
        from pharos.observability.otlp import write_otlp_json

        out_path = write_otlp_json(spans, otlp, service_name=f"pharos:{run_id}")
        console.print(
            f"[green]Exported {len(spans)} spans to {out_path} (OTLP/JSON)[/green]"
        )
        return

    if interactive:
        from pharos.observability.tui import interactive_view
        interactive_view(run_id, spans)
        return
    # Render
    from pharos.observability.trace import Span

    backend = ConsoleTraceBackend()
    for s_dict in spans:
        span = Span(
            id=s_dict["id"],
            trace_id=s_dict["trace_id"],
            parent_span_id=s_dict["parent_span_id"],
            name=s_dict["name"],
            started_at=s_dict["started_at"],
            ended_at=s_dict["ended_at"],
            status=s_dict["status"],
            attributes=s_dict.get("attributes", {}),
            error=s_dict.get("error"),
        )
        # Reconstruct events
        for ev in s_dict.get("events", []):
            span.events.append(ev)
        backend.write(span)
    console.print(backend.render())


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run id to replay"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    re_run: bool = typer.Option(
        False,
        "--re-run",
        help=(
            "Re-execute the graph with recorded LLM outputs "
            "(no network calls)"
        ),
    ),
    graph: Path = typer.Option(
        None,
        "--graph",
        "-g",
        help="Path to the graph YAML (required for --re-run)",
    ),
) -> None:
    """Inspect a recorded run, or re-execute it with cache replay.

    Without `--re-run`: show every entity's emitted text from the
    recorded run (deterministic, no re-execution).

    With `--re-run`: load the graph at `--graph`, swap LLM providers
    for `ReplayProvider`, and run it. LLM calls return the same text
    that was recorded — no network, no cost.
    """
    if re_run:
        if graph is None:
            console.print(
                "[red]--re-run requires --graph <path>[/red]"
            )
            raise typer.Exit(code=1)
        asyncio.run(_replay_rerun(graph, run_id))
        return

    from pharos.runtime import replay_run_summary

    summary = replay_run_summary(run_id)
    if summary["entity_count"] == 0:
        console.print(f"[red]No run with id {run_id!r}[/red]")
        console.print("Try [cyan]pharos trace list[/cyan] to see available runs.")
        raise typer.Exit(code=1)

    if json_out:
        typer.echo(json.dumps(summary, indent=2, default=str))
        return

    console.print(
        f"\n[bold]Replay:[/bold] {summary['run_id']}  "
        f"[cyan]director={summary['director'] or '?'}[/cyan]  "
        f"entities={summary['entity_count']}  "
        f"total={summary['total_duration_ms']:.1f} ms"
    )
    table = Table(title="Per-entity outputs (recorded)")
    table.add_column("#", justify="right")
    table.add_column("node", style="cyan")
    table.add_column("step_id", style="dim")
    table.add_column("duration", justify="right")
    table.add_column("output_text", style="green")
    for i, ent in enumerate(summary["entities"], 1):
        out = ent["output_text"]
        if len(out) > 200:
            out = out[:197] + "..."
        table.add_row(
            str(i),
            ent["node_id"],
            ent["step_id"][-12:] if ent["step_id"] else "",
            f"{ent['duration_ms']:.1f} ms",
            out or "(no text)",
        )
    console.print(table)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id to resume from"),
    graph: Path = typer.Option(
        ..., "--graph", "-g", help="Path to the graph YAML"
    ),
    input: str = typer.Option(
        "", "--input", "-i", help="Prompt for `__in__.prompt` (for live fires)"
    ),
    input_extra: list[str] = typer.Option(
        [], "--input-extra", help="Extra seed as port=value (repeatable)"
    ),
    grant: list[str] = typer.Option(
        [], "--grant", help="Permission to grant live fires (repeatable)"
    ),
    answer: list[str] = typer.Option(
        [],
        "--answer",
        help="Preset a human node's answer as node=value (repeatable)",
    ),
) -> None:
    """Continue a partially-recorded run: replay completed fires, run the rest.

    Fires that were recorded in `run_id` are re-emitted from the cache (no
    network, no cost); fires that were never reached execute live. Useful to
    pick up a long run that failed or was interrupted part-way.
    """
    from pharos.runtime import RunReplayer, get_run_director

    replayer = RunReplayer.load(run_id, resume=True)
    if replayer is None:
        console.print(f"[red]No recorded outputs for run {run_id!r}[/red]")
        console.print("Try [cyan]pharos trace list[/cyan] to see available runs.")
        raise typer.Exit(code=1)

    g, raw = load_graph(graph)
    seed_map: dict[str, str] = {}
    if input:
        seed_map["prompt"] = input
    for item in input_extra:
        if "=" in item:
            k, _, v = item.partition("=")
            seed_map[k.strip()] = v
    _seed_inputs(g, seed_map)
    _apply_answers(g, answer)

    director_name = get_run_director(run_id) or raw.get("director", "fn")
    console.print(
        f"[cyan]Resuming run {run_id}: recorded fires replay, the rest run "
        f"live (director={director_name}).[/cyan]"
    )
    result, _ = asyncio.run(
        _run_with_trace(
            g,
            director_name,
            want_trace=True,
            granted_permissions=set(grant),
            replayer=replayer,
        )
    )
    _print_summary(g, result, director_name)


async def _replay_rerun(
    graph: Path,
    run_id: str,
    seed_input: str = "",
) -> None:
    """Re-execute a graph from a recorded run with zero network calls.

    Preferred path (general): if the run recorded an entity-output cache,
    attach a `RunReplayer` so EVERY entity (LLM / shell / python / tool)
    replays its recorded outputs — no entity actually executes, so shell
    commands and file writes are not re-run.

    Fallback (legacy runs): if there is no output cache, swap each
    LLMAgent's provider for a `ReplayProvider` and re-execute (only LLM
    calls avoid the network; other entities run for real).
    """
    from pharos.runtime import RunReplayer, get_run_director

    replayer = RunReplayer.load(run_id)
    if replayer is not None:
        g, _raw = load_graph(graph)
        _seed_input(g, seed_input or "(replay)")
        director_name = get_run_director(run_id) or "fn"
        console.print(
            f"[cyan]Replaying run {run_id} from recorded entity outputs "
            f"(no execution, director={director_name}).[/cyan]"
        )
        result, _ = await _run_with_trace(
            g, director_name, want_trace=True, replayer=replayer
        )
        _print_summary(g, result, director_name)
        return

    # ---- legacy fallback: LLM-only provider swap ----
    from pharos.entities.llm import LLMAgent
    from pharos.llm.providers.replay import ReplayProvider
    from pharos.llm.types import Model, ModelCost
    from pharos.runtime import extract_cached_outputs

    g, _raw = load_graph(graph)
    cache = extract_cached_outputs(run_id)
    if not cache:
        console.print(
            f"[yellow]No cached outputs for run {run_id!r}[/yellow]"
        )
    _seed_input(g, seed_input or "(replay)")

    swap_count = 0
    replay_model = Model(
        id="replay",
        name="Replay (no network)",
        api="replay",
        provider="replay",
        base_url="",
        cost=ModelCost(input=0.0, output=0.0),
        context_window=128_000,
        max_tokens=8_192,
    )
    for node_id, node in g.nodes.items():
        if node.instance is None:
            continue
        if not isinstance(node.instance, LLMAgent):
            continue
        node.instance.provider = ReplayProvider(
            node_id=node_id, cache=cache
        )
        node.instance.model = replay_model
        node.instance._initialized = True  # type: ignore[attr-defined]
        swap_count += 1

    console.print(
        f"[cyan]Replaying run {run_id} with {swap_count} LLM agent(s) "
        f"using cached outputs (no network).[/cyan]"
    )
    result, _ = await _run_with_trace(g, "fn", want_trace=True)
    _print_summary(g, result, "fn")


def _parse_kv(items: list[str]) -> dict[str, str]:
    """Parse a list of ``key=value`` strings into a dict (first ``=`` splits)."""
    out: dict[str, str] = {}
    for item in items:
        if "=" in item:
            k, _, v = item.partition("=")
            out[k.strip()] = v
    return out


@app.command()
def test(
    target: Path = typer.Argument(
        ..., help="Fixture JSON file, or a directory of *.fixture.json"
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Re-run the real model(s) and diff against the golden (Phase 3)",
    ),
    update: bool = typer.Option(
        False, "--update", help="Re-bless the golden fixture from a --live run"
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    junit: Path | None = typer.Option(
        None, "--junit", help="Write a JUnit XML report to this path"
    ),
) -> None:
    """Run Agent-CI fixtures as a regression gate.

    Offline (default): replay each fixture's recorded outputs with zero network
    cost, recompute its chain_digest, and evaluate its assertions. Exits non-zero
    if any fixture drifts or an assertion fails — drop-in for CI.

    Record a fixture first with: pharos run <graph> ... --record-fixture fx.json
    """
    from pharos.testing.fixture import Fixture

    paths = _fixture_paths(target)
    if not paths:
        console.print(f"[red]No fixture(s) found at {target}[/red]")
        raise typer.Exit(code=2)

    results = []
    for p in paths:
        fx = Fixture.load(p)
        if live:
            from pharos.testing.runner import test_live

            res = asyncio.run(test_live(fx, p, update=update))
        else:
            from pharos.testing.runner import test_offline

            res = asyncio.run(test_offline(fx))
        results.append((p, fx, res))

    if junit is not None:
        _write_junit(junit, results)
    if json_out:
        typer.echo(
            json.dumps(
                [r.to_dict() for _, _, r in results], indent=2, default=str
            )
        )
    else:
        _render_test_results(results, live=live)

    if not all(r.passed for _, _, r in results):
        raise typer.Exit(code=1)


def _fixture_paths(target: Path) -> list[Path]:
    """Resolve a fixture target to a sorted list of fixture files."""
    if target.is_dir():
        return sorted(target.glob("*.fixture.json")) or sorted(
            target.glob("*.json")
        )
    return [target] if target.exists() else []


def _render_test_results(results: list[Any], *, live: bool) -> None:
    n_pass = sum(1 for _, _, r in results if r.passed)
    for path, _fx, r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        console.print(f"\n{status} [cyan]{escape(r.name)}[/cyan]  ({escape(str(path))})")
        if not r.graph_ok:
            console.print(
                "  [yellow]warning:[/yellow] graph file changed since record; "
                "offline replay can't see behavior changes — run "
                "[cyan]--live --update[/cyan] to re-bless."
            )
        if not r.digest_ok:
            console.print(
                f"  [red]chain_digest drift[/red] "
                f"expected {r.expected_digest[:12]}… got {r.actual_digest[:12]}…"
            )
        for a in r.assertion_results:
            mark = "[green]ok[/green]" if a.ok else "[red]fail[/red]"
            console.print(
                f"  {mark} {escape(a.target)} {a.op}: {escape(a.detail)}"
            )
        if r.error:
            console.print(f"  [red]error:[/red] {escape(r.error)}")
        drift = getattr(r, "drift", None)
        if live and drift is not None and drift.has_changes():
            _render_run_diff(drift, "golden", "live")
    color = "green" if n_pass == len(results) else "red"
    console.print(
        f"\n[{color}]{n_pass}/{len(results)} fixtures passed[/{color}]"
    )


def _write_junit(path: Path, results: list[Any]) -> None:
    """Write a minimal JUnit XML report (one testcase per fixture)."""
    import xml.etree.ElementTree as ET

    n_fail = sum(1 for _, _, r in results if not r.passed)
    suite = ET.Element(
        "testsuite",
        name="pharos-agent-ci",
        tests=str(len(results)),
        failures=str(n_fail),
    )
    for _p, _fx, r in results:
        case = ET.SubElement(suite, "testcase", name=r.name, classname="pharos.test")
        if not r.passed:
            msgs = []
            if not r.digest_ok:
                msgs.append(
                    f"chain_digest drift: expected {r.expected_digest} "
                    f"got {r.actual_digest}"
                )
            msgs += [
                f"{a.target} {a.op}: {a.detail}"
                for a in r.assertion_results
                if not a.ok
            ]
            if r.error:
                msgs.append(r.error)
            failure = ET.SubElement(
                case, "failure", message="; ".join(msgs) or "failed"
            )
            failure.text = "\n".join(msgs)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="Baseline run id"),
    run_b: str = typer.Argument(..., help="Run id to compare against the baseline"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Structurally diff two recorded runs' per-node outputs.

    Aligns the two runs by (node, fire, port) and shows exactly which output
    changed — JSON field paths, text line diffs — instead of an opaque blob
    comparison. Descriptive by design: it always exits 0.
    """
    from pharos.runtime import get_run_outputs
    from pharos.testing.diff import diff_runs

    a = get_run_outputs(run_a)
    b = get_run_outputs(run_b)
    if not a and not b:
        console.print(
            f"[red]No recorded outputs for {run_a!r} or {run_b!r}[/red]"
        )
        console.print("Try [cyan]pharos trace list[/cyan] to see available runs.")
        raise typer.Exit(code=1)

    rd = diff_runs(a, b)
    if json_out:
        typer.echo(json.dumps(rd.to_dict(), indent=2, default=str))
        return
    _render_run_diff(rd, run_a, run_b)


def _render_run_diff(rd: Any, a_label: str, b_label: str) -> None:
    """Pretty-print a RunDiff to the console (used by `diff` and `test --live`)."""
    console.print(
        f"\n[bold]diff[/bold] [red]{escape(a_label)}[/red] "
        f"-> [green]{escape(b_label)}[/green]"
    )
    if not rd.has_changes():
        console.print("[green]No differences in recorded outputs.[/green]")
        return
    for pd in rd.port_diffs:
        head = f"{pd.node_id}:{pd.fire_index}.{pd.port}"
        console.print(
            f"\n[cyan]{escape(head)}[/cyan] [yellow]{pd.status}[/yellow] "
            f"({pd.kind})"
        )
        if pd.kind == "json" and pd.field_changes:
            for fc in pd.field_changes:
                console.print(
                    f"  {fc.kind} {escape(fc.path)}: "
                    f"{escape(repr(fc.before))} -> {escape(repr(fc.after))}"
                )
        elif pd.kind == "text" and pd.line_diff:
            for line in pd.line_diff:
                style = (
                    "green" if line.startswith("+")
                    else "red" if line.startswith("-")
                    else "dim"
                )
                console.print(f"  [{style}]{escape(line)}[/{style}]")
        else:
            console.print(f"  before: {escape(repr(pd.before))}")
            console.print(f"  after:  {escape(repr(pd.after))}")
    if rd.propagation:
        console.print("\n[bold]propagation[/bold]")
        for node in sorted(rd.propagation):
            downs = ", ".join(rd.propagation[node])
            console.print(f"  {escape(node)} -> {escape(downs)}")


def _print_summary(g: CompositeGraph, result: Any, director_name: str = "fn") -> None:
    console.print(
        f"\n[bold]Run:[/bold] {g.name}  "
        f"[cyan]director={director_name}[/cyan]  "
        f"[green]converged={result.converged}[/green]  "
        f"iterations={result.iterations}  "
        f"tokens={result.tokens_emitted}  "
        f"cost=${result.cost_usd:.4f}"
    )
    if result.error:
        console.print(f"  [red]error:[/red] {result.error}")
    # Print a small output table — every entity's output port buffer,
    # plus any tokens collected at virtual __out__ nodes.
    table = Table(title="Output ports")
    table.add_column("node", style="cyan")
    table.add_column("port", style="magenta")
    table.add_column("value", style="green")
    for node_id, node in g.nodes.items():
        if node.instance is None:
            continue
        for port_name, port in node.instance.outs.items():
            for t in port.peek_all():
                v = t.value.payload
                if isinstance(v, str) and len(v) > 200:
                    v = v[:197] + "..."
                table.add_row(node_id, port_name, str(v))
    # Virtual output nodes (collected during deliver_upstream)
    collected = getattr(g, "collected", None) or {}
    for node_id, ports in collected.items():
        for port_name, tokens in ports.items():
            for t in tokens:
                v = t.value.payload
                if isinstance(v, str) and len(v) > 200:
                    v = v[:197] + "..."
                table.add_row(node_id, port_name, str(v))
    if table.row_count:
        console.print(table)


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "tokens_emitted": result.tokens_emitted,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }


# Entry point: pyproject.toml references `pharos.cli:cli_main`
cli_main = app

if __name__ == "__main__":
    cli_main()
