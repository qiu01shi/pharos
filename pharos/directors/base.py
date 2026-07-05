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

import contextlib
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pharos.core.graph import CompositeGraph
from pharos.core.permissions import PermissionPolicy
from pharos.core.token import Token


class BudgetExceededError(Exception):
    """Raised when a run exceeds its cost/token budget in ``hard`` mode."""


@dataclass
class RunBudget:
    """A run-level spend cap enforced uniformly in ``safe_fire``.

    Governance in pharos has two axes: RBAC decides *what* a run may do;
    a budget decides *how much*. After each fire, its (tokens, cost) are
    charged here. ``hard`` mode aborts the run once a limit is crossed;
    ``soft`` mode only annotates the trace with a ``budget.exceeded`` event
    and lets the run finish.
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    mode: str = "hard"  # "hard" aborts; "soft" warns via a trace event
    spent_tokens: int = 0
    spent_cost: float = 0.0
    exceeded: bool = False

    def charge(self, tokens: int, cost: float, span: Any = None) -> None:
        """Add a fire's usage and enforce the cap (raise in hard mode)."""
        self.spent_tokens += tokens
        self.spent_cost += cost
        reason = self._over_limit()
        if reason is None:
            return
        self.exceeded = True
        if span is not None:
            span.record_event(
                "budget.exceeded",
                {
                    "reason": reason,
                    "spent_tokens": self.spent_tokens,
                    "spent_cost_usd": round(self.spent_cost, 6),
                    "mode": self.mode,
                },
            )
        if self.mode == "hard":
            raise BudgetExceededError(reason)

    def _over_limit(self) -> str | None:
        if self.max_tokens is not None and self.spent_tokens > self.max_tokens:
            return f"tokens {self.spent_tokens} > budget {self.max_tokens}"
        if (
            self.max_cost_usd is not None
            and self.spent_cost > self.max_cost_usd
        ):
            return (
                f"cost ${self.spent_cost:.6f} > "
                f"budget ${self.max_cost_usd:.6f}"
            )
        return None


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
    # Cross-cutting run services, wired by the Director/CLI. Optional so
    # library callers can run a graph with none of them.
    tracer: Any = None  # observability.trace.Tracer
    recorder: Any = None  # runtime.RunRecorder — captures entity outputs
    replayer: Any = None  # runtime.RunReplayer — replays recorded outputs
    budget: RunBudget | None = None  # optional run-level spend cap

    def policy(self) -> PermissionPolicy:
        """The authorisation policy for this run (from granted_permissions)."""
        return PermissionPolicy.from_grants(self.granted_permissions)


@dataclass
class FireContext:
    """Per-fire context passed to Entity.fire()."""

    run_id: str
    step_id: str
    iter: int = 0
    started_at: float = field(default_factory=time.time)
    # Permissions propagated from RunContext so entities can
    # check tool permissions during fire().
    granted_permissions: set[str] = field(default_factory=set)
    # Tracer propagated from RunContext so entities can open child spans
    # (e.g. LLMAgent tracing each tool execution as its own span).
    tracer: Any = None
    # Record/replay services propagated so composite entities (e.g.
    # SubgraphEntity) can recurse into a child run that participates in
    # the same recording / replay session under a namespaced prefix.
    recorder: Any = None
    replayer: Any = None

    def policy(self) -> PermissionPolicy:
        """The authorisation policy for this fire (from granted_permissions)."""
        return PermissionPolicy.from_grants(self.granted_permissions)


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


def check_permissions(entity: Any, run_ctx: RunContext) -> None:
    """Enforce an entity's `required_permissions` against the run's grants.

    Raises PermissionError if any required permission is missing. This
    MUST be called by every Director before an entity's first fire so
    that RBAC is enforced uniformly regardless of scheduling semantics
    (FN / SDF / DE). The decision is delegated to the run's
    `PermissionPolicy` so entity-level and tool-level checks share one
    code path and one alias table.
    """
    required: set[str] = getattr(entity, "required_permissions", set()) or set()
    subject = f"entity {entity.node_id!r} ({type(entity).__name__})"
    run_ctx.policy().check(required, subject=subject)


def input_lineage_digest(entity: Any) -> str | None:
    """Digest of the entity's current input head-token hashes.

    Snapshotted BEFORE ``fire()`` consumes the inputs, this becomes the
    ``prev_hash`` stamped onto the tokens the fire emits — so an output
    token's hash transitively depends on every input it was derived from.
    Returns None when the entity has no pending input (a source node).
    """
    parts: list[str] = []
    for name in sorted(entity.ins):
        head = entity.ins[name].peek()
        if head is not None:
            parts.append(f"{name}:{head.self_hash}")
    if not parts:
        return None
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def stamp_lineage(entity: Any, prev_digest: str | None) -> None:
    """Rewrite freshly emitted output tokens with real lineage.

    ``OutputPort.emit`` cannot know its owning node or the upstream it
    depended on, so it emits tokens with a placeholder ``origin`` and no
    ``prev_hash``. Here — inside the single shared fire path — we know
    both, so we replace each not-yet-stamped token with one carrying
    ``origin = "<node_id>.<port>"`` and ``prev_hash = prev_digest``.

    Idempotent: a token counts as stamped once its ``origin`` matches the
    normalized form, so nodes that re-fire (SDF/DE) don't re-stamp tokens
    carried over from earlier rounds — the head token keeps a stable hash.
    """
    for port_name, port in entity.outs.items():
        normalized = f"{entity.node_id}.{port_name}"
        rebuilt: deque[Token] = deque()
        changed = False
        for tok in port.buffer:
            if tok.origin != normalized:
                tok = Token(
                    value=tok.value,
                    origin=normalized,
                    ts=tok.ts,
                    prev_hash=(
                        tok.prev_hash if tok.prev_hash is not None else prev_digest
                    ),
                    run_id=tok.run_id,
                    iter=tok.iter,
                    is_partial=tok.is_partial,
                    cost_usd=tok.cost_usd,
                    metadata=tok.metadata,
                )
                changed = True
            rebuilt.append(tok)
        if changed:
            port.buffer = rebuilt


def build_edge_index(
    graph: CompositeGraph,
) -> dict[str, list[tuple[str, str]]]:
    """Index edges as src_node -> [(dst_node, dst_port), ...] for delivery."""
    index: dict[str, list[tuple[str, str]]] = {}
    for e in graph.edges:
        index.setdefault(e.src_node, []).append((e.dst_node, e.dst_port))
    return index


async def safe_fire(
    entity: Any, fire_ctx: FireContext, run_ctx: RunContext
) -> tuple[int, float] | None:
    """Run an entity's permission-check / setup / fire / record safely.

    This is the single firing path shared by FN / SDF / DE, so RBAC,
    tracing, replay, and metric collection behave identically no matter
    which Director drives the run. Returns (token_count, cost).

    Order of operations:
      1. Permission check (before any setup — fail early).
      2. Lazy setup() on first fire.
      3. Open a trace span (if a tracer is registered) and expose the
         tracer to the entity via FireContext so it can open children.
      4. Either replay recorded outputs (if a replayer has them for this
         entity+fire) or call fire(); then record outputs (if recording).
      5. finish the span and collect (tokens, cost).
    """
    from pharos.observability.trace import current_span

    replayer = getattr(run_ctx, "replayer", None)
    recorder = getattr(run_ctx, "recorder", None)

    # Stable per-entity fire index so record/replay keys line up even when
    # a node fires many times (SDF/DE iterations).
    fire_index: int = getattr(entity, "_fire_count", 0)
    entity._fire_count = fire_index + 1

    # Decide, for THIS fire, whether to replay a recorded output or run live:
    #   * no replayer                       -> live
    #   * replayer has this fire recorded    -> replay (never executes)
    #   * replayer in resume mode, no record -> live (continue past checkpoint)
    #   * replayer, not resume, no record    -> skip (pure replay emits nothing)
    replay_this = replayer is not None and replayer.has(
        entity.node_id, fire_index
    )
    live = replayer is None or (
        getattr(replayer, "resume", False) and not replay_this
    )

    # Live fires are permission-gated and set up; replayed fires are not
    # (they never execute, so they need no client / API key / grant).
    if live:
        check_permissions(entity, run_ctx)
        if not getattr(entity, "_initialized", False):
            await entity.setup(run_ctx)
            entity._initialized = True

    tracer = getattr(run_ctx, "tracer", None)
    fire_ctx.tracer = tracer
    # Expose record/replay services so composite entities can recurse
    # into a child run under the same session.
    fire_ctx.recorder = recorder
    fire_ctx.replayer = replayer
    parent = current_span()
    span = None
    if tracer is not None:
        span = tracer.start_span(
            f"entity.fire.{entity.node_id}",
            parent=parent,
            attributes={
                "entity": entity.node_id,
                "entity_class": type(entity).__name__,
                "step_id": fire_ctx.step_id,
                "iter": fire_ctx.iter,
                "fire_index": fire_index,
            },
        )

    # Snapshot input lineage BEFORE fire() consumes the input ports, so the
    # tokens this fire emits can carry a prev_hash derived from their inputs.
    prev_digest = input_lineage_digest(entity)

    metrics = (0, 0.0)
    try:
        if replay_this:
            # Re-emit recorded outputs; the entity never executes.
            assert replayer is not None
            replayer.apply(entity, entity.node_id, fire_index)
        elif live:
            await entity.fire(fire_ctx)
        # else: pure-replay of a fire with no recorded output -> emit nothing.
        # Stamp real origin + lineage onto freshly emitted tokens (shared by
        # live and replay paths so recorded and replayed hashes agree).
        stamp_lineage(entity, prev_digest)
        if recorder is not None:
            recorder.capture(entity, entity.node_id, fire_index)
        # Metrics + budget are charged inside the try so a hard-budget abort
        # can annotate this fire's span before it closes.
        metrics = collect_metrics(entity)
        budget = getattr(run_ctx, "budget", None)
        if budget is not None:
            budget.charge(metrics[0], metrics[1], span=span)
    except BaseException as e:
        if span is not None:
            span.record_exception(e)
        raise
    finally:
        if span is not None and tracer is not None:
            tracer.finish_span(span)

    return metrics


def collect_metrics(entity: Any) -> tuple[int, float]:
    """Read (token_count, cost) from an entity after it fires.

    Prefers entity-level cumulative counters (LLMAgent tracks
    `total_tokens` / `total_cost` directly); falls back to counting
    output-port tokens for entities that don't. Shared by all Directors
    so the reported numbers are consistent across scheduling semantics.
    """
    token_count = getattr(entity, "total_tokens", 0) or sum(
        len(p) for p in entity.outs.values()
    )
    cost = getattr(entity, "total_cost", 0.0) or sum(
        t.cost_usd for p in entity.outs.values() for t in p.peek_all()
    )
    return (token_count, cost)


async def teardown_all(graph: CompositeGraph) -> None:
    """Release resources for every initialized entity in the graph.

    Calls each entity's `teardown()` (e.g. LLMAgent closes its HTTP
    client). Errors are swallowed so one entity's failed teardown does
    not mask the run result or block the others. Directors call this in
    a `finally` block so providers are always closed, even on error.
    """
    for node in graph.nodes.values():
        inst = node.instance
        if inst is None:
            continue
        if not getattr(inst, "_initialized", False):
            continue
        with contextlib.suppress(Exception):
            await inst.teardown()
        inst._initialized = False  # type: ignore[attr-defined]
        inst._fire_count = 0  # type: ignore[attr-defined]


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
        n: graph._nx.in_degree(n)
        for n in graph._nx.nodes
    }
    successors: dict[str, list[str]] = {n: [] for n in in_degree}
    for src, dst in graph._nx.edges:
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
    "build_edge_index",
    "check_permissions",
    "collect_metrics",
    "deliver_upstream",
    "input_lineage_digest",
    "safe_fire",
    "stamp_lineage",
    "teardown_all",
    "topo_layers",
]
