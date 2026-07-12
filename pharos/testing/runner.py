"""runner — record a golden fixture and run the offline regression gate.

Two entry points back ``pharos test``:

- ``record_fixture`` runs a graph live once and captures it as a ``Fixture``.
- ``test_offline`` replays a fixture's recorded outputs with zero network /
  zero cost, recomputes the ``chain_digest``, and evaluates the fixture's
  assertions — producing a pass/fail ``TestResult`` for CI.

The live drift check (``test_live``) is added in Phase 3; it shares the graph
loading and assertion machinery here.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors import make_director
from pharos.directors.base import RunBudget, RunContext
from pharos.ir import load_graph_from_text
from pharos.runtime import RunRecorder, RunReplayer
from pharos.testing.diff import RunDiff, diff_runs
from pharos.testing.digest import Outputs, chain_digest
from pharos.testing.fixture import Assertion, Fixture

# Output ports whose change signals a behavioral/structural regression (L1),
# as opposed to free-text ports (text/draft/thinking) where wording drift is
# expected and treated as a warning, not a failure.
_STRUCTURAL_PORTS = {"tool_calls", "error"}

_VAR_PATTERN = re.compile(r'"([^"\n]*?)\$\{([A-Za-z_][A-Za-z0-9_]*)\}([^"\n]*?)"')


# ---------- graph loading / seeding (self-contained, no CLI dependency) ----------


def _apply_vars(text: str, var_map: dict[str, str]) -> str:
    """Substitute ``${NAME}`` inside double-quoted YAML strings."""
    if not var_map:
        return text

    def repl(m: re.Match[str]) -> str:
        before, name, after = m.group(1), m.group(2), m.group(3)
        if name not in var_map:
            raise ValueError(f"graph references ${{{name}}} but var was not provided")
        return f'"{before}{var_map[name]}{after}"'

    return _VAR_PATTERN.sub(repl, text)


def load_graph_for_fixture(
    graph_path: str | Path, var: dict[str, str] | None = None
) -> tuple[CompositeGraph, dict[str, Any]]:
    """Read + var-substitute + load a graph the same way the CLI ``run`` does."""
    path = Path(graph_path)
    text = _apply_vars(path.read_text(encoding="utf-8"), var or {})
    return load_graph_from_text(text, base_dir=path.parent)


def seed_inputs(graph: CompositeGraph, seed: dict[str, str]) -> None:
    """Emit ``seed`` values into ``__in__.<port>`` edges (as text tokens)."""
    if not seed:
        return
    for edge in graph.edges:
        if edge.src_node != "__in__" or edge.src_port not in seed:
            continue
        tgt = graph.node(edge.dst_node)
        if tgt.instance is None or edge.dst_port not in tgt.instance.ins:
            continue
        tgt.instance.ins[edge.dst_port].emit(
            TypedValue(type="text", payload=seed[edge.src_port])
        )


def collected_outputs(graph: CompositeGraph) -> dict[str, list[dict[str, Any]]]:
    """Capture virtual ``__out__`` tokens as output records.

    The recorder only captures entities that fire; ``__out__`` has no instance,
    so its tokens live on ``graph.collected``. Including them in the outputs map
    is what lets the offline gate catch routing/edge regressions (a graph change
    that reroutes tokens changes ``__out__`` even when node outputs are replayed)
    and lets assertions target ``__out__.<port>``.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    collected = getattr(graph, "collected", {}) or {}
    for node_id, ports in collected.items():
        recs: list[dict[str, Any]] = []
        for port_name, tokens in ports.items():
            for tok in tokens:
                recs.append(
                    {
                        "port": port_name,
                        "type": tok.value.type,
                        "payload": tok.value.payload,
                        "self_hash": tok.self_hash,
                        "prev_hash": tok.prev_hash,
                        "origin": tok.origin,
                        "ts": tok.ts,
                        "run_id": tok.run_id,
                        "iter": tok.iter,
                        "is_partial": tok.is_partial,
                        "cost_usd": tok.cost_usd,
                        "metadata": tok.metadata,
                    }
                )
        if recs:
            out[f"{node_id}:0"] = recs
    return out


# ---------- execution ----------


@dataclass
class RunOutcome:
    result: Any
    outputs: dict[str, list[dict[str, Any]]]
    graph: CompositeGraph


async def _execute(
    graph: CompositeGraph,
    director_name: str,
    *,
    grant: set[str] | None = None,
    replayer: RunReplayer | None = None,
    budget: RunBudget | None = None,
) -> RunOutcome:
    recorder = RunRecorder()
    ctx = RunContext(
        run_id=str(uuid.uuid4()),
        granted_permissions=grant or set(),
        recorder=recorder,
        replayer=replayer,
        budget=budget,
    )
    director = make_director(director_name, max_iters=20, converge_k=2)
    result = await director.run(graph, ctx)
    outputs = recorder.to_dict()
    outputs.update(collected_outputs(graph))
    return RunOutcome(result=result, outputs=outputs, graph=graph)


async def record_fixture(
    graph_path: str | Path,
    *,
    name: str,
    seed: dict[str, str] | None = None,
    grant: list[str] | None = None,
    var: dict[str, str] | None = None,
    director_override: str | None = None,
    assertions: list[Assertion] | None = None,
) -> Fixture:
    """Run a graph live once and capture it as a golden ``Fixture``."""
    graph, raw = load_graph_for_fixture(graph_path, var)
    seed_inputs(graph, seed or {})
    director_name = director_override or raw.get("director", "fn")
    outcome = await _execute(graph, director_name, grant=set(grant or []))
    return Fixture.build(
        name=name,
        graph_path=graph_path,
        director=director_name,
        outputs=outcome.outputs,
        seed=seed,
        grant=grant,
        var=var,
        assertions=assertions,
    )


def fixture_from_outputs(
    graph: CompositeGraph,
    graph_path: str | Path,
    director: str,
    entity_outputs: Outputs,
    *,
    name: str,
    seed: dict[str, str] | None = None,
    grant: list[str] | None = None,
    var: dict[str, str] | None = None,
    assertions: list[Assertion] | None = None,
) -> Fixture:
    """Build a fixture from an already-executed run's entity outputs + graph.

    Used by ``pharos run --record-fixture`` so recording reuses the CLI's full
    live run (with trace/persistence) rather than executing a second time. The
    virtual ``__out__`` tokens are merged in from ``graph.collected``.
    """
    outputs: dict[str, list[dict[str, Any]]] = {
        k: list(v) for k, v in entity_outputs.items()
    }
    outputs.update(collected_outputs(graph))
    return Fixture.build(
        name=name,
        graph_path=graph_path,
        director=director,
        outputs=outputs,
        seed=seed,
        grant=grant,
        var=var,
        assertions=assertions,
    )


# ---------- assertions + verdict ----------


@dataclass
class AssertionResult:
    target: str
    op: str
    ok: bool
    detail: str
    policy: str = "pinned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op": self.op,
            "ok": self.ok,
            "detail": self.detail,
            "policy": self.policy,
        }


@dataclass
class TestResult:
    name: str
    passed: bool
    digest_ok: bool
    graph_ok: bool
    expected_digest: str
    actual_digest: str
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: str | None = None
    mode: str = "offline"
    # Live-mode structured diff vs the golden (None in offline mode).
    drift: RunDiff | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "digest_ok": self.digest_ok,
            "graph_ok": self.graph_ok,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "assertions": [a.to_dict() for a in self.assertion_results],
            "error": self.error,
            "mode": self.mode,
            "drift": self.drift.to_dict() if self.drift is not None else None,
        }


def resolve_target(outputs: Outputs, target: str) -> list[Any]:
    """Resolve ``"<node>.<port>"`` / ``"<node>:<fire>.<port>"`` to payloads.

    A bare ``<node>.<port>`` defaults to fire index 0.
    """
    base, _, port = target.rpartition(".")
    if not base:
        return []
    key = base if ":" in base else f"{base}:0"
    return [rec.get("payload") for rec in outputs.get(key, []) if rec.get("port") == port]


def run_assertions(fixture: Fixture, outputs: Outputs) -> list[AssertionResult]:
    """Evaluate every declared assertion against a run's outputs."""
    results: list[AssertionResult] = []
    for a in fixture.assertions:
        ok, detail = a.check(resolve_target(outputs, a.target))
        results.append(
            AssertionResult(
                target=a.target, op=a.op, ok=ok, detail=detail, policy=a.policy
            )
        )
    return results


async def test_offline(fixture: Fixture) -> TestResult:
    """Replay a fixture offline, recompute the digest, and run assertions.

    Zero network / zero cost: recorded node outputs are re-emitted rather than
    executed. The verdict fails if the recomputed ``chain_digest`` drifts from
    the golden (a runtime/graph-structure regression) or any declared assertion
    fails. A changed graph file is surfaced as ``graph_ok=False`` (a warning to
    re-bless with ``--live``), not an automatic failure.
    """
    graph_ok = fixture.graph_matches()
    try:
        graph, _raw = load_graph_for_fixture(fixture.graph_path, fixture.var)
        seed_inputs(graph, fixture.seed)
        replayer = RunReplayer(dict(fixture.outputs))
        outcome = await _execute(graph, fixture.director, replayer=replayer)
    except Exception as e:  # loading/replay failure is a test failure, not a crash
        return TestResult(
            name=fixture.name,
            passed=False,
            digest_ok=False,
            graph_ok=graph_ok,
            expected_digest=fixture.chain_digest,
            actual_digest="",
            error=f"{type(e).__name__}: {e}",
        )

    actual_digest = chain_digest(outcome.outputs, version=fixture.version)
    digest_ok = actual_digest == fixture.chain_digest
    assertion_results = run_assertions(fixture, outcome.outputs)
    # v2 makes the graph integrity pin part of the gate.  Legacy v1 fixtures
    # retain their historical warning-only behavior until re-recorded.
    graph_gate_ok = graph_ok or fixture.version <= 1
    passed = graph_gate_ok and digest_ok and all(r.ok for r in assertion_results)
    return TestResult(
        name=fixture.name,
        passed=passed,
        digest_ok=digest_ok,
        graph_ok=graph_ok,
        expected_digest=fixture.chain_digest,
        actual_digest=actual_digest,
        assertion_results=assertion_results,
        mode="offline",
    )


def structural_drift(drift: RunDiff) -> list[str]:
    """Reasons a diff counts as a behavioral/structural regression (L1).

    Differences ARE NOT drift by default. This layer flags only the changes
    that break behavior regardless of wording: a node stops/starts emitting a
    port, its tool-call set/order changes, an error appears, or a JSON output's
    *shape* (added/removed fields) changes. Pure free-text wording changes and
    JSON leaf-value changes are left to assertions / tolerance (WARN, not FAIL).
    """
    reasons: list[str] = []
    for pd in drift.port_diffs:
        where = f"{pd.node_id}.{pd.port}"
        if pd.status in ("added", "removed"):
            reasons.append(f"{where} {pd.status}")
        elif pd.port in _STRUCTURAL_PORTS:
            reasons.append(f"{where} changed ({pd.port})")
        elif pd.port == "usage":
            # Token counts vary run-to-run (not structural), but a rise in
            # self-heal repair rounds means the model needed more coaxing to
            # produce valid structured output — a behavioral regression.
            for fc in pd.field_changes:
                if fc.path.endswith("repair_attempts") and _as_int(
                    fc.after
                ) > _as_int(fc.before):
                    reasons.append(
                        f"{where} repair_attempts {fc.before}->{fc.after}"
                    )
        elif pd.kind == "json" and any(
            fc.kind in ("added", "removed") for fc in pd.field_changes
        ):
            reasons.append(f"{where} json shape changed")
    return reasons


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


async def test_live(
    fixture: Fixture,
    fixture_path: str | Path | None = None,
    *,
    update: bool = False,
) -> TestResult:
    """Re-run the real model(s) and judge drift against the golden.

    Verdict (v1, conservative): FAIL if a declared non-ignored assertion fails
    or a structural/behavioral invariant broke (see ``structural_drift``);
    otherwise PASS, even when free text changed (the diff still reports it).
    With ``update`` the golden is re-blessed from this live run and the result
    is forced to pass.
    """
    graph_ok = fixture.graph_matches()
    try:
        graph, _raw = load_graph_for_fixture(fixture.graph_path, fixture.var)
        seed_inputs(graph, fixture.seed)
        outcome = await _execute(
            graph, fixture.director, grant=set(fixture.grant)
        )
    except Exception as e:
        return TestResult(
            name=fixture.name,
            passed=False,
            digest_ok=False,
            graph_ok=graph_ok,
            expected_digest=fixture.chain_digest,
            actual_digest="",
            error=f"{type(e).__name__}: {e}",
            mode="live",
        )

    actual_digest = chain_digest(outcome.outputs, version=fixture.version)
    digest_ok = actual_digest == fixture.chain_digest
    drift = diff_runs(fixture.outputs, outcome.outputs, graph=graph)
    assertion_results = run_assertions(fixture, outcome.outputs)

    assertion_fail = any(
        not r.ok for r in assertion_results if r.policy != "ignored"
    )
    passed = not assertion_fail and not structural_drift(drift)

    if update and fixture_path is not None:
        Fixture.build(
            name=fixture.name,
            graph_path=fixture.graph_path,
            director=fixture.director,
            outputs=outcome.outputs,
            seed=fixture.seed,
            grant=fixture.grant,
            var=fixture.var,
            assertions=fixture.assertions,
        ).save(fixture_path)
        passed = True

    return TestResult(
        name=fixture.name,
        passed=passed,
        digest_ok=digest_ok,
        graph_ok=graph_ok,
        expected_digest=fixture.chain_digest,
        actual_digest=actual_digest,
        assertion_results=assertion_results,
        mode="live",
        drift=drift,
    )


__all__ = [
    "AssertionResult",
    "RunOutcome",
    "TestResult",
    "collected_outputs",
    "fixture_from_outputs",
    "load_graph_for_fixture",
    "record_fixture",
    "resolve_target",
    "run_assertions",
    "seed_inputs",
    "structural_drift",
    "test_live",
    "test_offline",
]
