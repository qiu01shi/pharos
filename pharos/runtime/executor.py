"""Shared graph execution service used by the CLI and local Runtime API."""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pharos.core.graph import CompositeGraph
from pharos.directors import make_director
from pharos.directors.base import RunBudget, RunContext, RunResult
from pharos.observability.trace import InMemoryTracer
from pharos.runtime import RunRecorder, record_run


@dataclass
class ExecutionOutcome:
    run_id: str
    result: RunResult
    spans: list[Any]
    outputs: dict[str, list[dict[str, Any]]]


async def execute_graph(
    graph: CompositeGraph,
    director_name: str,
    *,
    run_id: str | None = None,
    max_iters: int = 20,
    converge_k: int = 2,
    granted_permissions: set[str] | None = None,
    replayer: Any = None,
    budget: RunBudget | None = None,
    event_sink: Any = None,
    persist: bool = True,
) -> ExecutionOutcome:
    """Execute one graph with the canonical trace/record/budget services."""
    effective_run_id = run_id or str(uuid.uuid4())
    tracer = InMemoryTracer()
    resume_mode = replayer is not None and getattr(replayer, "resume", False)
    recorder = RunRecorder() if (replayer is None or resume_mode) else None
    ctx = RunContext(
        run_id=effective_run_id,
        granted_permissions=granted_permissions or set(),
        tracer=tracer,
        recorder=recorder,
        replayer=replayer,
        budget=budget,
        event_sink=event_sink,
    )
    director = make_director(
        director_name, max_iters=max_iters, converge_k=converge_k
    )
    result = await director.run(graph, ctx)
    outputs = recorder.to_dict() if recorder is not None else {}
    if persist:
        record_run(
            effective_run_id,
            tracer.spans,
            outputs=outputs,
            director=director_name,
        )
        await _index_run_sqlite(
            effective_run_id, tracer.spans, director_name, result
        )
    return ExecutionOutcome(
        run_id=effective_run_id,
        result=result,
        spans=tracer.spans,
        outputs=outputs,
    )


async def _index_run_sqlite(
    run_id: str, spans: list[Any], director_name: str, result: RunResult
) -> None:
    """Best-effort cross-run indexing; storage failures never fail a run."""
    with contextlib.suppress(Exception):
        from pharos.observability.backend.sqlite import SQLiteTraceBackend

        await SQLiteTraceBackend().index_run(
            run_id,
            spans,
            director=director_name,
            total_tokens=result.tokens_emitted or 0,
            total_cost=result.cost_usd or 0.0,
            status="error" if result.error else "ok",
            error=result.error,
            recorded_at=datetime.now().isoformat(),
        )


__all__ = ["ExecutionOutcome", "execute_graph"]
