"""Local Runtime API for Pharos Studio.

The server binds to loopback by default. Studio sends graph source to the
runtime for canonical validation/execution and subscribes to an SSE stream for
run and per-fire lifecycle events.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pharos import __version__
from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunBudget
from pharos.ir import load_graph_from_text
from pharos.runtime.executor import execute_graph

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class GraphSourceRequest(BaseModel):
    graph: str = Field(min_length=1, description="YAML or JSON graph source")
    source_name: str = "studio.yaml"
    base_dir: str | None = None


class RunCreateRequest(GraphSourceRequest):
    input: str = ""
    inputs: dict[str, str] = Field(default_factory=dict)
    grants: list[str] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)
    max_iterations: int | None = Field(default=None, ge=1)
    convergence_k: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    budget_mode: Literal["hard", "soft"] | None = None


@dataclass
class RunJob:
    run_id: str
    graph_name: str
    source_name: str
    director: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    result: dict[str, Any] | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    spans: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def append(self, event: dict[str, Any]) -> None:
        async with self.condition:
            item = {
                "seq": len(self.events) + 1,
                "run_id": self.run_id,
                "ts": time.time(),
                **event,
            }
            self.events.append(item)
            self.condition.notify_all()

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "graph_name": self.graph_name,
            "source_name": self.source_name,
            "director": self.director,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "result": self.result,
            "outputs": self.outputs,
            "spans": self.spans,
        }
        if include_events:
            payload["events"] = self.events
        return payload


class RunManager:
    def __init__(self) -> None:
        self.jobs: dict[str, RunJob] = {}

    async def create(self, request: RunCreateRequest) -> RunJob:
        graph, raw = _load_and_validate(request)
        _seed_inputs(graph, request.input, request.inputs)
        _apply_answers(graph, request.answers)

        director = str(raw.get("director", "fn"))
        execution = raw.get("execution", {}) or {}
        max_iterations = request.max_iterations or int(
            execution.get("max_iterations", 20)
        )
        convergence_k = request.convergence_k or int(
            execution.get("convergence_k", 2)
        )
        budget = _build_budget(request, raw.get("budget"))
        run_id = str(uuid.uuid4())
        job = RunJob(
            run_id=run_id,
            graph_name=graph.name,
            source_name=request.source_name,
            director=director,
        )
        self.jobs[run_id] = job
        await job.append({"type": "run.queued", "status": "queued"})
        job.task = asyncio.create_task(
            self._execute(
                job,
                graph,
                director=director,
                max_iterations=max_iterations,
                convergence_k=convergence_k,
                grants=set(request.grants),
                budget=budget,
            )
        )
        self._trim()
        return job

    async def _execute(
        self,
        job: RunJob,
        graph: CompositeGraph,
        *,
        director: str,
        max_iterations: int,
        convergence_k: int,
        grants: set[str],
        budget: RunBudget | None,
    ) -> None:
        job.status = "running"
        job.started_at = time.time()
        await job.append({"type": "run.started", "status": "running"})
        try:
            outcome = await execute_graph(
                graph,
                director,
                run_id=job.run_id,
                max_iters=max_iterations,
                converge_k=convergence_k,
                granted_permissions=grants,
                budget=budget,
                event_sink=job.append,
            )
            job.result = asdict(outcome.result)
            job.spans = [_span_to_dict(span) for span in outcome.spans]
            job.outputs = _graph_outputs(graph)
            job.ended_at = time.time()
            if outcome.result.error:
                job.status = "failed"
                await job.append(
                    {
                        "type": "run.failed",
                        "status": "failed",
                        "error": outcome.result.error,
                        "result": job.result,
                        "outputs": job.outputs,
                    }
                )
            else:
                job.status = "completed"
                await job.append(
                    {
                        "type": "run.completed",
                        "status": "completed",
                        "result": job.result,
                        "outputs": job.outputs,
                    }
                )
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.ended_at = time.time()
            await job.append({"type": "run.cancelled", "status": "cancelled"})
            raise
        except BaseException as exc:
            job.status = "failed"
            job.ended_at = time.time()
            job.result = {"error": f"{type(exc).__name__}: {exc}"}
            await job.append(
                {
                    "type": "run.failed",
                    "status": "failed",
                    "error": job.result["error"],
                    "result": job.result,
                }
            )

    def get(self, run_id: str) -> RunJob:
        job = self.jobs.get(run_id)
        if job is None:
            raise KeyError(run_id)
        return job

    async def cancel(self, run_id: str) -> RunJob:
        job = self.get(run_id)
        if job.task is not None and not job.task.done():
            job.task.cancel()
        return job

    def _trim(self) -> None:
        if len(self.jobs) <= 100:
            return
        completed = sorted(
            (job for job in self.jobs.values() if job.status in TERMINAL_STATUSES),
            key=lambda item: item.created_at,
        )
        for job in completed[: len(self.jobs) - 100]:
            self.jobs.pop(job.run_id, None)


def _load_and_validate(
    request: GraphSourceRequest,
) -> tuple[CompositeGraph, dict[str, Any]]:
    base_dir = Path(request.base_dir).expanduser() if request.base_dir else Path.cwd()
    try:
        graph, raw = load_graph_from_text(request.graph, base_dir=base_dir)
        errors = graph.validate()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return graph, raw


def _seed_inputs(
    graph: CompositeGraph, prompt: str, extra: dict[str, str]
) -> None:
    values = dict(extra)
    if prompt:
        values["prompt"] = prompt
    for edge in graph.edges:
        if edge.src_node != graph.INPUT_NODE_ID or edge.src_port not in values:
            continue
        target = graph.node(edge.dst_node).instance
        if target is not None and edge.dst_port in target.ins:
            target.ins[edge.dst_port].emit(
                TypedValue(type="text", payload=values[edge.src_port])
            )


def _apply_answers(graph: CompositeGraph, answers: dict[str, str]) -> None:
    for node_id, answer in answers.items():
        node = graph.nodes.get(node_id)
        instance = node.instance if node is not None else None
        if instance is not None and hasattr(instance, "answer"):
            instance.answer = answer


def _build_budget(
    request: RunCreateRequest, raw_budget: Any
) -> RunBudget | None:
    config = raw_budget if isinstance(raw_budget, dict) else {}
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else config.get("max_tokens")
    )
    max_cost = (
        request.max_cost_usd
        if request.max_cost_usd is not None
        else config.get("max_cost_usd")
    )
    if max_tokens is None and max_cost is None:
        return None
    return RunBudget(
        max_tokens=max_tokens,
        max_cost_usd=max_cost,
        mode=request.budget_mode or config.get("mode", "hard"),
    )


def _span_to_dict(span: Any) -> dict[str, Any]:
    return {
        "id": span.id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "duration_ms": span.duration_ms,
        "status": span.status,
        "attributes": span.attributes,
        "events": [asdict(event) for event in span.events],
        "error": span.error,
    }


def _graph_outputs(graph: CompositeGraph) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node_id, ports in graph.collected.items():
        result[node_id] = {
            port: [token.value.payload for token in tokens]
            for port, tokens in ports.items()
        }
    return result


async def _event_stream(job: RunJob, after: int = 0):
    index = max(0, after)
    while True:
        heartbeat = False
        async with job.condition:
            if index >= len(job.events) and job.status not in TERMINAL_STATUSES:
                try:
                    await asyncio.wait_for(job.condition.wait(), timeout=15)
                except TimeoutError:
                    heartbeat = True
            batch = job.events[index:]
            index = len(job.events)
            terminal = job.status in TERMINAL_STATUSES
        if heartbeat and not batch:
            yield ": keep-alive\n\n"
        for event in batch:
            yield (
                f"id: {event['seq']}\n"
                f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            )
        if terminal and index >= len(job.events):
            break


def create_app(manager: RunManager | None = None) -> FastAPI:
    run_manager = manager or RunManager()
    app = FastAPI(
        title="Pharos Local Runtime API",
        version=__version__,
        description="Loopback control plane for Pharos Studio.",
    )
    app.state.run_manager = run_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "workspace": str(Path.cwd()),
        }

    @app.post("/api/graphs/validate")
    async def validate_graph(request: GraphSourceRequest) -> dict[str, Any]:
        graph, raw = _load_and_validate(request)
        return {
            "valid": True,
            "name": graph.name,
            "director": raw.get("director", "fn"),
            "node_count": len(raw.get("nodes", [])),
            "edge_count": len(raw.get("edges", [])),
        }

    @app.post("/api/runs", status_code=202)
    async def create_run(request: RunCreateRequest) -> dict[str, Any]:
        job = await run_manager.create(request)
        return job.snapshot(include_events=False)

    @app.get("/api/runs")
    async def list_active_runs() -> list[dict[str, Any]]:
        jobs = sorted(
            run_manager.jobs.values(), key=lambda item: item.created_at, reverse=True
        )
        return [job.snapshot(include_events=False) for job in jobs]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        try:
            return run_manager.get(run_id).snapshot()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            job = run_manager.get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        resume_after = after
        if last_event_id is not None:
            with contextlib.suppress(ValueError):
                resume_after = max(resume_after, int(last_event_id))
        return StreamingResponse(
            _event_stream(job, after=resume_after),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            job = await run_manager.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return job.snapshot(include_events=False)

    return app


app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["RunCreateRequest", "RunManager", "app", "create_app", "run_server"]
