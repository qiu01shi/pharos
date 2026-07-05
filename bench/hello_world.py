"""Hello world: 100 fan-out entities firing concurrently.

This is the P0 performance baseline. Goal: P95 < 2s for 100 actors
each doing a 10ms FauxProvider LLM call (no real network).

The graph is a true fan-out: 1 source → 100 parallel leaves → 1 sink.
All 100 leaves run in a single layer (asyncio.gather). Total latency
should be approx slowest leaf, not 100 x leaf_latency.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.llm.providers.faux import FauxConfig, FauxProvider
from pharos.llm.types import Context, StreamOptions, UserMessage
from pharos.observability.backend.console import ConsoleTraceBackend
from pharos.observability.trace import InMemoryTracer


@entity
class BroadcastSource(Entity):
    """Emits one token that downstream fan-out will copy via edges."""

    outs = {"prompt": OutputPort(name="prompt", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        self.outs["prompt"].emit(TypedValue(type="text", payload="hello"))


@entity
class LLMLoop(Entity):
    """One-shot LLM call to the shared FauxProvider."""

    ins = {"prompt": InputPort(name="prompt", accepted_types=["text"])}
    outs = {"text": OutputPort(name="text", accepted_types=["text"])}

    def __init__(self, node_id, provider, model):
        super().__init__(node_id=node_id)
        self.provider = provider
        self.model = model

    async def fire(self, ctx):  # type: ignore[override]
        toks = self.ins["prompt"].consume()
        if not toks:
            return
        prompt = toks[0].value.payload
        ctx_in = Context(messages=[UserMessage(content=prompt)])
        msg = await self.provider.complete(self.model, ctx_in, StreamOptions())
        self.outs["text"].emit(TypedValue(type="text", payload=msg.text()))


@entity
class CollectorSink(Entity):
    """Counts how many text tokens it received."""

    ins = {"text": InputPort(name="text", accepted_types=["text"])}
    outs = {"count": OutputPort(name="count", accepted_types=["int"])}

    def __init__(self, node_id):
        super().__init__(node_id=node_id)
        self.received = 0

    async def fire(self, ctx):  # type: ignore[override]
        n = len(self.ins["text"].consume())
        self.received += n
        self.outs["count"].emit(TypedValue(type="int", payload=self.received))


async def run_once(n_leaves: int) -> tuple[float, int]:
    """Build the graph, run it once, return (wallclock_seconds, leaves_fired)."""
    provider = FauxProvider(FauxConfig(latency_seconds=0.01))
    model = (await provider.list_models())[0]

    g = CompositeGraph(name="hello")
    source = BroadcastSource(node_id="source")
    g.add_entity("source", source)

    for i in range(n_leaves):
        leaf = LLMLoop(node_id=f"leaf_{i}", provider=provider, model=model)
        g.add_entity(f"leaf_{i}", leaf)

    sink = CollectorSink(node_id="sink")
    g.add_entity("sink", sink)

    # Fan out: source → every leaf
    for i in range(n_leaves):
        g.connect("source.prompt", f"leaf_{i}.prompt")
        g.connect(f"leaf_{i}.text", "sink.text")

    # Seed source buffer (in real use this comes from __in__)
    g.nodes["source"].instance.outs["prompt"].emit(  # type: ignore[union-attr]
        TypedValue(type="text", payload="kickoff")
    )

    director = FNDirector()
    tracer = InMemoryTracer()
    backend = ConsoleTraceBackend()
    run_ctx = RunContext(
        run_id=str(uuid.uuid4()),
        config={"tracer": tracer},
    )
    # Attach tracer to the context so _safe_fire can find it
    run_ctx.tracer = tracer  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    result = await director.run(g, run_ctx)
    elapsed = time.perf_counter() - t0

    # Push spans into the backend for tree rendering
    for s in tracer.spans:
        backend.write(s)
    return elapsed, sink.received, result, backend


async def main() -> None:
    N = 100
    print(f"pharos P0 performance baseline: {N} leaves, 10ms FauxProvider each")
    print("=" * 60)

    # Warmup
    await run_once(5)

    # Measure
    runs = 5
    times: list[float] = []
    counts: list[int] = []
    last_backend: ConsoleTraceBackend | None = None
    for i in range(runs):
        elapsed, leaves, result, backend = await run_once(N)
        times.append(elapsed)
        counts.append(leaves)
        last_backend = backend
        print(
            f"  run {i + 1}: {elapsed * 1000:7.1f} ms  "
            f"(leaves_fired={leaves}, "
            f"iterations={result.iterations}, "
            f"converged={result.converged})"
        )

    times_sorted = sorted(times)
    p50 = times_sorted[len(times) // 2] * 1000
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    p95 = times_sorted[p95_idx] * 1000
    mean = statistics.mean(times) * 1000
    print()
    print(f"  mean: {mean:7.1f} ms")
    print(f"  p50:  {p50:7.1f} ms")
    print(f"  p95:  {p95:7.1f} ms")
    print()
    target_p95 = 2000.0
    if p95 < target_p95:
        print(f"  PASS: p95 {p95:.1f} ms < target {target_p95:.0f} ms")
    else:
        print(f"  FAIL: p95 {p95:.1f} ms >= target {target_p95:.0f} ms")

    # Show a sample trace (small subset)
    if last_backend is not None and last_backend.spans:
        sample = last_backend.spans[:5]
        print()
        print(f"  sample trace ({len(last_backend.spans)} spans total, first 5):")
        for s in sample:
            attrs = " ".join(f"{k}={v}" for k, v in s.attributes.items())
            print(f"    [{s.duration_ms:6.1f}ms] {s.name}  {attrs}")


if __name__ == "__main__":
    asyncio.run(main())
