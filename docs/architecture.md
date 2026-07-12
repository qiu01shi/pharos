# pharos — Architecture

pharos is a typed dataflow runtime for LLM workflows. It models an
LLM pipeline as a directed graph: nodes are typed **Entities**
(LLM calls, shell commands, custom logic), edges are typed **Ports**,
and a **Director** drives the firing cycle.

The design is intentionally inspired by [Ptolemy II][ptolemy] but
rebuilt for the LLM era: streaming by default, observability as a
first-class concern, and no Java runtime.

[ptolemy]: https://ptolemy.berkeley.edu/ptolemyii/

---

## Why pharos exists

Most "multi-agent" frameworks today are prompt-orchestrators: they
hand strings between LLM calls, with no type checks, no replay, and
no way to see what actually happened when something goes wrong.

pharos takes a different stance:

1. **Types cross LLM boundaries.** If a downstream node expects a
   JSON object with field `{"file": str, "line": int}`, an upstream
   that emits `{"path": str, "ln": int}` is rejected at the **port**
   — not at runtime 5 minutes later when an LLM agent produces a
   bad patch. (`pharos/core/port.py`)

2. **Every token carries lineage.** A `Token` is frozen and includes
   `prev_hash` + `self_hash` + `ts` + `run_id` + `cost_usd`. Its content
   hash deliberately excludes wall-clock and run identity, so deterministic
   graph executions can be compared across runs. (`pharos/core/token.py`)

3. **Trace is built in, not bolted on.** Every `Entity.fire()` is
   wrapped in a `Span` by the Director. Every port emit is logged.
   You can replay a run later by re-executing the trace. (`pharos/directors/fn.py`, `pharos/observability/trace.py`)

4. **Directors are pluggable.** `FN` (one-shot, topologically), `DE`
   (discrete-event), and `SDF` (synchronous rounds with feedback loops)
   ship today. Additional semantics should be added only for validated use
   cases.

5. **No subprocess boundary for LLM calls.** subprocess startup is
   500ms — the original `pi-director` (the project's predecessor)
   paid this for every actor fire. pharos keeps a long-lived
   `httpx.AsyncClient` per provider and uses `asyncio.gather` for
   concurrency. The 100-actor benchmark runs in ~14ms, vs. ~1000ms
   sequentially.

---

## Layered architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CLI (pharos run / validate / list-providers / doctor)        │
│      Typer + Rich                                           │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│  IR (YAML → Graph)                                           │
│      Pydantic schema + load_graph()                         │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│  Runtime                                                     │
│    Director (FN, DE, SDF) ── schedules fires                  │
│    Tracer  ── records every span (OTel-compatible)            │
│    Cost   ── aggregates token usage and $$                  │
│    Replay ── reproduces a run from trace                     │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│  Core                                                        │
│    Token / TypedValue  (frozen, hash-chained)               │
│    Port / InputPort / OutputPort (deque + schema)           │
│    Entity (setup / fire / teardown) + @entity decorator     │
│    Edge / CompositeGraph (NetworkX-backed)                  │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│  LLM Layer                                                   │
│    types (Usage, CostRecord, Model, StreamEvent, ...)       │
│    LLMProvider Protocol (stream / complete / list_models)    │
│    OpenAIProvider (Responses + Chat Completions)             │
│    GLMProvider   (Volcengine Ark, inherits OpenAI)           │
│    FauxProvider  (mock for tests / benchmarks)               │
└──────────────────────────────────────────────────────────────┘
```

Every layer is testable in isolation; dependencies point downward.

---

## Core abstractions

### Token

```python
@dataclass(frozen=True)
class Token:
    value: TypedValue       # type + payload
    origin: str             # "node_id.port_name"
    ts: float
    prev_hash: str | None
    self_hash: str          # sha256 of canonical(value + prev_hash + origin + iter)
    run_id: str
    iter: int
    is_partial: bool
    cost_usd: float
```

`self_hash` enables a critical property: **replay equivalence**. Two
runs of the same graph with the same LLM and input should produce
the same `self_hash` for the head token — if not, something non-
deterministic happened (LLM temperature > 0, network jitter, etc.).

### Port

```python
@dataclass
class InputPort:
    name: str
    accepted_types: list[str]       # strict type whitelist
    capacity: int | None
    overflow: "backpressure" | "drop_oldest" | "drop_newest"
    buffer: deque[Token]
    metrics: ...
```

`Port.receive()` validates the token's `type` tag against
`accepted_types` and raises `PortContractViolation` on mismatch.
This is the LLM hallucination guard.

### Entity

```python
@entity
class LLMAgent(Entity):
    ins  = {"prompt": InputPort("prompt", accepted_types=["text"])}
    outs = {"text": OutputPort("text", accepted_types=["text"]),
            "draft": OutputPort("draft", accepted_types=["text"]),
            "usage": OutputPort("usage", accepted_types=["json"])}
    async def fire(self, ctx):
        # Read self.ins, write self.outs
        ...
```

Lifecycle: `setup()` once → `fire()` many → `teardown()` once.
Long-lived resources (HTTP clients) belong in `setup()`.

### CompositeGraph

```python
g = CompositeGraph(name="hello")
g.add_entity("agent", LLMAgent(...))
g.add_entity("runner", ShellEntity(...))
g.connect("agent.text", "runner.command")
```

Plus virtual `__in__` and `__out__` nodes for graph-level I/O.
Token delivery to a virtual `__out__` is collected onto
`graph.collected` so the CLI can display results.

---

## The Director abstraction

A `Director` is the scheduler. Three semantics currently ship:

| Director | When it fires | When it stops |
| --- | --- | --- |
| `FN` (Function) | Topological order, same layer runs concurrently | One pass |
| `DE` (Discrete Event) | Whenever an input port receives a new token | All buffers empty |
| `SDF` (Synchronous Data Flow) | Production/consumption rates; supports feedback loops | `max_iter` or fixed-point reached (K-of-N token-hash) |

New Directors require explicit firing, termination, checkpoint, and replay
contracts before they are added.

---

## Observability

Every `Entity.fire()` is wrapped in a `Span` by the Director:

```python
span = tracer.start_span(
    "entity.fire.leaf_42",
    parent=current_span(),
    attributes={"entity": "leaf_42", "entity_class": "LLMLoop"},
)
try:
    await entity.fire(fire_ctx)
finally:
    tracer.finish_span(span)
```

`Span` carries `trace_id` (inherited from parent or new at root),
`parent_span_id`, `attributes`, `events` (timeline annotations),
and a `duration_ms`. Compatible with OpenTelemetry semantic
conventions — a future OTel exporter can ship spans to any OTel
backend without code changes.

The `ConsoleTraceBackend` renders spans as a tree:

```
trace_0001_a1b2c3
├─ entity.fire.source  (BroadcastSource)  0.1 ms
├─ entity.fire.leaf_0  (LLMLoop)         11.2 ms
├─ entity.fire.leaf_1  (LLMLoop)         11.2 ms
...
└─ entity.fire.sink    (CollectorSink)     0.1 ms
```

---

## LLM Provider design

`LLMProvider` is a Protocol:

```python
class LLMProvider(Protocol):
    name: str
    async def list_models(self) -> list[Model]: ...
    def stream(
        self, model: Model, context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
    async def complete(
        self, model: Model, context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage: ...
    async def close(self) -> None: ...
```

The contract: **errors terminate the stream via a `StreamEvent(type="error", error=...)`, not via exceptions**. This matches the pattern used by `pi-ai`, Anthropic's SDK, and OpenAI's streaming — the agent loop in `LLMAgent` is a single `async for` over events, with error handling inline.

`acomplete_from_stream` is the helper that turns a stream into a
final `AssistantMessage`. **Important**: it explicitly `await
events.aclose()` in a `finally` block, because `break` from an
`async for` does NOT run the generator's `finally` (asyncio
generator semantics). Without this, providers that record state in
their `finally` block (like `FauxProvider`) lose data.

### Why subclass for new providers

`GLMProvider(OpenAIProvider)` — Volcengine Ark's API is OpenAI-
compatible, so we just override `__init__` to swap the base URL and
API key lookup. Same for `DeepSeekProvider` (future) and any other
OpenAI-compatible endpoint. This keeps each new provider to ~50
lines.

`AnthropicProvider` is independent because the Anthropic Messages
API has different request/response shape (e.g. `thinking` budget
configuration).

---

## Performance

The hello-world benchmark (`bench/hello_world.py`) runs **100
concurrent actors** doing 10ms FauxProvider calls each.

```
mean: 15.0 ms
p50:  13.9 ms
p95:  14.0 ms
target: 2000 ms  (exceeded by 140x)
```

Compare to pi-director (the predecessor): 1000ms sequentially
because each `fire()` `subprocess.run`'d a `pi` interpreter.

The factor that matters: **asyncio.gather on the same topological
layer**. As long as 100 entities share a depth, they run in
parallel. The "100x fan-out" works because they don't actually
block each other — the 14ms is dominated by the single FauxProvider
call.

---

## What's NOT in pharos (by design)

- ❌ **No general agent OS.** pharos is a platform primitive, not a
  consumer product.
- ⚠️ **Studio is a local inspection workbench, not yet a full editor.** It can
  load YAML/JSON graphs and recorded runs, validate references, inspect nodes,
  and overlay execution traces. Editing and round-trip persistence come next.
- ⚠️ **Remote/container execution is a leaf-node boundary.** Any language can
  implement `pharos.worker/v1`, but the Pharos process still owns scheduling,
  retries, budgets, authorization, and graph state.
- ❌ **No automatic distributed execution.** Single-process by
  design. Distributed comes much later (P9+).
- ❌ **No LangChain dependency.** The LLM layer is implemented
  directly against provider HTTP APIs (~500 lines per provider).
  LangChain's abstractions don't fit the typed-port model.

---

## Roadmap

| Phase | What ships | Status |
| --- | --- | --- |
| **Foundation** | Core, FN/DE/SDF, providers, trace/replay, Agent CI | ✅ shipped |
| **Correctness** | Per-fire accounting, stable convergence, graph reset, typed compile checks | ✅ shipped |
| **Authoring** | Versioned IR, Python builder, team templates, Studio inspector | ✅ foundation |
| **Interop** | Container/remote entities and worker protocol | ✅ foundation |
| **Scale** | Durable distributed control plane | demand-driven |

See `docs/P0-COMPLETE.md` for the detailed P0 retrospective.
