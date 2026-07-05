# pharos P0 — Complete

**Status:** All P0 tasks done. **121 tests pass, ruff 0 errors, P95 = 14ms for 100 concurrent actors** (target was 2000ms).

---

## What was built

### Core abstractions (`pharos/core/`)
- **`Token`** — typed payload + canonical hash chain for replay
- **`TypedValue`** — schema-tagged data
- **`Port` / `InputPort` / `OutputPort`** — `deque[Token]` buffers, strict schema validation, capacity + 3 overflow strategies
- **`Entity`** — ABC with `setup/fire/teardown` lifecycle, `@entity` decorator collects declared ports
- **`Edge`** — directed connection
- **`CompositeGraph`** — NetworkX-backed graph with `topo_order`, `topo_layers`, cycle detection, transparent port exports

### LLM layer (`pharos/llm/`)
- **`types.py`** — Pydantic models for `Usage` / `CostRecord` / `Message` / `StreamEvent` / `Model` / etc. All frozen.
- **`LLMProvider` Protocol** — unified `stream()` / `complete()` / `list_models()` / `close()`
- **`acomplete_from_stream`** — the helper that taught us: **always `aclose()` after `break` in async-for** to run finally blocks
- **`FauxProvider`** — controllable mock with delay / error injection / scripted responses / call records
- **`registry.py`** — name → class lookup with self-registration
- **Common utils**: `with_retry` (exponential backoff + jitter), `estimate_tokens` (CJK-aware)

### Directors (`pharos/directors/`)
- **`base.py`** — `RunContext` / `FireContext` / `RunResult` / `topo_layers()` / `deliver_upstream()` (with **fan-out replication**)
- **`FNDirector`** — one-shot topological execution with **same-layer `asyncio.gather` concurrency** + automatic span wrapping
- ✅ 9 tests for FN + base helpers

### Observability (`pharos/observability/`)
- **`Span`** — OTel-compatible span with attributes / events / status / duration
- **`InMemoryTracer`** — reference impl with `ContextVar` for `current_span` / `current_trace_id`
- **`EventBus`** — in-process pub/sub for business events
- **`StructuredLogger`** — JSON line logger, auto-links to active trace
- **`Metrics`** — counters / gauges / histograms with labels + percentile summary
- **`ConsoleTraceBackend`** — pretty-prints span trees
- ✅ 24 tests

### Benchmarks (`bench/hello_world.py`)
- 100 concurrent actors doing 10ms FauxProvider calls
- **Mean = 15.0ms, P50 = 13.9ms, P95 = 14.0ms** (target was 2000ms)
- 100 leaves all fire (concurrent, not serialized)
- 102 spans captured end-to-end

### Tooling
- `pyproject.toml` with uv, ruff, mypy, pytest
- 121 tests across 5 test modules
- README + .gitignore + .python-version

---

## Key engineering decisions (and why)

### 1. Async generator cleanup is non-obvious
First version of `acomplete_from_stream` lost the FauxProvider's call records because `break` in `async for` doesn't run the generator's `finally` block. Fix: explicit `await events.aclose()` in a `finally`. This will matter for every real provider (HTTP cleanup, span finalization, etc).

### 2. Fan-out must replicate, not move
First version of `deliver_upstream` moved all tokens from a source port to one downstream at a time, losing tokens when a source fed multiple sinks. Fix: snapshot tokens once, then push copies to every downstream edge.

### 3. `@entity` decorator must recurse into `ins`/`outs` dicts
The dict-of-ports declaration style means the decorator has to dig into the dict to find the Port objects, not just look at top-level attributes. This took one debug round to figure out.

### 4. Trace is a first-class concern, not a debug tool
Every Entity fire is automatically wrapped in a span by `_safe_fire`. The `tracer` is plumbed via `RunContext.tracer` (duck-typed attribute), so any Tracer backend works without coupling.

### 5. Concurrent fan-out is real
The 70× speedup (1000ms serial → 14ms pharos) validates that the asyncio.gather + topological layer approach delivers the kind of concurrency users expect from "multi-agent" frameworks — without the complexity of a full scheduler.

---

## What's NOT in P0 (intentionally deferred to P1+)

- ❌ DE / PN / SDF / CT directors (FN covers ~80% of use cases)
- ❌ Real LLM providers (GLM / OpenAI / Anthropic / DeepSeek / MiniMax)
- ❌ YAML IR loader + CLI (`pharos run graph.yaml`)
- ❌ SQLite trace backend
- ❌ Replay command
- ❌ Document upload for `why-trace.md` / `architecture.md`
- ❌ ToolEntity / HumanEntity / MemoryEntity
- ❌ Router Entity
- ❌ Real bug-fix case study (场景 A)

---

## File map

```
/Users/heshuwen/pharos/
├── pyproject.toml             uv + ruff + mypy + pytest config
├── README.md                  quickstart
├── pharos/
│   ├── __init__.py
│   ├── core/                  Token / Port / Entity / Graph (6 files)
│   ├── llm/                   LLM types + Provider Protocol + FauxProvider (5 files)
│   ├── llm/providers/         faux.py
│   ├── directors/             base + fn (2 files)
│   ├── observability/         trace / metrics / logs / events / console backend (5 files)
│   └── observability/backend/ console.py
├── tests/                     121 tests across 6 modules
├── bench/hello_world.py       100-actor baseline, prints trace sample
├── docs/                      empty — docs deferred
└── graphs/                    empty — YAML IR deferred
```

---

## How to reproduce the numbers

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 121 tests in ~0.4s
uv run python bench/hello_world.py  # 100 actors, P95 ~14ms
```
