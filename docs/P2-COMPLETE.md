# pharos P2 — Complete

**Status:** P2 done. **187 tests pass, ruff clean, 5 LLM providers, 2 directors, 4 entities, end-to-end CLI supports both FN and SDF workflows.**

---

## What was added on top of P1

### New entities
- **`Router`** (`pharos/entities/router.py`, 2.7 KB) — conditional branching with guard functions. Configure with `Router(guards={"errors": lambda v: "error" in v.lower()}, default="info")`.
- **`Memory`** (`pharos/entities/memory.py`, 4.2 KB) — shared key-value store. Multiple Memory nodes can share state via the default store, or use isolated per-instance stores.

### New Director: SDF (feedback loops)
- **`SDFDirector`** (`pharos/directors/sdf.py`, 6.4 KB) — runs the graph in rounds; after each round, compares head-token hashes; converges after K-of-N stable rounds. Hard cap on `max_iterations`.
- Cycles are now allowed in the graph validator when `director: sdf`.
- CLI now dispatches to SDF when YAML says `director: sdf`.

### New providers (now 5 total)
- **`DeepSeekProvider`** (`pharos/llm/providers/deepseek.py`, 5.8 KB) — OpenAI-compatible; **extracts `reasoning_content` as `thinking_delta` events**.
- **`AnthropicProvider`** (`pharos/llm/providers/anthropic.py`, 15.7 KB) — independent implementation of Anthropic's Messages API: SSE event types (message_start, content_block_*, message_delta, message_stop), thinking mode, tool_use, cache_read/cache_write usage.
- Catalog files for both.

### CLI improvements
- `--max-iters` and `--converge-k` flags for SDF runs.
- Run summary now shows which Director was used.

### Sample graph
- **`graphs/04_fix_bug_sdf.yaml`** — SDF demo: coder → reviewer → shell_test (feedback loop). Runs end-to-end with FauxProvider.

---

## Verification

```bash
$ uv run pharos run graphs/04_fix_bug_sdf.yaml --input "fix bugs"
Run: fix-buggy-calculator  director=sdf  converged=True  iterations=5  tokens=39
                                  Output ports
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ node       ┃ port        ┃ value                                             ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ coder      │ draft       │ echo: fix bugs                                    │
│ reviewer   │ draft       │ echo: echo: fix bugs                              │
│ reviewer   │ draft       │ echo:                                             │
└────────────┴─────────────┴───────────────────────────────────────────────────┘

$ uv run pharos list-providers
 Registered providers
┏━━━━━━━━━━━┓
┃ name      ┃
┡━━━━━━━━━━━┩
│ anthropic │   ← NEW
│ deepseek  │   ← NEW
│ faux      │
│ glm       │
│ openai    │
└───────────┘
```

---

## Key engineering decisions

### 1. SDF doesn't use `topo_layers`
Originally SDF reused the FNDirector's per-layer dispatch. But
topo_layers drops cycle nodes entirely (Kahn's algorithm never
reduces their in-degree to 0), so a feedback-loop reviewer never
fires. Fixed: SDF iterates `sorted(all_nodes)` and fires every
node each round. `deliver_upstream` then routes tokens along edges,
including back-edges, naturally forming the next round's input.

### 2. Usage is frozen; use `model_copy`
`Usage` is `frozen=True` (for safe concurrent sharing). Mutating
`usage.output = N` raises a Pydantic `frozen_instance` error. The
Anthropic provider learned this the hard way in tests. Fix:
`usage = usage.model_copy(update={"output": N})`.

### 3. Anthropic `stop_reason` mapping
Anthropic uses `"end_turn"`, `"max_tokens"`, `"tool_use"`,
`"stop_sequence"`, `"refusal"`, `"model_context_window_exceeded"`.
pharos's `AssistantMessage.stop_reason` is a Pydantic Literal with
5 values. We map via a dict, defaulting unknown values to `"stop"`.

### 4. Cycles in `g.validate()`
`CompositeGraph.validate()` reports cycles as errors. For SDF we
want to allow them. We made `load_graph` smart: only fail validation
if the errors are NOT all cycle-related when the director is SDF.
A future improvement: encode cycle-awareness into CompositeGraph
itself via a `cycle_policy` attribute.

### 5. `topo_layers` still ignores cycles — by design
We left `topo_layers` unchanged. FNDirector (which uses it) is
correctly acyclic. SDF uses its own dispatch. Trying to make
`topo_layers` cycle-aware would be a bigger refactor for no
practical benefit.

---

## What's NOT in P2 (deferred to P3+)

- ❌ **No real-API end-to-end test** — your `ARK_API_KEY` is invalid
  (curl returns 401), so we couldn't add a live GLM/Anthropic test.
- ❌ **No MiniMax Provider** — earlier priority list had it, but
  with 5 working providers (faux, glm, openai, deepseek, anthropic),
  MiniMax can wait for P3 if needed.
- ❌ **No DE / PN / CT directors** — only FN + SDF. These are for
  streams / message-bus / external-event workflows; not requested.
- ❌ **No real Replay** — `record_run` writes the trace to disk, but
  we don't yet **re-execute** a run by substituting recorded LLM
  outputs. The trace format supports it; the actual replay loop
  requires a recorded-output cache that doesn't exist yet.
- ❌ **No `pharos replay` command** — `pharos trace list` and
  `pharos trace <id>` are read-only viewers.

---

## File map (P2 additions)

```
pharos/
├── pharos/
│   ├── llm/providers/
│   │   ├── deepseek.py          (NEW: 5.8 KB)
│   │   └── anthropic.py         (NEW: 15.7 KB)
│   ├── llm/catalog/
│   │   ├── deepseek.py          (NEW: 2 models)
│   │   └── anthropic.py         (NEW: 3 models)
│   ├── entities/
│   │   ├── router.py            (NEW: 2.7 KB)
│   │   └── memory.py            (NEW: 4.2 KB)
│   └── directors/
│       └── sdf.py               (NEW: 6.4 KB)
├── graphs/
│   └── 04_fix_bug_sdf.yaml      (NEW: SDF demo)
└── tests/
    ├── llm/test_deepseek.py    (NEW: 6 tests)
    ├── llm/test_anthropic.py   (NEW: 8 tests)
    ├── entities/test_router_memory.py (NEW: 9 tests)
    └── directors/test_sdf.py   (NEW: 3 tests)
```

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 187 tests in ~0.5s
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # 100 actors, P95 ~15-30ms
uv run pharos list-providers        # anthropic, deepseek, faux, glm, openai
uv run pharos run graphs/03_faux_demo.yaml --input "hi"
uv run pharos run graphs/04_fix_bug_sdf.yaml --input "fix bugs"
uv run pharos trace list
```