# pharos P5 — Complete

**Status:** P5 done. **218 tests pass, ruff clean, byte-equal replay via ReplayProvider.**

---

## What was added on top of P4

### `ReplayProvider` — replay recorded LLM outputs without network

```python
from pharos.llm.providers.replay import ReplayProvider
from pharos.runtime import extract_cached_outputs

cache = extract_cached_outputs("run-uuid-1234")
provider = ReplayProvider(node_id="agent", cache=cache)
# When this provider is used in a graph, it returns the recorded
# output for that node/step — no network call, no cost.
```

The provider's `stream()` yields the same `StreamEvent` sequence
that was captured by `record_run`: `start` → `text_delta` × N →
`done`. The done event's `message` carries the full
`AssistantMessage` reconstructed from the recorded attributes.

### `extract_cached_outputs` now extracts the full event sequence

Previously (P3) this only accumulated text. P5 captures:
- All `StreamEvent`s in order
- The final `AssistantMessage` (with `TextContent`,
  `ThinkingContent`)
- `ToolCall` events
- Usage dict (input/output/cache_read/cache_write)
- Model + provider names

### `pharos replay --re-run --graph <path> <run_id>`

```bash
$ uv run pharos run graphs/03_faux_demo.yaml --input "hi"
Run: hello-faux  director=fn  converged=True  iterations=3  tokens=3
  __out__.response → echo: hi

$ uv run pharos replay --re-run --graph graphs/03_faux_demo.yaml <run-id>
Replaying run <run-id> with 1 LLM agent(s) using cached outputs (no network).
Run: hello-faux  director=fn  converged=True  iterations=3  tokens=3
  __out__.response → echo: hi       ← same output, no API call
```

CLI semantics:
- `--re-run` flag swaps every `LLMAgent`'s provider for a
  `ReplayProvider` keyed by the node's id.
- `--graph` specifies which YAML to re-run (the run alone doesn't
  contain the graph structure).
- A placeholder prompt "(replay)" is seeded so the graph fires.
  The actual input text doesn't matter — `ReplayProvider` ignores
  the context and replays the cached output.

---

## Verification

```bash
$ uv run pharos run graphs/03_faux_demo.yaml --input "hi"   # original
Run: hello-faux  director=fn  converged=True  iterations=3  tokens=3
  __out__.response → echo: hi

$ uv run pharos replay --re-run --graph graphs/03_faux_demo.yaml <run-id>
Replaying run <run-id> with 1 LLM agent(s) using cached outputs (no network).
Run: hello-faux  director=fn  converged=True  iterations=3  tokens=3
  __out__.response → echo: hi       ← byte-equal
```

`test_full_replay_round_trip` proves this in a unit test:
- Records a run with a `FauxProvider` that returns `"original"`.
- Builds a fresh graph using `ReplayProvider` and the cached output.
- Asserts the replayed output equals `"original"`.

---

## Key engineering decisions

### 1. Step counter starts at 1, not 0
The Director generates `step_id = "<run_id>:N"` where N starts at
1. `extract_cached_outputs` keys cache by `node_id:step_index`.
If `ReplayProvider._step_counter` started at 0, it would always
miss the first cache entry. Lesson: match the producer's
numbering.

### 2. `ReplayProvider` is a **single-node** provider
Each `LLMAgent` instance gets its own `ReplayProvider` keyed by
node_id. This is intentional — different nodes have different
cached outputs. A graph with 3 LLM agents needs 3 ReplayProvider
instances. CLI auto-creates them in `_replay_rerun`.

### 3. Events are reconstructed from asdict, not pickling
`record_run` calls `asdict(span_event)` to serialize the recorded
events. On replay, `_dict_to_stream_event` rebuilds `StreamEvent`
objects from those dicts. We don't `pickle` because:
- JSON-serializable
- Forward-compatible (older replays can read newer traces, with
  unknown event types safely ignored)
- Smaller on disk

### 4. Placeholder prompt
`LLMAgent.fire()` returns early if `prompt_tokens` is empty. So
`_replay_rerun` injects a literal `"(replay)"` to ensure the
graph fires. The replay ignores this anyway — `ReplayProvider`
returns cached output regardless of `Context.messages`.

### 5. `extract_cached_outputs` filters to LLMAgent only
A graph may contain non-LLM entities (e.g. `ShellEntity`,
`Memory`, `Router`). Replay only makes sense for LLM outputs.
We filter by `entity_class == "LLMAgent"` at extraction time.

---

## Limitations

- **No execution of custom Python entities on replay** — only LLM
  agents get their output replayed. A `WordCounter` from P3 will
  re-execute normally (it's deterministic). A `ShellEntity`
  *will* re-run its shell command (we don't capture stdout). For
  full replay including shell, we'd need to capture stdout in
  `ShellEntity` too.
- **No replay of LLM tool calls yet** — `ToolCall` events are
  recorded but `ReplayProvider` doesn't synthesize tool execution.
  A tool-call replay would need a stub `ToolExecutor` that
  returns the recorded tool result.
- **Re-run uses FNDirector** — SDF/DE replay is not yet wired into
  the CLI. The replay cache itself works for any director (it's
  keyed by `(node_id, step_index)`), but the CLI hardcodes FN.

---

## File map (P5 additions)

```
pharos/
├── pharos/
│   ├── llm/providers/replay.py    (NEW: ReplayProvider, ~250 lines)
│   ├── runtime/__init__.py        (extract_cached_outputs: full event capture)
│   └── cli.py                     (added `replay --re-run` + `_replay_rerun`)
└── tests/
    └── llm/test_replay.py         (NEW: 9 tests)
```

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 218 tests in ~10s
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # P95 ≈ 11ms (baseline unchanged)

# Record a run
uv run pharos run graphs/03_faux_demo.yaml --input "hi"

# Find the run id
uv run pharos trace list

# Replay (no network!)
uv run pharos replay --re-run --graph graphs/03_faux_demo.yaml <run-id>
```