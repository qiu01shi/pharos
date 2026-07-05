# pharos P4 — Complete

**Status:** P4 done. **209 tests pass, ruff clean, 3 Directors (FN/SDF/DE), 5 providers, 6 entities + custom Python.**

---

## What was added on top of P3

### DE (Discrete Event) Director

```yaml
name: de-demo
director: de   # ← new
nodes:
  - id: agent
    type: faux
    provider: faux
    ...
edges:
  - { src: __in__.prompt, dst: agent.prompt }
  - { src: agent.text,    dst: __out__.response }
```

`DEDirector` (`pharos/directors/de.py`) fires a node only when:
- Its `ins` buffer has pending tokens, OR
- It is a source-like node (no inputs declared, has outputs) that
  hasn't fired yet.

The run converges when a full pass leaves nothing eligible to fire
(the system is quiescent). Hard cap on `max_iterations`.

Use case: incremental workflows where events trickle in and
downstream entities react only when there's new work.

### CLI / IR support

- `DirectorName` literal accepts `"de"`.
- `pharos run` auto-dispatches to `DEDirector` when YAML says
  `director: de`.
- CLI `--max-iters` flag now also caps DE runs.

### Sample graph
- `graphs/06_de_demo.yaml` — single faux LLM, end-to-end DE run.

---

## Verification

```bash
$ uv run pharos run graphs/06_de_demo.yaml --input "hi"
Run: de-demo  director=de  converged=True  iterations=2  tokens=3
  agent.draft    → echo: hi
  __out__.response → echo: hi
```

---

## Key engineering decisions

### 1. DE firing rule is **trigger-driven**, not topological
Unlike FN (every node fires once) and SDF (every node fires each
round), DE asks: "does this node have pending input?" Sources are
the only exception — they fire exactly once when first seen.

This means a 3-layer pipeline with one input token goes:
- Round 1: source fires (it's a source), then counter fires (it
  got input). Both contribute. Round 2: nothing has input →
  converged.

### 2. `fired_sources` set for source idempotency
A source fires at most once per run, tracked by `fired_sources`.
Without this, a no-input source would re-fire every round and
never quiesce.

### 3. `topo_layers` is still used for order
Even though not every layer has a fire-eligible node, we walk the
layers in order so that data flows top-down (source → middle →
sink). An entity in layer 2 can only fire after its upstream
inputs have been delivered.

### 4. DE without self-loops is the intended pattern
If you need a node to keep firing (e.g. polling), use a feedback
edge from a downstream port. SDF is the right tool for feedback
loops with convergence guarantees; DE is the right tool for
"fire when input arrives".

---

## Known limitations

- **No self-loop support**: `topo_layers` skips cycle nodes, so a
  graph with a self-loop will only fire the source-like endpoint
  once. This is the same behavior as FNDirector; SDFDirector is
  the correct tool for cycles.
- **No priority scheduling**: all eligible nodes in a layer fire
  concurrently (asyncio.gather). For a real-time-ish DE scheduler
  you'd want priorities and timestamps, which is out of scope.
- **No input event types**: DE fires on the presence of tokens,
  not on token type or metadata. A future version could read
  `token.value.type` to dispatch.

---

## File map (P4 additions)

```
pharos/
├── pharos/
│   ├── directors/de.py         (NEW: DEDirector, 5.4 KB)
│   └── directors/__init__.py   (added DEDirector export)
├── graphs/
│   └── 06_de_demo.yaml         (NEW)
└── tests/
    └── directors/test_de.py    (NEW: 5 tests)
```

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 209 tests in ~0.5s
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # P95 ≈ 11ms (baseline unchanged)
uv run pharos run graphs/06_de_demo.yaml --input "hi"
uv run pharos run graphs/04_fix_bug_sdf.yaml --input "fix bug"  # SDF still works
uv run pharos run graphs/03_faux_demo.yaml --input "hi"        # FN still works
```