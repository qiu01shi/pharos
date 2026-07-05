# pharos P3 — Complete

**Status:** P3 done. **204 tests pass, ruff clean, custom Python entities loadable from YAML, deterministic replay inspects recorded runs.**

---

## What was added on top of P2

### Custom Python entities from YAML

```yaml
nodes:
  - id: counter
    type: python
    class: "my_pkg.entities:WordCounter"
    params:
      prefix: ">>> "
```

The IR loader now supports a new `type: python` node. The `class`
field is a `module.path:ClassName` reference. The class is imported
on demand and must subclass `pharos.core.entity.Entity`. Optional
`params` are passed as keyword arguments to `__init__`.

Errors are clear:
- Missing `class:` → `ValueError("python class spec must be ...")`
- Bad format → same message
- Import error → `python node 'X': cannot import module 'Y': ...`
- Wrong class → `python node 'X': ... has no class 'Y'`
- Not an Entity subclass → `python node 'X': class 'Y' is not an Entity subclass`

Examples: `graphs/05_python_word_counter.yaml`, `graphs/05b_python_prefix.yaml`.
Test fixtures live in `tests/_user_entities.py` (`WordCounter`,
`PrefixAdder`, `Doubler`).

### Deterministic replay

`pharos replay <run_id>` extracts each entity's emitted text from
the recorded trace (no network, no re-execution) and prints a
per-step table.

```
$ uv run pharos run graphs/03_faux_demo.yaml --input "hello world"
Run: hello-faux  director=fn  converged=True  iterations=3  tokens=3

$ uv run pharos replay <run_id>
            Per-entity outputs (recorded)
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ node  ┃ step_id      ┃ duration ┃ output_text       ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ 1 │ agent │ 139b26f0a3:1 │  10.4 ms │ echo: hello world │
└───┴───────┴──────────────┴──────────┴───────────────────┘
```

Implementation:
- `LLMAgent.fire()` now forwards every `StreamEvent` it consumes
  to the active `Span.events` (using `_current_span_fn()` and the
  `SpanEvent` class). This makes replay possible without changing
  the LLMProvider Protocol.
- `pharos.runtime.replay_run_summary(run_id)` walks the recorded
  spans, picks `entity.fire.*` ones, and extracts the last text
  emitted (either accumulated `text_delta` events or the `done`
  event's `message.content` text).

---

## Verification

```bash
$ uv run pharos run graphs/05b_python_prefix.yaml --input "world"
Run: prefix-adder-demo  director=fn  converged=True  iterations=3
       Output ports
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ node    ┃ port   ┃ value     ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ __out__ │ result │ >>> world │
└─────────┴────────┴───────────┘
```

```bash
$ uv run pharos --help
  ...
╭─ Commands ──────────────────────────────────────────╮
│ run / validate / list-providers / doctor           │
│ trace / replay                                     │
╰────────────────────────────────────────────────────╯
```

---

## Key engineering decisions

### 1. Lazy span import in `LLMAgent`
Putting `from pharos.observability.trace import ...` at the top
of `pharos.entities.llm` would create a soft dependency. We
import inside the function so LLMAgent can be unit-tested
without spinning up observability.

### 2. Event capture is opt-in
`Span` is only populated when a tracer is attached to the
`RunContext`. Without a tracer, `_current_span_fn()` returns
None and the forwarder is a no-op. This keeps the cost at
zero for runs that don't trace.

### 3. `replay_run_summary` walks the recorded JSON
We didn't add a "playback" provider; instead we **read the
recorded events back out**. Pros: no cache plumbing, no replay
driver. Cons: this is **inspection replay**, not byte-equal
re-execution. A future `ReplayProvider` would let us re-run a
graph without network calls — but that's P4+.

### 4. Class path uses `:` to avoid `.` ambiguity
`module.path:ClassName` reads cleanly: `pkg.mod:Cls` is
unambiguous (one `:` separates module from class).

### 5. Non-Entity classes are rejected at IR time
If someone tries `type: python, class: "my_module:MyHelper"`,
the loader fails fast with a clear message. This catches a
common mistake (forgetting the `@entity` decorator) at config
time, not at run time.

---

## Known limitations

- **Python entities don't emit to trace events** — only
  `LLMAgent` does. `replay` shows `(no text)` for pure Python
  entities. A future change: have `Entity` itself record
  `fire()` start/end events.
- **No replayed graph execution** — replay is inspection only.
  Re-executing requires either (a) a `ReplayProvider` that
  substitutes recorded outputs, or (b) a recording driver
  that calls `_safe_fire` with mocked providers.
- **Custom Python entities from YAML need the module on
  `PYTHONPATH`** — we don't yet auto-add the project root.

---

## File map (P3 additions)

```
pharos/
├── pharos/
│   ├── ir/__init__.py           (added PythonNodeSpec + _build_python_entity)
│   ├── runtime/__init__.py      (added replay_run_summary, _extract_entity_output_text)
│   ├── entities/llm.py          (added span event forwarding)
│   └── cli.py                   (added `replay` subcommand)
├── graphs/
│   ├── 05_python_word_counter.yaml  (NEW)
│   └── 05b_python_prefix.yaml       (NEW)
└── tests/
    ├── _user_entities.py        (NEW: WordCounter, PrefixAdder, Doubler)
    ├── ir/test_python.py        (NEW: 11 tests)
    └── runtime/test_replay_summary.py (NEW: 6 tests)
```

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 204 tests in ~0.5s
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # 100 actors, P95 ~15ms
uv run pharos run graphs/05b_python_prefix.yaml --input "world"
uv run pharos trace list
uv run pharos replay <run_id>       # P3
```