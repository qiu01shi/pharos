# pharos P6 — Complete

**Status:** P6 done. **231 tests pass (2 integration skipped), ruff clean, TUI viewer + integration test scaffolding.**

---

## What was added on top of P5

### TUI viewer (`pharos trace <id> --interactive`)

A new optional `--interactive` flag on `pharos trace` opens a Rich
panel view of a recorded run:

```bash
$ uv run pharos trace --interactive f7de0448-...
╭───────────────────────────────────────────────────╮
│ Run: f7de0448-...  spans=1  entities=1  total=10.3 ms  │
╰───────────────────────────────────────────────────╯

entity.fire.agent  10.3 ms  entity=agent  a5df6d8473:1
```

Features:
- Run summary panel (spans, entities, total duration)
- Per-span line with color-coded duration:
  - green: < 5 ms (fast)
  - yellow: 5-50 ms (medium)
  - red: > 50 ms (slow)
- Step ID and entity name shown inline
- Tree structure with proper indentation (`├─` / `└─`)
- Falls back gracefully in non-TTY environments

Implementation: `pharos/observability/tui.py` provides:
- `build_span_tree(spans)` — flat list → tree
- `render_tree_compact(roots)` — single-line-per-span tree
- `render_span_detail(span)` — Rich Panel for one span
- `render_summary(run_id, spans, tree)` — top-of-screen summary
- `interactive_view(run_id, spans)` — entry point used by CLI

### Integration test scaffolding

Real LLM provider tests under `tests/llm/test_integration.py`,
gated by the `integration` pytest marker:

```python
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIIntegration:
    async def test_basic_call_gpt_4o_mini(self):
        ...
```

To run:
```bash
OPENAI_API_KEY=sk-... uv run pytest -m integration
GLM_API_KEY=...       uv run pytest -m integration
```

Each test makes ONE real call to keep cost low. If a key is
invalid or the API has changed, tests fail with the real error
— useful for catching upstream breakage.

---

## Verification

```bash
$ uv run pharos run graphs/03_faux_demo.yaml --input "hi"
$ R=$(ls -t ~/.pharos/runs/*.json | head -1 | xargs basename | sed 's/.json//')
$ uv run pharos trace --interactive "$R"
╭────────────────────────────────────────────────────╮
│ Run: f7de0448-...  spans=1  entities=1  total=10.3 ms │
╰────────────────────────────────────────────────────╯
entity.fire.agent  10.3 ms  entity=agent  a5df6d8473:1

$ uv run pytest
SKIPPED [1] tests/llm/test_integration.py:65: OPENAI_API_KEY not set
SKIPPED [1] tests/llm/test_integration.py:37: GLM_API_KEY not set
==================== 231 passed, 2 skipped in 10.68s =====================
```

---

## Key engineering decisions

### 1. TUI is text-first, not a real keyboard-driven UI
A real TUI (Textual) would let users navigate with arrow keys.
We chose a simpler approach: a Rich Panel summary + a sorted
list of spans. This avoids the Textual dependency, works in any
terminal, and degrades gracefully to stdout-only when not a TTY.

If users need full keyboard navigation, P7+ could swap the
backend to Textual without changing the data layer.

### 2. `duration_ms` is computed from `started_at/ended_at`
`Span.duration_ms` is a `@property` on the dataclass — `asdict()`
does NOT preserve properties. The TUI's `build_span_tree` computes
the value from timestamps, which works for both in-memory and
serialized traces.

### 3. Integration tests are skipped by default
We don't want CI to fail when no API keys are configured. The
`integration` marker + `pytest.skipif` lets users opt in:

```bash
# Default: skip integration tests
uv run pytest

# Opt-in: run only integration
uv run pytest -m integration

# Run everything (skips when keys absent)
uv run pytest -m 'integration or not integration'
```

### 4. TUI doesn't replay — for that, use `pharos replay`
The TUI is read-only. To re-execute a run, use
`pharos replay --re-run --graph <path> <run_id>` (P5). Keeping
the TUI read-only means it can be reused for other views (e.g.
a future TUI for analyzing prompt content).

---

## Limitations

- **TUI is one-shot, not interactive** — arrow-key navigation is
  not implemented. A future Textual-based version could add it.
- **No deep links** — you can't jump to a specific span by ID
  from the CLI. A `--span <id>` flag is straightforward to add.
- **Integration tests don't run in CI** — we rely on the
  provider mocks for unit-level coverage. Real API tests are
  opt-in for cost reasons.

---

## File map (P6 additions)

```
pharos/
├── pharos/
│   ├── observability/tui.py       (NEW: TUI viewer, ~200 lines)
│   └── cli.py                     (added `trace --interactive` flag)
└── tests/
    ├── observability/test_tui.py  (NEW: 13 tests)
    └── llm/test_integration.py    (NEW: 2 integration tests, skipped by default)
```

`pyproject.toml` adds the `integration` marker:
```toml
markers = [
    "integration: tests that hit real LLM provider APIs (deselect with '-m \"not integration\"')",
]
```

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 231 tests, 2 skipped (integration)
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # P95 ≈ 10ms (baseline unchanged)

# Record a run, then view it
uv run pharos run graphs/03_faux_demo.yaml --input "hi"
uv run pharos trace list
uv run pharos trace --interactive <run_id>

# Run integration tests (need real keys)
OPENAI_API_KEY=sk-... uv run pytest -m integration
```