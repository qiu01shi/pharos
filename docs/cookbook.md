# pharos — Cookbook

Real workflows you can run today.

> All examples use the **FauxProvider** (mock LLM) so you can run
> them without any API keys. Swap `provider: faux` → `provider:
> glm` (or `openai`) to use a real model.

---

## 1. Hello: a single LLM call

`graphs/03_faux_demo.yaml`:

```yaml
name: hello-faux
director: fn
nodes:
  - id: agent
    type: faux
    provider: faux
    model: faux-fast
    system: "You are brief."
    max_tokens: 100
edges:
  - { src: __in__.prompt, dst: agent.prompt }
  - { src: agent.text, dst: __out__.response }
```

Run it:

```bash
pharos run graphs/03_faux_demo.yaml --input "say hi"
```

Output:

```
Run: hello-faux  converged=True  iterations=3  tokens=3  cost=$0.0000
         Output ports
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ node    ┃ port     ┃ value                ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ agent   │ draft    │ echo: say hi         │
│ agent   │ usage    │ {input: 50, ...}     │
│ __out__ │ response │ echo: say hi         │
└─────────┴──────────┴──────────────────────┘
```

Ports explained:
- `agent.draft` — partial text (streamed live)
- `agent.usage` — token cost breakdown
- `__out__.response` — final aggregated result

---

## 2. LLM → Shell: have the LLM run a command

`graphs/02_llm_then_shell.yaml`:

```yaml
name: llm-then-shell
director: fn
nodes:
  - id: agent
    type: llm
    provider: glm
    model: glm-4.5-air
    system: "Reply with a single shell command. No commentary."
    max_tokens: 100
  - id: runner
    type: shell
    timeout: 30
edges:
  - { src: __in__.task,    dst: agent.prompt }
  - { src: agent.text,     dst: runner.command }
  - { src: runner.stdout,  dst: __out__.output }
  - { src: runner.exit_code, dst: __out__.exit_code }
```

The LLM produces a shell command (e.g. `ls -la`), the `runner`
entity executes it, and the result flows to `__out__`.

---

## 3. Streaming output (live progress)

Connect downstream entities to `agent.draft` to see the LLM's
partial response before it finishes:

```yaml
- { src: agent.draft, dst: logger.input }
```

(With a custom Logger entity that prints to console.)

---

## 4. Multi-model voting (3 reviewers)

```yaml
name: code-review
director: fn
nodes:
  - id: reviewer_a
    type: llm
    provider: glm
    model: glm-4.5-air
    system: "You focus on security issues."
  - id: reviewer_b
    type: llm
    provider: glm
    model: glm-4.5
    system: "You focus on performance."
  - id: reviewer_c
    type: llm
    provider: openai
    model: gpt-4o-mini
    system: "You focus on readability."
  - id: aggregator
    type: llm
    provider: glm
    model: glm-4.5-air
    system: "Aggregate three review reports into a final list."
edges:
  - { src: __in__.diff,           dst: reviewer_a.prompt }
  - { src: __in__.diff,           dst: reviewer_b.prompt }
  - { src: __in__.diff,           dst: reviewer_c.prompt }
  - { src: reviewer_a.text,       dst: aggregator.prompt }
  - { src: reviewer_b.text,       dst: aggregator.prompt }
  - { src: reviewer_c.text,       dst: aggregator.prompt }
  - { src: aggregator.text,       dst: __out__.review  }
```

All three reviewers run in **parallel** (same topological layer).
Total wall time ≈ max(reviewer_a, reviewer_b, reviewer_c), not the
sum.

---

## 5. Custom Entity

```python
from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue

@entity
class UpperCaser(Entity):
    ins = {"text": InputPort("text", accepted_types=["text"])}
    outs = {"shouted": OutputPort("shouted", accepted_types=["text"])}

    async def fire(self, ctx):
        for t in self.ins["text"].consume():
            self.outs["shouted"].emit(
                TypedValue("text", t.value.payload.upper() + "!")
            )
```

Use it in a graph:

```python
from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.ir import load_graph  # for YAML graphs

g = CompositeGraph("upper")
g.add_entity("shout", UpperCaser("shout"))
# Inject input directly
g.nodes["__in__"].instance.ins["prompt"].emit(  # type: ignore[union-attr]
    TypedValue("text", "hello")
)
# (you'd connect __in__.prompt -> shout.text in a real flow)
```

---

## 6. Tracing every fire

`pharos trace` is not yet a separate command, but every run already
captures spans via `InMemoryTracer`. To see them programmatically:

```python
from pharos.observability.trace import InMemoryTracer
from pharos.observability.backend.console import ConsoleTraceBackend

tracer = InMemoryTracer()
# ... attach to run_ctx.tracer before director.run() ...
backend = ConsoleTraceBackend()
for s in tracer.spans:
    backend.write(s)
print(backend.render())
```

Output:

```
trace_0001_a1b2c3
├─ entity.fire.source  (BroadcastSource)  0.1 ms
├─ entity.fire.leaf_0  (LLMLoop)         11.2 ms
├─ entity.fire.leaf_1  (LLMLoop)         11.2 ms
...
```

---

## 7. End-to-end debugging with token hashes

```python
from pharos.core.token import Token, TypedValue

t1 = Token(value=TypedValue("text", "hello"), origin="a.x")
t2 = Token(value=TypedValue("text", "hello"), origin="a.x")
assert t1.self_hash == t2.self_hash   # deterministic
```

If two runs of the same graph produce different head-token hashes
**with the same LLM temperature = 0**, something non-deterministic
happened. The first place to check: is your LLM config in
`temperature=0`?

---

## Coming soon

- **Reuse with SDF Director** — feedback loops where a `reviewer`
  rejects a `coder`'s output and forces another iteration.
- **Replay** — re-execute a recorded run byte-for-byte (no LLM call).
- **TUI viewer** — browse traces interactively.
