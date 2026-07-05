# pharos — Quickstart

> **5 minutes to your first traced LLM workflow.**

pharos is a Python runtime for LLM workflows. It treats an LLM
pipeline as a typed dataflow graph: nodes are LLM calls / shell
commands / custom logic, edges are typed ports, and a Director
drives the firing cycle. Every fire is recorded in a trace so you
can replay, audit, and debug.

---

## 1. Install

```bash
git clone https://github.com/your-org/pharos
cd pharos
uv sync --all-extras
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

---

## 2. Run the demo

```bash
uv run pharos run graphs/03_faux_demo.yaml --input "say hi"
```

You should see:

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

This uses the `FauxProvider` (mock LLM, no API keys needed). For
real models, set `GLM_API_KEY` or `OPENAI_API_KEY` and change the
graph's `provider` field.

---

## 3. Run with a real LLM

```bash
# Volcengine Ark (GLM family)
export ARK_API_KEY=your-key

# OpenAI
export OPENAI_API_KEY=sk-...

# Run a real workflow
uv run pharos run graphs/02_llm_then_shell.yaml \
    --input "list the 3 largest Python files in the current dir"
```

The `02_llm_then_shell.yaml` graph has the LLM produce a shell
command, then runs it.

---

## 4. Other commands

```bash
# Validate a graph without running
uv run pharos validate graphs/02_llm_then_shell.yaml

# List registered LLM providers
uv run pharos list-providers
# faux   glm   openai

# Check your environment
uv run pharos doctor
```

---

## 5. Write your first graph

`hello.yaml`:

```yaml
name: my-first-graph
director: fn
nodes:
  - id: greeter
    type: llm
    provider: glm
    model: glm-4.5-air
    system: "Reply in 3 words max."
    max_tokens: 50
edges:
  - { src: __in__.prompt, dst: greeter.prompt }
  - { src: greeter.text,  dst: __out__.response }
```

```bash
uv run pharos run hello.yaml --input "hi there"
```

---

## 6. Write your first custom Entity

```python
# my_entities.py
from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue

@entity
class WordCounter(Entity):
    ins  = {"text": InputPort("text", accepted_types=["text"])}
    outs = {"count": OutputPort("count", accepted_types=["int"])}

    async def fire(self, ctx):
        for t in self.ins["text"].consume():
            n = len(t.value.payload.split())
            self.outs["count"].emit(TypedValue("int", n))
```

Use it in a graph:

```yaml
nodes:
  - id: counter
    type: python:my_entities:WordCounter
edges:
  - { src: __in__.text,   dst: counter.text }
  - { src: counter.count, dst: __out__.result }
```

*(Note: `python:module:Class` resolution for custom entities is
planned for P2. For now, build your graph programmatically.)*

---

## What's next?

- **Read [docs/architecture.md](docs/architecture.md)** for the full
  design rationale.
- **Read [docs/cookbook.md](docs/cookbook.md)** for more workflows.
- **Run the benchmark**: `uv run python bench/hello_world.py`
  (100 concurrent actors, P95 ≈ 14ms).
- **Read the existing P0 retrospective** in
  `docs/P0-COMPLETE.md`.
