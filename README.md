# pharos

> **Typed dataflow runtime for LLM workflows with first-class trace and replay.**

pharos models an LLM pipeline as a typed graph: nodes are LLM calls
/ shell commands / custom logic, edges are typed ports, and a
Director drives the firing cycle. Every fire is wrapped in a
span; every token is hash-chained; replays reproduce a run
byte-for-byte.

Built to answer two real pain points:
- **Multi-step LLM workflows are untraceable.** A bug-fix pipeline
  that runs "analyze → patch → test → review" gives you no way to
  see which step spent the tokens or which one hung.
- **Multi-agent frameworks don't compose.** Each one reinvents
  port types, scheduling, and error recovery.

pharos is the missing primitive: a runtime that knows what your
agents said, in what order, and at what cost.

---

## Why pharos

| What you get | What you don't get |
| --- | --- |
| ✅ Type-checked ports (LLM output schema validated at boundary) | ❌ A drop-in replacement for LangChain |
| ✅ Streaming by default (Supervisor text is consumed live by workers) | ❌ Visual drag-and-drop editor (planned P5) |
| ✅ Per-step trace + cost (find which actor is slow/expensive) | ❌ Multi-language runtime (Python only) |
| ✅ Replay (re-execute a recorded run without LLM calls) | ❌ Distributed execution (planned P9+) |
| ✅ 100 concurrent actors in **14ms** (vs. ~1000ms sequential) | ❌ Auto-fixing of broken graphs |

---

## 5-minute quickstart

```bash
git clone https://github.com/your-org/pharos
cd pharos
uv sync --all-extras

# Run the demo (no API key needed)
uv run pharos run graphs/03_faux_demo.yaml --input "say hi"
```

See [docs/quickstart.md](docs/quickstart.md) for the full tour.

---

## Hello workflow

```yaml
# hello.yaml
name: hello
director: fn
nodes:
  - id: greeter
    type: llm
    provider: glm
    model: glm-4.5-air
    system: "Reply in 3 words max."
edges:
  - { src: __in__.prompt, dst: greeter.prompt }
  - { src: greeter.text,  dst: __out__.response }
```

```bash
pharos run hello.yaml --input "hi there"
# → "Hi there friend."
```

---

## Documentation

- [Quickstart](docs/quickstart.md) — install + first run
- [Architecture](docs/architecture.md) — design rationale, layers, contracts
- [Cookbook](docs/cookbook.md) — real workflows (multi-reviewer, custom entities, streaming, etc.)
- [P0 Retrospective](docs/P0-COMPLETE.md) — what's done, what isn't, and the key engineering decisions

---

## Current status

**P0–P3 done.**

| Phase | What | Status |
| --- | --- | --- |
| **P0** | Core abstractions, FN Director, FauxProvider, trace, perf baseline | ✅ |
| **P1** | OpenAI + GLM providers, YAML IR, CLI, trace persistence | ✅ |
| **P2** | Anthropic + DeepSeek providers, Router + Memory entities, SDF Director (feedback loops) | ✅ |
| **P3** | Custom Python entities from YAML, deterministic replay inspection | ✅ |
| **P4+** | DE / PN / CT directors, real Replay, TUI viewer | pending |

**Numbers**:
- 204 tests, 0 lint errors
- 100 concurrent actors: P95 ≈ 12ms (target 2000ms)
- 5 LLM providers: faux / glm / openai / deepseek / anthropic
- 2 Directors: FN / SDF
- 5 Entities: LLMAgent / ShellEntity / Router / Memory / (custom Python)
- 6 CLI subcommands: run / validate / list-providers / doctor / trace / replay

See [docs/P0-COMPLETE.md](docs/P0-COMPLETE.md), [P1](docs/P1-COMPLETE.md), [P2](docs/P2-COMPLETE.md), [P3](docs/P3-COMPLETE.md).

## Project layout

```
pharos/
├── pharos/
│   ├── core/              Token / Port / Entity / Graph (typed dataflow)
│   ├── llm/               LLMProvider Protocol + OpenAI / GLM / Faux
│   ├── llm/catalog/       Model metadata per provider
│   ├── directors/         FN (DE/PN/SDF/CT in later phases)
│   ├── entities/          LLMAgent, ShellEntity, more later
│   ├── observability/     Trace / Metrics / Logs / Events
│   ├── ir/                YAML graph loader
│   ├── runtime/           (Replay comes in P2)
│   └── cli.py             `pharos` command
├── graphs/                Sample workflows
├── tests/                 151 tests across 9 modules
├── bench/                 Performance baselines
└── docs/                  This documentation
```

---

## License

MIT
