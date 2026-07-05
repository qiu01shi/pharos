# pharos P8 — Complete (Tool Calling)

**Status:** P8 done. **263 tests pass, ruff clean, LLMAgent supports real tool-calling loops.**

---

## What was added

### 1. `ToolRegistry` + `ToolCallResult`

`pharos/entities/tools.py` — a registry of named, JSON-Schema-described
callables the LLM can request.

```python
from pharos.entities.tools import ToolRegistry

reg = ToolRegistry()
reg.register(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    fn=lambda city: f"sunny in {city}",
    required_permission="weather:read",  # optional
)
```

Tools can declare `required_permission`. Execution fails with
`is_error=True` if the run context doesn't have it.

### 2. Built-in tools (`tools_builtins.py`)

Three defaults:
- `echo(text) -> text`
- `get_time() -> ISO 8601 string`
- `add_numbers(numbers: list[number]) -> string`

```python
from pharos.entities.tools_builtins import register_builtins
register_builtins(reg)
```

### 3. LLMAgent tool-call loop

The biggest change. `LLMAgent.fire()` is now refactored:

```
loop up to max_tool_iterations + 1:
    stream one LLM round → (text, thinking, tool_calls, usage, error)
    if no tool_calls or error → break
    for each tool_call:
        execute via ToolRegistry
        append ToolResultMessage to messages
    loop
```

- Tool results flow back as `ToolResultMessage` so the LLM
  sees them in the next round.
- Each tool's `tool.execute.start` and `tool.execute.end`
  events are recorded onto the active trace span (so replay
  includes tool output).
- `max_tool_iterations` (default 5) prevents infinite loops.

### 4. `LLMEntityConfig` extended

```python
@dataclass
class LLMEntityConfig:
    provider_class: type[LLMProvider]
    provider_kwargs: dict[str, Any]
    model_id: str
    # ... existing fields ...
    tool_registry: ToolRegistry | None = None
    max_tool_iterations: int = 5
```

`tool_registry=None` → agent behaves as before (no tool calling).
`tool_registry=reg` → full tool loop enabled.

---

## Verification

The new `TestToolCallLoop::test_faux_emits_toolcall_then_text` proves
the loop end-to-end:

```python
# Custom provider: round 1 emits tool_call, round 2 emits text
agent = LLMAgent(
    "agent",
    config=LLMEntityConfig(
        provider_class=_ToolCallingProvider,
        provider_kwargs={},
        model_id="tc-1",
        tool_registry=reg,
        max_tool_iterations=3,
    ),
)
await FNDirector().run(g, ctx)
# → agent.text = ["got: hello"]   (round 2's final answer)
# → tool_calls = [{"id": "call-1", "name": "echo", "arguments": {...}}]
# → LLM was called 2 times
```

Other notable tests:
- `test_max_iterations_limits_loop` — agent emitting tool calls
  forever stops at `max_tool_iterations + 1`.
- `test_tool_error_propagates` — a tool that raises doesn't crash
  the run; `is_error=True` flows back to the LLM.
- `test_permission_check_denied` — a tool declaring
  `required_permission` returns `permission denied` if not granted.

---

## Key engineering decisions

### 1. Tool-loop is in LLMAgent, not the provider
Provider returns tool_calls; LLMAgent decides what to do with them.
This keeps providers stateless and makes the loop testable without
network. Different providers (OpenAI, Anthropic, etc.) just need to
emit `toolcall_end` events with a `ToolCall` — no provider changes.

### 2. `_stream_one_round` extracted
Pulled out the "stream one round" logic from the original `fire()`
so the loop is `while True: round = stream_one(); if no tools break`.
Easier to test, easier to reason about.

### 3. ToolRegistry errors are ToolCallResults, not exceptions
A tool that throws `ZeroDivisionError` becomes a `ToolCallResult`
with `is_error=True`. The LLM sees it as a tool message and can
react ("let me try again"). The run continues.

### 4. Permission check at execute time, not register time
A tool's `required_permission` is checked each time it's invoked,
not when registered. This lets the same registry be used in runs
with different permission sets without rebuilding the registry.

### 5. Loop bound via `max_tool_iterations` (default 5)
LLMs occasionally hallucinate into infinite tool-call loops. A
configurable upper bound prevents runaway cost. The P8 default of 5
covers most real-world scenarios (e.g. agent uses 1-3 calls, then
finalizes answer).

---

## Limitations (deferred to P9+)

- **Provider tool-spec sending not wired**: `LLMAgent` declares
  tools to the LLM via `Context.tools`, but most providers
  ignore this field when serializing to wire format. To make
  OpenAI/Anthropic actually call tools, each provider needs its
  own `tools` serialization (OpenAI's `tools` array, Anthropic's
  `tools` array). Currently, this works only with custom /
  test providers.
- **ReplayToolExecutor not implemented**: `ReplayProvider` emits
  the original `toolcall_end` event, but the LLM won't actually
  loop on it during replay — we'd need to capture and replay the
  tool execution separately. This is the next step.
- **No async tools**: `ToolRegistry.execute` is async but every
  built-in tool is sync. Real-world tools (HTTP calls, DB queries)
  would benefit from `async def fn(...)`.
- **No parallel tool calls**: If the LLM emits 3 tool calls in one
  round, we execute them serially. `asyncio.gather` would speed
  this up.

---

## File map (P8 additions)

```
pharos/
├── pharos/entities/
│   ├── tools.py                  NEW: ToolRegistry, ToolCallResult, ToolError
│   ├── tools_builtins.py         NEW: echo / get_time / add_numbers
│   └── llm.py                    MODIFIED: tool-call loop, _stream_one_round
└── tests/entities/
    └── test_tools.py             NEW: 15 tests (registry, builtins, end-to-end loop)
```

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 263 passed, 2 skipped
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # P95 ≈ 11ms (baseline unchanged)
```

To enable tool calling in your own LLM:
```python
from pharos.entities.llm import LLMAgent, LLMEntityConfig
from pharos.entities.tools import ToolRegistry
from pharos.entities.tools_builtins import register_builtins

reg = ToolRegistry()
register_builtins(reg)

agent = LLMAgent(
    "my_agent",
    config=LLMEntityConfig(
        provider_class=YourProvider,
        provider_kwargs={"api_key": "..."},
        model_id="your-model",
        tool_registry=reg,
        max_tool_iterations=5,
    ),
)
```