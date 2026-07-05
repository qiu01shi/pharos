"""Full workflow: write code, run it, capture output."""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, "/Users/heshuwen/pharos")

from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.llm import LLMAgent, LLMEntityConfig
from pharos.entities.tools import ToolRegistry
from pharos.entities.tools_coding import register_coding_tools
from pharos.llm.providers.minimax import MiniMaxProvider
from pharos.observability.trace import InMemoryTracer


async def main():
    workdir = tempfile.mkdtemp(prefix="pharos-pi-coding-")
    os.chdir(workdir)

    reg = ToolRegistry()
    register_coding_tools(reg)
    agent = LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt=(
                "You are a coding agent. Use the write and bash tools. "
                "After running code, reply with the captured output."
            ),
            max_tokens=1500,
            tool_registry=reg,
            max_tool_iterations=20,
        ),
    )

    g = CompositeGraph("workflow")
    g.add_entity("coder", agent)
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(
            type="text",
            payload=(
                "Task: write a Python script called greet.py that takes a "
                "name and prints 'Hello, <name>!'. The script should call "
                "print() with 'Hello, World!' when run as 'python3 greet.py'.\n\n"
                "Steps:\n"
                "1. Use write tool to create greet.py with the code\n"
                "2. Use bash tool to run 'python3 greet.py'\n"
                "3. Reply with what bash printed"
            ),
        )
    )

    tracer = InMemoryTracer()
    ctx = RunContext(
        run_id="pi-coding-test",
        granted_permissions={"bash:execute", "fs:read", "fs:write"},
    )
    ctx.tracer = tracer
    result = await FNDirector().run(g, ctx)

    print(f"converged: {result.converged}")
    print("\n--- Tool calls ---")
    for t in agent.outs["tool_calls"].peek_all():
        for c in t.value.payload:
            print(f"  {c['name']}({c['arguments']})")
    print("\n--- Tool results ---")
    for span in tracer.spans:
        for ev in span.events:
            if ev.name == "tool.execute.end":
                attrs = ev.attributes
                ok = "OK" if not attrs["is_error"] else "ERROR"
                out = attrs["output"][:200] if attrs["output"] else attrs.get("error", "")
                print(f"  [{ok}] {attrs['tool_name']}: {out}")
    print("\n--- Files ---")
    for f in sorted(os.listdir(workdir)):
        if os.path.isfile(os.path.join(workdir, f)):
            print(f"=== {f} ===")
            print(open(os.path.join(workdir, f)).read())


if __name__ == "__main__":
    asyncio.run(main())