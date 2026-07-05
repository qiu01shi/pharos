"""Quick test: Multi-round tool calling with MiniMax."""
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


async def main():
    workdir = tempfile.mkdtemp(prefix="pharos-")
    os.chdir(workdir)
    print(f"workdir: {workdir}")

    reg = ToolRegistry()
    register_coding_tools(reg)

    agent = LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt=(
                "You are a file manager agent. "
                "Use the write tool to create files, "
                "then use the read tool to read them back. "
                "Always use tools — never say the content from memory."
            ),
            max_tokens=500,
            tool_registry=reg,
            max_tool_iterations=10,
        ),
    )

    g = CompositeGraph("test")
    g.add_entity("coder", agent)
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(
            type="text",
            payload=(
                "Step 1: Use the write tool to create a file called "
                "'hello.txt' with content 'Hello from MiniMax!'. "
                "Step 2: Use the read tool to read hello.txt. "
                "Reply with the content you read."
            ),
        )
    )

    d = FNDirector()
    ctx = RunContext(
        run_id="minimax-multi-tool",
        granted_permissions={"bash:execute", "fs:read", "fs:write"},
    )
    result = await d.run(g, ctx)

    print(f"\nconverged={result.converged}, error={result.error}")
    print("\n--- tool_calls ---")
    for t in agent.outs["tool_calls"].peek_all():
        for c in t.value.payload:
            print(f"  {c['name']}({c['arguments']})")
    print("\n--- text ---")
    for t in agent.outs["text"].peek_all():
        print(t.value.payload)
    print("\n--- hello.txt ---")
    if os.path.exists("hello.txt"):
        with open("hello.txt") as fh:
            print(fh.read())
    else:
        print("(not created)")

    import shutil
    shutil.rmtree(workdir)


if __name__ == "__main__":
    asyncio.run(main())