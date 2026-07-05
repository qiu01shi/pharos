"""End-to-end: MiniMax coding agent reads/writes files via tools."""
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
    # Create a temp working directory with a file to edit
    workdir = tempfile.mkdtemp(prefix="pharos-coding-")
    os.chdir(workdir)
    print(f"=== Working directory: {workdir} ===")

    # Pre-create a file with a typo
    with open("README.md", "w") as f:
        f.write("# Project\n\nThis project has a tpyo in it.\n")

    # Build the coding agent
    reg = ToolRegistry()
    register_coding_tools(reg)

    agent = LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt=(
                "You are a coding agent with tools: bash, read, write, edit, "
                "delete, glob, grep. When asked to work with files, USE the "
                "tools. Be concise."
            ),
            max_tokens=1000,
            tool_registry=reg,
            max_tool_iterations=10,
        ),
    )

    g = CompositeGraph("coding-agent-test")
    g.add_entity("coder", agent)
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(
            type="text",
            payload=(
                "Use the read tool to read README.md. "
                "Then use the edit tool to fix the typo 'tpyo' -> 'typo'. "
                "Then use the read tool again to confirm. "
                "Reply with exactly: DONE or the error you got."
            ),
        )
    )

    d = FNDirector()
    ctx = RunContext(
        run_id="minimax-coding-test",
        granted_permissions={"bash:execute", "fs:read", "fs:write"},
    )
    result = await d.run(g, ctx)

    print()
    print("=== RESULT ===")
    print(f"converged={result.converged}, error={result.error}")
    print()
    print("=== agent.text (final answer) ===")
    for t in agent.outs["text"].peek_all():
        print(t.value.payload)
    print()
    print("=== agent.tool_calls ===")
    for t in agent.outs["tool_calls"].peek_all():
        calls = t.value.payload
        for c in calls:
            print(f"  {c['name']}({c['arguments']})")
    print()
    print("=== README.md final content ===")
    with open("README.md") as f:
        print(f.read())

    # Cleanup
    import shutil

    shutil.rmtree(workdir)


if __name__ == "__main__":
    asyncio.run(main())