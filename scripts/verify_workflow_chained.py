"""Multi-step coding workflow via explicit agent chaining (workaround for
MiniMax tendency to skip tools on long multi-step prompts).

Workflow:
  1. Agent A: write code (write tool)
  2. Agent B: run code (bash tool), show output
"""
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


def make_agent():
    reg = ToolRegistry()
    register_coding_tools(reg)
    return LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt="Use tools. No prose.",
            max_tokens=400,
            tool_registry=reg,
            max_tool_iterations=5,
        ),
    )


async def run_step(agent, task, perms):
    g = CompositeGraph("step")
    g.add_entity("coder", agent)
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(type="text", payload=task)
    )
    ctx = RunContext(run_id="step", granted_permissions=perms)
    _ = await FNDirector().run(g, ctx)
    text = agent.outs["text"].peek_all()
    return text[-1].value.payload if text else ""


async def main():
    workdir = tempfile.mkdtemp(prefix="pharos-workflow-")
    os.chdir(workdir)
    print(f"Working in: {workdir}")

    # Step 1: write code
    print("\n=== Step 1: write greet.py ===")
    writer = make_agent()
    step1 = await run_step(
        writer,
        'Use the write tool to create greet.py with content: '
        '\'name = "World"\\nprint(f"Hello, {name}!")\'. Reply DONE.',
        {"fs:write"},
    )
    print(f"Step 1 result: {step1[:200]}")
    target = os.path.join(workdir, "greet.py")
    if os.path.exists(target):
        print(f"  ✓ File created: {open(target).read()!r}")
    else:
        print("  ✗ File NOT created")

    # Step 2: run it
    print("\n=== Step 2: bash python3 greet.py ===")
    runner = make_agent()
    step2 = await run_step(
        runner, "Use the bash tool to run: python3 greet.py", {"bash:execute"}
    )
    print(f"Step 2 result: {step2[:200]}")


if __name__ == "__main__":
    asyncio.run(main())