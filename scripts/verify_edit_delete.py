"""Focused test: edit and delete via MiniMax — using explicit 2-step prompt."""
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


async def test_edit_directly():
    """Test edit by directly calling the registry (no LLM)."""
    from pharos.entities.tools import ToolRegistry

    reg = ToolRegistry()
    register_coding_tools(reg)

    workdir = tempfile.mkdtemp(prefix="edit-test-")
    f = os.path.join(workdir, "code.py")
    with open(f, "w") as fh:
        fh.write("def hello():\n    print('world')\n")

    # Call edit directly
    result = await reg.execute(
        "edit",
        {
            "path": f,
            "old_string": "print('world')",
            "new_string": "print('hello world')",
        },
        granted_permissions={"fs:write"},
    )
    print(f"edit result: {result.output}")
    print(f"file content after edit: {open(f).read()}")


async def test_edit_via_minimax():
    """Test edit via real MiniMax API."""
    workdir = tempfile.mkdtemp(prefix="pharos-edit-")
    os.chdir(workdir)
    target = os.path.join(workdir, "code.txt")
    with open(target, "w") as f:
        f.write("the quick brown fox\n")

    reg = ToolRegistry()
    register_coding_tools(reg)
    agent = LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt="Use tools only. Reply DONE when task complete.",
            max_tokens=400,
            tool_registry=reg,
            max_tool_iterations=10,
        ),
    )
    g = CompositeGraph("t")
    g.add_entity("coder", agent)
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(
            type="text",
            payload=(
                f"Use the edit tool to replace 'brown fox' with 'RED FOX' "
                f"in {target}. After editing, reply DONE."
            ),
        )
    )
    tracer = InMemoryTracer()
    ctx = RunContext(
        run_id="edit-minimax", granted_permissions={"fs:read", "fs:write"}
    )
    ctx.tracer = tracer
    result = await FNDirector().run(g, ctx)

    print(f"converged: {result.converged}")
    print("calls:")
    for t in agent.outs["tool_calls"].peek_all():
        for c in t.value.payload:
            print(f"  {c['name']}({c['arguments']})")
    print("tool results:")
    for span in tracer.spans:
        for ev in span.events:
            if ev.name == "tool.execute.end":
                print(f"  {ev.attributes['tool_name']}: {ev.attributes['output'][:100]}")
    print(f"file content: {open(target).read()}")


async def test_delete_via_minimax():
    """Test delete via real MiniMax API."""
    workdir = tempfile.mkdtemp(prefix="pharos-del-")
    os.chdir(workdir)
    target = os.path.join(workdir, "doomed.txt")
    with open(target, "w") as f:
        f.write("bye\n")

    reg = ToolRegistry()
    register_coding_tools(reg)
    agent = LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt="Use tools only. Reply DELETED when task complete.",
            max_tokens=400,
            tool_registry=reg,
            max_tool_iterations=10,
        ),
    )
    g = CompositeGraph("t")
    g.add_entity("coder", agent)
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(
            type="text",
            payload=(
                f"Use the delete tool to remove {target}. "
                f"After deleting, reply DELETED."
            ),
        )
    )
    ctx = RunContext(
        run_id="del-minimax", granted_permissions={"fs:write"}
    )
    result = await FNDirector().run(g, ctx)

    print(f"converged: {result.converged}")
    print("calls:")
    for t in agent.outs["tool_calls"].peek_all():
        for c in t.value.payload:
            print(f"  {c['name']}({c['arguments']})")
    print(f"file exists: {os.path.exists(target)}")


async def main():
    print("=" * 50)
    print("Direct registry test (no LLM)")
    print("=" * 50)
    await test_edit_directly()

    print("\n" + "=" * 50)
    print("Edit via MiniMax")
    print("=" * 50)
    await test_edit_via_minimax()

    print("\n" + "=" * 50)
    print("Delete via MiniMax")
    print("=" * 50)
    await test_delete_via_minimax()


if __name__ == "__main__":
    asyncio.run(main())