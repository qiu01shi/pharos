"""End-to-end verify each coding tool with real MiniMax API."""
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


def make_agent():
    reg = ToolRegistry()
    register_coding_tools(reg)
    return LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt="You are a coding agent. ALWAYS use tools. Reply briefly.",
            max_tokens=600,
            tool_registry=reg,
            max_tool_iterations=10,
        ),
    )


async def run_task(perms, task, setup=None):
    workdir = tempfile.mkdtemp(prefix="pharos-test-")
    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if setup:
            setup(workdir)
        agent = make_agent()
        g = CompositeGraph("t")
        g.add_entity("coder", agent)
        g.nodes["coder"].instance.ins["prompt"].emit(
            TypedValue(type="text", payload=task)
        )
        tracer = InMemoryTracer()
        ctx = RunContext(
            run_id=f"test-{id(workdir)}", granted_permissions=perms
        )
        ctx.tracer = tracer
        result = await FNDirector().run(g, ctx)
        calls, tool_results = [], []
        for t in agent.outs["tool_calls"].peek_all():
            for c in t.value.payload:
                calls.append(c)
        for span in tracer.spans:
            for ev in span.events:
                if ev.name == "tool.execute.end":
                    tool_results.append(ev.attributes)
        return {
            "workdir": workdir,
            "converged": result.converged,
            "calls": calls,
            "results": tool_results,
        }
    finally:
        os.chdir(old_cwd)


async def main():
    print("=" * 60)
    print("TEST 1: bash — list /tmp")
    print("=" * 60)
    r = await run_task(
        {"bash:execute"}, "Use bash tool to run: ls -la /tmp | head -3"
    )
    print(f"calls: {r['calls']}")
    print(f"results: {r['results']}")
    print(f"converged: {r['converged']}")

    print("\n" + "=" * 60)
    print("TEST 2: read — read /etc/hosts")
    print("=" * 60)
    r = await run_task(
        set(),
        "Use read tool to read /etc/hosts. Reply with what you found.",
    )
    print(f"calls: {r['calls']}")
    print(f"results: {r['results']}")

    print("\n" + "=" * 60)
    print("TEST 3: write — create a file")
    print("=" * 60)
    r = await run_task(
        {"fs:write"},
        "Use write tool to create test.txt with content 'WRITE_OK_123'. "
        "Reply DONE when done.",
    )
    print(f"calls: {r['calls']}")
    print(f"results: {r['results']}")
    target = os.path.join(r["workdir"], "test.txt")
    if os.path.exists(target):
        with open(target) as f:
            print(f"VERIFIED on disk: '{f.read()}'")
    else:
        print("VERIFIED on disk: FILE NOT CREATED")

    print("\n" + "=" * 60)
    print("TEST 4: edit — fix a typo")
    print("=" * 60)
    r = await run_task(
        {"fs:read", "fs:write"},
        "Step 1: Use write tool to create note.txt with content 'I have a tpyo.'\n"
        "Step 2: Use edit tool to replace 'tpyo' with 'typo' in note.txt\n"
        "Step 3: Use read tool to read note.txt and show the content\n"
        "Reply with the final content of note.txt.",
    )
    print(f"calls: {r['calls']}")
    target = os.path.join(r["workdir"], "note.txt")
    if os.path.exists(target):
        with open(target) as f:
            print(f"VERIFIED on disk: '{f.read()}'")
    else:
        print("VERIFIED on disk: FILE NOT CREATED")

    print("\n" + "=" * 60)
    print("TEST 5: glob — list *.txt files")
    print("=" * 60)

    def setup_glob(d):
        for i in range(3):
            open(os.path.join(d, f"file{i}.txt"), "w").write(str(i))

    r = await run_task(
        {"fs:read"},
        "Use glob tool with pattern *.txt. Reply with the file list.",
        setup=setup_glob,
    )
    print(f"calls: {r['calls']}")
    print(f"results: {r['results']}")
    print(f"files in workdir: {sorted(os.listdir(r['workdir']))}")

    print("\n" + "=" * 60)
    print("TEST 6: grep — find 'def' in *.py")
    print("=" * 60)

    def setup_grep(d):
        open(
            os.path.join(d, "code.py"), "w"
        ).write("def foo():\n    pass\n\ndef bar():\n    pass\n")

    r = await run_task(
        {"fs:read"},
        "Use grep tool to find pattern 'def' in code.py. Reply with matches.",
        setup=setup_grep,
    )
    print(f"calls: {r['calls']}")
    print(f"results: {r['results']}")

    print("\n" + "=" * 60)
    print("TEST 7: delete — remove a file")
    print("=" * 60)

    def setup_delete(d):
        open(os.path.join(d, "doomed.txt"), "w").write("bye")

    r = await run_task(
        {"fs:write", "fs:read"},
        "Step 1: Use read tool to read doomed.txt\n"
        "Step 2: Use delete tool to remove doomed.txt\n"
        "Step 3: Reply with: deleted or NOT_DELETED",
        setup=setup_delete,
    )
    print(f"calls: {r['calls']}")
    target = os.path.join(r["workdir"], "doomed.txt")
    print(f"VERIFIED on disk: exists={os.path.exists(target)}")


if __name__ == "__main__":
    asyncio.run(main())