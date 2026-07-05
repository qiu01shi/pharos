"""MiniMax tends to skip tools on verbose multi-step prompts.
Try a tighter, more direct prompt."""
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
    reg = ToolRegistry()
    register_coding_tools(reg)
    agent = LLMAgent(
        "coder",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt="Use tools. No prose.",
            max_tokens=400,
            tool_registry=reg,
            max_tool_iterations=10,
        ),
    )
    g = CompositeGraph("t")
    g.add_entity("coder", agent)
    # Direct, single-tool task
    g.nodes["coder"].instance.ins["prompt"].emit(
        TypedValue(
            type="text",
            payload='Use the bash tool to run: echo "output_is_42"',
        )
    )
    ctx = RunContext(
        run_id="x", granted_permissions={"bash:execute"}
    )
    result = await FNDirector().run(g, ctx)
    print("converged:", result.converged)
    for t in agent.outs["tool_calls"].peek_all():
        for c in t.value.payload:
            print(f"  {c['name']}({c['arguments']})")
    text = agent.outs["text"].peek_all()
    if text:
        print("text:", text[-1].value.payload[:300])


if __name__ == "__main__":
    asyncio.run(main())