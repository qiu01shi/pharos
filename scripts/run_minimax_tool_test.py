"""Run a MiniMax Agent with tool calling end-to-end."""
import asyncio

from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.llm import LLMAgent, LLMEntityConfig
from pharos.entities.tools import ToolRegistry
from pharos.entities.tools_builtins import register_builtins
from pharos.llm.providers.minimax import MiniMaxProvider


async def main():
    reg = ToolRegistry()
    register_builtins(reg)
    # Add a custom tool so we can see something interesting
    reg.register(
        name="reverse_string",
        description="Reverse the input string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=lambda text: text[::-1],
    )

    agent = LLMAgent(
        "agent",
        config=LLMEntityConfig(
            provider_class=MiniMaxProvider,
            provider_kwargs={},
            model_id="MiniMax-Text-01",
            system_prompt=(
                "You have access to tools. When a tool can answer the question, "
                "call it. Reply concisely with the final answer only."
            ),
            max_tokens=500,
            tool_registry=reg,
            max_tool_iterations=3,
        ),
    )
    g = CompositeGraph("minimax-tool-test")
    g.add_entity("agent", agent)
    g.nodes["agent"].instance.ins["prompt"].emit(  # type: ignore[union-attr]
        TypedValue(
            type="text",
            payload="用 reverse_string 工具反转字符串 'hello world',告诉我结果",
        )
    )

    d = FNDirector()
    result = await d.run(g, RunContext(run_id="minimax-tool-test"))
    print("---RESULT---")
    print(f"converged={result.converged}, error={result.error}")
    print()
    print("---agent.text (final answer)---")
    for t in agent.outs["text"].peek_all():
        print(f"  {t.value.payload!r}")
    print()
    print("---agent.tool_calls (LLM requested)---")
    for t in agent.outs["tool_calls"].peek_all():
        print(f"  {t.value.payload!r}")
    print()
    print("---agent.usage---")
    for t in agent.outs["usage"].peek_all():
        print(f"  {t.value.payload}")


if __name__ == "__main__":
    asyncio.run(main())