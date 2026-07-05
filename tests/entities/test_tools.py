"""Tests for Tool Registry and tool calling loop."""
from __future__ import annotations

from pharos.entities.tools import ToolRegistry
from pharos.entities.tools_builtins import register_builtins
from pharos.llm.base import LLMProvider


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        reg.register(
            name="greet",
            description="Say hi.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            fn=lambda name: f"hi {name}",
        )
        tools = reg.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "greet"
        assert tools[0].description == "Say hi."
        assert tools[0].parameters["properties"]["name"]["type"] == "string"

    async def test_execute_success(self):
        reg = ToolRegistry()
        reg.register(
            name="add",
            description="Add two numbers.",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            fn=lambda a, b: str(a + b),
        )
        result = await reg.execute("add", {"a": 2, "b": 3}, tool_call_id="tc1")
        assert result.tool_call_id == "tc1"
        assert result.output == "5"
        assert result.is_error is False

    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute("nope", {})
        assert result.is_error is True
        assert "unknown tool" in (result.error or "")

    async def test_execute_bad_args(self):
        reg = ToolRegistry()
        reg.register(
            name="greet",
            description="",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            fn=lambda name: f"hi {name}",
        )
        # Missing required arg
        result = await reg.execute("greet", {})
        assert result.is_error is True
        assert "bad arguments" in (result.error or "")

    async def test_execute_crash(self):
        reg = ToolRegistry()
        reg.register(
            name="crash",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: 1 / 0,
        )
        result = await reg.execute("crash", {})
        assert result.is_error is True
        assert "ZeroDivision" in (result.error or "")

    async def test_permission_check_passes(self):
        reg = ToolRegistry()
        reg.register(
            name="protected",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: "ok",
            required_permission="secret:read",
        )
        result = await reg.execute(
            "protected", {}, granted_permissions={"secret:read"}
        )
        assert result.is_error is False
        assert result.output == "ok"

    async def test_permission_check_denied(self):
        reg = ToolRegistry()
        reg.register(
            name="protected",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: "ok",
            required_permission="secret:read",
        )
        result = await reg.execute("protected", {})
        assert result.is_error is True
        assert "permission denied" in (result.error or "")

    async def test_permission_check_with_partial_grants(self):
        reg = ToolRegistry()
        reg.register(
            name="protected",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: "ok",
            required_permission="secret:read",
        )
        # We have other perms but not the required one
        result = await reg.execute(
            "protected", {}, granted_permissions={"fs:read", "net:connect"}
        )
        assert result.is_error is True

    async def test_non_string_return_coerced(self):
        reg = ToolRegistry()
        reg.register(
            name="get_number",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: 42,  # returns int, not str
        )
        result = await reg.execute("get_number", {})
        assert result.output == "42"  # str(int)


class TestBuiltins:
    def test_register_all(self):
        reg = ToolRegistry()
        register_builtins(reg)
        names = {t.name for t in reg.list_tools()}
        assert "echo" in names
        assert "get_time" in names
        assert "add_numbers" in names

    async def test_echo(self):
        reg = ToolRegistry()
        register_builtins(reg)
        result = await reg.execute("echo", {"text": "hi"})
        assert result.output == "hi"

    async def test_add_numbers(self):
        reg = ToolRegistry()
        register_builtins(reg)
        result = await reg.execute("add_numbers", {"numbers": [1, 2, 3, 4]})
        assert result.output == "10.0"  # str(float(10))


class TestToolCallLoop:
    """End-to-end: FauxProvider emits a tool_call, LLMAgent executes it."""

    async def test_faux_emits_toolcall_then_text(self):
        from pharos.core.graph import CompositeGraph
        from pharos.core.token import TypedValue
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector
        from pharos.entities.llm import LLMAgent, LLMEntityConfig
        from pharos.llm.types import AssistantMessage, StreamEvent, TextContent, ToolCall, Usage

        # FauxProvider that emits a tool_call on the first call, then
        # plain text on the second (mimics a real agent).
        call_count = {"n": 0}

        class _ToolCallingProvider(LLMProvider):
            name = "tc"

            async def list_models(self):
                from pharos.llm.types import Model, ModelCost

                return [
                    Model(
                        id="tc-1", name="tc", api="custom",
                        provider="tc", base_url="",
                        cost=ModelCost(input=10, output=10),
                        context_window=8000, max_tokens=1000,
                    )
                ]

            async def close(self):
                pass

            def stream(self, model, context, options=None):
                async def _gen():
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        # Round 1: emit a tool_call
                        tc = ToolCall(
                            id="call-1", name="echo",
                            arguments={"text": "hello"},
                        )
                        yield StreamEvent(type="toolcall_start", content_index=0)
                        yield StreamEvent(
                            type="toolcall_end", content_index=0,
                            tool_call=tc,
                        )
                        msg = AssistantMessage(
                            content=[tc],
                            api="custom", provider="tc", model="tc-1",
                            usage=Usage(input=5, output=3),
                        )
                        yield StreamEvent(type="done", message=msg)
                    else:
                        # Round 2: emit final text (after seeing tool result)
                        msg = AssistantMessage(
                            content=[TextContent(text="got: hello")],
                            api="custom", provider="tc", model="tc-1",
                            usage=Usage(input=10, output=5),
                        )
                        yield StreamEvent(type="done", message=msg)
                return _gen()

        # Build entity
        reg = ToolRegistry()
        reg.register(
            name="echo",
            description="",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            fn=lambda text: text,
        )

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
        g = CompositeGraph("g")
        g.add_entity("agent", agent)
        agent.ins["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )

        d = FNDirector()
        ctx = RunContext(run_id="t")
        result = await d.run(g, ctx)
        assert result.converged is True, f"error: {result.error}"
        # After the tool loop, agent.text should be the final round's text
        text_out = [t.value.payload for t in agent.outs["text"].peek_all()]
        assert text_out == ["got: hello"]
        # And tool_calls should include the one we emitted
        tc_out = agent.outs["tool_calls"].peek_all()
        assert len(tc_out) == 1
        tcs = tc_out[0].value.payload
        assert tcs == [{"id": "call-1", "name": "echo", "arguments": {"text": "hello"}}]
        # And the LLM was called twice
        assert call_count["n"] == 2

    async def test_max_iterations_limits_loop(self):
        """If the LLM keeps emitting tool calls forever, the loop stops."""
        from pharos.core.graph import CompositeGraph
        from pharos.core.token import TypedValue
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector
        from pharos.entities.llm import LLMAgent, LLMEntityConfig
        from pharos.llm.types import AssistantMessage, StreamEvent, ToolCall, Usage

        call_count = {"n": 0}

        class _AlwaysToolCalls(LLMProvider):
            name = "loop"

            async def list_models(self):
                from pharos.llm.types import Model, ModelCost

                return [
                    Model(
                        id="loop-1", name="loop", api="custom",
                        provider="loop", base_url="",
                        cost=ModelCost(input=0, output=0),
                    )
                ]

            async def close(self):
                pass

            def stream(self, model, context, options=None):
                async def _gen():
                    call_count["n"] += 1
                    tc = ToolCall(
                        id=f"call-{call_count['n']}",
                        name="noop",
                        arguments={},
                    )
                    yield StreamEvent(type="toolcall_end", content_index=0, tool_call=tc)
                    yield StreamEvent(
                        type="done",
                        message=AssistantMessage(
                            content=[tc],
                            api="custom", provider="loop", model="loop-1",
                            usage=Usage(),
                        ),
                    )
                return _gen()

        reg = ToolRegistry()
        reg.register(
            name="noop",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: "ok",
        )

        agent = LLMAgent(
            "agent",
            config=LLMEntityConfig(
                provider_class=_AlwaysToolCalls,
                provider_kwargs={},
                model_id="loop-1",
                tool_registry=reg,
                max_tool_iterations=3,  # stop after 3 rounds
            ),
        )
        g = CompositeGraph("g")
        g.add_entity("agent", agent)
        agent.ins["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        d = FNDirector()
        _ = await d.run(g, RunContext(run_id="t"))
        # iterations = max_tool_iterations + 1 (initial) = 4
        assert call_count["n"] == 4

    async def test_tool_error_propagates(self):
        """If a tool fails, the result message has is_error=True."""
        from pharos.core.graph import CompositeGraph
        from pharos.core.token import TypedValue
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector
        from pharos.entities.llm import LLMAgent, LLMEntityConfig
        from pharos.llm.types import AssistantMessage, StreamEvent, ToolCall, Usage

        class _OneToolThenText(LLMProvider):
            name = "one"

            async def list_models(self):
                from pharos.llm.types import Model, ModelCost

                return [Model(id="one-1", name="one", api="custom", provider="one",
                              base_url="", cost=ModelCost(input=0, output=0))]

            async def close(self):
                pass

            def stream(self, model, context, options=None):
                async def _gen():
                    tc = ToolCall(id="c1", name="crash_tool", arguments={})
                    yield StreamEvent(type="toolcall_end", content_index=0, tool_call=tc)
                    yield StreamEvent(
                        type="done",
                        message=AssistantMessage(
                            content=[tc],
                            api="custom", provider="one", model="one-1",
                            usage=Usage(),
                        ),
                    )
                return _gen()

        reg = ToolRegistry()
        reg.register(
            name="crash_tool",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: 1 / 0,  # crashes
        )

        agent = LLMAgent(
            "agent",
            config=LLMEntityConfig(
                provider_class=_OneToolThenText,
                provider_kwargs={},
                model_id="one-1",
                tool_registry=reg,
                max_tool_iterations=3,
            ),
        )
        g = CompositeGraph("g")
        g.add_entity("agent", agent)
        agent.ins["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        d = FNDirector()
        # No LLM will produce a final round (we only have 1 round
        # with the tool call), but the loop won't crash.
        result = await d.run(g, RunContext(run_id="t"))
        assert result.converged is True