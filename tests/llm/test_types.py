"""Tests for pharos.llm.types."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from pharos.llm.types import (
    AssistantMessage,
    Context,
    CostRecord,
    ImageContent,
    Model,
    ModelCost,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class TestUsage:
    def test_default_zero(self):
        u = Usage()
        assert u.input == 0
        assert u.output == 0
        assert u.total == 0

    def test_total_sums_input_output(self):
        u = Usage(input=10, output=20, cache_read=5, cache_write=3)
        assert u.total == 30  # cache does NOT count toward input/output

    def test_frozen(self):
        u = Usage(input=10)
        with pytest.raises(ValidationError):
            u.input = 20  # type: ignore[misc]

    def test_reasoning_is_optional(self):
        u = Usage(input=10, output=20, reasoning=5)
        assert u.reasoning == 5


class TestCostRecord:
    def test_from_usage_basic(self):
        cost = ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
        usage = Usage(input=1_000_000, output=500_000, cache_read=200_000)
        record = CostRecord.from_usage(usage, cost)
        assert record.input == 3.0
        assert record.output == 7.5
        assert record.cache_read == pytest.approx(0.06)
        assert record.total == pytest.approx(3.0 + 7.5 + 0.06)

    def test_from_usage_zero(self):
        cost = ModelCost(input=3.0, output=15.0)
        record = CostRecord.from_usage(Usage(), cost)
        assert record.total == 0.0


class TestContentBlocks:
    def test_text_content(self):
        tc = TextContent(text="hello")
        assert tc.type == "text"
        assert tc.text == "hello"

    def test_thinking_content(self):
        tc = ThinkingContent(thinking="reasoning here", signature="abc")
        assert tc.type == "thinking"
        assert tc.signature == "abc"

    def test_image_content_default_mime(self):
        ic = ImageContent(data="data:image/png;base64,...")
        assert ic.mime_type == "image/png"

    def test_tool_call(self):
        tc = ToolCall(id="1", name="read", arguments={"path": "/tmp"})
        assert tc.type == "tool_call"
        assert tc.arguments == {"path": "/tmp"}


class TestMessages:
    def test_user_message(self):
        m = UserMessage(content="hi")
        assert m.role == "user"
        assert m.content == "hi"
        assert m.timestamp > 0

    def test_user_message_with_images(self):
        m = UserMessage(content=[TextContent(text="what is this?"), ImageContent(data="x")])
        assert len(m.content) == 2

    def test_assistant_message_text_helper(self):
        m = AssistantMessage(
            content=[TextContent(text="hello "), TextContent(text="world")],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert m.text() == "hello world"
        assert m.stop_reason == "stop"

    def test_assistant_message_empty(self):
        m = AssistantMessage()
        assert m.text() == ""
        assert m.content == []

    def test_tool_result_message(self):
        m = ToolResultMessage(
            tool_call_id="1",
            tool_name="read",
            content=[TextContent(text="file content")],
            is_error=False,
        )
        assert m.role == "tool_result"


class TestTool:
    def test_minimal(self):
        t = Tool(name="read", description="read a file")
        assert t.name == "read"
        assert t.parameters == {}

    def test_with_parameters(self):
        t = Tool(
            name="bash",
            description="run shell",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        )
        assert "properties" in t.parameters


class TestContext:
    def test_empty(self):
        c = Context()
        assert c.system_prompt is None
        assert c.messages == []
        assert c.tools == []

    def test_with_messages(self):
        c = Context(
            system_prompt="be brief",
            messages=[UserMessage(content="hi")],
        )
        assert c.system_prompt == "be brief"
        assert len(c.messages) == 1


class TestStreamEvent:
    def test_start_event(self):
        e = StreamEvent(type="start", partial=AssistantMessage())
        assert e.type == "start"
        assert e.partial is not None

    def test_text_delta(self):
        e = StreamEvent(type="text_delta", delta="hello")
        assert e.delta == "hello"

    def test_done_with_message(self):
        msg = AssistantMessage(content=[TextContent(text="complete")])
        e = StreamEvent(type="done", message=msg)
        assert e.type == "done"
        assert e.message is not None

    def test_error_event(self):
        e = StreamEvent(type="error", error="rate limit")
        assert e.error == "rate limit"

    def test_toolcall_end(self):
        tc = ToolCall(id="1", name="bash", arguments={"command": "ls"})
        e = StreamEvent(type="toolcall_end", tool_call=tc)
        assert e.tool_call is not None


class TestModel:
    def test_minimal(self):
        m = Model(
            id="gpt-4o",
            name="GPT-4o",
            api="openai-responses",
            provider="openai",
            base_url="https://api.openai.com/v1",
            cost=ModelCost(input=2.5, output=10.0),
        )
        assert m.context_window == 200_000  # default
        assert m.reasoning is False

    def test_full(self):
        m = Model(
            id="claude-sonnet-4",
            name="Claude Sonnet 4",
            api="anthropic-messages",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input=["text", "image"],
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            context_window=200_000,
            max_tokens=8192,
            headers={"anthropic-version": "2023-06-01"},
        )
        assert m.reasoning is True
        assert "image" in m.input


class TestStreamOptions:
    def test_defaults(self):
        o = StreamOptions()
        assert o.temperature is None
        assert o.cache_retention == "short"
        assert o.extra == {}

    def test_with_thinking(self):
        o = StreamOptions(thinking_level="high", max_tokens=4096)
        assert o.thinking_level == "high"
        assert o.max_tokens == 4096

    def test_invalid_thinking_level(self):
        with pytest.raises(ValidationError):
            StreamOptions(thinking_level="ultra")  # type: ignore[arg-type]


class TestImmutability:
    """All token-bearing types must be frozen to support safe concurrent sharing."""

    def test_usage_frozen(self):
        u = Usage(input=1)
        with pytest.raises(ValidationError):
            u.input = 2  # type: ignore[misc]

    def test_text_content_frozen(self):
        t = TextContent(text="x")
        with pytest.raises(ValidationError):
            t.text = "y"  # type: ignore[misc]

    def test_tool_call_frozen(self):
        tc = ToolCall(id="1", name="x")
        with pytest.raises(ValidationError):
            tc.name = "y"  # type: ignore[misc]
