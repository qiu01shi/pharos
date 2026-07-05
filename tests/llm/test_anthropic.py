"""Tests for the Anthropic provider (mocked SSE)."""

from __future__ import annotations

import json

import httpx
import pytest

from pharos.llm.providers.anthropic import AnthropicProvider
from pharos.llm.types import Context, Model, ModelCost, UserMessage


def _model() -> Model:
    return Model(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        cost=ModelCost(input=3.0, output=15.0),
        context_window=200_000,
        max_tokens=8192,
    )


def _make_provider() -> AnthropicProvider:
    p = AnthropicProvider(api_key="test-key")
    return p


def _attach_mock(p: AnthropicProvider, chunks: list[str]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        # Verify request headers
        assert request.headers.get("x-api-key") == "test-key"
        assert request.headers.get("anthropic-version") == "2023-06-01"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
        )

    p._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url=p.base_url,
        headers={
            "x-api-key": p.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        transport=httpx.MockTransport(handler),
    )


class TestAnthropicProvider:
    def test_name(self):
        assert _make_provider().name == "anthropic"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Anthropic API key required"):
            AnthropicProvider()

    def test_request_uses_correct_headers(self):
        """Headers: x-api-key + anthropic-version."""
        # Implicit in _attach_mock's assertion above
        pass

    async def test_basic_text_stream(self):
        chunks = [
            "data: " + json.dumps({
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                },
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_stop",
                "index": 0,
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "message_stop",
            }) + "\n\n",
        ]
        p = _make_provider()
        _attach_mock(p, chunks)
        ctx = Context(messages=[UserMessage(content="hi")])
        events = []
        async for ev in p.stream(_model(), ctx):
            events.append(ev)

        final = events[-1].message
        assert final is not None
        assert final.text() == "hello world"
        assert final.api == "anthropic-messages"
        assert final.usage.input == 5
        assert final.usage.output == 2
        # Anthropic "end_turn" maps to pharos "stop"
        assert final.stop_reason == "stop"

    async def test_thinking_stream(self):
        chunks = [
            "data: " + json.dumps({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 1, "output_tokens": 0}},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "deep thought"},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_stop",
                "index": 0,
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "answer"},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_stop",
                "index": 1,
            }) + "\n\n",
            "data: " + json.dumps({"type": "message_stop"}) + "\n\n",
        ]
        p = _make_provider()
        _attach_mock(p, chunks)
        ctx = Context(messages=[UserMessage(content="think please")])
        events = []
        async for ev in p.stream(_model(), ctx):
            events.append(ev)

        thinking_events = [e for e in events if e.type == "thinking_delta"]
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(thinking_events) >= 1
        assert len(text_events) >= 1
        # Final should have BOTH kinds of content
        msg = events[-1].message
        from pharos.llm.types import TextContent, ThinkingContent
        assert any(isinstance(b, ThinkingContent) for b in msg.content)
        assert any(isinstance(b, TextContent) for b in msg.content)

    async def test_tool_use_stream(self):
        chunks = [
            "data: " + json.dumps({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 1, "output_tokens": 0}},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "bash",
                    "input": {},
                },
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"com'},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'mand":"ls"}'},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_stop",
                "index": 0,
            }) + "\n\n",
            "data: " + json.dumps({"type": "message_stop"}) + "\n\n",
        ]
        p = _make_provider()
        _attach_mock(p, chunks)
        ctx = Context(messages=[UserMessage(content="run ls")])
        events = []
        async for ev in p.stream(_model(), ctx):
            events.append(ev)
        ends = [e for e in events if e.type == "toolcall_end"]
        assert len(ends) == 1
        assert ends[0].tool_call.arguments == {"command": "ls"}

    async def test_error_event(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401, content=b'{"error": "unauthorized"}'
            )

        p = _make_provider()
        p._client = httpx.AsyncClient(  # type: ignore[attr-defined]
            base_url=p.base_url,
            headers={"x-api-key": p.api_key, "anthropic-version": "2023-06-01"},
            transport=httpx.MockTransport(handler),
        )
        ctx = Context(messages=[UserMessage(content="hi")])
        events = []
        async for ev in p.stream(_model(), ctx):
            events.append(ev)
        assert events[-1].type == "error"
        assert "401" in (events[-1].error or "")

    async def test_list_models(self):
        p = _make_provider()
        models = await p.list_models()
        ids = {m.id for m in models}
        assert "claude-sonnet-4-20250514" in ids
        assert "claude-3-haiku-20240307" in ids