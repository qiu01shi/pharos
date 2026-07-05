"""Tests for DeepSeek provider.

DeepSeek emits `reasoning_content` alongside `content` for thinking
models. The provider must extract this as `thinking_delta` events.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pharos.llm.providers.deepseek import DeepSeekProvider
from pharos.llm.providers.openai import OpenAIProvider
from pharos.llm.types import Context, UserMessage


def _make_provider() -> DeepSeekProvider:
    p = DeepSeekProvider(api_key="test-key")
    return p


def _attach_mock_client(p: DeepSeekProvider, body_chunks: list[str]) -> None:
    """Replace p._client with one that returns the given SSE chunks."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(body_chunks).encode(),
        )

    p._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url=p.base_url,
        headers={"Authorization": f"Bearer {p.api_key}"},
        transport=httpx.MockTransport(handler),
    )


class TestDeepSeekProvider:
    def test_inherits_openai(self):
        assert issubclass(DeepSeekProvider, OpenAIProvider)

    def test_name(self):
        assert _make_provider().name == "deepseek"

    def test_base_url(self):
        assert "deepseek.com" in _make_provider().base_url

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DeepSeek API key required"):
            DeepSeekProvider()

    async def test_stream_extracts_reasoning_content(self):
        """DeepSeek streams `reasoning_content` for thinking models."""
        chunks = [
            "data: " + json.dumps({
                "choices": [{"delta": {"reasoning_content": "think A "}, "index": 0}]
            }) + "\n\n",
            "data: " + json.dumps({
                "choices": [{"delta": {"content": "answer A"}, "index": 0}]
            }) + "\n\n",
            "data: " + json.dumps({
                "choices": [{"delta": {"reasoning_content": "think B"}, "index": 0}]
            }) + "\n\n",
            "data: " + json.dumps({
                "choices": [{"delta": {"content": " answer B"}, "index": 0}]
            }) + "\n\n",
            "data: " + json.dumps({
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 22,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                }
            }) + "\n\n",
            "data: [DONE]\n\n",
        ]
        p = _make_provider()
        _attach_mock_client(p, chunks)
        model = (await p.list_models())[1]  # deepseek-reasoner
        ctx = Context(messages=[UserMessage(content="hi")])
        events = []
        async for ev in p.stream(model, ctx):
            events.append(ev)

        # Verify we got both thinking_delta and text_delta
        thinking_events = [e for e in events if e.type == "thinking_delta"]
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(thinking_events) >= 1
        assert len(text_events) >= 1
        # The final message should have BOTH thinking and text content
        done_event = [e for e in events if e.type == "done"][-1]
        msg = done_event.message
        kinds = [type(b).__name__ for b in msg.content]
        assert "ThinkingContent" in kinds
        assert "TextContent" in kinds

        # Usage should reflect reasoning_tokens
        assert msg.usage.reasoning == 5

    async def test_list_models(self):
        p = _make_provider()
        models = await p.list_models()
        assert len(models) >= 2
        ids = {m.id for m in models}
        assert "deepseek-chat" in ids
        assert "deepseek-reasoner" in ids