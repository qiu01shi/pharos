"""Tests for the OpenAI provider.

These tests use httpx.MockTransport to simulate OpenAI's API without
needing a real API key. The real provider is exercised manually via
integration tests (test_openai_live.py) that skip without OPENAI_API_KEY.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pharos.llm.providers.openai import OpenAIProvider
from pharos.llm.types import (
    Context,
    Model,
    ModelCost,
    UserMessage,
)


def _gpt4o() -> Model:
    return Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(input=2.5, output=10.0),
    )


def _gpt35() -> Model:
    return Model(
        id="gpt-3.5-turbo",
        name="GPT-3.5",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(input=0.5, output=1.5),
    )


def _ctx(prompt: str = "hi") -> Context:
    return Context(messages=[UserMessage(content=prompt)])


def _make_provider(handler) -> OpenAIProvider:
    """Create an OpenAIProvider whose HTTP client is replaced with a mock."""
    p = OpenAIProvider(api_key="test-key")
    transport = httpx.MockTransport(handler)
    p._client = httpx.AsyncClient(
        base_url=p.base_url,
        headers={"Authorization": f"Bearer {p.api_key}"},
        transport=transport,
        timeout=httpx.Timeout(30.0),
    )
    return p


# ============== Responses API ==============


class TestResponsesAPI:
    async def test_basic_text(self):
        sse_chunks = [
            "data: " + json.dumps(
                {"type": "response.output_text.delta", "delta": "Hello"}
            ),
            "\n\n",
            "data: " + json.dumps(
                {"type": "response.output_text.delta", "delta": " world"}
            ),
            "\n\n",
            "data: " + json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 5,
                            "output_tokens": 2,
                        }
                    },
                }
            ),
            "\n\n",
        ]
        body_holder: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body_holder.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join(sse_chunks).encode(),
            )

        p = _make_provider(handler)
        model = _gpt4o()
        events = []
        async for ev in p.stream(model, _ctx("hi")):
            events.append(ev)

        # Last event should be done
        assert events[-1].type == "done"
        final = events[-1].message
        assert final is not None
        assert final.text() == "Hello world"
        assert final.usage.input == 5
        assert final.usage.output == 2
        assert final.api == "openai-responses"

        # Body should have correct shape
        body = body_holder[0]
        assert body["model"] == "gpt-4o"
        assert body["stream"] is True
        assert body["input"][0]["role"] == "user"
        assert body["input"][0]["content"] == "hi"

    async def test_error_event(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429, content=b'{"error": "rate limited"}'
            )

        p = _make_provider(handler)
        model = _gpt4o()
        events = []
        async for ev in p.stream(model, _ctx()):
            events.append(ev)
        # Must end with an error event
        assert events[-1].type == "error"
        assert "429" in (events[-1].error or "")

    async def test_tool_call(self):
        sse_chunks = [
            "data: " + json.dumps(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "bash",
                    },
                }
            ),
            "\n\n",
            "data: " + json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_1",
                    "delta": '{"com',
                }
            ),
            "\n\n",
            "data: " + json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_1",
                    "delta": 'mand":"ls"}',
                }
            ),
            "\n\n",
            "data: " + json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_1",
                }
            ),
            "\n\n",
            "data: " + json.dumps(
                {"type": "response.completed", "response": {"usage": {}}}
            ),
            "\n\n",
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join(sse_chunks).encode(),
            )

        p = _make_provider(handler)
        model = _gpt4o()
        events = []
        async for ev in p.stream(model, _ctx()):
            events.append(ev)

        # Find the toolcall_end event
        ends = [e for e in events if e.type == "toolcall_end"]
        assert len(ends) == 1
        tc = ends[0].tool_call
        assert tc is not None
        assert tc.name == "bash"
        assert tc.arguments == {"command": "ls"}


# ============== Chat Completions API ==============


class TestChatCompletionsAPI:
    async def test_basic_text(self):
        sse_chunks = [
            "data: " + json.dumps(
                {
                    "choices": [
                        {"delta": {"content": "hi "}, "index": 0}
                    ]
                }
            ),
            "\n\n",
            "data: " + json.dumps(
                {
                    "choices": [
                        {"delta": {"content": "there"}, "index": 0}
                    ]
                }
            ),
            "\n\n",
            "data: " + json.dumps(
                {"usage": {"prompt_tokens": 7, "completion_tokens": 2}}
            ),
            "\n\n",
            "data: [DONE]\n\n",
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "gpt-3.5-turbo"
            assert body["stream"] is True
            assert body["messages"][0]["role"] == "user"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join(sse_chunks).encode(),
            )

        p = _make_provider(handler)
        model = _gpt35()
        events = []
        async for ev in p.stream(model, _ctx("ping")):
            events.append(ev)
        final = events[-1].message
        assert final is not None
        assert final.text() == "hi there"
        assert final.usage.input == 7
        assert final.usage.output == 2
        assert final.api == "openai-completions"

    async def test_message_conversion(self):
        """Verify system_prompt becomes the first message."""
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b'data: [DONE]\n\n',
            )

        p = _make_provider(handler)
        ctx = Context(
            system_prompt="be brief",
            messages=[UserMessage(content="hi")],
        )
        async for _ in p.stream(_gpt35(), ctx):
            pass
        body = captured[0]
        assert body["messages"][0] == {"role": "system", "content": "be brief"}
        assert body["messages"][1]["role"] == "user"


# ============== list_models ==============


class TestListModels:
    async def test_returns_models(self):
        p = OpenAIProvider(api_key="k")
        models = await p.list_models()
        assert len(models) >= 5
        ids = {m.id for m in models}
        assert "gpt-4o" in ids
        assert "o3" in ids

    def test_missing_key_raises(self):
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                OpenAIProvider()
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


# ============== close ==============


class TestClose:
    async def test_close_releases_client(self):
        p = OpenAIProvider(api_key="k")
        # Manually attach a client so close() has something to release
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        p._client = httpx.AsyncClient(  # type: ignore[attr-defined]
            base_url=p.base_url,
            headers={"Authorization": f"Bearer {p.api_key}"},
            transport=httpx.MockTransport(handler),
        )
        assert p._client is not None  # type: ignore[attr-defined]
        await p.close()
        assert p._client is None  # type: ignore[attr-defined]
