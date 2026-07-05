"""Tests for GLM provider.

Uses httpx.MockTransport; no real API key required.
"""

from __future__ import annotations

import httpx
import pytest

from pharos.llm.providers.glm import GLMProvider
from pharos.llm.providers.openai import OpenAIProvider


def _make_provider(api_key: str = "test-key") -> GLMProvider:
    p = GLMProvider(api_key=api_key)
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    b'data: [DONE]\n\n',
        )
    p._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url=p.base_url,
        headers={"Authorization": f"Bearer {p.api_key}"},
        transport=httpx.MockTransport(handler),
    )
    return p


class TestGLMProvider:
    def test_inherits_openai(self):
        assert issubclass(GLMProvider, OpenAIProvider)

    def test_name(self):
        p = GLMProvider(api_key="k")
        assert p.name == "glm"

    def test_base_url(self):
        p = GLMProvider(api_key="k")
        assert "volces.com" in p.base_url

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "from-env")
        p = GLMProvider()
        assert p.api_key == "from-env"

    def test_ark_env_var_fallback(self, monkeypatch):
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        monkeypatch.setenv("ARK_API_KEY", "from-ark-env")
        p = GLMProvider()
        assert p.api_key == "from-ark-env"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GLM API key required"):
            GLMProvider()

    async def test_stream_inherits_from_openai(self):
        p = _make_provider()
        model = (await p.list_models())[0]
        from pharos.llm.types import Context, UserMessage
        ctx = Context(messages=[UserMessage(content="hi")])
        events = []
        async for ev in p.stream(model, ctx):
            events.append(ev)
        final = events[-1].message
        assert final is not None
        assert final.text() == "ok"

    async def test_list_models(self):
        p = GLMProvider(api_key="k")
        models = await p.list_models()
        assert len(models) >= 1
        ids = {m.id for m in models}
        assert "glm-4.5-air" in ids or "glm-4.5" in ids
