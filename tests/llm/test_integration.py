"""Integration tests against real LLM providers.

These tests are SKIPPED by default (require real API keys). To run:

    OPENAI_API_KEY=sk-... uv run pytest -m integration tests/llm/test_integration.py

If a key isn't set, that provider's tests are skipped. If the
provider rejects the key (e.g. invalid), the test fails with the
real error — that lets us catch upstream API changes early.

Each test makes ONE real call, keeps cost low.
"""

from __future__ import annotations

import os

import pytest

from pharos.llm.providers.anthropic import AnthropicProvider
from pharos.llm.providers.deepseek import DeepSeekProvider
from pharos.llm.providers.glm import GLMProvider
from pharos.llm.providers.minimax import MiniMaxProvider
from pharos.llm.providers.openai import OpenAIProvider
from pharos.llm.types import Context, UserMessage

pytestmark = pytest.mark.integration


# ---------- helpers ----------


def _require_key(env_var: str) -> str:
    """Skip the test if the API key env var is missing or empty."""
    val = os.environ.get(env_var, "").strip()
    if not val:
        pytest.skip(f"{env_var} not set; skipping real-LLM test")
    return val


async def _one_call(provider, model_id: str, prompt: str = "Reply with just the word 'pong'") -> str:
    """Make one real call, return the assistant text."""
    models = await provider.list_models()
    model = next((m for m in models if m.id == model_id), None)
    if model is None:
        pytest.skip(f"model {model_id!r} not in {provider.name}'s catalog")
    ctx = Context(messages=[UserMessage(content=prompt)])
    chunks: list[str] = []
    async for ev in provider.stream(model, ctx):
        if ev.type == "text_delta" and ev.delta:
            chunks.append(ev.delta)
        elif ev.type == "error":
            pytest.fail(f"{provider.name} error: {ev.error}")
    return "".join(chunks)


# ---------- OpenAI ----------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIIntegration:
    async def test_basic_call_gpt_4o_mini(self):
        key = _require_key("OPENAI_API_KEY")
        p = OpenAIProvider(api_key=key)
        try:
            text = await _one_call(p, "gpt-4o-mini")
            assert text, "empty response"
            assert isinstance(text, str)
            # Cost should be > 0 (real LLM call)
            models = await p.list_models()
            gpt4o_mini = next(m for m in models if m.id == "gpt-4o-mini")
            assert gpt4o_mini.cost.input > 0
        finally:
            await p.close()


# ---------- GLM / Volcengine Ark ----------


@pytest.mark.skipif(
    not os.environ.get("GLM_API_KEY") and not os.environ.get("ARK_API_KEY"),
    reason="GLM_API_KEY / ARK_API_KEY not set",
)
class TestGLMIntegration:
    async def test_basic_call_glm_4_5_air(self):
        key = os.environ.get("GLM_API_KEY") or _require_key("ARK_API_KEY")
        p = GLMProvider(api_key=key)
        try:
            text = await _one_call(p, "glm-4.5-air")
            assert text, "empty response"
            assert isinstance(text, str)
        finally:
            await p.close()


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAnthropicIntegration:
    async def test_basic_call(self):
        p = AnthropicProvider(api_key=_require_key("ANTHROPIC_API_KEY"))
        try:
            assert await _one_call(p, "claude-sonnet-4-20250514")
        finally:
            await p.close()


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)
class TestDeepSeekIntegration:
    async def test_basic_call(self):
        p = DeepSeekProvider(api_key=_require_key("DEEPSEEK_API_KEY"))
        try:
            assert await _one_call(p, "deepseek-chat")
        finally:
            await p.close()


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_CN_API_KEY"),
    reason="MINIMAX_CN_API_KEY not set",
)
class TestMiniMaxIntegration:
    async def test_basic_call(self):
        p = MiniMaxProvider(api_key=_require_key("MINIMAX_CN_API_KEY"))
        try:
            assert await _one_call(p, "MiniMax-Text-01")
        finally:
            await p.close()
