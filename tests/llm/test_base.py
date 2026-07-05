"""Tests for pharos.llm.base, retry, estimate."""
from __future__ import annotations

import asyncio

import pytest

from pharos.llm.base import LLMProvider
from pharos.llm.estimate import estimate_messages_tokens, estimate_tokens
from pharos.llm.retry import with_retry
from pharos.llm.types import (
    Context,
    Model,
    ModelCost,
    StreamEvent,
    UserMessage,
)


def _model() -> Model:
    return Model(
        id="test",
        name="Test",
        api="test",
        provider="test",
        base_url="local://",
        cost=ModelCost(input=0.0, output=0.0),
    )


def _ctx(prompt: str = "hi") -> Context:
    return Context(messages=[UserMessage(content=prompt)])


class TestEstimate:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short_ascii(self):
        # "hello" -> 5/4 = 1.25 -> max(1, 1) = 1
        assert estimate_tokens("hello") == 1

    def test_long_ascii(self):
        text = "x" * 400
        assert estimate_tokens(text) == 100

    def test_cjk(self):
        # 4 CJK chars / 1.5 = 2.67 -> int(2.67) = 2
        assert estimate_tokens("你好世界") == 2

    def test_messages_estimator(self):
        msgs = [UserMessage(content="hello world")]
        # 11 chars / 4 = 2 + 4 framing = 6
        n = estimate_messages_tokens(msgs)
        assert n > 0


class TestRetry:
    def test_succeeds_first_try(self):
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        async def run():
            return await with_retry(fn, max_attempts=3, base_delay=0.01)

        result = asyncio.run(run())
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        attempts = []

        async def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient")
            return "ok"

        async def run():
            return await with_retry(
                fn,
                max_attempts=5,
                base_delay=0.01,
                retryable_exceptions=(RuntimeError,),
            )

        result = asyncio.run(run())
        assert result == "ok"
        assert len(attempts) == 3

    def test_raises_after_max(self):
        async def fn():
            raise RuntimeError("always fails")

        async def run():
            return await with_retry(
                fn, max_attempts=3, base_delay=0.01,
                retryable_exceptions=(RuntimeError,),
            )

        with pytest.raises(RuntimeError, match="always fails"):
            asyncio.run(run())

    def test_non_retryable_propagates(self):
        async def fn():
            raise ValueError("fatal")

        async def run():
            return await with_retry(
                fn, max_attempts=3, base_delay=0.01,
                retryable_exceptions=(RuntimeError,),
            )

        with pytest.raises(ValueError):
            asyncio.run(run())


class TestStreamContract:
    """Verify the event-stream contract used by all providers."""

    async def test_stream_terminates_with_done(self):
        from pharos.llm.providers.faux import FauxProvider

        p = FauxProvider()
        model = (await p.list_models())[0]

        events: list[StreamEvent] = []
        async for ev in p.stream(model, _ctx()):
            events.append(ev)

        # Must end with 'done' carrying a message
        assert events[-1].type == "done"
        assert events[-1].message is not None

        # Must include start and at least one text_delta
        types = {e.type for e in events}
        assert "start" in types
        assert "text_delta" in types
        assert "text_end" in types

    async def test_complete_helper(self):
        from pharos.llm.providers.faux import FauxProvider

        p = FauxProvider()
        model = (await p.list_models())[0]
        msg = await p.complete(model, _ctx("hello"))
        assert msg.text().startswith("echo:")
        assert "hello" in msg.text()

    async def test_recorded_calls(self):
        from pharos.llm.providers.faux import FauxProvider

        p = FauxProvider()
        model = (await p.list_models())[0]
        await p.complete(model, _ctx("foo"))
        await p.complete(model, _ctx("bar"))
        assert p.call_count == 2
        assert p.last_call() is not None
        assert "bar" in p.last_call().prompt  # type: ignore[union-attr]


class TestRegistry:
    def test_faux_registered(self):
        from pharos.llm.registry import (
            create_provider,
            list_providers,
        )

        assert "faux" in list_providers()
        p = create_provider("faux")
        assert isinstance(p, LLMProvider)

    def test_unknown_raises(self):
        from pharos.llm.registry import get_provider_class

        with pytest.raises(KeyError, match="unknown provider"):
            get_provider_class("nonexistent")

    def test_duplicate_registration_raises(self):
        from pharos.llm.providers.faux import FauxProvider
        from pharos.llm.registry import register_provider

        with pytest.raises(ValueError, match="already registered"):
            register_provider("faux", FauxProvider)
