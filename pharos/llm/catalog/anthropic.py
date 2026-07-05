"""Anthropic model catalog."""

from __future__ import annotations

from pharos.llm.types import Model, ModelCost

MODELS: list[Model] = [
    Model(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200_000,
        max_tokens=8_192,
    ),
    Model(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        input=["text", "image"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200_000,
        max_tokens=8_192,
    ),
    Model(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        input=["text", "image"],
        cost=ModelCost(input=0.25, output=1.25, cache_read=0.03, cache_write=0.3),
        context_window=200_000,
        max_tokens=4_096,
    ),
]