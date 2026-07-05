"""OpenAI model catalog.

Static metadata for the OpenAI models pharos routes to. Prices and
context windows are reviewed quarterly; bump when OpenAI changes them.
"""

from __future__ import annotations

from pharos.llm.types import Model, ModelCost

# Reference pricing: USD per 1M tokens
MODELS: list[Model] = [
    Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        input=["text", "image"],
        cost=ModelCost(input=2.5, output=10.0, cache_read=1.25),
        context_window=128_000,
        max_tokens=16_384,
    ),
    Model(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        input=["text", "image"],
        cost=ModelCost(input=0.15, output=0.6, cache_read=0.075),
        context_window=128_000,
        max_tokens=16_384,
    ),
    Model(
        id="gpt-4.1",
        name="GPT-4.1",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        input=["text", "image"],
        cost=ModelCost(input=2.0, output=8.0, cache_read=0.5),
        context_window=1_000_000,
        max_tokens=32_768,
    ),
    Model(
        id="gpt-4.1-mini",
        name="GPT-4.1 mini",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        input=["text", "image"],
        cost=ModelCost(input=0.4, output=1.6, cache_read=0.1),
        context_window=1_000_000,
        max_tokens=32_768,
    ),
    Model(
        id="o3",
        name="o3",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=10.0, output=40.0, cache_read=2.5),
        context_window=200_000,
        max_tokens=100_000,
    ),
    Model(
        id="o4-mini",
        name="o4-mini",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=1.1, output=4.4, cache_read=0.275),
        context_window=200_000,
        max_tokens=100_000,
    ),
    # Legacy Chat Completions (kept for compatibility with older code)
    Model(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        input=["text"],
        cost=ModelCost(input=0.5, output=1.5),
        context_window=16_385,
        max_tokens=4_096,
    ),
]
