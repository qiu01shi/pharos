"""DeepSeek model catalog."""

from __future__ import annotations

from pharos.llm.types import Model, ModelCost

MODELS: list[Model] = [
    Model(
        id="deepseek-chat",
        name="DeepSeek V3 Chat",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        input=["text"],
        cost=ModelCost(input=0.14, output=0.28, cache_read=0.014),
        context_window=128_000,
        max_tokens=8_192,
    ),
    Model(
        id="deepseek-reasoner",
        name="DeepSeek R1 Reasoning",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0.55, output=2.19, cache_read=0.14),
        context_window=128_000,
        max_tokens=64_000,
    ),
]