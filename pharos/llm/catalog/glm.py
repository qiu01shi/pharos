"""GLM model catalog (Volcengine Ark)."""

from __future__ import annotations

from pharos.llm.types import Model, ModelCost

# Reference pricing on Volcengine Ark; bump when GLM changes.
# The model IDs are the "endpoint IDs" you create in the Ark console,
# but we provide a few common ones for the public inference endpoints.
MODELS: list[Model] = [
    Model(
        id="glm-4.5",
        name="GLM-4.5",
        api="openai-completions",
        provider="glm",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        input=["text"],
        cost=ModelCost(input=0.6, output=2.2),
        context_window=128_000,
        max_tokens=32_000,
    ),
    Model(
        id="glm-4.5-air",
        name="GLM-4.5 Air",
        api="openai-completions",
        provider="glm",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        input=["text"],
        cost=ModelCost(input=0.2, output=0.8),
        context_window=128_000,
        max_tokens=32_000,
    ),
    Model(
        id="glm-4.6",
        name="GLM-4.6",
        api="openai-completions",
        provider="glm",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0.6, output=2.2),
        context_window=200_000,
        max_tokens=32_000,
    ),
]
