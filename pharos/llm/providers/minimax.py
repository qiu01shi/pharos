"""MiniMax provider — Anthropic-compatible API at minimaxi.com.

Inherits from AnthropicProvider and overrides only:
  - base_url: https://api.minimaxi.com/anthropic
  - env var: MINIMAX_CN_API_KEY
  - name: "minimax"

The wire protocol (messages, SSE event types, thinking mode)
is identical to Anthropic Messages API, so no other changes.
"""

from __future__ import annotations

import os

from pharos.llm.providers.anthropic import AnthropicProvider
from pharos.llm.types import Model, ModelCost

_MINIMAX_CATALOG: list[Model] = [
    Model(
        id="MiniMax-Text-01",
        name="MiniMax Text 01",
        api="anthropic-messages",
        provider="minimax",
        base_url="https://api.minimaxi.com/anthropic",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=1.0, output=8.0, cache_read=0.1, cache_write=1.25),
        context_window=1_000_000,
        max_tokens=64_000,
    ),
]


class MiniMaxProvider(AnthropicProvider):
    """Anthropic-compatible client for api.minimaxi.com."""

    name = "minimax"
    BASE_URL = "https://api.minimaxi.com/anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        key = api_key or os.environ.get("MINIMAX_CN_API_KEY", "")
        if not key:
            raise ValueError(
                "MiniMax API key required "
                "(set MINIMAX_CN_API_KEY in env or ~/.pharos/.env, "
                "or pass api_key=)"
            )
        super().__init__(
            api_key=key,
            base_url=base_url or self.BASE_URL,
            timeout=timeout,
        )

    async def list_models(self) -> list[Model]:
        return list(_MINIMAX_CATALOG)


__all__ = ["MiniMaxProvider"]