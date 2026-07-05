"""GLM provider — Volcengine Ark (OpenAI-compatible).

GLM (智谱) models are served through Volcengine Ark's
OpenAI-compatible endpoint. We inherit OpenAIProvider and override
just the base URL + auth.

Env: GLM_API_KEY (Volcengine Ark API key, e.g. "xxxxxxxx-xxxx-...")
"""

from __future__ import annotations

import os

from pharos.llm.providers.openai import OpenAIProvider


class GLMProvider(OpenAIProvider):
    """GLM via Volcengine Ark. Drop-in subclass of OpenAIProvider."""

    name = "glm"
    # Volcengine Ark OpenAI-compatible endpoint
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        key = api_key or os.environ.get("GLM_API_KEY", "")
        # Fall back to ARK_API_KEY for users who set the canonical name
        if not key:
            key = os.environ.get("ARK_API_KEY", "")
        if not key:
            raise ValueError(
                "GLM API key required (set GLM_API_KEY or ARK_API_KEY or pass api_key=)"
            )
        super().__init__(
            api_key=key,
            base_url=base_url or self.BASE_URL,
            timeout=timeout,
        )

    async def list_models(self):
        from pharos.llm.catalog.glm import MODELS
        return list(MODELS)


__all__ = ["GLMProvider"]
