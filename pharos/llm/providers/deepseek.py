"""DeepSeek provider — OpenAI-compatible with custom thinking format.

DeepSeek uses OpenAI Chat Completions but emits `reasoning_content`
in addition to `content`. We subclass OpenAIProvider and add a thin
override that extracts `reasoning_content` as thinking_delta events.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from pharos.llm.providers.openai import (
    OpenAIProvider,
    _convert_messages_chat,
    _convert_tools_chat,
)
from pharos.llm.types import (
    AssistantMessage,
    Model,
    StreamEvent,
    TextContent,
    ThinkingContent,
    Usage,
)


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek via OpenAI-compatible API."""

    name = "deepseek"
    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError(
                "DeepSeek API key required (set DEEPSEEK_API_KEY or pass api_key=)"
            )
        super().__init__(api_key=key, base_url=base_url or self.BASE_URL, timeout=timeout)

    async def list_models(self) -> list[Model]:
        from pharos.llm.catalog.deepseek import MODELS
        return list(MODELS)

    async def _stream_chat(
        self,
        model: Model,
        context: Any,
        options: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Override Chat Completions to extract `reasoning_content`.

        DeepSeek's API returns delta.reasoning_content alongside
        delta.content for thinking-mode models. We yield it as
        `thinking_delta` events so consumers (LLMAgent) capture it.
        """
        client = await self._get_client()
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        messages.extend(_convert_messages_chat(context.messages))
        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if context.tools:
            body["tools"] = _convert_tools_chat(context.tools)
        if options and options.temperature is not None:
            body["temperature"] = options.temperature
        if options and options.max_tokens:
            body["max_tokens"] = options.max_tokens

        usage = Usage()
        text_buf = ""
        thinking_buf = ""
        try:
            async with client.stream(
                "POST", "/chat/completions", json=body,
            ) as resp:
                if resp.status_code >= 400:
                    body_b = await resp.aread()
                    raise RuntimeError(
                        f"DeepSeek {resp.status_code}: {body_b.decode(errors='ignore')[:300]}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = ev.get("choices", [])
                    for ch in choices:
                        delta = ch.get("delta", {})
                        # DeepSeek's reasoning field
                        if delta.get("reasoning_content"):
                            thinking_buf += delta["reasoning_content"]
                            yield StreamEvent(
                                type="thinking_delta",
                                content_index=0,
                                delta=delta["reasoning_content"],
                            )
                        if delta.get("content"):
                            text_buf += delta["content"]
                            yield StreamEvent(
                                type="text_delta",
                                content_index=0,
                                delta=delta["content"],
                            )
                    u = ev.get("usage")
                    if u:
                        usage = Usage(
                            input=u.get("prompt_tokens", 0),
                            output=u.get("completion_tokens", 0),
                            cache_read=u.get(
                                "prompt_tokens_details", {}
                            ).get("cached_tokens", 0),
                            reasoning=(
                                u.get("completion_tokens_details", {}).get(
                                    "reasoning_tokens", 0
                                )
                            ),
                        )

            content: list[Any] = []
            if thinking_buf:
                content.append(ThinkingContent(thinking=thinking_buf))
            if text_buf:
                content.append(TextContent(text=text_buf))
            final = AssistantMessage(
                content=content,
                api="openai-completions",
                provider=self.name,
                model=model.id,
                usage=usage,
                stop_reason="stop",
            )
            yield StreamEvent(type="done", message=final)
        except httpx.HTTPError as e:
            yield StreamEvent(type="error", error=f"http: {e}")
        except Exception as e:
            yield StreamEvent(type="error", error=str(e))


__all__ = ["DeepSeekProvider"]