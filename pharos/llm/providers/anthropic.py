"""Anthropic provider — Messages API with thinking support.

Anthropic's Messages API is structurally similar to OpenAI Chat
Completions but with notable differences:
- Headers: `x-api-key` + `anthropic-version`
- System prompt is a top-level field, not a message
- SSE events: message_start, content_block_start/delta/stop,
  message_delta, message_stop
- Thinking: `thinking: {type: "enabled", budget_tokens: N}`
- Tool use: `input_json_delta` accumulates to a ToolCall
- Usage: input_tokens, output_tokens, cache_creation_input_tokens,
  cache_read_input_tokens
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from pharos.llm.base import LLMProvider
from pharos.llm.types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _convert_messages(context: Context) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert pharos messages → Anthropic (system, messages[]) format."""
    system: str | None = None
    messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    for m in context.messages:
        if isinstance(m, UserMessage):
            content = m.content
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                # List of TextContent / ImageContent
                blocks = []
                for block in content:
                    if isinstance(block, TextContent):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, type(content[0])) and hasattr(block, "data"):
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.mime_type,  # type: ignore[union-attr]
                                "data": block.data,  # type: ignore[union-attr]
                            },
                        })
                messages.append({"role": "user", "content": blocks})
        elif isinstance(m, AssistantMessage):
            # Concatenate text + emit tool_use blocks
            blocks = []
            for b in m.content:
                if isinstance(b, TextContent):
                    blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ThinkingContent):
                    if b.signature:
                        blocks.append({
                            "type": "thinking",
                            "thinking": b.thinking,
                            "signature": b.signature,
                        })
                elif isinstance(b, ToolCall):
                    blocks.append({
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.arguments,
                    })
            messages.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolResultMessage):
            content_blocks = []
            for b in m.content:
                if isinstance(b, TextContent):
                    content_blocks.append({"type": "text", "text": b.text})
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": content_blocks,
                "is_error": m.is_error,
            })

    # Append any pending tool results as a final user message
    if pending_tool_results:
        messages.append({"role": "user", "content": pending_tool_results})

    return system, messages


def _convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


# Map Anthropic stop_reason values to pharos's Literal["stop",
# "length", "tool_use", "error", "aborted"].
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "refusal": "error",
    "model_context_window_exceeded": "length",
}


def _map_stop_reason(anthropic_reason: str) -> str:
    return _STOP_REASON_MAP.get(anthropic_reason, "stop")


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API client."""

    name = "anthropic"
    BASE_URL = "https://api.anthropic.com"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required (set ANTHROPIC_API_KEY or pass api_key=)"
            )
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(
                    connect=10.0, read=self.timeout,
                    write=10.0, pool=10.0,
                ),
                trust_env=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> list[Model]:
        from pharos.llm.catalog.anthropic import MODELS
        return list(MODELS)

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(model, context, options)

    async def _stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None,
    ) -> AsyncIterator[StreamEvent]:
        client = await self._get_client()
        system, messages = _convert_messages(context)

        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "max_tokens": (
                options.max_tokens if options and options.max_tokens
                else model.max_tokens
            ),
            "stream": True,
        }
        if system:
            body["system"] = system
        if context.tools:
            body["tools"] = _convert_tools(context.tools)
        if options and options.temperature is not None:
            body["temperature"] = options.temperature
        if options and options.thinking_level and options.thinking_level != "off":
            budget = {
                "minimal": 1024,
                "low": 2048,
                "medium": 4096,
                "high": 8192,
            }.get(options.thinking_level, 4096)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}

        usage = Usage()
        text_buf = ""
        thinking_buf = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        stop_reason = "stop"

        try:
            async with client.stream(
                "POST", "/v1/messages", json=body,
            ) as resp:
                if resp.status_code >= 400:
                    body_b = await resp.aread()
                    raise RuntimeError(
                        f"Anthropic {resp.status_code}: {body_b.decode(errors='ignore')[:300]}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    etype = ev.get("type", "")

                    if etype == "message_start":
                        msg_usage = ev.get("message", {}).get("usage", {})
                        usage = Usage(
                            input=msg_usage.get("input_tokens", 0),
                            output=msg_usage.get("output_tokens", 0),
                            cache_read=msg_usage.get(
                                "cache_read_input_tokens", 0
                            ),
                            cache_write=msg_usage.get(
                                "cache_creation_input_tokens", 0
                            ),
                            cache_write_1h=msg_usage.get(
                                "cache_creation_input_tokens_1h", 0
                            ),
                        )
                        yield StreamEvent(
                            type="start",
                            partial=AssistantMessage(
                                api="anthropic-messages",
                                provider=self.name,
                                model=model.id,
                                usage=usage,
                            ),
                        )

                    elif etype == "content_block_start":
                        block = ev.get("content_block", {})
                        block_type = block.get("type", "")
                        idx = ev.get("index", 0)
                        if block_type == "text":
                            yield StreamEvent(
                                type="text_start",
                                content_index=idx,
                            )
                        elif block_type == "thinking":
                            yield StreamEvent(
                                type="thinking_start",
                                content_index=idx,
                            )
                        elif block_type == "tool_use":
                            tool_calls[idx] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input_json": "",
                            }
                            yield StreamEvent(
                                type="toolcall_start",
                                content_index=idx,
                            )

                    elif etype == "content_block_delta":
                        delta = ev.get("delta", {})
                        idx = ev.get("index", 0)
                        delta_type = delta.get("type", "")
                        if delta_type == "text_delta":
                            text_buf += delta.get("text", "")
                            yield StreamEvent(
                                type="text_delta",
                                content_index=idx,
                                delta=delta.get("text", ""),
                            )
                        elif delta_type == "thinking_delta":
                            thinking_buf += delta.get("thinking", "")
                            yield StreamEvent(
                                type="thinking_delta",
                                content_index=idx,
                                delta=delta.get("thinking", ""),
                            )
                        elif delta_type == "input_json_delta":
                            if idx in tool_calls:
                                tool_calls[idx]["input_json"] += (
                                    delta.get("partial_json", "")
                                )
                                yield StreamEvent(
                                    type="toolcall_delta",
                                    content_index=idx,
                                    delta=delta.get("partial_json", ""),
                                )

                    elif etype == "content_block_stop":
                        idx = ev.get("index", 0)
                        if idx in tool_calls:
                            try:
                                args = json.loads(
                                    tool_calls[idx]["input_json"] or "{}"
                                )
                            except json.JSONDecodeError:
                                args = {"_raw": tool_calls[idx]["input_json"]}
                            yield StreamEvent(
                                type="toolcall_end",
                                content_index=idx,
                                tool_call=ToolCall(
                                    id=tool_calls[idx]["id"],
                                    name=tool_calls[idx]["name"],
                                    arguments=args,
                                ),
                            )
                        elif text_buf and idx == 0:
                            # Only emit text_end once (for the main text block)
                            pass

                    elif etype == "message_delta":
                        msg_delta = ev.get("delta", {})
                        sr = msg_delta.get("stop_reason")
                        if sr:
                            # Anthropic uses "end_turn" / "max_tokens" /
                            # "tool_use" / "stop_sequence"; map to pharos
                            # literals.
                            stop_reason = _map_stop_reason(sr)
                        msg_usage = ev.get("usage", {})
                        if msg_usage.get("output_tokens"):
                            # Usage is frozen; replace via model_copy
                            usage = usage.model_copy(
                                update={
                                    "output": msg_usage["output_tokens"],
                                    "reasoning": (
                                        msg_usage.get(
                                            "output_tokens_details", {}
                                        ).get("reasoning_tokens", 0)
                                        or usage.reasoning
                                    ),
                                }
                            )

                    elif etype == "message_stop":
                        content: list[Any] = []
                        if thinking_buf:
                            content.append(
                                ThinkingContent(thinking=thinking_buf)
                            )
                        if text_buf:
                            content.append(TextContent(text=text_buf))
                        for tc in tool_calls.values():
                            try:
                                args = json.loads(tc["input_json"] or "{}")
                            except json.JSONDecodeError:
                                args = {"_raw": tc["input_json"]}
                            content.append(
                                ToolCall(
                                    id=tc["id"],
                                    name=tc["name"],
                                    arguments=args,
                                )
                            )
                        final = AssistantMessage(
                            content=content,
                            api="anthropic-messages",
                            provider=self.name,
                            model=model.id,
                            usage=usage,
                            stop_reason=stop_reason,  # type: ignore[arg-type]
                        )
                        yield StreamEvent(type="done", message=final)
                        return
        except httpx.HTTPError as e:
            yield StreamEvent(type="error", error=f"http: {e}")
            return
        except Exception as e:
            yield StreamEvent(type="error", error=str(e))
            return

        # If we exit the SSE stream without a message_stop, emit a final.
        content = []
        if thinking_buf:
            content.append(ThinkingContent(thinking=thinking_buf))
        if text_buf:
            content.append(TextContent(text=text_buf))
        final = AssistantMessage(
            content=content,
            api="anthropic-messages",
            provider=self.name,
            model=model.id,
            usage=usage,
            stop_reason=stop_reason,  # type: ignore[arg-type]
        )
        yield StreamEvent(type="done", message=final)


__all__ = ["AnthropicProvider"]