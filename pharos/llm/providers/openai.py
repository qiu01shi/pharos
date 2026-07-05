"""OpenAI provider — Responses + Chat Completions APIs.

Two API surfaces, selected by `model.api`:
- "openai-responses"      → POST /v1/responses   (newer, supports tools/structured)
- "openai-completions"    → POST /v1/chat/completions  (legacy, broadly compatible)

Both are async-streams of SSE. We use httpx.AsyncClient with a connection
pool so concurrent calls don't pay TCP setup cost.

For testing without real API calls, see `FauxProvider`.
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
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _content_to_openai_parts(
    content: str | list[Any],
) -> str | list[dict[str, Any]]:
    """Convert pharos content into OpenAI's content field shape.

    OpenAI Chat Completions: str | list[{type: text|image_url, ...}]
    OpenAI Responses: list[{type: input_text|input_image|...}]
    """
    if isinstance(content, str):
        return content
    out: list[dict[str, Any]] = []
    for block in content:
        t = getattr(block, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(block, "text", "")})
        elif t == "image":
            data = getattr(block, "data", "")
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data},
                }
            )
    return out


def _convert_messages_responses(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Map pharos messages → OpenAI Responses input array.

    Responses API uses role: user/assistant/system/developer and
    item types: message, function_call, function_call_output.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "role", "user")
        if isinstance(m, UserMessage):
            out.append(
                {
                    "role": role,
                    "content": _content_to_openai_parts(m.content),
                }
            )
        elif isinstance(m, AssistantMessage):
            # Concatenate text + tool calls into a single assistant item
            text = "".join(
                b.text for b in m.content if isinstance(b, TextContent)
            )
            out.append(
                {
                    "role": "assistant",
                    "content": text,
                }
            )
            for b in m.content:
                if isinstance(b, ToolCall):
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": b.id,
                            "name": b.name,
                            "arguments": json.dumps(b.arguments),
                        }
                    )
        elif isinstance(m, ToolResultMessage):
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": "".join(
                        b.text for b in m.content if isinstance(b, TextContent)
                    ),
                }
            )
    return out


def _convert_messages_chat(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Map pharos messages → OpenAI Chat Completions messages array."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append(
                {
                    "role": "user",
                    "content": _content_to_openai_parts(m.content),
                }
            )
        elif isinstance(m, AssistantMessage):
            text = "".join(
                b.text for b in m.content if isinstance(b, TextContent)
            )
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.arguments),
                    },
                }
                for b in m.content
                if isinstance(b, ToolCall)
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif isinstance(m, ToolResultMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": "".join(
                        b.text for b in m.content if isinstance(b, TextContent)
                    ),
                }
            )
    return out


def _convert_tools_responses(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


def _convert_tools_chat(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


class OpenAIProvider(LLMProvider):
    """OpenAI provider. Inheritable — GLMProvider / DeepSeekProvider reuse this."""

    name = "openai"
    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required (set OPENAI_API_KEY or pass api_key=)"
            )
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # trust_env=False: don't auto-pick up HTTP_PROXY etc.
            # (SOCKS proxies require optional `socksio`; we don't need them)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
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
        # Catalog imported lazily to avoid hard dep cycle
        from pharos.llm.catalog.openai import MODELS
        return list(MODELS)

    # ----- dispatch -----

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if model.api == "openai-responses":
            return self._stream_responses(model, context, options)
        return self._stream_chat(model, context, options)

    # ----- Responses API -----

    async def _stream_responses(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None,
    ) -> AsyncIterator[StreamEvent]:
        client = await self._get_client()
        body: dict[str, Any] = {
            "model": model.id,
            "input": _convert_messages_responses(context.messages),
            "stream": True,
        }
        if context.system_prompt:
            body["instructions"] = context.system_prompt
        if context.tools:
            body["tools"] = _convert_tools_responses(context.tools)
        if options:
            if options.temperature is not None:
                body["temperature"] = options.temperature
            if options.max_tokens:
                body["max_output_tokens"] = options.max_tokens
            if options.thinking_level and options.thinking_level != "off":
                effort = options.thinking_level
                # OpenAI reasoning models only support "minimal"/"low"/"medium"/"high"
                if effort in ("minimal", "low", "medium", "high"):
                    body["reasoning"] = {"effort": effort}
        usage = Usage()
        text_buf = ""
        tool_calls: dict[str, dict[str, Any]] = {}
        try:
            async with client.stream(
                "POST", "/responses", json=body,
            ) as resp:
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    raise RuntimeError(
                        f"OpenAI {resp.status_code}: "
                        f"{err_body.decode(errors='ignore')[:300]}"
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
                    etype = ev.get("type", "")
                    if etype == "response.output_text.delta":
                        delta = ev.get("delta", "")
                        text_buf += delta
                        yield StreamEvent(
                            type="text_delta",
                            content_index=0,
                            delta=delta,
                        )
                    elif etype == "response.output_item.added":
                        item = ev.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id") or item.get("id", "")
                            tool_calls[call_id] = {
                                "id": call_id,
                                "name": item.get("name", ""),
                                "arguments": "",
                            }
                            yield StreamEvent(
                                type="toolcall_start",
                                content_index=len(tool_calls) - 1,
                            )
                    elif etype == "response.function_call_arguments.delta":
                        call_id = ev.get("call_id", "")
                        if call_id in tool_calls:
                            tool_calls[call_id]["arguments"] += ev.get("delta", "")
                            yield StreamEvent(
                                type="toolcall_delta",
                                content_index=list(tool_calls).index(call_id),
                                delta=ev.get("delta", ""),
                            )
                    elif etype == "response.function_call_arguments.done":
                        call_id = ev.get("call_id", "")
                        if call_id in tool_calls:
                            import contextlib

                            with contextlib.suppress(json.JSONDecodeError):
                                tool_calls[call_id]["arguments"] = json.loads(
                                    tool_calls[call_id]["arguments"] or "{}"
                                )
                            yield StreamEvent(
                                type="toolcall_end",
                                content_index=list(tool_calls).index(call_id),
                                tool_call=ToolCall(**tool_calls[call_id]),
                            )
                    elif etype == "response.completed":
                        resp_obj = ev.get("response", {})
                        u = resp_obj.get("usage", {})
                        usage = Usage(
                            input=u.get("input_tokens", 0),
                            output=u.get("output_tokens", 0),
                            cache_read=u.get("input_tokens_details", {}).get(
                                "cached_tokens", 0
                            ),
                        )
                        # Mark the end
                        final = AssistantMessage(
                            content=[TextContent(text=text_buf)] if text_buf else [],
                            api="openai-responses",
                            provider=self.name,
                            model=model.id,
                            usage=usage,
                            stop_reason="stop",
                        )
                        yield StreamEvent(type="done", message=final)
                        return
        except httpx.HTTPError as e:
            yield StreamEvent(type="error", error=f"http: {e}")
            return
        except Exception as e:
            yield StreamEvent(type="error", error=str(e))
            return

        # Stream ended without explicit completion — emit a final message
        # from what we accumulated.
        final = AssistantMessage(
            content=[TextContent(text=text_buf)] if text_buf else [],
            api="openai-responses",
            provider=self.name,
            model=model.id,
            usage=usage,
            stop_reason="stop",
        )
        yield StreamEvent(type="done", message=final)

    # ----- Chat Completions API -----

    async def _stream_chat(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None,
    ) -> AsyncIterator[StreamEvent]:
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
        if options:
            if options.temperature is not None:
                body["temperature"] = options.temperature
            if options.max_tokens:
                body["max_tokens"] = options.max_tokens
        usage = Usage()
        text_buf = ""
        tool_calls: dict[str, dict[str, Any]] = {}
        try:
            async with client.stream(
                "POST", "/chat/completions", json=body,
            ) as resp:
                if resp.status_code >= 400:
                    body_b = await resp.aread()
                    raise RuntimeError(
                        f"OpenAI {resp.status_code}: {body_b.decode(errors='ignore')[:300]}"
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
                        if delta.get("content"):
                            text_buf += delta["content"]
                            yield StreamEvent(
                                type="text_delta",
                                content_index=0,
                                delta=delta["content"],
                            )
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                key = str(idx)
                                if key not in tool_calls:
                                    tool_calls[key] = {
                                        "id": tc.get("id", ""),
                                        "name": "",
                                        "arguments": "",
                                    }
                                    yield StreamEvent(
                                        type="toolcall_start",
                                        content_index=idx,
                                    )
                                if tc.get("id"):
                                    tool_calls[key]["id"] = tc["id"]
                                if tc.get("function", {}).get("name"):
                                    tool_calls[key]["name"] = tc["function"]["name"]
                                if tc.get("function", {}).get("arguments"):
                                    tool_calls[key]["arguments"] += tc["function"][
                                        "arguments"
                                    ]
                                    yield StreamEvent(
                                        type="toolcall_delta",
                                        content_index=idx,
                                        delta=tc["function"]["arguments"],
                                    )
                    u = ev.get("usage")
                    if u:
                        usage = Usage(
                            input=u.get("prompt_tokens", 0),
                            output=u.get("completion_tokens", 0),
                            cache_read=u.get("prompt_tokens_details", {}).get(
                                "cached_tokens", 0
                            ),
                        )
            # Build final
            content: list[Any] = []
            if text_buf:
                content.append(TextContent(text=text_buf))
            for _k, tc in tool_calls.items():
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc["arguments"]}
                content.append(
                    ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                )
            for k, tc in tool_calls.items():
                yield StreamEvent(
                    type="toolcall_end",
                    content_index=int(k),
                    tool_call=ToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=(
                            json.loads(tc["arguments"])
                            if tc["arguments"]
                            else {}
                        ),
                    ),
                )
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


__all__ = ["OpenAIProvider"]
