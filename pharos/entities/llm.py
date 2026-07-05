"""LLMAgent — wraps a LLMProvider as a pharos Entity.

Behavior:
- `setup()` creates the LLM client (long-lived).
- `fire()` consumes one or more prompt tokens, calls the LLM,
  emits one `text` token and one `usage` token.
- Streams partial text to `draft` so downstream entities can react
  before the final response arrives.
- If `tools` are configured, the LLM may emit `tool_call` tokens
  that get echoed out of the `tool_calls` port.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.llm.base import LLMProvider
from pharos.llm.types import (
    Context,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pharos.observability.trace import SpanEvent as _SpanEvent
from pharos.observability.trace import current_span as _current_span_fn


@dataclass
class LLMEntityConfig:
    """Configuration for an LLMAgent.

    `provider_class` is a class (not an instance) because the Director
    instantiates the Entity on first setup. We hold the *params* to
    pass at instantiation time.
    """

    provider_class: type[LLMProvider]
    provider_kwargs: dict[str, Any]
    model_id: str
    system_prompt: str = ""
    tools: list[Any] = None  # list[Tool] once provided
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_level: str | None = None

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = []


@entity
class LLMAgent(Entity):
    """An Entity that calls an LLM.

    Ports:
        ins:
            prompt: TextPort        — user prompt
            system_override: TextPort — optional system-prompt override
        outs:
            text: TextPort         — final LLM response
            draft: TextPort        — streamed partial response (live)
            thinking: TextPort     — thinking/reasoning (if model emits any)
            tool_calls: JsonPort   — list of ToolCall (if model emits any)
            usage: JsonPort        — Usage object
    """

    ins = {
        "prompt": InputPort(name="prompt", accepted_types=["text"]),
        "system_override": InputPort(
            name="system_override", accepted_types=["text"]
        ),
    }
    outs = {
        "text": OutputPort(name="text", accepted_types=["text"]),
        "draft": OutputPort(name="draft", accepted_types=["text"]),
        "thinking": OutputPort(name="thinking", accepted_types=["text"]),
        "tool_calls": OutputPort(name="tool_calls", accepted_types=["json"]),
        "usage": OutputPort(name="usage", accepted_types=["json"]),
    }

    def __init__(
        self,
        node_id: str,
        config: LLMEntityConfig,
    ) -> None:
        super().__init__(node_id=node_id)
        self.config = config
        self.provider: LLMProvider | None = None
        self.model: Model | None = None

    async def setup(self, ctx) -> None:  # type: ignore[override]
        # Construct the provider (this may fail if API key is missing;
        # we let the error bubble — the Director will mark the run failed).
        self.provider = self.config.provider_class(
            **self.config.provider_kwargs
        )
        models = await self.provider.list_models()
        # Find the requested model, or fall back to the first one
        for m in models:
            if m.id == self.config.model_id:
                self.model = m
                break
        if self.model is None:
            self.model = models[0]

    async def teardown(self) -> None:  # type: ignore[override]
        if self.provider is not None:
            await self.provider.close()
            self.provider = None

    async def fire(self, ctx) -> None:  # type: ignore[override]
        if self.provider is None or self.model is None:
            await self.setup(ctx)
        assert self.provider is not None
        assert self.model is not None

        # Consume prompts (concatenate multiple)
        prompt_tokens = self.ins["prompt"].consume()
        if not prompt_tokens:
            return
        user_text = "".join(t.value.payload for t in prompt_tokens)

        # Optional system override
        sys_override = self.ins["system_override"].consume()
        system_prompt = self.config.system_prompt
        if sys_override:
            system_prompt = "".join(t.value.payload for t in sys_override)

        context = Context(
            system_prompt=system_prompt,
            messages=[UserMessage(content=user_text)],
            tools=self.config.tools,
        )
        options = StreamOptions(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking_level=self.config.thinking_level,  # type: ignore[arg-type]
        )

        accumulated = ""
        accumulated_thinking = ""
        tool_calls: list[ToolCall] = []
        usage = Usage()
        # Record events onto the active span (if any) for replay.
        span = _current_span_fn()

        # Stream the response
        async for ev in self.provider.stream(self.model, context, options):
            # Forward event to the trace span for replay support.
            if span is not None:
                span.events.append(
                    _SpanEvent(
                        name=ev.type,
                        ts=time.time(),
                        attributes={
                            "delta": ev.delta,
                            "content_index": ev.content_index,
                            "tool_call": (
                                ev.tool_call.model_dump()
                                if ev.tool_call
                                else None
                            ),
                            "message": (
                                ev.message.model_dump()
                                if ev.message
                                else None
                            ),
                            "error": ev.error,
                        },
                    )
                )
            if ev.type == "text_delta" and ev.delta:
                accumulated += ev.delta
                # Live-stream: emit a partial token to `draft` on every delta
                self.outs["draft"].emit(
                    TypedValue(type="text", payload=accumulated)
                )
            elif ev.type == "thinking_delta" and ev.delta:
                accumulated_thinking += ev.delta
            elif ev.type == "toolcall_end" and ev.tool_call is not None:
                tool_calls.append(ev.tool_call)
            elif ev.type == "done" and ev.message is not None:
                # Prefer provider-reported usage; fall back to whatever
                # we accumulated.
                if ev.message.usage.input or ev.message.usage.output:
                    usage = ev.message.usage
                # If the provider gave us a final message with text
                # content, prefer that (it may include tool calls etc.)
                for block in ev.message.content:
                    if isinstance(block, TextContent) and not accumulated:
                        accumulated = block.text
                    elif isinstance(block, ThinkingContent) and not accumulated_thinking:
                        accumulated_thinking = block.thinking
                    elif isinstance(block, ToolCall) and block not in tool_calls:
                        tool_calls.append(block)
            elif ev.type == "error":
                # Surface the error but still emit a `text` token so
                # downstream isn't left hanging.
                self.outs["text"].emit(
                    TypedValue(type="text", payload=f"[error: {ev.error}]")
                )
                return

        # Final outputs
        if accumulated:
            self.outs["text"].emit(
                TypedValue(type="text", payload=accumulated)
            )
        if accumulated_thinking:
            self.outs["thinking"].emit(
                TypedValue(type="text", payload=accumulated_thinking)
            )
        if tool_calls:
            self.outs["tool_calls"].emit(
                TypedValue(
                    type="json",
                    payload=[
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in tool_calls
                    ],
                )
            )
        if usage.input or usage.output:
            self.outs["usage"].emit(
                TypedValue(
                    type="json",
                    payload={
                        "input": usage.input,
                        "output": usage.output,
                        "cache_read": usage.cache_read,
                        "cache_write": usage.cache_write,
                        "total": usage.total,
                    },
                )
            )


__all__ = ["LLMAgent", "LLMEntityConfig"]
