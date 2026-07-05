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
    # If set, LLMAgent will execute tool calls and feed the
    # results back to the LLM in a loop, up to `max_tool_iterations`.
    # The registry's tools are sent to the LLM via the
    # provider's tool-spec protocol.
    tool_registry: Any = None  # pharos.entities.tools.ToolRegistry
    max_tool_iterations: int = 5

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
        # Cumulative token count and cost across all fire() invocations.
        # The Director reads these after each fire to update the run's
        # total_tokens / total_cost.
        self.total_tokens: int = 0
        self.total_cost: float = 0.0

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

        # Build the tool spec. ToolRegistry.list_tools() returns
        # list[Tool] (pharos.llm.types.Tool), which is what
        # Context.tools and providers expect.
        context_tools: list[Any] = self.config.tools or []
        if self.config.tool_registry is not None:
            context_tools = self.config.tool_registry.list_tools()

        # Messages start with the user's prompt; tool results are
        # appended as we go through the tool-call loop.
        from pharos.llm.types import AssistantMessage as _AM
        from pharos.llm.types import ToolResultMessage as _TR

        messages: list[Any] = [UserMessage(content=user_text)]
        options = StreamOptions(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking_level=self.config.thinking_level,  # type: ignore[arg-type]
        )

        # Accumulators across all LLM calls in this fire()
        accumulated = ""
        accumulated_thinking = ""
        all_tool_calls: list[ToolCall] = []
        total_usage = Usage()
        # Record events onto the active span (if any) for replay.
        span = _current_span_fn()

        # Tool-call loop: keep calling LLM until it stops emitting
        # tool calls, or max_tool_iterations is hit.
        granted = getattr(ctx, "granted_permissions", set()) or set()
        for _iteration in range(self.config.max_tool_iterations + 1):
            context = Context(
                system_prompt=system_prompt,
                messages=list(messages),
                tools=context_tools,
            )
            (
                text_delta,
                think_delta,
                tool_calls_this_round,
                round_usage,
                round_error,
                round_text_final,
            ) = await self._stream_one_round(context, options, span)

            if text_delta:
                accumulated += text_delta
            if think_delta:
                accumulated_thinking += think_delta
            if round_text_final and not accumulated:
                accumulated = round_text_final
            if round_usage.input or round_usage.output:
                total_usage = round_usage
            all_tool_calls.extend(tool_calls_this_round)

            # If we have an error or no tool calls, we're done.
            if round_error:
                self.outs["text"].emit(
                    TypedValue(type="text", payload=f"[error: {round_error}]")
                )
                return

            if not tool_calls_this_round:
                # No more tool calls → LLM produced final answer
                break

            # Execute tool calls and append results to messages.
            # Persist what the LLM said (it might be partial reasoning).
            if text_delta or round_text_final:
                messages.append(
                    _AM(
                        content=[TextContent(text=text_delta or round_text_final)],
                        api="",
                        provider="",
                        model="",
                    )
                )
            # Track assistant's tool_call declarations so the
            # provider round-trip stays consistent.
            messages.append(
                _AM(
                    content=[
                        ToolCall(
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                        for tc in tool_calls_this_round
                    ],
                    api="",
                    provider="",
                    model="",
                )
            )

            if self.config.tool_registry is None:
                # Tool calls emitted but no registry configured;
                # surface them but don't execute.
                break

            # Execute all tool calls in this round in parallel.
            tool_call_dicts = [
                {"name": tc.name, "arguments": tc.arguments, "tool_call_id": tc.id}
                for tc in tool_calls_this_round
            ]
            if span is not None:
                for tc in tool_calls_this_round:
                    span.events.append(
                        _SpanEvent(
                            name="tool.execute.start",
                            ts=time.time(),
                            attributes={
                                "tool_name": tc.name,
                                "arguments": tc.arguments,
                            },
                        )
                    )
            results = await self.config.tool_registry.execute_batch(
                tool_call_dicts,
                granted_permissions=granted,
            )
            for tc, result in zip(tool_calls_this_round, results, strict=True):
                if span is not None:
                    span.events.append(
                        _SpanEvent(
                            name="tool.execute.end",
                            ts=time.time(),
                            attributes={
                                "tool_call_id": tc.id,
                                "tool_name": tc.name,
                                "output": result.output,
                                "is_error": result.is_error,
                                "error": result.error,
                            },
                        )
                    )
                messages.append(
                    _TR(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text=result.output)],
                        is_error=result.is_error,
                    )
                )
        # Final outputs
        if accumulated:
            self.outs["text"].emit(
                TypedValue(type="text", payload=accumulated)
            )
        if accumulated_thinking:
            self.outs["thinking"].emit(
                TypedValue(type="text", payload=accumulated_thinking)
            )
        if all_tool_calls:
            self.outs["tool_calls"].emit(
                TypedValue(
                    type="json",
                    payload=[
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in all_tool_calls
                    ],
                )
            )
        if total_usage.input or total_usage.output:
            self.outs["usage"].emit(
                TypedValue(
                    type="json",
                    payload={
                        "input": total_usage.input,
                        "output": total_usage.output,
                        "cache_read": total_usage.cache_read,
                        "cache_write": total_usage.cache_write,
                        "total": total_usage.total,
                    },
                )
            )

        # Update cumulative token/cost for the Director to read.
        self.total_tokens += total_usage.total
        # Cost = sum of input_cost + output_cost. catalog defines
        # these in $/million tokens; convert to dollars.
        cost = (total_usage.input * self.model.cost.input
                + total_usage.output * self.model.cost.output) / 1_000_000
        self.total_cost += cost

    async def _stream_one_round(
        self,
        context: Context,
        options: StreamOptions,
        span: Any,
    ) -> tuple[str, str, list[ToolCall], Usage, str | None, str]:
        """Stream one LLM round. Returns (text_delta, think_delta,
        tool_calls, usage, error_message, final_text).

        `final_text` is the final text from the done event's
        AssistantMessage (may differ from the accumulated
        `text_delta` if the provider coalesces deltas).
        """
        assert self.provider is not None
        assert self.model is not None
        text_delta = ""
        think_delta = ""
        tool_calls: list[ToolCall] = []
        usage = Usage()
        error_msg: str | None = None
        final_text = ""

        try:
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
                    text_delta += ev.delta
                    # Live-stream: emit a partial token to `draft`
                    self.outs["draft"].emit(
                        TypedValue(type="text", payload=text_delta)
                    )
                elif ev.type == "thinking_delta" and ev.delta:
                    think_delta += ev.delta
                elif ev.type == "toolcall_end" and ev.tool_call is not None:
                    tool_calls.append(ev.tool_call)
                elif ev.type == "done" and ev.message is not None:
                    if ev.message.usage.input or ev.message.usage.output:
                        usage = ev.message.usage
                    for block in ev.message.content:
                        if isinstance(block, TextContent) and not final_text:
                            final_text = block.text
                elif ev.type == "error":
                    error_msg = ev.error
                    break
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

        return text_delta, think_delta, tool_calls, usage, error_msg, final_text


__all__ = ["LLMAgent", "LLMEntityConfig"]
