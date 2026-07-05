"""LLMProvider protocol — the unified interface every provider implements.

Design notes:
- `stream()` is the primary path; `complete()` is derived (stream + collect).
- Errors terminate the stream via `StreamEvent(type="error", error=...)`,
  they are never raised mid-stream. This matches pi-ai's contract.
- `close()` releases any long-lived resources (HTTP clients, etc).
- `name` is a stable identifier used by config/registry code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pharos.llm.types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    StreamOptions,
)


def collect_stream(
    events: AsyncIterator[StreamEvent],
) -> tuple[AssistantMessage | None, list[StreamEvent]]:
    """Drain a stream to completion. Returns the final message and all events.

    Helper for `complete()`. Collects ALL events so callers can also
    forward them to a tracer.
    """
    import asyncio

    async def _drain() -> tuple[AssistantMessage | None, list[StreamEvent]]:
        collected: list[StreamEvent] = []
        final: AssistantMessage | None = None
        async for event in events:
            collected.append(event)
            if event.type == "done" and event.message is not None:
                final = event.message
                break
            if event.type == "error":
                if event.partial is not None:
                    final = event.partial
                break
        return final, collected

    return asyncio.run(_drain())


async def acomplete_from_stream(
    events: AsyncIterator[StreamEvent],
) -> AssistantMessage:
    """Async version of collect_stream that returns just the final message.

    Raises RuntimeError if the stream ends without a `done` event.
    Always cleans up the generator via aclose() to ensure finally blocks
    in provider implementations run (e.g. FauxProvider records calls).
    """
    final: AssistantMessage | None = None
    try:
        async for event in events:
            if event.type == "done" and event.message is not None:
                final = event.message
                break
            if event.type == "error":
                if event.partial is not None:
                    final = event.partial
                break
    finally:
        # Critical: aclose() runs the generator's finally block, so
        # providers that record state (e.g. FauxProvider) get a chance
        # to finalize. Without this, `break` in async for skips the
        # finally block (asyncio generator semantics).
        await events.aclose()
    if final is None:
        raise RuntimeError("stream ended without done event")
    return final


@runtime_checkable
class LLMProvider(Protocol):
    """All LLM providers implement this protocol.

    Implementations:
    - pharos.llm.providers.faux.FauxProvider
    - pharos.llm.providers.openai.OpenAIProvider
    - pharos.llm.providers.anthropic.AnthropicProvider
    - pharos.llm.providers.glm.GLMProvider  (inherits OpenAI)
    - pharos.llm.providers.deepseek.DeepSeekProvider  (inherits OpenAI)
    - pharos.llm.providers.minimax.MiniMaxProvider  (inherits Anthropic)
    """

    name: str

    async def list_models(self) -> list[Model]:
        """Return all models this provider exposes."""
        ...

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion. Errors via `error` event, not exceptions.

        Implementations are async generators — declare as
            async def stream(self, model, context, options=None):
                yield StreamEvent(...)
        The `def` here in the protocol is intentional.
        """
        ...

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        """Non-streaming shortcut. Equivalent to stream + collect."""
        events = self.stream(model, context, options)
        return await acomplete_from_stream(events)

    async def close(self) -> None:
        """Release any long-lived resources (HTTP clients, etc)."""
        ...


__all__ = [
    "LLMProvider",
    "acomplete_from_stream",
    "collect_stream",
]
