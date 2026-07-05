"""Simple in-process event bus.

Use case: emit high-level business events (e.g. "agent.completed",
"run.failed") that dashboards, monitors, or other agents can subscribe to.

Not for inter-process pub/sub — use a message broker for that.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

# Handler may be sync or async
Handler = Callable[[dict[str, Any]], Awaitable[None] | None]


class EventBus:
    """In-process pub/sub for business events.

    Example:
        bus = EventBus()
        @bus.subscribe("agent.completed")
        async def on_done(payload):
            print(payload)
        await bus.publish("agent.completed", {"agent": "echo"})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(self, topic: str) -> Callable[[Handler], Handler]:
        """Decorator: register a handler for a topic."""
        def decorator(fn: Handler) -> Handler:
            self._handlers[topic].append(fn)
            return fn
        return decorator

    async def publish(self, topic: str, payload: dict[str, Any] | None = None) -> None:
        """Fire all handlers for a topic. Errors in one handler do not stop others."""
        payload = payload or {}
        for handler in list(self._handlers.get(topic, [])):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # Swallow — we never want one bad handler to kill the run.
                # Real backends would log here; for now the in-process
                # bus is for ephemeral signaling only.
                pass

    def subscriber_count(self, topic: str) -> int:
        return len(self._handlers.get(topic, []))


__all__ = ["EventBus", "Handler"]
