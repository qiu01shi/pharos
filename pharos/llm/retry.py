"""Exponential backoff with jitter for transient LLM errors."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Run `fn` with exponential backoff + jitter.

    Args:
        fn: Zero-arg async callable.
        max_attempts: Total attempts including the first.
        base_delay: Initial sleep on first retry (seconds).
        max_delay: Cap on sleep duration.
        retryable_exceptions: Exceptions that trigger a retry. Anything
            else propagates immediately.
        on_retry: Optional callback (attempt_num, exception, sleep_seconds)
            invoked just before each sleep. Useful for logging.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 0.5 + random.random()  # 0.5x..1.5x jitter
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


__all__ = ["with_retry"]
