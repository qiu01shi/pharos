"""Typed ports with strict schema validation and bounded buffers.

Design notes:
- `deque` (not list) so consume(n) is O(1) for any prefix size.
- Each port declares accepted `type` tags; mismatches raise
  PortContractViolation (the LLM-hallucination guard).
- `capacity` is optional but recommended; overflow strategies include
  backpressure (raise) and drop-oldest.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from pharos.core.token import Token, TypedValue

OverflowStrategy = Literal["backpressure", "drop_oldest", "drop_newest"]


class PortContractViolation(Exception):
    """Raised when a token's type doesn't match a port's accepted types.

    This is the primary defense against LLM hallucinations propagating
    downstream. LLM outputs are untyped; the receiving port enforces
    the contract.
    """


class PortCapacityExceeded(Exception):
    """Raised when a port is at capacity and the strategy is backpressure."""


@dataclass
class _PortBase:
    """Common buffer + validation logic for input and output ports."""

    name: str
    accepted_types: list[str] = field(default_factory=list)
    capacity: int | None = None
    overflow: OverflowStrategy = "backpressure"
    buffer: deque[Token] = field(default_factory=deque)
    total_received: int = 0
    total_emitted: int = 0
    total_dropped: int = 0

    def _check_type(self, value: TypedValue) -> None:
        if self.accepted_types and value.type not in self.accepted_types:
            raise PortContractViolation(
                f"port {self.name!r} accepts {self.accepted_types}, "
                f"got {value.type!r}"
            )

    def _enqueue(self, token: Token) -> None:
        if self.capacity is not None and len(self.buffer) >= self.capacity:
            if self.overflow == "backpressure":
                raise PortCapacityExceeded(
                    f"port {self.name!r} at capacity {self.capacity}"
                )
            if self.overflow == "drop_oldest" and self.buffer:
                self.buffer.popleft()
                self.total_dropped += 1
            elif self.overflow == "drop_newest":
                self.total_dropped += 1
                return
        self.buffer.append(token)

    def receive(self, token: Token) -> None:
        """Push a token into the port. Validates type and capacity."""
        self._check_type(token.value)
        self.total_received += 1
        self._enqueue(token)

    def emit(self, value: TypedValue) -> Token:
        """Create a Token from a TypedValue and push. Returns the token."""
        self._check_type(value)
        token = Token(value=value, origin=f"port:{self.name}", ts=time.time())
        self.total_emitted += 1
        self._enqueue(token)
        return token

    def consume(self, n: int | None = None) -> list[Token]:
        """Pop up to n tokens (default: all). Returns what was popped."""
        if n is None:
            out = list(self.buffer)
            self.buffer.clear()
        else:
            out = [self.buffer.popleft() for _ in range(min(n, len(self.buffer)))]
        return out

    def peek(self) -> Token | None:
        """Return the head token without removing it."""
        return self.buffer[0] if self.buffer else None

    def peek_all(self) -> list[Token]:
        """Return a snapshot of the buffer (does not pop)."""
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def __iter__(self) -> Iterator[Token]:
        # Iterate over a copy; consume() to drain.
        return iter(list(self.buffer))

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0


@dataclass
class InputPort(_PortBase):
    """Receives tokens from upstream entities (or from __in__)."""


@dataclass
class OutputPort(_PortBase):
    """Sends tokens to downstream entities (or to __out__)."""


# Aliases / re-exports for clarity
Port = _PortBase  # generic; prefer InputPort/OutputPort

__all__ = [
    "InputPort",
    "OutputPort",
    "OverflowStrategy",
    "Port",
    "PortCapacityExceeded",
    "PortContractViolation",
]
