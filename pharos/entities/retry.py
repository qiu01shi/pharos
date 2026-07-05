"""RetryEntity — re-fire any inner entity on failure, with backoff.

LLM calls, shell commands, and tool executions are the flaky parts of a
workflow: a provider 500s, a network blips, a subprocess is killed. Rather
than teach every entity how to retry, we wrap one in ``RetryEntity`` — the
same "wrap an entity" pattern used by ``SubgraphEntity``. The Director
schedules the wrapper like any other node, so permission checks, tracing and
record/replay in ``safe_fire`` apply to the wrapper as a whole.

Semantics:
  * ``fire()`` snapshots this node's inputs once, then runs the inner entity
    up to ``max_attempts`` times, re-seeding the same inputs each attempt.
  * An attempt "fails" when ``inner.fire()`` raises. On failure we wait
    ``backoff_s`` seconds (if any attempts remain) and try again.
  * On success the inner entity's output tokens are moved out unchanged
    (via ``receive`` so cost/lineage survive). On exhaustion the last
    exception is re-raised, so the Director marks the run failed.

Retries re-run side effects (a shell command runs again). That is inherent
to retrying; use it where the operation is idempotent or the cost of a
duplicate is acceptable.

Wire it in YAML by adding a ``retry`` block to any node:

    - id: flaky
      type: llm
      provider: minimax
      model: MiniMax-Text-01
      retry: { max_attempts: 3, backoff_s: 0.5 }
"""

from __future__ import annotations

import asyncio

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort


@entity
class RetryEntity(Entity):
    """Wrap an entity so its ``fire()`` is retried on exception."""

    def __init__(
        self,
        node_id: str,
        inner: Entity,
        max_attempts: int = 3,
        backoff_s: float = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.inner = inner
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        # Observability: how many attempts the last fire() consumed.
        self.attempts_used = 0
        # Roll-up metrics the Director reads, mirroring inner if present.
        self.total_tokens = 0
        self.total_cost = 0.0

        # Mirror the inner entity's ports so edges connect to this wrapper.
        self.ins = {
            name: InputPort(
                name=p.name,
                accepted_types=list(p.accepted_types),
                capacity=p.capacity,
                overflow=p.overflow,
            )
            for name, p in inner.ins.items()
        }
        self.outs = {
            name: OutputPort(
                name=p.name,
                accepted_types=list(p.accepted_types),
                capacity=p.capacity,
                overflow=p.overflow,
            )
            for name, p in inner.outs.items()
        }
        # Enforce the inner entity's permissions at the graph level.
        self.required_permissions = set(
            getattr(inner, "required_permissions", set()) or set()
        )

    async def setup(self, ctx) -> None:
        await self.inner.setup(ctx)

    async def teardown(self) -> None:
        await self.inner.teardown()

    async def fire(self, ctx) -> None:
        # Snapshot inputs once so every attempt sees the same data.
        snapshot = {name: port.consume() for name, port in self.ins.items()}

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            # Reset inner ports, then re-seed this attempt's inputs.
            for ip in self.inner.ins.values():
                ip.consume()
            for op in self.inner.outs.values():
                op.consume()
            for name, tokens in snapshot.items():
                if name in self.inner.ins:
                    for tok in tokens:
                        self.inner.ins[name].receive(tok)

            try:
                await self.inner.fire(ctx)
            except Exception as exc:  # retry any failure
                last_exc = exc
                self.attempts_used = attempt + 1
                if attempt + 1 < self.max_attempts and self.backoff_s > 0:
                    await asyncio.sleep(self.backoff_s)
                continue

            # Success: move inner outputs out unchanged (preserve cost/lineage).
            for name, op in self.inner.outs.items():
                if name in self.outs:
                    for tok in op.consume():
                        self.outs[name].receive(tok)
            self.attempts_used = attempt + 1
            self.total_tokens = getattr(self.inner, "total_tokens", 0)
            self.total_cost = getattr(self.inner, "total_cost", 0.0)
            return

        assert last_exc is not None
        raise last_exc


__all__ = ["RetryEntity"]
