"""HumanEntity — human-in-the-loop approval / input as a graph node.

Some workflows must pause for a person: approve a risky shell command, pick
between options, or supply a missing value. HumanEntity models that as an
ordinary node so it is scheduled, RBAC-gated (``human:input``), traced, and
recorded like any other entity — no Director changes required.

Resolution order for the answer:
  1. ``self.answer`` — set from YAML (``answer:``) or the CLI (``--answer
     node=value``). This is the automation / resume path.
  2. interactive ``input()`` — only if ``interactive=True`` and stdin is a TTY.
  3. otherwise the fire records a ``human.pending`` trace event and raises
     ``HumanInputRequired``. The run pauses with a partial recording; supply
     the answer and continue with ``pharos resume ... --answer node=value``.
"""

from __future__ import annotations

import sys

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.observability.trace import current_span


class HumanInputRequired(Exception):
    """Raised when a HumanEntity has no answer available yet (run pauses)."""


@entity
class HumanEntity(Entity):
    """Pause a run for human approval / input.

    Ports:
        ins:  prompt: TextPort    — the question shown to the human
        outs: response: TextPort  — the human's answer
    """

    required_permissions = {"human:input"}
    ins = {"prompt": InputPort(name="prompt", accepted_types=["text"])}
    outs = {"response": OutputPort(name="response", accepted_types=["text"])}

    def __init__(
        self,
        node_id: str,
        answer: str | None = None,
        interactive: bool = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.answer = answer
        self.interactive = interactive

    async def fire(self, ctx) -> None:
        question = "".join(
            str(t.value.payload) for t in self.ins["prompt"].consume()
        )
        answer = self.answer
        if answer is None and self.interactive and sys.stdin.isatty():
            answer = input(
                f"[pharos human:{self.node_id}] {question}\n> "
            )

        span = current_span()
        if answer is None:
            if span is not None:
                span.record_event(
                    "human.pending",
                    {"node_id": self.node_id, "question": question},
                )
            raise HumanInputRequired(
                f"{self.node_id}: awaiting human input"
                + (f" for {question!r}" if question else "")
            )

        if span is not None:
            span.record_event(
                "human.answered", {"node_id": self.node_id}
            )
        self.outs["response"].emit(TypedValue(type="text", payload=answer))


__all__ = ["HumanEntity", "HumanInputRequired"]
