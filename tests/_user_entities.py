"""User-defined pharos entities used in tests.

In a real project, this would live under your own package. We put
it under tests/ so pharos's own tests can exercise the
`type: python` loader path.
"""

from __future__ import annotations

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue


@entity
class WordCounter(Entity):
    """Counts whitespace-separated words in the input text."""

    ins = {"text": InputPort(name="text", accepted_types=["text"])}
    outs = {"count": OutputPort(name="count", accepted_types=["int"])}

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["text"].consume():
            n = len((t.value.payload or "").split())
            self.outs["count"].emit(TypedValue(type="int", payload=n))


@entity
class PrefixAdder(Entity):
    """Prepends a fixed prefix to the input. Configured via constructor kwargs."""

    ins = {"in": InputPort(name="in", accepted_types=["text"])}
    outs = {"out": OutputPort(name="out", accepted_types=["text"])}

    def __init__(self, node_id: str, prefix: str = ">>> ") -> None:
        super().__init__(node_id=node_id)
        self._prefix = prefix

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["in"].consume():
            self.outs["out"].emit(
                TypedValue(type="text", payload=self._prefix + t.value.payload)
            )


@entity
class Doubler(Entity):
    """Doubles any int it receives."""

    ins = {"in": InputPort(name="in", accepted_types=["int"])}
    outs = {"out": OutputPort(name="out", accepted_types=["int"])}

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["in"].consume():
            v = t.value.payload
            self.outs["out"].emit(TypedValue(type="int", payload=v * 2))