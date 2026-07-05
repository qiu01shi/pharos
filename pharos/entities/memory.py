"""Memory Entity — shared key-value + (optional) vector store.

Ports:
    ins:
        read_key:   TextPort  — key to look up; emits to `value` port
        write_key:  TextPort  — key to write
        write_val:  TextPort  — value to associate with the most recent write_key
        clear:      TextPort  — set payload to "all" to wipe, otherwise wipe just that key
    outs:
        value:      TextPort (or JsonPort if value is dict)
        stored:     TextPort  — JSON {key: count, ...} of current contents
        missing:    TextPort  — emitted when a read_key was not found
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue


@dataclass
class _MemoryState:
    """In-memory KV store. Shared across all instances of this Entity class
    by default (so multiple Memory nodes see the same dict).

    Override with a class-level override or pass `store=...` to
    the constructor to use a per-instance dict.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, k: str) -> Any | None:
        return self.data.get(k)

    def set(self, k: str, v: Any) -> None:
        self.data[k] = v

    def delete(self, k: str) -> None:
        self.data.pop(k, None)

    def clear(self) -> None:
        self.data.clear()

    def keys(self) -> list[str]:
        return list(self.data.keys())


# Global default store; tests can replace this
_DEFAULT_STORE = _MemoryState()


@entity
class Memory(Entity):
    """A simple key-value memory shared across all Memory instances."""

    ins = {
        "read_key": InputPort(name="read_key", accepted_types=["text"]),
        "write_key": InputPort(name="write_key", accepted_types=["text"]),
        "write_val": InputPort(name="write_val", accepted_types=["text"]),
        "clear": InputPort(name="clear", accepted_types=["text"]),
    }
    outs = {
        "value": OutputPort(name="value", accepted_types=["text"]),
        "stored": OutputPort(name="stored", accepted_types=["json"]),
        "missing": OutputPort(name="missing", accepted_types=["text"]),
    }

    def __init__(
        self,
        node_id: str,
        store: _MemoryState | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self._store = store if store is not None else _DEFAULT_STORE

    async def fire(self, ctx) -> None:  # type: ignore[override]
        # Order: write → clear → read. This matches the "execute
        # side effects, then read" mental model and means a clear
        # performed in the same fire() can wipe a previous write.
        write_keys = self.ins["write_key"].consume()
        write_vals = self.ins["write_val"].consume()
        # Pair them up (most recent write_val goes to the last write_key)
        if write_vals:
            val = write_vals[-1].value.payload
            # Try to parse as JSON, fall back to string
            try:
                parsed = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                parsed = val
            for kt in write_keys:
                self._store.set(kt.value.payload, parsed)

        # Clear (after writes so a clear can wipe a prior write)
        clears = self.ins["clear"].consume()
        for t in clears:
            payload = t.value.payload
            if payload == "all":
                self._store.clear()
            elif isinstance(payload, str) and payload:
                self._store.delete(payload)

        # Read
        for kt in self.ins["read_key"].consume():
            key = kt.value.payload
            v = self._store.get(key)
            if v is None:
                self.outs["missing"].emit(  # type: ignore[attr-defined]
                    TypedValue(type="text", payload=key)
                )
            else:
                payload = v if isinstance(v, str) else json.dumps(v)
                self.outs["value"].emit(  # type: ignore[attr-defined]
                    TypedValue(type="text", payload=payload)
                )

        # Always emit the current `stored` summary
        self.outs["stored"].emit(  # type: ignore[attr-defined]
            TypedValue(type="json", payload=dict(self._store.data))
        )


__all__ = ["Memory", "_MemoryState"]
