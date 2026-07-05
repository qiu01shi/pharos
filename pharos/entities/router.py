"""Router Entity — conditional branching.

Reads one (or more) input tokens, evaluates a list of guard
functions, and emits the input value to the corresponding output
port. If no guard matches, emits to the `default` port (if any).

Ports:
    ins:
        in:   TextPort  — the value to route
    outs:
        route_a / route_b / route_c / ... — branch outputs
        default                          — fallback
"""

from __future__ import annotations

from collections.abc import Callable

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue


@entity
class Router(Entity):
    """Branch on input value via guard functions.

    Configure via:
        Router(guards={
            "route_a": lambda v: "error" in v.lower(),
            "route_b": lambda v: v.startswith("?"),
        })
    """

    ins = {"in": InputPort(name="in", accepted_types=["text"])}
    outs: dict[str, OutputPort] = {}  # dynamically populated

    def __init__(
        self,
        node_id: str,
        guards: dict[str, Callable[[str], bool]],
        default: str = "default",
    ) -> None:
        super().__init__(node_id=node_id)
        # Populate ins / outs dynamically based on the guards
        self._guards = guards
        self._default = default if default not in guards else "default"
        # Re-declare ports on this instance
        for name in [*list(guards.keys()), self._default]:
            if name not in self.outs:
                self.outs[name] = OutputPort(  # type: ignore[attr-defined]
                    name=name, accepted_types=["text"]
                )

    async def fire(self, ctx) -> None:  # type: ignore[override]
        toks = self.ins["in"].consume()
        for tok in toks:
            value = tok.value.payload
            if not isinstance(value, str):
                # Router currently only routes strings; pass through to default
                self.outs[self._default].emit(  # type: ignore[attr-defined]
                    TypedValue(type="text", payload=str(value))
                )
                continue
            routed = False
            for port_name, guard in self._guards.items():
                if guard(value):
                    self.outs[port_name].emit(  # type: ignore[attr-defined]
                        TypedValue(type="text", payload=value)
                    )
                    routed = True
                    break
            if not routed and self._default in self.outs:
                self.outs[self._default].emit(  # type: ignore[attr-defined]
                    TypedValue(type="text", payload=value)
                )


__all__ = ["Router"]
