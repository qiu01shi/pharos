"""Entity — a typed actor with declared I/O.

An Entity is a node in the dataflow graph. It declares its input/output
ports at the class level and implements `fire()`. The Director calls
`fire()` when the entity is ready (all upstream data available).

Lifecycle:
    setup(ctx)  -> create long-lived resources (LLM clients, DB conns)
    fire(ctx)   -> do one unit of work; called potentially many times
    teardown()  -> release resources

Entities are intended to be reusable across many runs. They should not
hold per-run state. Use ports to communicate.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from pharos.core.port import InputPort, OutputPort


@dataclass(frozen=True)
class EntitySpec:
    """Static description of an entity's interface."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class Entity(ABC):
    """Base class for all pharos entities.

    Subclass and either:
      - Decorate with @entity and declare ports as class attributes, OR
      - Override _declare_ports() in __init__ (dynamic declaration)

    Then implement `async def fire(ctx)`.

    Permissions:
      Subclasses MAY declare `required_permissions` as a class
      attribute — a set of permission strings the entity needs to
      run. Examples: `{"shell:execute"}`, `{"fs:read"}`. The Director
      checks these against `RunContext.granted_permissions` before
      calling `setup()`; missing permissions raise PermissionError.
    """

    spec: ClassVar[EntitySpec]
    # Port *declarations* harvested by @entity live on the class; each
    # instance gets its own cloned buffers in `ins` / `outs` below.
    _declared_ins: ClassVar[dict[str, InputPort]] = {}
    _declared_outs: ClassVar[dict[str, OutputPort]] = {}
    # Per-instance ports (own buffers) and permission set. These are
    # instance attributes so subclasses may set them dynamically in
    # __init__ (e.g. Router, SubgraphEntity) without fighting mypy.
    ins: dict[str, InputPort]
    outs: dict[str, OutputPort]
    required_permissions: set[str] = set()
    node_id: str

    def __init__(self, node_id: str | None = None) -> None:
        self.node_id = node_id or self.__class__.__name__
        # Per-instance port dicts (so multiple instances don't share buffers).
        self.ins = {
            name: _clone_port(port)
            for name, port in type(self)._declared_ins.items()
        }
        self.outs = {
            name: _clone_port(port)
            for name, port in type(self)._declared_outs.items()
        }

    @abstractmethod
    async def fire(self, ctx: FireContext) -> None:
        """Do one unit of work. Subclasses must implement.

        Read from self.ins, write to self.outs. The Director is responsible
        for delivering upstream tokens to your input ports before calling
        fire(), and for draining your output ports to downstream after.
        """

    async def setup(self, ctx: RunContext) -> None:
        """Override to acquire long-lived resources (e.g. an LLM client)."""
        return None

    async def teardown(self) -> None:
        """Override to release resources acquired in setup()."""
        return None


_PortT = TypeVar("_PortT", InputPort, OutputPort)


def _clone_port(port: _PortT) -> _PortT:
    """Make a per-instance copy of a class-level port declaration.

    Each entity instance needs its own buffer. The class-level port
    holds the schema (accepted_types, capacity) but not the buffer.
    Generic over the two port kinds so callers keep the concrete type.
    """
    cls = type(port)
    return cls(
        name=port.name,
        accepted_types=list(port.accepted_types),
        capacity=port.capacity,
        overflow=port.overflow,
    )


def entity(cls: type[Entity]) -> type[Entity]:
    """Class decorator: collect declared ports, build EntitySpec.

    Two declaration styles are supported:

    1. Dict form (preferred):
        @entity
        class Echo(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    2. Bare-attribute form:
        @entity
        class Echo(Entity):
            x = InputPort(name="x", accepted_types=["text"])
            y = OutputPort(name="y", accepted_types=["text"])

    In style (1), the keys of the dict become the port names. The Port
    objects' own `name` attribute is for diagnostics only.
    """
    inputs: dict[str, InputPort] = {}
    outputs: dict[str, OutputPort] = {}
    for attr_name in dir(cls):
        if attr_name.startswith("_"):
            continue
        # The Entity base class declares `ins` and `outs` as ClassVars.
        # Skip them only if they are still the base class defaults
        # (empty dicts). If the subclass has set them to a dict of ports,
        # we need to harvest the ports below.
        if attr_name in ("ins", "outs"):
            value = cls.__dict__.get(attr_name)
            if isinstance(value, dict):
                for port_name, port in value.items():
                    if isinstance(port, InputPort):
                        inputs[port_name] = port
                    elif isinstance(port, OutputPort):
                        outputs[port_name] = port
            continue
        value = cls.__dict__.get(attr_name)
        if isinstance(value, InputPort):
            inputs[attr_name] = value
        elif isinstance(value, OutputPort):
            outputs[attr_name] = value
        elif isinstance(value, dict):
            # Style (1) for non-standard attribute names
            for port_name, port in value.items():
                if isinstance(port, InputPort):
                    inputs[port_name] = port
                elif isinstance(port, OutputPort):
                    outputs[port_name] = port

    cls._declared_ins = inputs
    cls._declared_outs = outputs
    cls.spec = EntitySpec(
        name=cls.__name__,
        inputs=tuple(inputs.keys()),
        outputs=tuple(outputs.keys()),
    )

    # Ensure fire() is async
    if not inspect.iscoroutinefunction(cls.__dict__.get("fire")):
        raise TypeError(
            f"{cls.__name__}.fire() must be defined as `async def fire(...)`"
        )

    return cls


# Forward reference types — concrete definitions live in pharos.runtime
class FireContext:  # placeholder; real type is in runtime
    """Per-fire call context (set by Director)."""

    run_id: str
    iter: int
    step_id: str


class RunContext:  # placeholder; real type is in runtime
    """Per-run context (set by Director before setup)."""

    run_id: str
    started_at: float
    config: dict[str, Any]


__all__ = ["Entity", "EntitySpec", "FireContext", "RunContext", "entity"]
