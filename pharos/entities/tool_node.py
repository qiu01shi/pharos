"""ToolEntity — a single tool exposed as a first-class graph node.

Historically a tool could only run *inside* an ``LLMAgent``'s tool-call
loop, invisible to the Director: it couldn't be scheduled as a node, its
permission was checked only inside ``ToolRegistry`` (not by the Director),
and it appeared in the trace as an event rather than a node.

``ToolEntity`` promotes one registered tool to a graph node so it can:
  * be wired with typed ports and scheduled by any Director;
  * declare ``required_permissions`` so the Director enforces RBAC at the
    graph level (same code path as every other entity);
  * be traced and replayed like any other entity.

Ports:
    ins:
        args:   JsonPort — the tool arguments (a JSON object), OR
        input:  TextPort — convenience single-string arg (mapped to the
                first required string parameter, else ``"input"``)
    outs:
        output: TextPort — the tool's string result
        error:  TextPort — populated instead of ``output`` on failure
"""

from __future__ import annotations

from typing import Any

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.entities.tools import ToolRegistry


@entity
class ToolEntity(Entity):
    """Wrap a single registered tool as a schedulable graph node."""

    ins = {
        "args": InputPort(name="args", accepted_types=["json"]),
        "input": InputPort(name="input", accepted_types=["text"]),
    }
    outs = {
        "output": OutputPort(name="output", accepted_types=["text"]),
        "error": OutputPort(name="error", accepted_types=["text"]),
    }

    def __init__(
        self,
        node_id: str,
        tool_name: str,
        registry: ToolRegistry,
        arg_key: str | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.tool_name = tool_name
        self.registry = registry
        # Which parameter the `input` text port feeds when `args` is absent.
        self.arg_key = arg_key or self._default_arg_key()
        # Declare the tool's permission on the instance so the Director's
        # RBAC enforces it at the graph level (not just the registry).
        required = registry.get(tool_name).get("required_permission")
        self.required_permissions = {required} if required else set()

    def _default_arg_key(self) -> str:
        params = self.registry.get(self.tool_name).get("parameters", {})
        required = params.get("required") or []
        if required:
            return str(required[0])
        props = params.get("properties") or {}
        return next(iter(props), "input")

    async def fire(self, ctx) -> None:  # type: ignore[override]
        args: dict[str, Any] = {}
        arg_tokens = self.ins["args"].consume()
        if arg_tokens:
            payload = arg_tokens[-1].value.payload
            if isinstance(payload, dict):
                args = dict(payload)
        else:
            text_tokens = self.ins["input"].consume()
            if text_tokens:
                args = {
                    self.arg_key: "".join(t.value.payload for t in text_tokens)
                }
        if not args and not arg_tokens:
            return

        granted = getattr(ctx, "granted_permissions", set()) or set()
        result = await self.registry.execute(
            self.tool_name, args, granted_permissions=granted
        )
        if result.is_error:
            self.outs["error"].emit(
                TypedValue(
                    type="text",
                    payload=result.error or result.output or "tool error",
                )
            )
        else:
            self.outs["output"].emit(
                TypedValue(type="text", payload=result.output)
            )


__all__ = ["ToolEntity"]
