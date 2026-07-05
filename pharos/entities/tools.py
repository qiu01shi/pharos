"""Tool Protocol + Registry.

A Tool is a callable that takes a dict of arguments and returns a
string result. Tools are registered with a JSON Schema describing
their arguments, which is sent to the LLM as part of the context.

Usage:
    from pharos.entities.tools import ToolRegistry

    reg = ToolRegistry()
    reg.register(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
            },
            "required": ["city"],
        },
        fn=lambda city, unit="c": f"sunny in {city}, 72°{unit}",
    )

    # Or register built-in tools (see tools_builtins.py)
    from pharos.entities.tools_builtins import register_builtins
    register_builtins(reg)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """Raised when a tool call fails (bad args, exception, etc.)."""

    def __init__(self, name: str, message: str, is_recoverable: bool = True):
        super().__init__(f"{name}: {message}")
        self.name = name
        self.message = message
        self.is_recoverable = is_recoverable


@dataclass
class ToolCallResult:
    """Result of executing a tool call."""

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False
    error: str | None = None

    def to_message_dict(self) -> dict[str, Any]:
        """Render as a ToolResultMessage-shaped dict for serialization."""
        d: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.output,
        }
        if self.is_error:
            d["is_error"] = True
        if self.error:
            d["error"] = self.error
        return d


class ToolRegistry:
    """A collection of tools the LLM can call.

    Tools are registered via `register()` and looked up by name.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
        required_permission: str | None = None,
    ) -> None:
        """Register a tool callable.

        Args:
            name: Tool identifier (must match what the LLM emits).
            description: Human-readable; sent to the LLM.
            parameters: JSON Schema for arguments.
            fn: The callable. Must accept kwargs matching the schema.
            required_permission: If set, callers must have this
                permission to invoke the tool (checked at
                execute() time, not registration).
        """
        self._tools[name] = {
            "description": description,
            "parameters": parameters,
            "fn": fn,
            "required_permission": required_permission,
        }

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise ToolError(name, f"unknown tool: {name!r}")
        return self._tools[name]

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tools as a list of JSON-Schema-compatible dicts.

        This is what gets sent to the LLM via `Context.tools`.
        """
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for name, info in self._tools.items()
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
        granted_permissions: set[str] | None = None,
    ) -> ToolCallResult:
        """Execute a tool call.

        Catches all exceptions and returns them as `is_error=True`
        ToolCallResults so the LLM can react. Permission errors
        (raised before execution) come back as is_error=True too.
        """
        try:
            tool = self.get(name)
        except ToolError as e:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=str(e),
            )

        # Permission check
        required = tool.get("required_permission")
        if required and (granted_permissions is None or required not in granted_permissions):
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=f"permission denied: requires {required!r}",
            )

        # Execute
        try:
            result = tool["fn"](**arguments)
            # Tool output should be a string; coerce if not
            output = result if isinstance(result, str) else str(result)
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output=output,
            )
        except TypeError as e:
            # Bad arguments
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=f"bad arguments: {e}",
            )
        except Exception as e:
            # Tool crashed
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=f"{type(e).__name__}: {e}",
            )


__all__ = ["ToolCallResult", "ToolError", "ToolRegistry"]