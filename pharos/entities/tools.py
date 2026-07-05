"""Tool Protocol + Registry — supports sync and async tools.

A Tool is a callable + JSON Schema. Tools are registered with the
registry and sent to the LLM via `Context.tools` as `Tool` objects.

Sync fn:  fn(**args) -> str
Async fn: async fn(**args) -> str

Permission:
    Tools can declare `required_permission`. At execute() time,
    the caller's `granted_permissions` must contain it, else
    `is_error=True` is returned.

Parallel execution:
    `execute_batch()` runs multiple tool calls concurrently
    via asyncio.gather. LLMAgent uses this when the LLM emits
    multiple tool_calls in one round.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pharos.llm.types import Tool


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

    Each tool is registered with:
        - name (str)
        - description (str)
        - parameters (JSON Schema dict)
        - fn (sync callable or async coroutine fn)
        - required_permission (optional str)

    `list_tools()` returns `list[Tool]` (pharos.llm.types.Tool),
    which is what providers expect in `Context.tools`.
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

    def list_tools(self) -> list[Tool]:
        """Return the tools as `Tool` objects for `Context.tools`."""
        return [
            Tool(
                name=name,
                description=info["description"],
                parameters=info["parameters"],
            )
            for name, info in self._tools.items()
        ]

    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
        granted_permissions: set[str] | None = None,
    ) -> ToolCallResult:
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
        if required and (
            granted_permissions is None or required not in granted_permissions
        ):
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=f"permission denied: requires {required!r}",
            )

        # Execute — detect async fn and await if needed
        fn = tool["fn"]
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**arguments)
            else:
                result = fn(**arguments)
            output = result if isinstance(result, str) else str(result)
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output=output,
            )
        except TypeError as e:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=f"bad arguments: {e}",
            )
        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                output="",
                is_error=True,
                error=f"{type(e).__name__}: {e}",
            )

    async def execute_batch(
        self,
        calls: list[dict[str, Any]],
        granted_permissions: set[str] | None = None,
    ) -> list[ToolCallResult]:
        """Execute multiple tool calls in parallel.

        Each call is a dict with keys: name, arguments, tool_call_id.
        """
        coros = [
            self.execute(
                name=c["name"],
                arguments=c.get("arguments", {}),
                tool_call_id=c.get("tool_call_id", ""),
                granted_permissions=granted_permissions,
            )
            for c in calls
        ]
        return await asyncio.gather(*coros)


__all__ = ["ToolCallResult", "ToolError", "ToolRegistry"]