"""Built-in tool pack: simple utilities the LLM can call out-of-the-box.

Each `register_X(reg)` adds one tool. Call `register_builtins(reg)` to
register the full default set.

Available tools:
    - echo: return the input string (trivial, for testing)
    - get_time: return the current ISO timestamp
    - add_numbers: sum a list of numbers (math example)
"""

from __future__ import annotations

import datetime

from pharos.entities.tools import ToolRegistry


def _register_echo(reg: ToolRegistry) -> None:
    reg.register(
        name="echo",
        description="Return the input string unchanged. Useful for testing.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"},
            },
            "required": ["text"],
        },
        fn=lambda text: text,
    )


def _register_get_time(reg: ToolRegistry) -> None:
    reg.register(
        name="get_time",
        description="Return the current UTC time as an ISO 8601 string.",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )


def _register_add_numbers(reg: ToolRegistry) -> None:
    reg.register(
        name="add_numbers",
        description="Add a list of numbers and return the sum.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numbers to sum",
                },
            },
            "required": ["numbers"],
        },
        fn=lambda numbers: str(sum(float(x) for x in numbers)),
    )


def register_builtins(reg: ToolRegistry) -> None:
    """Register the default built-in tool set on `reg`."""
    _register_echo(reg)
    _register_get_time(reg)
    _register_add_numbers(reg)


__all__ = ["register_builtins"]