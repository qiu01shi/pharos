"""Minimal JSON Schema validator — the "types cross LLM boundaries" contract.

pharos ports already gate the coarse *type tag* (text / json / int). This module
adds *shape* validation for `json` payloads so a port can declare, e.g.,
``{"type": "object", "required": ["file", "line"], ...}`` and reject an LLM
that emits ``{"path": ..., "ln": ...}`` — exactly the promise made in
``docs/architecture.md``.

It is intentionally a small, self-contained subset (no new dependency): the
keywords LLM structured output actually needs — ``type`` (object / array /
string / number / integer / boolean / null, or a list of them), ``properties``,
``required``, ``items``, and ``enum``. Unknown keywords are ignored rather than
rejected, so a fuller JSON Schema still validates its supported parts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# type name -> predicate. `integer`/`number` exclude bool because in Python
# `bool` is an `int` subclass and an LLM emitting `true` for an integer field
# should fail the contract.
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Validate ``payload`` against ``schema``.

    Returns a list of human-readable error messages; an empty list means the
    payload conforms. Never raises for a malformed schema — unsupported
    keywords are simply skipped.
    """
    errors: list[str] = []
    _validate(payload, schema, "$", errors)
    return errors


def _validate(
    value: Any, schema: Any, path: str, errors: list[str]
) -> None:
    if not isinstance(schema, dict):
        return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")

    declared = schema.get("type")
    if declared is not None:
        allowed = declared if isinstance(declared, list) else [declared]
        if not any(_TYPE_CHECKS.get(t, _always_ok)(value) for t in allowed):
            errors.append(
                f"{path}: expected type {declared!r}, got "
                f"{type(value).__name__}"
            )
            # A wrong type makes nested checks meaningless.
            return

    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], subschema, f"{path}.{key}", errors)

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]", errors)


def _always_ok(_value: Any) -> bool:
    return True


__all__ = ["validate"]
