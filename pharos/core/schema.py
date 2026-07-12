"""JSON Schema 2020-12 validation and conservative contract comparison."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def check_schema(
    schema: dict[str, Any], *, require_constraints: bool = False
) -> None:
    """Raise ``ValueError`` when ``schema`` is not valid Draft 2020-12.

    JSON Schema intentionally permits extension keywords, so a shorthand such
    as ``{file: str}`` is technically valid but constrains nothing.  Authoring
    entry points can set ``require_constraints`` to catch that common mistake.
    """
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        path = ".".join(str(part) for part in exc.absolute_schema_path)
        where = f" at {path}" if path else ""
        raise ValueError(f"invalid JSON Schema{where}: {exc.message}") from exc
    if require_constraints and not set(schema).intersection(
        {
            "$ref",
            "allOf",
            "anyOf",
            "const",
            "enum",
            "not",
            "oneOf",
            "properties",
            "required",
            "type",
        }
    ):
        raise ValueError(
            "JSON Schema declares no constraints; use type/properties/required "
            "instead of field-name shorthand"
        )


def validate(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Return deterministic, human-readable validation errors."""
    try:
        check_schema(schema)
    except ValueError as exc:
        return [f"$schema: {exc}"]

    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [_format_error(error) for error in errors]


def compatibility_errors(
    producer: dict[str, Any], consumer: dict[str, Any]
) -> list[str]:
    """Find compile-time incompatibilities between two JSON contracts.

    Full JSON Schema subsumption is expensive and sometimes undecidable.  The
    compiler therefore rejects only contradictions it can prove: disjoint root
    types, consumer-required object fields that the producer does not promise,
    and disjoint declared types for shared properties.
    """
    check_schema(producer)
    check_schema(consumer)
    errors: list[str] = []
    producer_types = _types(producer.get("type"))
    consumer_types = _types(consumer.get("type"))
    if producer_types and consumer_types and producer_types.isdisjoint(consumer_types):
        errors.append(
            f"producer types {sorted(producer_types)} do not satisfy "
            f"consumer types {sorted(consumer_types)}"
        )
        return errors

    if "object" in producer_types and "object" in consumer_types:
        promised = set(producer.get("required", []))
        required = set(consumer.get("required", []))
        missing = required - promised
        if missing:
            errors.append(f"producer does not require consumer fields {sorted(missing)}")

        producer_props = producer.get("properties", {})
        consumer_props = consumer.get("properties", {})
        for name in sorted(set(producer_props) & set(consumer_props)):
            left = _types(producer_props[name].get("type"))
            right = _types(consumer_props[name].get("type"))
            if left and right and left.isdisjoint(right):
                errors.append(
                    f"property {name!r} producer types {sorted(left)} do not "
                    f"satisfy consumer types {sorted(right)}"
                )
    return errors


def _types(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _json_path(parts: Iterable[Any]) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _format_error(error: Any) -> str:
    path = _json_path(error.absolute_path)
    if error.validator == "type":
        return f"{path}: expected type {error.validator_value!r}, got {type(error.instance).__name__}"
    if error.validator == "required":
        missing = str(error.message).split("'")[1]
        return f"{path}: missing required property {missing!r}"
    if error.validator == "enum":
        return f"{path}: {error.instance!r} is not one of {error.validator_value!r}"
    return f"{path}: {error.message}"


__all__ = ["check_schema", "compatibility_errors", "validate"]
