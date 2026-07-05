"""Type system helpers — Pydantic-backed schema validation for ports.

This module is a thin layer over Pydantic that ports and the IR layer
use to declare what kinds of values they accept. Concrete validation
happens at the port boundary (`Port.receive`); this module provides
utilities to build schemas and to assert compatibility between an
upstream output type and a downstream input type.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def accepts_schema(
    port_accepted_types: list[str], value_type: str
) -> bool:
    """True iff `value_type` is in the port's allowed types (or port is permissive)."""
    if not port_accepted_types:
        return True  # permissive: accepts any
    return value_type in port_accepted_types


def model_dump_safe(model: BaseModel | dict[str, Any] | None) -> dict[str, Any] | None:
    """Pydantic model → dict, robust to None / plain dict / nested models."""
    if model is None:
        return None
    if isinstance(model, BaseModel):
        return model.model_dump()
    if isinstance(model, dict):
        return dict(model)
    raise TypeError(f"cannot dump {type(model).__name__} as dict")


def schema_name(model: type[BaseModel] | None) -> str:
    """Stable name for a Pydantic model. None → ''."""
    if model is None:
        return ""
    return model.__name__


__all__ = ["accepts_schema", "model_dump_safe", "schema_name"]
