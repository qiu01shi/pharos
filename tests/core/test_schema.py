"""Tests for the JSON Schema 2020-12 validator."""
from __future__ import annotations

from pharos.core.schema import validate


class TestScalarTypes:
    def test_string_ok_and_mismatch(self):
        assert validate("hi", {"type": "string"}) == []
        errs = validate(42, {"type": "string"})
        assert errs and "expected type 'string'" in errs[0]

    def test_integer_rejects_bool(self):
        # bool is an int subclass in Python; the contract must not accept it.
        assert validate(3, {"type": "integer"}) == []
        assert validate(True, {"type": "integer"})

    def test_number_accepts_int_and_float(self):
        assert validate(1, {"type": "number"}) == []
        assert validate(1.5, {"type": "number"}) == []

    def test_null_and_boolean(self):
        assert validate(None, {"type": "null"}) == []
        assert validate(True, {"type": "boolean"}) == []

    def test_union_type(self):
        schema = {"type": ["string", "null"]}
        assert validate("x", schema) == []
        assert validate(None, schema) == []
        assert validate(5, schema)


class TestObjects:
    schema = {
        "type": "object",
        "required": ["file", "line"],
        "properties": {
            "file": {"type": "string"},
            "line": {"type": "integer"},
        },
    }

    def test_valid_object(self):
        assert validate({"file": "a.py", "line": 10}, self.schema) == []

    def test_missing_required(self):
        errs = validate({"file": "a.py"}, self.schema)
        assert any("missing required property 'line'" in e for e in errs)

    def test_wrong_property_type(self):
        errs = validate({"file": "a.py", "line": "ten"}, self.schema)
        assert any("$.line" in e for e in errs)

    def test_non_object(self):
        errs = validate(["not", "an", "object"], self.schema)
        assert errs and "expected type 'object'" in errs[0]


class TestArraysAndEnums:
    def test_array_items(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        assert validate([1, 2, 3], schema) == []
        errs = validate([1, "two", 3], schema)
        assert any("$[1]" in e for e in errs)

    def test_enum(self):
        schema = {"enum": ["low", "high"]}
        assert validate("low", schema) == []
        assert validate("mid", schema)

    def test_standard_string_constraints_are_enforced(self):
        assert validate("x", {"type": "string", "minLength": 1}) == []
        assert validate("x", {"type": "string", "minLength": 2})

    def test_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        }
        assert validate({"ok": True}, schema) == []
        assert validate({"extra": True}, schema)
