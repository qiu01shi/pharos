"""Tests for `type: python` IR loader."""

from __future__ import annotations

import uuid

import pytest

from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.ir import PythonNodeSpec, _build_python_entity


class TestPythonNodeSpec:
    def test_class_alias_works(self):
        spec = PythonNodeSpec.model_validate(
            {"id": "x", "type": "python", "class": "pkg.mod:Foo"}
        )
        assert spec.class_ == "pkg.mod:Foo"
        assert spec.id == "x"
        assert spec.params == {}

    def test_params_default(self):
        spec = PythonNodeSpec.model_validate(
            {
                "id": "x",
                "type": "python",
                "class": "pkg.mod:Foo",
                "params": {"a": 1},
            }
        )
        assert spec.params == {"a": 1}

    def test_missing_class_raises(self):
        with pytest.raises(ValueError):
            PythonNodeSpec.model_validate({"id": "x", "type": "python"})

    def test_malformed_class_raises(self):
        with pytest.raises(ValueError, match=r"module\.path:ClassName"):
            PythonNodeSpec.model_validate(
                {"id": "x", "type": "python", "class": "no_colon"}
            )


class TestPythonEntityLoading:
    def test_load_word_counter(self):
        # The test's own user-entities module
        spec = PythonNodeSpec.model_validate(
            {
                "id": "wc",
                "type": "python",
                "class": "tests._user_entities:WordCounter",
            }
        )
        ent = _build_python_entity("wc", spec)
        assert ent.node_id == "wc"
        assert ent.outs["count"].accepted_types == ["int"]

    def test_load_with_params(self):
        spec = PythonNodeSpec.model_validate(
            {
                "id": "pa",
                "type": "python",
                "class": "tests._user_entities:PrefixAdder",
                "params": {"prefix": "!! "},
            }
        )
        ent = _build_python_entity("pa", spec)
        assert ent.node_id == "pa"

    def test_unknown_module_raises(self):
        spec = PythonNodeSpec.model_validate(
            {"id": "x", "type": "python", "class": "no.such.module:Foo"}
        )
        with pytest.raises(ValueError, match="cannot import"):
            _build_python_entity("x", spec)

    def test_unknown_class_raises(self):
        spec = PythonNodeSpec.model_validate(
            {
                "id": "x",
                "type": "python",
                "class": "tests._user_entities:NoSuchClass",
            }
        )
        with pytest.raises(ValueError, match="no class"):
            _build_python_entity("x", spec)

    def test_non_entity_class_raises(self):
        # Pick something that exists but isn't an Entity
        spec = PythonNodeSpec.model_validate(
            {
                "id": "x",
                "type": "python",
                "class": "pharos.ir:GraphSpec",
            }
        )
        with pytest.raises(ValueError, match="not an Entity"):
            _build_python_entity("x", spec)


class TestPythonEntityEndToEnd:
    async def test_word_counter_via_load_graph(self):
        from pharos.ir import load_graph

        g, _raw = load_graph(
            "graphs/05_python_word_counter.yaml"
        )
        node = g.node("wc").instance
        from pharos.core.token import TypedValue

        node.ins["text"].emit(TypedValue(type="text", payload="hello world foo"))
        result = await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        # After deliver_upstream, the count token moved to __out__.count
        # (virtual node, collected on graph.collected).
        collected = getattr(g, "collected", {}).get("__out__", {})
        counts = [t.value.payload for t in collected.get("count", [])]
        assert counts == [3]
        assert result.converged is True

    async def test_prefix_adder_with_params(self):
        from pharos.ir import load_graph

        g, _ = load_graph("graphs/05b_python_prefix.yaml")
        node = g.node("pa").instance
        from pharos.core.token import TypedValue

        node.ins["in"].emit(TypedValue(type="text", payload="hi"))
        await FNDirector().run(g, RunContext(run_id=str(uuid.uuid4())))
        collected = getattr(g, "collected", {}).get("__out__", {})
        out = [t.value.payload for t in collected.get("result", [])]
        assert out == [">>> hi"]