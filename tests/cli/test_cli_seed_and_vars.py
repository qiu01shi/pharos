"""Tests for --input-extra (multi-port seed) and --var (template substitution)."""
from __future__ import annotations

import pytest

from pharos.cli import _seed_inputs, _substitute_vars


class TestSeedInputs:
    def test_empty_seeds_nothing(self):
        from pharos.core.graph import CompositeGraph

        g = CompositeGraph("g")
        _seed_inputs(g, {})  # Should not raise

    def test_single_port(self, tmp_path):
        from pharos.core.graph import CompositeGraph

        g = CompositeGraph("g")
        # Build a target with input port
        from pharos.core.entity import Entity, entity
        from pharos.core.port import InputPort

        @entity
        class _Sink(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                pass

        g.add_entity("sink", _Sink("sink"))
        from pharos.core.graph import Edge

        g.edges.append(Edge("__in__", "x", "sink", "x"))
        _seed_inputs(g, {"x": "hello"})
        sink = g.node("sink").instance
        assert sink is not None
        assert sink.ins["x"].peek_all()[0].value.payload == "hello"

    def test_multi_port(self, tmp_path):
        from pharos.core.entity import Entity, entity
        from pharos.core.graph import CompositeGraph
        from pharos.core.port import InputPort

        @entity
        class _Sink(Entity):
            ins = {
                "text": InputPort(name="text", accepted_types=["text"]),
                "msg": InputPort(name="msg", accepted_types=["text"]),
            }

            async def fire(self, ctx):  # type: ignore[override]
                pass

        g = CompositeGraph("g")
        g.add_entity("sink", _Sink("sink"))
        from pharos.core.graph import Edge

        g.edges.append(Edge("__in__", "text", "sink", "text"))
        g.edges.append(Edge("__in__", "msg", "sink", "msg"))
        _seed_inputs(g, {"text": "alpha", "msg": "beta"})
        sink = g.node("sink").instance
        assert sink is not None
        ins = sink.ins
        assert ins["text"].peek_all()[0].value.payload == "alpha"
        assert ins["msg"].peek_all()[0].value.payload == "beta"

    def test_unknown_port_ignored(self):
        from pharos.core.entity import Entity, entity
        from pharos.core.graph import CompositeGraph
        from pharos.core.port import InputPort

        @entity
        class _Sink(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                pass

        g = CompositeGraph("g")
        g.add_entity("sink", _Sink("sink"))
        from pharos.core.graph import Edge

        g.edges.append(Edge("__in__", "x", "sink", "x"))
        # "unknown" is in seed_map but no edge originates from
        # __in__.unknown → should be silently ignored.
        _seed_inputs(g, {"x": "hi", "unknown": "skip"})
        sink = g.node("sink").instance
        assert sink is not None
        assert sink.ins["x"].peek_all()[0].value.payload == "hi"


class TestSubstituteVars:
    def test_no_vars_passthrough(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text('name: g\nmodel: "no-vars"\n')
        out = _substitute_vars(f, [])
        assert out == f.read_text()

    def test_simple_var(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text('model: "${name}"\n')
        out = _substitute_vars(f, ["name=claude"])
        assert 'model: "claude"' in out

    def test_multiple_vars(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text('a: "${x}"\nb: "${y}"\n')
        out = _substitute_vars(f, ["x=1", "y=2"])
        assert '"1"' in out
        assert '"2"' in out

    def test_unknown_var_raises(self, tmp_path):
        # When user passes at least one --var, missing references
        # should fail fast.
        f = tmp_path / "g.yaml"
        f.write_text('model: "${missing}"\n')
        with pytest.raises(ValueError, match=r"missing"):
            _substitute_vars(f, ["unrelated=foo"])

    def test_ignores_comment_vars(self, tmp_path):
        """Comments should not be substituted."""
        f = tmp_path / "g.yaml"
        # The comment contains ${comment_var} but it's outside quotes
        f.write_text('# uses ${comment_var}\nmodel: "${real_var}"\n')
        out = _substitute_vars(f, ["real_var=foo"])
        # Comment unchanged
        assert "uses ${comment_var}" in out
        # Real var replaced
        assert '"foo"' in out

    def test_ignores_unquoted_var(self, tmp_path):
        """${...} not inside a quoted string is left alone."""
        f = tmp_path / "g.yaml"
        f.write_text("model: ${bare}\nquoted: \"${in_quotes}\"\n")
        out = _substitute_vars(f, ["bare=ignored", "in_quotes=ok"])
        # Unquoted ${bare} stays
        assert "model: ${bare}" in out
        # Quoted replaced
        assert '"ok"' in out

    def test_invalid_var_format_raises(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text("model: ok\n")
        with pytest.raises(ValueError, match=r"Invalid --var"):
            _substitute_vars(f, ["no_equals_sign"])