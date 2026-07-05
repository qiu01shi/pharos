"""Tests for pharos.testing.fixture (Assertion + Fixture JSON roundtrip)."""
from __future__ import annotations

from pharos.testing.fixture import Assertion, Fixture

GRAPH = "graphs/03_faux_demo.yaml"


class TestAssertion:
    def test_equals(self):
        ok, _ = Assertion("x", "equals", "hi").check(["hi"])
        assert ok
        ok, _ = Assertion("x", "equals", "hi").check(["bye"])
        assert not ok

    def test_contains_text(self):
        ok, _ = Assertion("x", "contains", "ell").check(["hello"])
        assert ok

    def test_contains_list_of_tool_calls(self):
        # tool_calls payload is a list of dicts; contains checks membership.
        ok, _ = Assertion("x", "contains", {"name": "read"}).check(
            [[{"name": "read"}, {"name": "edit"}]]
        )
        assert ok

    def test_not_contains(self):
        ok, _ = Assertion("x", "not_contains", "cannot").check(["done"])
        assert ok
        ok, _ = Assertion("x", "not_contains", "cannot").check(["I cannot do that"])
        assert not ok

    def test_regex(self):
        ok, _ = Assertion("x", "regex", r"\d{2}").check(["line 42"])
        assert ok

    def test_schema_ok(self):
        a = Assertion("x", "schema", {"type": "object", "required": ["line"]})
        ok, _ = a.check([{"line": 42}])
        assert ok

    def test_schema_fail(self):
        a = Assertion("x", "schema", {"type": "object", "required": ["line"]})
        ok, detail = a.check([{"nope": 1}])
        assert not ok
        assert "line" in detail

    def test_missing_target_fails_except_not_contains(self):
        assert Assertion("x", "equals", "v").check([])[0] is False
        assert Assertion("x", "not_contains", "v").check([])[0] is True


class TestFixtureRoundtrip:
    def test_save_load(self, tmp_path):
        fx = Fixture.build(
            name="t",
            graph_path=GRAPH,
            director="fn",
            outputs={"a:0": [{"port": "y", "type": "text", "payload": "hi"}]},
            seed={"prompt": "x"},
            grant=["fs:read"],
            assertions=[Assertion("a.y", "contains", "hi")],
        )
        p = tmp_path / "t.fixture.json"
        fx.save(p)
        loaded = Fixture.load(p)
        assert loaded.name == "t"
        assert loaded.chain_digest == fx.chain_digest
        assert loaded.director == "fn"
        assert loaded.seed == {"prompt": "x"}
        assert loaded.assertions[0].target == "a.y"
        # The graph file is unchanged, so the integrity pin matches.
        assert loaded.graph_matches()

    def test_graph_change_detected(self, tmp_path):
        fx = Fixture.build(
            name="t", graph_path=GRAPH, director="fn", outputs={}
        )
        fx.graph_sha256 = "0" * 64  # simulate the graph having changed
        assert fx.graph_matches() is False
