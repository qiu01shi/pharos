"""Tests for pharos.testing.diff (structured run diff)."""
from __future__ import annotations

from pharos.testing.diff import diff_json, diff_runs


class TestDiffRuns:
    def test_no_changes(self):
        a = {"n:0": [{"port": "y", "type": "text", "payload": "x"}]}
        rd = diff_runs(a, dict(a))
        assert rd.has_changes() is False
        assert rd.changed_nodes == set()

    def test_json_field_change(self):
        a = {
            "patch:0": [
                {"port": "json", "type": "json", "payload": {"line": 42, "file": "a.py"}}
            ]
        }
        b = {
            "patch:0": [
                {"port": "json", "type": "json", "payload": {"line": 43, "file": "a.py"}}
            ]
        }
        rd = diff_runs(a, b)
        assert rd.has_changes()
        assert "patch" in rd.changed_nodes
        pd = rd.port_diffs[0]
        assert pd.kind == "json"
        paths = {fc.path: (fc.kind, fc.before, fc.after) for fc in pd.field_changes}
        assert paths["$.line"] == ("changed", 42, 43)
        assert "$.file" not in paths  # unchanged field not reported

    def test_added_port(self):
        a = {"n:0": [{"port": "text", "type": "text", "payload": "hi"}]}
        b = {
            "n:0": [
                {"port": "text", "type": "text", "payload": "hi"},
                {"port": "error", "type": "text", "payload": "boom"},
            ]
        }
        rd = diff_runs(a, b)
        statuses = {(pd.port, pd.status) for pd in rd.port_diffs}
        assert ("error", "added") in statuses

    def test_removed_key(self):
        a = {
            "n:0": [{"port": "y", "type": "text", "payload": "hi"}],
            "extra:0": [{"port": "z", "type": "text", "payload": "gone"}],
        }
        b = {"n:0": [{"port": "y", "type": "text", "payload": "hi"}]}
        rd = diff_runs(a, b)
        statuses = {(pd.node_id, pd.port, pd.status) for pd in rd.port_diffs}
        assert ("extra", "z", "removed") in statuses

    def test_text_line_diff(self):
        a = {"n:0": [{"port": "text", "type": "text", "payload": "line1\nline2"}]}
        b = {"n:0": [{"port": "text", "type": "text", "payload": "line1\nCHANGED"}]}
        rd = diff_runs(a, b)
        pd = rd.port_diffs[0]
        assert pd.kind == "text"
        assert any("CHANGED" in ln for ln in pd.line_diff)

    def test_tool_calls_dropped_is_detected(self):
        # The motivating drift: a prompt edit makes coder stop calling `edit`.
        a = {
            "coder:0": [
                {
                    "port": "tool_calls",
                    "type": "json",
                    "payload": [{"name": "read"}, {"name": "edit"}],
                }
            ]
        }
        b = {
            "coder:0": [
                {"port": "tool_calls", "type": "json", "payload": [{"name": "read"}]}
            ]
        }
        rd = diff_runs(a, b)
        assert "coder" in rd.changed_nodes
        pd = rd.port_diffs[0]
        # The dropped second tool call shows up as a removed list element.
        assert any(fc.path == "$[1]" and fc.kind == "removed" for fc in pd.field_changes)

    def test_propagation_with_graph(self):
        from pharos.core.entity import Entity, entity
        from pharos.core.graph import CompositeGraph
        from pharos.core.port import InputPort, OutputPort

        @entity
        class _E(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                ...

        g = CompositeGraph(name="g")
        g.add_entity("coder", _E("coder"))
        g.add_entity("reviewer", _E("reviewer"))
        g.connect("__in__.p", "coder.x")
        g.connect("coder.y", "reviewer.x")
        g.connect("reviewer.y", "__out__.done")

        a = {"coder:0": [{"port": "y", "type": "text", "payload": "v1"}]}
        b = {"coder:0": [{"port": "y", "type": "text", "payload": "v2"}]}
        rd = diff_runs(a, b, graph=g)
        assert "reviewer" in rd.propagation["coder"]
        assert "__out__" in rd.propagation["coder"]

    def test_to_dict_roundtrips(self):
        a = {"n:0": [{"port": "y", "type": "json", "payload": {"k": 1}}]}
        b = {"n:0": [{"port": "y", "type": "json", "payload": {"k": 2}}]}
        d = diff_runs(a, b).to_dict()
        assert d["has_changes"] is True
        assert d["changed_nodes"] == ["n"]
        assert d["port_diffs"][0]["field_changes"][0]["path"] == "$.k"


class TestDiffJson:
    def test_nested_paths(self):
        before = {"a": {"b": [1, 2]}, "c": "keep"}
        after = {"a": {"b": [1, 3]}, "c": "keep", "d": "new"}
        changes = {fc.path: fc.kind for fc in diff_json(before, after)}
        assert changes["$.a.b[1]"] == "changed"
        assert changes["$.d"] == "added"
        assert "$.c" not in changes
