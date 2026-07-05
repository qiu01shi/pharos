"""Tests for the IR loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from pharos.ir import GraphSpec, LLMNodeSpec, ShellNodeSpec, load_graph

REPO = Path(__file__).resolve().parent.parent.parent


class TestLoader:
    def test_load_single_llm(self):
        g, _raw = load_graph(REPO / "graphs/01_single_llm.yaml")
        assert g.name == "hello-llm"
        assert "agent" in g.nodes
        node = g.node("agent")
        assert node.instance is not None
        # Edges include __in__ and __out__
        assert any("__in__" in e.src_node for e in g.edges)
        assert any("__out__" in e.dst_node for e in g.edges)

    def test_load_llm_then_shell(self):
        g, _raw = load_graph(REPO / "graphs/02_llm_then_shell.yaml")
        assert g.name == "llm-then-shell"
        assert "agent" in g.nodes
        assert "runner" in g.nodes

    def test_unknown_provider_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "director: fn\n"
            "nodes:\n"
            "  - id: a\n"
            "    type: llm\n"
            "    provider: notreal\n"
            "    model: x\n"
            "edges: []\n"
        )
        with pytest.raises(ValueError, match="provider"):
            load_graph(path)

    def test_unknown_node_type_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "director: fn\n"
            "nodes:\n"
            "  - id: a\n"
            "    type: portal\n"
            "edges: []\n"
        )
        with pytest.raises(ValueError, match="unknown node type"):
            load_graph(path)

    def test_duplicate_ids_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "director: fn\n"
            "nodes:\n"
            "  - id: a\n"
            "    type: shell\n"
            "  - id: a\n"
            "    type: shell\n"
            "edges: []\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_graph(path)

    def test_graph_validation_runs(self, tmp_path):
        # A graph with an edge to a nonexistent node should fail validation
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "director: fn\n"
            "nodes:\n"
            "  - id: a\n"
            "    type: shell\n"
            "edges:\n"
            "  - { src: a.x, dst: ghost.y }\n"
        )
        with pytest.raises(ValueError, match="not in graph"):
            load_graph(path)


class TestNodeSpecs:
    def test_llm_spec_minimal(self):
        s = LLMNodeSpec(id="x", provider="glm", model="glm-4.5-air")
        assert s.type == "llm"
        assert s.system == ""
        assert s.temperature is None

    def test_llm_spec_invalid_provider(self):
        # `provider` is a free string so newly registered providers work
        # in YAML without editing the schema; the spec itself accepts any
        # name. Unknown names are rejected at graph-load / resolution time.
        spec = LLMNodeSpec(id="x", provider="bogus", model="y")
        assert spec.provider == "bogus"

        from pharos.ir import load_graph_from_dict

        with pytest.raises(ValueError, match="unknown LLM provider"):
            load_graph_from_dict(
                {
                    "name": "g",
                    "nodes": [
                        {
                            "id": "x",
                            "type": "llm",
                            "provider": "bogus",
                            "model": "y",
                        }
                    ],
                    "edges": [],
                }
            )

    def test_shell_spec_defaults(self):
        s = ShellNodeSpec(id="x")
        assert s.timeout == 300.0
        assert s.cwd is None


class TestGraphSpec:
    def test_minimal(self):
        s = GraphSpec.model_validate(
            {"name": "x", "nodes": [{"id": "a", "type": "shell"}], "edges": []}
        )
        assert s.director == "fn"
        assert len(s.nodes) == 1
