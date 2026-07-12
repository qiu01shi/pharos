"""Python authoring and team macros compile through the canonical IR."""

from __future__ import annotations

import pytest

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.ir import GraphBuilder, TeamMemberSpec, TeamSpec, TerminationSpec


class TestGraphBuilder:
    def test_builds_same_versioned_graph_ir(self):
        builder = GraphBuilder("hello", director="fn", metadata={"owner": "qa"})
        agent = builder.llm(
            "agent", provider="faux", model="faux-fast", system="brief"
        )
        builder.connect(builder.input.output("prompt"), agent.input("prompt"))
        builder.connect(agent.output("text"), builder.output.input("response"))

        raw = builder.to_dict()
        graph, compiled_raw = builder.build()

        assert raw["apiVersion"] == "pharos.ai/v1"
        assert raw["kind"] == "Graph"
        assert raw["metadata"] == {"owner": "qa"}
        assert isinstance(graph, CompositeGraph)
        assert compiled_raw == raw
        assert "apiVersion: pharos.ai/v1" in builder.to_yaml()

    def test_duplicate_node_rejected_while_authoring(self):
        builder = GraphBuilder("dupe")
        builder.shell("x")
        with pytest.raises(ValueError, match="duplicate"):
            builder.shell("x")


class TestStaticContracts:
    def test_disjoint_port_types_fail_at_connect_time(self):
        @entity
        class TextSource(Entity):
            outs = {"value": OutputPort("value", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                return None

        @entity
        class IntSink(Entity):
            ins = {"value": InputPort("value", accepted_types=["int"])}

            async def fire(self, ctx):  # type: ignore[override]
                return None

        graph = CompositeGraph("bad")
        graph.add_entity("source", TextSource("source"))
        graph.add_entity("sink", IntSink("sink"))
        with pytest.raises(ValueError, match="type contract mismatch"):
            graph.connect("source.value", "sink.value")

    def test_json_schema_contract_checks_required_fields(self):
        @entity
        class Producer(Entity):
            outs = {
                "value": OutputPort(
                    "value",
                    accepted_types=["json"],
                    schema={
                        "type": "object",
                        "properties": {"file": {"type": "string"}},
                        "required": ["file"],
                    },
                )
            }

            async def fire(self, ctx):  # type: ignore[override]
                return None

        @entity
        class Consumer(Entity):
            ins = {
                "value": InputPort(
                    "value",
                    accepted_types=["json"],
                    schema={
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                        },
                        "required": ["file", "line"],
                    },
                )
            }

            async def fire(self, ctx):  # type: ignore[override]
                return None

        graph = CompositeGraph("bad-schema")
        graph.add_entity("producer", Producer("producer"))
        graph.add_entity("consumer", Consumer("consumer"))
        with pytest.raises(ValueError, match=r"does not require.*line"):
            graph.connect("producer.value", "consumer.value")


class TestTeamSpec:
    def test_pipeline_compiles_to_fn_graph(self):
        team = TeamSpec(
            name="writers",
            members=[
                TeamMemberSpec(id="draft", provider="faux", model="faux-fast"),
                TeamMemberSpec(id="edit", provider="faux", model="faux-fast"),
            ],
        )
        raw = team.to_dict()
        assert raw["director"] == "fn"
        assert raw["metadata"]["team_pattern"] == "pipeline"
        assert raw["edges"][-1]["dst"] == "__out__.response"

    def test_reflection_compiles_to_sdf_feedback(self):
        team = TeamSpec(
            name="review",
            pattern="reflection",
            members=[
                TeamMemberSpec(id="coder", provider="faux", model="faux-fast"),
                TeamMemberSpec(id="reviewer", provider="faux", model="faux-fast"),
            ],
            termination=TerminationSpec(max_iterations=8, convergence_k=3),
        )
        raw = team.to_dict()
        assert raw["director"] == "sdf"
        assert raw["execution"] == {"max_iterations": 8, "convergence_k": 3}
        assert {"src": "reviewer.text", "dst": "coder.prompt"} in raw["edges"]
        graph, _ = team.build()
        assert graph.has_cycle()

    def test_reflection_requires_exactly_two_members(self):
        with pytest.raises(ValueError, match="producer and reviewer"):
            TeamSpec(
                name="bad",
                pattern="reflection",
                members=[
                    TeamMemberSpec(id="solo", provider="faux", model="faux-fast")
                ],
            )
