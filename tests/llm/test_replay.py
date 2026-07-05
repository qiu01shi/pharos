"""Tests for the ReplayProvider — byte-equal replay of recorded LLM outputs."""
from __future__ import annotations

from pharos.llm.providers.replay import ReplayProvider, _dict_to_stream_event
from pharos.llm.types import (
    Context,
    Model,
    ModelCost,
    UserMessage,
)


def _make_model() -> Model:
    return Model(
        id="replay",
        name="Replay (no network)",
        api="replay",
        provider="replay",
        base_url="",
        cost=ModelCost(input=0.0, output=0.0),
        context_window=128_000,
        max_tokens=8_192,
    )


def _simple_cache(text: str = "hello world") -> dict[str, dict]:
    return {
        "agent:1": {
            "node_id": "agent",
            "step_index": 1,
            "events": [
                {"type": "start", "partial": None},
                {"type": "text_delta", "delta": text + " "},
                {"type": "text_delta", "delta": "tail"},
                {"type": "done", "message": {
                    "content": [{"type": "text", "text": text}],
                    "model": "glm-4.5-air",
                    "provider": "glm",
                    "stop_reason": "stop",
                    "usage": {"input": 5, "output": 2},
                }},
            ],
            "final_text": text,
            "usage": {"input": 5, "output": 2},
            "model": "glm-4.5-air",
            "provider": "glm",
        }
    }


class TestDictToStreamEvent:
    def test_text_delta(self):
        ev = _dict_to_stream_event({"type": "text_delta", "delta": "hi"})
        assert ev is not None
        assert ev.type == "text_delta"
        assert ev.delta == "hi"

    def test_done_with_text_message(self):
        ev = _dict_to_stream_event({
            "type": "done",
            "message": {
                "content": [{"type": "text", "text": "x"}],
                "model": "m",
                "provider": "p",
                "stop_reason": "stop",
                "usage": {"input": 1, "output": 2},
            },
        })
        assert ev is not None
        assert ev.type == "done"
        assert ev.message is not None
        assert ev.message.text() == "x"
        assert ev.message.usage.input == 1
        assert ev.message.usage.output == 2
        assert ev.message.model == "m"

    def test_done_with_thinking(self):
        ev = _dict_to_stream_event({
            "type": "done",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "deep thought"},
                    {"type": "text", "text": "answer"},
                ],
                "model": "m", "provider": "p", "stop_reason": "stop",
            },
        })
        assert ev is not None
        kinds = [type(b).__name__ for b in ev.message.content]
        assert "ThinkingContent" in kinds
        assert "TextContent" in kinds

    def test_done_with_tool_call(self):
        ev = _dict_to_stream_event({
            "type": "done",
            "tool_call": {
                "id": "tc1", "name": "bash",
                "arguments": {"command": "ls"},
            },
        })
        assert ev is not None
        assert ev.tool_call is not None
        assert ev.tool_call.name == "bash"
        assert ev.tool_call.arguments == {"command": "ls"}


class TestReplayProvider:
    async def test_basic_replay(self):
        p = ReplayProvider(node_id="agent", cache=_simple_cache("hi"))
        events = []
        async for ev in p.stream(_make_model(), Context(messages=[UserMessage(content="x")])):
            events.append(ev)
        assert len(events) == 4
        assert events[0].type == "start"
        last = events[-1]
        assert last.type == "done"
        assert last.message is not None
        assert last.message.text() == "hi"

    async def test_step_counter_increments(self):
        """Two stream() calls should consume two cache entries if
        available."""
        cache = {
            "agent:1": {**_simple_cache("first")["agent:1"]},
            "agent:2": {
                "node_id": "agent", "step_index": 2,
                "events": [
                    {"type": "start", "partial": None},
                    {"type": "done", "message": {
                        "content": [{"type": "text", "text": "second"}],
                        "model": "m", "provider": "p", "stop_reason": "stop",
                    }},
                ],
                "final_text": "second",
                "usage": {},
                "model": "m", "provider": "p",
            },
        }
        p = ReplayProvider(node_id="agent", cache=cache)
        # First call: get "first"
        ev1 = []
        async for ev in p.stream(_make_model(), Context(messages=[UserMessage(content="x")])):
            ev1.append(ev)
        assert ev1[-1].message.text() == "first"
        # Second call: get "second"
        ev2 = []
        async for ev in p.stream(_make_model(), Context(messages=[UserMessage(content="x")])):
            ev2.append(ev)
        assert ev2[-1].message.text() == "second"

    async def test_missing_cache_key_yields_error(self):
        cache = _simple_cache()
        # Different node id → no cache entries
        p = ReplayProvider(node_id="other", cache=cache)
        events = []
        async for ev in p.stream(_make_model(), Context(messages=[UserMessage(content="x")])):
            events.append(ev)
        # Should yield an error event
        assert any(e.type == "error" for e in events)

    async def test_list_models(self):
        p = ReplayProvider(node_id="agent", cache=_simple_cache())
        models = await p.list_models()
        assert len(models) == 1
        assert models[0].id == "replay"
        assert models[0].api == "replay"


class TestReplayProviderInGraph:
    """End-to-end: replay a record + re-run a graph with replay."""

    async def test_full_replay_round_trip(self):
        """Record a run, then replay it through the FNDirector."""
        from pharos.core.graph import CompositeGraph
        from pharos.core.token import TypedValue
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector
        from pharos.entities.llm import LLMAgent, LLMEntityConfig
        from pharos.llm.providers.faux import FauxConfig, FauxProvider
        from pharos.observability.trace import InMemoryTracer
        from pharos.runtime import (
            extract_cached_outputs,
            record_run,
        )

        # 1. Record a run
        g = CompositeGraph("rec")
        g.add_entity(
            "agent",
            LLMAgent(
                "agent",
                config=LLMEntityConfig(
                    provider_class=FauxProvider,
                    provider_kwargs={
                        "config": FauxConfig(
                            response_mode="scripted",
                            scripted_responses=["original"],
                            latency_seconds=10,
                            input_tokens=5,
                            output_tokens=3,
                        )
                    },
                    model_id="faux-fast",
                ),
            ),
        )
        g.nodes["agent"].instance.ins["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        g.nodes["__out__"] = g.nodes["__out__"]
        g.collected = {}  # type: ignore[attr-defined]

        from pharos.core.graph import Edge

        g.edges.append(Edge(src_node="agent", src_port="text", dst_node="__out__", dst_port="response"))

        tracer = InMemoryTracer()
        ctx = RunContext(run_id="rid-1")
        ctx.tracer = tracer  # type: ignore[attr-defined]
        await FNDirector().run(g, ctx)

        # Record spans
        record_run("rid-1", tracer.spans)

        # 2. Build replay cache
        cache = extract_cached_outputs("rid-1")
        assert "agent:1" in cache, list(cache.keys())

        # 3. Build a NEW graph using ReplayProvider
        g2 = CompositeGraph("replay")
        g2.add_entity(
            "agent",
            LLMAgent(
                "agent",
                config=LLMEntityConfig(
                    provider_class=ReplayProvider,
                    provider_kwargs={
                        "node_id": "agent",
                        "cache": cache,
                    },
                    model_id="replay",
                ),
            ),
        )
        g2.nodes["agent"].instance.ins["prompt"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        g2.collected = {}  # type: ignore[attr-defined]
        from pharos.core.graph import Edge as Edge2

        g2.edges.append(Edge2(src_node="agent", src_port="text", dst_node="__out__", dst_port="response"))

        result = await FNDirector().run(g2, RunContext(run_id="rid-replay"))
        assert result.converged is True
        # The replayed output should match the original
        replayed = g2.collected["__out__"]["response"]  # type: ignore[index]
        assert len(replayed) == 1
        assert replayed[0].value.payload == "original"