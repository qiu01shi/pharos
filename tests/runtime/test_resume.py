"""Tests for resume — continue a partially-recorded run.

A resume-mode ``RunReplayer`` re-emits recorded fires and lets everything else
run live. Completed work replays cheaply (no execution); the remainder runs,
and the resumed run is itself fully recorded. We use a custom counting entity
so "did this fire actually execute?" survives teardown.
"""
from __future__ import annotations

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.ir import load_graph_from_dict
from pharos.runtime import RunRecorder, RunReplayer


@entity
class _CountingEcho(Entity):
    """Echo `tag:input` to `y`, counting how many times it actually fires."""

    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    def __init__(self, node_id: str, tag: str = "out") -> None:
        super().__init__(node_id=node_id)
        self.tag = tag
        self.fire_calls = 0

    async def fire(self, ctx) -> None:  # type: ignore[override]
        self.fire_calls += 1
        for t in self.ins["x"].consume():
            self.outs["y"].emit(
                TypedValue(type="text", payload=f"{self.tag}:{t.value.payload}")
            )


def _two_node_graph() -> dict:
    def node(nid: str, tag: str) -> dict:
        return {
            "id": nid,
            "type": "python",
            "class": "tests.runtime.test_resume:_CountingEcho",
            "params": {"tag": tag},
        }

    return {
        "name": "resume-graph",
        "director": "fn",
        "nodes": [node("a", "A"), node("b", "B")],
        "edges": [
            {"src": "__in__.msg", "dst": "a.x"},
            {"src": "a.y", "dst": "b.x"},
            {"src": "b.y", "dst": "__out__.result"},
        ],
    }


def _seed(g, text: str) -> None:
    for e in g.edges:
        if e.src_node == "__in__" and e.src_port == "msg":
            g.node(e.dst_node).instance.ins[e.dst_port].emit(
                TypedValue(type="text", payload=text)
            )


class TestResume:
    async def test_recorded_fire_replays_unrecorded_runs_live(self):
        # Checkpoint: only node "a" completed (its output is cached).
        partial = {"a:0": [{"port": "y", "type": "text", "payload": "A:go"}]}
        g, _ = load_graph_from_dict(_two_node_graph())
        _seed(g, "go")

        recorder = RunRecorder()
        replayer = RunReplayer(partial, resume=True)
        ctx = RunContext(run_id="r2", recorder=recorder, replayer=replayer)
        result = await FNDirector().run(g, ctx)

        assert result.converged is True
        a = g.node("a").instance
        b = g.node("b").instance
        # "a" was replayed from cache — it never executed.
        assert a.fire_calls == 0  # type: ignore[attr-defined]
        # "b" ran live, consuming a's replayed output as its input.
        assert b.fire_calls == 1  # type: ignore[attr-defined]

        out = getattr(g, "collected", {}).get("__out__", {}).get("result", [])
        assert [t.value.payload for t in out] == ["B:A:go"]

        # The resumed run recorded BOTH fires (replayed + live).
        data = recorder.to_dict()
        assert "a:0" in data and "b:0" in data

    async def test_pure_replay_emits_nothing_for_unrecorded(self):
        # Without resume, an unrecorded fire produces no tokens (legacy path).
        partial = {"a:0": [{"port": "y", "type": "text", "payload": "A:go"}]}
        g, _ = load_graph_from_dict(_two_node_graph())
        _seed(g, "go")

        replayer = RunReplayer(partial, resume=False)
        ctx = RunContext(run_id="r3", replayer=replayer)
        result = await FNDirector().run(g, ctx)

        assert result.converged is True
        b = g.node("b").instance
        assert b.fire_calls == 0  # type: ignore[attr-defined]
        out = getattr(g, "collected", {}).get("__out__", {}).get("result", [])
        assert out == []
