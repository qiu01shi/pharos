"""Tests for HumanEntity — human-in-the-loop input as a graph node.

Covers: RBAC gating, preset answer, pause-and-resume (record partial run,
then continue live with the answer supplied).
"""
from __future__ import annotations

import pytest

from pharos.core.token import TypedValue
from pharos.directors.base import FireContext, RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.human import HumanEntity, HumanInputRequired
from pharos.ir import load_graph_from_dict
from pharos.runtime import RunRecorder, RunReplayer


async def _fire(entity, granted=None):
    await entity.setup(RunContext(run_id="t"))
    await entity.fire(
        FireContext(run_id="t", step_id="s", granted_permissions=granted or set())
    )


class TestHumanEntityUnit:
    def test_requires_permission(self):
        assert HumanEntity("h").required_permissions == {"human:input"}

    async def test_preset_answer_emitted(self):
        h = HumanEntity("h", answer="approved")
        h.ins["prompt"].emit(TypedValue(type="text", payload="OK to deploy?"))
        await _fire(h)
        out = h.outs["response"].peek_all()
        assert [t.value.payload for t in out] == ["approved"]

    async def test_pauses_without_answer(self):
        h = HumanEntity("h")  # no answer, not interactive
        h.ins["prompt"].emit(TypedValue(type="text", payload="?"))
        with pytest.raises(HumanInputRequired):
            await _fire(h)


def _graph() -> dict:
    return {
        "name": "human-graph",
        "director": "fn",
        "nodes": [
            {"id": "gate", "type": "human"},
            {
                "id": "after",
                "type": "python",
                "class": "tests.entities.test_human:_Echo",
            },
        ],
        "edges": [
            {"src": "__in__.prompt", "dst": "gate.prompt"},
            {"src": "gate.response", "dst": "after.x"},
            {"src": "after.y", "dst": "__out__.result"},
        ],
    }


# Custom entity used by the resume graph above.
from pharos.core.entity import Entity, entity  # noqa: E402
from pharos.core.port import InputPort, OutputPort  # noqa: E402


@entity
class _Echo(Entity):
    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    async def fire(self, ctx) -> None:  # type: ignore[override]
        for t in self.ins["x"].consume():
            self.outs["y"].emit(
                TypedValue(type="text", payload=f"done:{t.value.payload}")
            )


def _seed(g, text: str) -> None:
    for e in g.edges:
        if e.src_node == "__in__" and e.src_port == "prompt":
            g.node(e.dst_node).instance.ins[e.dst_port].emit(
                TypedValue(type="text", payload=text)
            )


class TestHumanPauseResume:
    async def test_run_pauses_then_resumes_with_answer(self):
        # 1) First run pauses at the human gate (no answer): partial recording.
        g1, _ = load_graph_from_dict(_graph())
        _seed(g1, "ship it?")
        rec = RunRecorder()
        r1 = await FNDirector().run(
            g1,
            RunContext(
                run_id="run1",
                granted_permissions={"human:input"},
                recorder=rec,
            ),
        )
        assert r1.converged is False  # paused
        data = rec.to_dict()
        # The human gate did not complete, so `after` never recorded.
        assert "after:0" not in data

        # 2) Resume with the answer supplied -> gate runs live, `after` runs.
        g2, _ = load_graph_from_dict(_graph())
        _seed(g2, "ship it?")
        g2.node("gate").instance.answer = "yes"
        replayer = RunReplayer(data, resume=True)
        r2 = await FNDirector().run(
            g2,
            RunContext(
                run_id="run2",
                granted_permissions={"human:input"},
                recorder=RunRecorder(),
                replayer=replayer,
            ),
        )
        assert r2.converged is True
        out = getattr(g2, "collected", {}).get("__out__", {}).get("result", [])
        assert [t.value.payload for t in out] == ["done:yes"]

    async def test_missing_permission_denied(self):
        g, _ = load_graph_from_dict(_graph())
        _seed(g, "?")
        g.node("gate").instance.answer = "yes"
        # No human:input grant -> permission check fails before fire.
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is False
        assert r.error is not None
