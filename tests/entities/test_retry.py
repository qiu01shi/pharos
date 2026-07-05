"""Tests for RetryEntity — re-fire an inner entity on failure.

Covers: success after N failures, exhaustion re-raises, input re-seeding
across attempts, cost/lineage preservation, permission passthrough, backoff,
and IR `retry:` wiring end-to-end through a Director.
"""
from __future__ import annotations

import pytest

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import Token, TypedValue
from pharos.directors.base import FireContext, RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.retry import RetryEntity
from pharos.ir import load_graph_from_dict


@entity
class _Flaky(Entity):
    """Raises `fail_times` then echoes input to output (with cost)."""

    required_permissions = {"demo:run"}
    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    def __init__(self, node_id: str, fail_times: int = 0) -> None:
        super().__init__(node_id=node_id)
        self.fail_times = fail_times
        self.fire_calls = 0

    async def fire(self, ctx) -> None:
        self.fire_calls += 1
        if self.fire_calls <= self.fail_times:
            raise RuntimeError(f"boom #{self.fire_calls}")
        for t in self.ins["x"].consume():
            self.outs["y"].emit(
                TypedValue(type="text", payload=f"ok:{t.value.payload}")
            )


def _fire_ctx() -> FireContext:
    return FireContext(run_id="t", step_id="s", granted_permissions={"demo:run"})


async def _fire(entity: Entity) -> None:
    """Drive one fire() directly (bypassing the Director)."""
    await entity.setup(RunContext(run_id="t"))
    await entity.fire(_fire_ctx())


class TestRetryBehavior:
    async def test_succeeds_after_failures(self):
        inner = _Flaky("f", fail_times=2)
        r = RetryEntity("f", inner, max_attempts=3)
        r.ins["x"].emit(TypedValue(type="text", payload="hi"))
        await _fire(r)
        assert [t.value.payload for t in r.outs["y"].peek_all()] == ["ok:hi"]
        assert r.attempts_used == 3
        assert inner.fire_calls == 3

    async def test_exhaustion_reraises(self):
        inner = _Flaky("f", fail_times=99)
        r = RetryEntity("f", inner, max_attempts=3)
        r.ins["x"].emit(TypedValue(type="text", payload="hi"))
        with pytest.raises(RuntimeError, match="boom #3"):
            await _fire(r)
        assert r.attempts_used == 3
        assert inner.fire_calls == 3

    async def test_first_attempt_success(self):
        inner = _Flaky("f", fail_times=0)
        r = RetryEntity("f", inner, max_attempts=5)
        r.ins["x"].emit(TypedValue(type="text", payload="hi"))
        await _fire(r)
        assert r.attempts_used == 1
        assert inner.fire_calls == 1

    async def test_inputs_reseeded_each_attempt(self):
        # The inner consumes its input; without re-seeding, attempt 2 would
        # see an empty port and emit nothing.
        inner = _Flaky("f", fail_times=1)
        r = RetryEntity("f", inner, max_attempts=2)
        r.ins["x"].emit(TypedValue(type="text", payload="data"))
        await _fire(r)
        assert [t.value.payload for t in r.outs["y"].peek_all()] == ["ok:data"]

    async def test_invalid_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryEntity("f", _Flaky("f"), max_attempts=0)


class TestPortsAndMetrics:
    def test_mirrors_inner_ports_and_permissions(self):
        r = RetryEntity("f", _Flaky("f"), max_attempts=2)
        assert set(r.ins) == {"x"}
        assert set(r.outs) == {"y"}
        assert r.required_permissions == {"demo:run"}

    async def test_preserves_token_cost(self):
        @entity
        class _Costly(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

            async def fire(self, ctx) -> None:
                self.ins["x"].consume()
                # Emit a pre-built token carrying a cost, via receive().
                self.outs["y"].receive(
                    Token(
                        value=TypedValue(type="text", payload="z"),
                        origin="costly.y",
                        cost_usd=0.5,
                    )
                )

        inner = _Costly("c")
        r = RetryEntity("c", inner, max_attempts=1)
        r.ins["x"].emit(TypedValue(type="text", payload="hi"))
        await _fire(r)
        out = r.outs["y"].peek_all()
        assert len(out) == 1
        assert out[0].cost_usd == 0.5  # cost survived the wrapper


class TestRetryViaIR:
    def _graph_dict(self, attempts: int) -> dict:
        return {
            "name": "retry-ir",
            "director": "fn",
            "nodes": [
                {
                    "id": "n",
                    "type": "python",
                    "class": "tests.entities.test_retry:_Flaky",
                    "params": {"fail_times": 1},
                    "retry": {"max_attempts": attempts, "backoff_s": 0.0},
                }
            ],
            "edges": [
                {"src": "__in__.msg", "dst": "n.x"},
                {"src": "n.y", "dst": "__out__.done"},
            ],
        }

    def test_retry_block_wraps_node(self):
        g, _ = load_graph_from_dict(self._graph_dict(3))
        assert isinstance(g.node("n").instance, RetryEntity)

    def test_no_retry_block_is_plain(self):
        raw = self._graph_dict(3)
        del raw["nodes"][0]["retry"]
        g, _ = load_graph_from_dict(raw)
        assert not isinstance(g.node("n").instance, RetryEntity)

    async def test_end_to_end_through_director(self):
        g, _ = load_graph_from_dict(self._graph_dict(3))
        for e in g.edges:
            if e.src_node == "__in__" and e.src_port == "msg":
                g.node(e.dst_node).instance.ins[e.dst_port].emit(
                    TypedValue(type="text", payload="go")
                )
        r = await FNDirector().run(
            g, RunContext(run_id="t", granted_permissions={"demo:run"})
        )
        assert r.converged is True
        out = getattr(g, "collected", {}).get("__out__", {}).get("done", [])
        assert [t.value.payload for t in out] == ["ok:go"]
