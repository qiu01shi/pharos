"""Tests for run-level cost/token budgets enforced in safe_fire."""
from __future__ import annotations

import pytest

from pharos.core.token import TypedValue
from pharos.directors.base import (
    BudgetExceededError,
    RunBudget,
    RunContext,
)
from pharos.directors.fn import FNDirector
from pharos.directors.sdf import SDFDirector
from pharos.ir import load_graph_from_dict


def _two_llm_graph() -> dict:
    def node(nid: str) -> dict:
        return {
            "id": nid,
            "type": "faux",
            "provider": "faux",
            "model": "faux-fast",
            "params": {
                "latency_seconds": 0.0,
                "response_mode": "scripted",
                "scripted_responses": ["ok"],
                # 100 tokens per fire (50 in + 50 out).
                "input_tokens": 50,
                "output_tokens": 50,
            },
        }

    return {
        "name": "budget-graph",
        "director": "fn",
        "nodes": [node("a"), node("b")],
        "edges": [
            {"src": "__in__.prompt", "dst": "a.prompt"},
            {"src": "a.text", "dst": "b.prompt"},
            {"src": "b.text", "dst": "__out__.result"},
        ],
    }


def _seed(g, text: str) -> None:
    for e in g.edges:
        if e.src_node == "__in__" and e.src_port == "prompt":
            g.node(e.dst_node).instance.ins[e.dst_port].emit(
                TypedValue(type="text", payload=text)
            )


class TestRunBudgetUnit:
    def test_hard_raises_over_tokens(self):
        b = RunBudget(max_tokens=100, mode="hard")
        b.charge(60, 0.0)  # under
        with pytest.raises(BudgetExceededError, match="tokens 120 > budget 100"):
            b.charge(60, 0.0)

    def test_soft_flags_but_does_not_raise(self):
        b = RunBudget(max_cost_usd=0.01, mode="soft")
        b.charge(0, 0.05)
        assert b.exceeded is True
        assert b.spent_cost == pytest.approx(0.05)


class TestBudgetThroughDirector:
    async def test_sdf_charges_per_fire_delta_not_cumulative_total(self):
        raw = {
            "name": "one-call-many-rounds",
            "director": "sdf",
            "nodes": [_two_llm_graph()["nodes"][0]],
            "edges": [],
        }
        g, _ = load_graph_from_dict(raw)
        g.node("a").instance.ins["prompt"].emit(
            TypedValue(type="text", payload="once")
        )
        budget = RunBudget(max_tokens=10_000, mode="soft")

        result = await SDFDirector(max_iterations=3, convergence_k=99).run(
            g, RunContext(run_id="delta", budget=budget)
        )

        assert g.node("a").instance.total_tokens == 100
        assert result.tokens_emitted == 100
        assert budget.spent_tokens == 100

    async def test_graph_reset_allows_clean_reuse(self):
        g, _ = load_graph_from_dict(_two_llm_graph())
        _seed(g, "first")
        first = await FNDirector().run(g, RunContext(run_id="first"))
        assert first.tokens_emitted == 200
        assert len(g.collected["__out__"]["result"]) == 1

        g.reset_run_state()
        assert g.collected == {}
        assert g.node("a").instance.total_tokens == 0
        assert all(
            len(port) == 0
            for node in g.nodes.values()
            if node.instance is not None
            for port in (*node.instance.ins.values(), *node.instance.outs.values())
        )

        _seed(g, "second")
        second = await FNDirector().run(g, RunContext(run_id="second"))
        assert second.tokens_emitted == 200
        assert len(g.collected["__out__"]["result"]) == 1

    async def test_hard_budget_aborts_run(self):
        g, _ = load_graph_from_dict(_two_llm_graph())
        _seed(g, "go")
        # Each node spends 100 tokens; cap at 150 -> second node trips it.
        budget = RunBudget(max_tokens=150, mode="hard")
        r = await FNDirector().run(
            g, RunContext(run_id="t", budget=budget)
        )
        assert r.converged is False
        assert r.error is not None
        assert "budget" in r.error.lower()

    async def test_soft_budget_completes(self):
        g, _ = load_graph_from_dict(_two_llm_graph())
        _seed(g, "go")
        budget = RunBudget(max_tokens=150, mode="soft")
        r = await FNDirector().run(
            g, RunContext(run_id="t", budget=budget)
        )
        assert r.converged is True
        assert budget.exceeded is True
        assert budget.spent_tokens == 200

    async def test_generous_budget_no_effect(self):
        g, _ = load_graph_from_dict(_two_llm_graph())
        _seed(g, "go")
        budget = RunBudget(max_tokens=10_000, max_cost_usd=100.0)
        r = await FNDirector().run(
            g, RunContext(run_id="t", budget=budget)
        )
        assert r.converged is True
        assert budget.exceeded is False
