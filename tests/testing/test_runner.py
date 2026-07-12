"""Tests for pharos.testing.runner - record, offline gate, live drift verdict.

Uses the deterministic faux provider graph so the "live" path is exercised
without network or API keys.
"""
from __future__ import annotations

from pharos.testing import runner
from pharos.testing.diff import diff_runs
from pharos.testing.fixture import Assertion, Fixture
from pharos.testing.runner import structural_drift

GRAPH = "graphs/03_faux_demo.yaml"


async def _record(assertions=None) -> Fixture:
    return await runner.record_fixture(
        GRAPH, name="demo", seed={"prompt": "say hi"}, assertions=assertions
    )


class TestOfflineGate:
    async def test_passes_on_unchanged(self):
        fx = await _record()
        res = await runner.test_offline(fx)
        assert res.passed
        assert res.digest_ok
        assert res.graph_ok

    async def test_fails_on_digest_drift(self):
        fx = await _record()
        fx.chain_digest = "deadbeef"  # golden no longer matches replay
        res = await runner.test_offline(fx)
        assert not res.passed
        assert not res.digest_ok

    async def test_v2_fails_when_graph_integrity_pin_changes(self):
        fx = await _record()
        fx.graph_sha256 = "0" * 64
        res = await runner.test_offline(fx)
        assert not res.passed
        assert not res.graph_ok

    async def test_assertion_pass(self):
        fx = await _record(
            assertions=[Assertion("__out__.response", "contains", "echo")]
        )
        res = await runner.test_offline(fx)
        assert res.passed
        assert all(a.ok for a in res.assertion_results)

    async def test_assertion_fail(self):
        fx = await _record(
            assertions=[Assertion("__out__.response", "contains", "NOT-THERE")]
        )
        res = await runner.test_offline(fx)
        assert not res.passed
        assert any(not a.ok for a in res.assertion_results)


class TestLiveGate:
    async def test_faux_live_no_drift(self):
        fx = await _record()
        res = await runner.test_live(fx)
        assert res.passed
        assert res.drift is not None
        assert not res.drift.has_changes()

    async def test_update_reblesses(self, tmp_path):
        fx = await _record()
        fx.chain_digest = "stale"
        p = tmp_path / "demo.fixture.json"
        fx.save(p)
        res = await runner.test_live(fx, p, update=True)
        assert res.passed
        assert Fixture.load(p).chain_digest != "stale"


class TestCommittedFixture:
    """The checked-in example fixture must stay green (the CI self-test)."""

    async def test_offline_gate_passes(self):
        fx = Fixture.load("tests/agent/faux_demo.fixture.json")
        res = await runner.test_offline(fx)
        assert res.passed, res.to_dict()
        assert res.digest_ok
        assert all(a.ok for a in res.assertion_results)


class TestStructuralDrift:
    def test_tool_calls_change_is_structural(self):
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
        assert structural_drift(diff_runs(a, b))

    def test_text_wording_change_is_not_structural(self):
        a = {"n:0": [{"port": "text", "type": "text", "payload": "hello world"}]}
        b = {"n:0": [{"port": "text", "type": "text", "payload": "hi world"}]}
        assert not structural_drift(diff_runs(a, b))

    def test_removed_port_is_structural(self):
        a = {
            "n:0": [
                {"port": "text", "type": "text", "payload": "x"},
                {"port": "json", "type": "json", "payload": {"k": 1}},
            ]
        }
        b = {"n:0": [{"port": "text", "type": "text", "payload": "x"}]}
        assert structural_drift(diff_runs(a, b))

    def test_usage_token_change_is_not_structural(self):
        # Token counts legitimately vary run-to-run; not a regression.
        a = {"n:0": [{"port": "usage", "type": "json", "payload": {"total": 100}}]}
        b = {"n:0": [{"port": "usage", "type": "json", "payload": {"total": 120}}]}
        assert not structural_drift(diff_runs(a, b))

    def test_usage_repair_attempts_rise_is_structural(self):
        a = {
            "n:0": [
                {"port": "usage", "type": "json", "payload": {"repair_attempts": 0}}
            ]
        }
        b = {
            "n:0": [
                {"port": "usage", "type": "json", "payload": {"repair_attempts": 3}}
            ]
        }
        assert structural_drift(diff_runs(a, b))
