"""Tests for schema-validated ports + LLMAgent structured output.

Covers the "types cross LLM boundaries" contract:
  * a port with a `schema` rejects mis-shaped JSON on emit/receive;
  * LLMAgent emits the validated object on `json` when it conforms;
  * `on_invalid="raise"` raises (and composes with RetryEntity);
  * `on_invalid="error_port"` routes the failure to `error`;
  * the `json` port participates in record/replay byte-for-byte.
"""
from __future__ import annotations

import pytest

from pharos.core.port import OutputPort, PortContractViolation
from pharos.core.token import TypedValue
from pharos.directors.base import FireContext, RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.llm import LLMAgent, LLMEntityConfig
from pharos.ir import load_graph_from_dict
from pharos.llm.providers.faux import FauxConfig, FauxProvider
from pharos.runtime import RunRecorder, RunReplayer

_SCHEMA = {
    "type": "object",
    "required": ["file", "line"],
    "properties": {
        "file": {"type": "string"},
        "line": {"type": "integer"},
    },
}


# ---------- port-level schema enforcement ----------


class TestPortSchema:
    def test_valid_payload_emits(self):
        port = OutputPort(name="json", accepted_types=["json"], schema=_SCHEMA)
        tok = port.emit(TypedValue(type="json", payload={"file": "a.py", "line": 1}))
        assert tok.value.payload["line"] == 1

    def test_invalid_shape_rejected(self):
        port = OutputPort(name="json", accepted_types=["json"], schema=_SCHEMA)
        with pytest.raises(PortContractViolation, match="schema violation"):
            port.emit(TypedValue(type="json", payload={"file": "a.py"}))

    def test_schema_only_applies_to_json_tag(self):
        # A schema is irrelevant to non-json payloads (type tag guards those).
        port = OutputPort(name="p", accepted_types=["text", "json"], schema=_SCHEMA)
        # text payload bypasses schema entirely
        port.emit(TypedValue(type="text", payload="anything"))


# ---------- LLMAgent structured output ----------


def _agent(scripted: list[str], *, on_invalid: str = "raise") -> LLMAgent:
    cfg = LLMEntityConfig(
        provider_class=FauxProvider,  # type: ignore[arg-type]
        provider_kwargs={
            "config": FauxConfig(
                latency_seconds=0.0,
                response_mode="scripted",
                scripted_responses=scripted,
            )
        },
        model_id="faux-fast",
        output_schema=_SCHEMA,
        on_invalid=on_invalid,
    )
    return LLMAgent("ex", cfg)


async def _drive(agent: LLMAgent) -> None:
    await agent.setup(RunContext(run_id="t"))
    agent.ins["prompt"].emit(TypedValue(type="text", payload="go"))
    await agent.fire(FireContext(run_id="t", step_id="s"))


class TestLLMAgentStructured:
    async def test_json_port_bound_to_schema(self):
        agent = _agent(['{"file": "a.py", "line": 5}'])
        assert agent.outs["json"].schema == _SCHEMA

    async def test_valid_output_emitted_on_json_port(self):
        agent = _agent(['{"file": "a.py", "line": 5}'])
        await _drive(agent)
        out = agent.outs["json"].peek_all()
        assert len(out) == 1
        assert out[0].value.type == "json"
        assert out[0].value.payload == {"file": "a.py", "line": 5}

    async def test_json_extracted_from_code_fence(self):
        agent = _agent(['```json\n{"file": "a.py", "line": 7}\n```'])
        await _drive(agent)
        assert agent.outs["json"].peek_all()[0].value.payload["line"] == 7

    async def test_invalid_raises_by_default(self):
        agent = _agent(["not json at all"])
        with pytest.raises(PortContractViolation, match="structured output invalid"):
            await _drive(agent)

    async def test_schema_mismatch_raises(self):
        # Parses as JSON but violates the shape (line missing).
        agent = _agent(['{"file": "a.py"}'])
        with pytest.raises(PortContractViolation):
            await _drive(agent)

    async def test_error_port_mode_does_not_raise(self):
        agent = _agent(["totally not json"], on_invalid="error_port")
        await _drive(agent)
        assert not agent.outs["json"].peek_all()
        err = agent.outs["error"].peek_all()
        assert len(err) == 1
        assert "structured output invalid" in err[0].value.payload


# ---------- composition with RetryEntity through a Director ----------


def _retry_graph(scripted: list[str]) -> dict:
    return {
        "name": "structured-retry",
        "director": "fn",
        "nodes": [
            {
                "id": "ex",
                "type": "faux",
                "provider": "faux",
                "model": "faux-fast",
                "on_invalid": "raise",
                "retry": {"max_attempts": 2, "backoff_s": 0.0},
                "output_schema": _SCHEMA,
                "params": {
                    "latency_seconds": 0.0,
                    "response_mode": "scripted",
                    "scripted_responses": scripted,
                },
            }
        ],
        "edges": [
            {"src": "__in__.prompt", "dst": "ex.prompt"},
            {"src": "ex.json", "dst": "__out__.result"},
        ],
    }


class TestRetrySynergy:
    async def test_retry_recovers_from_invalid_output(self):
        # First attempt: invalid; second attempt: valid → run succeeds.
        g, _ = load_graph_from_dict(
            _retry_graph(["not json", '{"file": "m.py", "line": 42}'])
        )
        for e in g.edges:
            if e.src_node == "__in__" and e.src_port == "prompt":
                g.node(e.dst_node).instance.ins[e.dst_port].emit(
                    TypedValue(type="text", payload="go")
                )
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is True
        out = getattr(g, "collected", {}).get("__out__", {}).get("result", [])
        assert [t.value.payload for t in out] == [{"file": "m.py", "line": 42}]

    async def test_run_fails_when_retries_exhausted(self):
        g, _ = load_graph_from_dict(_retry_graph(["bad", "still bad"]))
        for e in g.edges:
            if e.src_node == "__in__" and e.src_port == "prompt":
                g.node(e.dst_node).instance.ins[e.dst_port].emit(
                    TypedValue(type="text", payload="go")
                )
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is False
        assert r.error is not None


# ---------- record / replay of the json port ----------


class TestReplay:
    async def test_json_port_recorded_and_replayed(self):
        agent = _agent(['{"file": "z.py", "line": 3}'])
        await _drive(agent)
        recorder = RunRecorder()
        recorder.capture(agent, "ex", 0)
        data = recorder.to_dict()
        recorded = data["ex:0"]
        assert any(rec["port"] == "json" for rec in recorded)

        # Replay onto a fresh agent (no provider): re-emits identical tokens,
        # and the json port's schema re-validates on the way out.
        fresh = _agent(["unused"])
        replayer = RunReplayer(data)
        replayer.apply(fresh, "ex", 0)
        replayed = fresh.outs["json"].peek_all()
        assert len(replayed) == 1
        assert replayed[0].value.payload == {"file": "z.py", "line": 3}
