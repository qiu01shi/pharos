"""Tests for business entities (LLMAgent, ShellEntity)."""

from __future__ import annotations

from pharos.core.graph import CompositeGraph
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.entities.llm import LLMEntityConfig
from pharos.entities.shell import ShellEntity
from pharos.llm.providers.faux import FauxConfig, FauxProvider

# ============== ShellEntity ==============


class TestShellEntity:
    async def test_echo(self):
        g = CompositeGraph("g")
        shell = ShellEntity(node_id="shell")
        g.add_entity("shell", shell)
        # Seed the command
        g.nodes["shell"].instance.ins["command"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="echo hello-pharos")
        )
        d = FNDirector()
        await d.run(g, RunContext(
            run_id="t",
            granted_permissions={"shell:execute"},
        ))
        stdout = [t.value.payload for t in shell.outs["stdout"].peek_all()]
        exit_code = [t.value.payload for t in shell.outs["exit_code"].peek_all()]
        assert stdout == ["hello-pharos\n"]
        assert exit_code == [0]

    async def test_nonzero_exit(self):
        g = CompositeGraph("g")
        shell = ShellEntity(node_id="shell")
        g.add_entity("shell", shell)
        g.nodes["shell"].instance.ins["command"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="exit 7")
        )
        d = FNDirector()
        await d.run(g, RunContext(
            run_id="t",
            granted_permissions={"shell:execute"},
        ))
        exit_code = [t.value.payload for t in shell.outs["exit_code"].peek_all()]
        assert exit_code == [7]

    async def test_stderr_capture(self):
        g = CompositeGraph("g")
        shell = ShellEntity(node_id="shell")
        g.add_entity("shell", shell)
        g.nodes["shell"].instance.ins["command"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="echo bad >&2")
        )
        d = FNDirector()
        await d.run(g, RunContext(
            run_id="t",
            granted_permissions={"shell:execute"},
        ))
        stderr = [t.value.payload for t in shell.outs["stderr"].peek_all()]
        assert any("bad" in s for s in stderr)


# ============== LLMAgent ==============


async def test_basic_with_faux():
    # Just verifies the config dataclass is importable + usable
    cfg = LLMEntityConfig(
        provider_class=FauxProvider,  # type: ignore[arg-type]
        provider_kwargs={
            "config": FauxConfig(
                latency_seconds=0.0,
                response_mode="scripted",
                scripted_responses=["hi"],
            )
        },
        model_id="faux-fast",
    )
    assert cfg.provider_class is FauxProvider
    assert cfg.model_id == "faux-fast"