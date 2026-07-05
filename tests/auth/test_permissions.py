"""Tests for permission declarations and RBAC enforcement."""
from __future__ import annotations

import uuid

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.de import DEDirector
from pharos.directors.fn import FNDirector
from pharos.directors.sdf import SDFDirector

# ---------- test entities ----------


@entity
class _Shell(Entity):
    """Mimics ShellEntity's permission requirement."""
    required_permissions = {"shell:execute"}
    ins = {"cmd": InputPort(name="cmd", accepted_types=["text"])}
    outs = {"out": OutputPort(name="out", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["cmd"].consume():
            self.outs["out"].emit(TypedValue(type="text", payload=f"ran: {t.value.payload}"))


@entity
class _Plain(Entity):
    """No required permissions."""
    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["x"].consume():
            self.outs["y"].emit(TypedValue(type="text", payload=t.value.payload))


# ---------- tests ----------


class TestPermissionDenied:
    async def test_missing_permission_raises(self):
        g = CompositeGraph("g")
        g.add_entity("shell", _Shell("shell"))
        g.nodes["shell"].instance.ins["cmd"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="ls")
        )
        d = FNDirector()
        # No permissions granted
        ctx = RunContext(run_id=str(uuid.uuid4()))
        result = await d.run(g, ctx)
        assert result.converged is False
        assert "PermissionError" in (result.error or "")
        assert "shell:execute" in (result.error or "")

    async def test_wrong_permission_raises(self):
        g = CompositeGraph("g")
        g.add_entity("shell", _Shell("shell"))
        g.nodes["shell"].instance.ins["cmd"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="ls")
        )
        d = FNDirector()
        ctx = RunContext(
            run_id=str(uuid.uuid4()),
            granted_permissions={"fs:read"},  # wrong perm
        )
        result = await d.run(g, ctx)
        assert result.converged is False
        assert "shell:execute" in (result.error or "")

    async def test_extra_permissions_ok(self):
        """Having MORE permissions than required is fine."""
        g = CompositeGraph("g")
        g.add_entity("shell", _Shell("shell"))
        g.nodes["shell"].instance.ins["cmd"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="ls")
        )
        d = FNDirector()
        ctx = RunContext(
            run_id=str(uuid.uuid4()),
            granted_permissions={"shell:execute", "fs:read", "net:connect"},
        )
        result = await d.run(g, ctx)
        assert result.converged is True
        # Shell output should appear
        shell_inst = g.node("shell").instance
        assert shell_inst is not None
        out = [
            t.value.payload
            for t in shell_inst.outs["out"].peek_all()
        ]
        assert out == ["ran: ls"]


class TestPermissionAllowed:
    async def test_no_required_permissions_runs_anyway(self):
        g = CompositeGraph("g")
        g.add_entity("plain", _Plain("plain"))
        g.nodes["plain"].instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hello")
        )
        d = FNDirector()
        # Empty grants is fine when no perms are required
        ctx = RunContext(run_id=str(uuid.uuid4()))
        result = await d.run(g, ctx)
        assert result.converged is True

    async def test_plain_entity_runs_with_explicit_empty_grants(self):
        g = CompositeGraph("g")
        g.add_entity("plain", _Plain("plain"))
        g.nodes["plain"].instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hello")
        )
        d = FNDirector()
        ctx = RunContext(
            run_id=str(uuid.uuid4()),
            granted_permissions=set(),
        )
        result = await d.run(g, ctx)
        assert result.converged is True


class TestMixedEntities:
    async def test_one_denied_blocks_run(self):
        """If ANY entity is denied, the run fails fast."""
        g = CompositeGraph("g")
        g.add_entity("plain", _Plain("plain"))
        g.add_entity("shell", _Shell("shell"))
        # plain.y (output) → shell.cmd (input)
        g.connect("plain.y", "shell.cmd")
        g.nodes["plain"].instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        d = FNDirector()
        # Granting nothing → shell will fail when fired
        ctx = RunContext(run_id=str(uuid.uuid4()))
        result = await d.run(g, ctx)
        # The plain entity runs first (no perms needed), then shell fails.
        # In practice the error propagates from the gather().
        assert result.converged is False
        assert "PermissionError" in (result.error or "")


class TestPermissionEnforcedAcrossDirectors:
    """RBAC must be enforced regardless of which Director drives the run.

    Regression guard: previously only FNDirector checked permissions, so
    SDF/DE silently ran permission-gated entities without a grant.
    """

    async def test_sdf_enforces_permissions(self):
        g = CompositeGraph("g")
        g.add_entity("shell", _Shell("shell"))
        g.nodes["shell"].instance.ins["cmd"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="ls")
        )
        d = SDFDirector(max_iterations=3)
        ctx = RunContext(run_id=str(uuid.uuid4()))  # no grant
        result = await d.run(g, ctx)
        assert result.converged is False
        assert "shell:execute" in (result.error or "")

    async def test_de_enforces_permissions(self):
        g = CompositeGraph("g")
        g.add_entity("shell", _Shell("shell"))
        g.nodes["shell"].instance.ins["cmd"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="ls")
        )
        d = DEDirector(max_iterations=3)
        ctx = RunContext(run_id=str(uuid.uuid4()))  # no grant
        result = await d.run(g, ctx)
        assert result.converged is False
        assert "shell:execute" in (result.error or "")


class TestTeardownReleasesResources:
    """Directors must call teardown() so providers close their clients."""

    async def test_teardown_called_after_run(self):
        torn_down: list[str] = []

        @entity
        class _Res(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

            async def setup(self, ctx):  # type: ignore[override]
                self._initialized = True

            async def fire(self, ctx):  # type: ignore[override]
                for t in self.ins["x"].consume():
                    self.outs["y"].emit(t.value)

            async def teardown(self):  # type: ignore[override]
                torn_down.append(self.node_id)

        g = CompositeGraph("g")
        g.add_entity("res", _Res("res"))
        g.nodes["res"].instance.ins["x"].emit(  # type: ignore[union-attr]
            TypedValue(type="text", payload="hi")
        )
        d = FNDirector()
        result = await d.run(g, RunContext(run_id=str(uuid.uuid4())))
        assert result.converged is True
        assert torn_down == ["res"]