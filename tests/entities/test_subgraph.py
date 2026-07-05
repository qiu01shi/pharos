"""Tests for SubgraphEntity — nested graph composability (A1 + B1).

Covers: recursive execution, heterogeneous scheduling (parent fn + child
sdf), permission union fail-fast + propagation, nested trace tree,
namespaced record/replay round-trip, and IR ref loading + cycle detection.
"""
from __future__ import annotations

import pytest

from pharos.core.entity import Entity, entity
from pharos.core.graph import CompositeGraph
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.directors.base import RunContext
from pharos.directors.fn import FNDirector
from pharos.observability.trace import InMemoryTracer
from pharos.runtime import RunRecorder, RunReplayer


@entity
class _Echo(Entity):
    """Echo input to output, counting fires (to detect replay skipping)."""

    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    def __init__(self, node_id: str, prefix: str = "out:") -> None:
        super().__init__(node_id=node_id)
        self.prefix = prefix
        self.calls = 0

    async def fire(self, ctx):  # type: ignore[override]
        self.calls += 1
        for t in self.ins["x"].consume():
            self.outs["y"].emit(
                TypedValue(type="text", payload=f"{self.prefix}{t.value.payload}")
            )


@entity
class _Priv(Entity):
    """An entity that requires a permission (like ShellEntity)."""

    required_permissions = {"shell:execute"}
    ins = {"x": InputPort(name="x", accepted_types=["text"])}
    outs = {"y": OutputPort(name="y", accepted_types=["text"])}

    async def fire(self, ctx):  # type: ignore[override]
        for t in self.ins["x"].consume():
            self.outs["y"].emit(TypedValue(type="text", payload=t.value.payload))


def _child(prefix: str = "out:") -> CompositeGraph:
    c = CompositeGraph("child")
    c.add_entity("inner", _Echo("inner", prefix=prefix))
    c.connect("__in__.text", "inner.x")
    c.connect("inner.y", "__out__.result")
    return c


def _parent_embedding(child: CompositeGraph, **kw) -> CompositeGraph:
    p = CompositeGraph("parent")
    p.add_subgraph("sub", child, **kw)
    p.connect("__in__.msg", "sub.text")
    p.connect("sub.result", "__out__.out")
    return p


def _seed(g: CompositeGraph, port: str, value: str) -> None:
    for e in g.edges:
        if e.src_node == "__in__" and e.src_port == port:
            g.node(e.dst_node).instance.ins[e.dst_port].emit(  # type: ignore[union-attr]
                TypedValue(type="text", payload=value)
            )


def _out(g: CompositeGraph, node: str, port: str) -> list:
    coll = getattr(g, "collected", {}).get(node, {})
    return [t.value.payload for t in coll.get(port, [])]


class TestRecursiveExecution:
    async def test_basic_nested_run(self):
        g = _parent_embedding(_child())
        _seed(g, "msg", "hello")
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is True
        assert _out(g, "__out__", "out") == ["out:hello"]

    async def test_subgraph_derives_ports(self):
        sub = _parent_embedding(_child()).node("sub").instance
        assert set(sub.ins) == {"text"}  # from child __in__.text
        assert set(sub.outs) == {"result"}  # from child __out__.result


class TestHeterogeneousScheduling:
    async def test_parent_fn_child_sdf(self):
        # Child uses SDF; parent uses FN. A deterministic echo converges.
        g = _parent_embedding(_child(), director_name="sdf", max_iters=5)
        _seed(g, "msg", "x")
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is True
        assert _out(g, "__out__", "out") == ["out:x"]


class TestPermissionUnion:
    def _priv_parent(self) -> CompositeGraph:
        c = CompositeGraph("child")
        c.add_entity("priv", _Priv("priv"))
        c.connect("__in__.text", "priv.x")
        c.connect("priv.y", "__out__.result")
        return _parent_embedding(c)

    async def test_union_surfaced_on_wrapper(self):
        g = self._priv_parent()
        assert g.node("sub").instance.required_permissions == {"shell:execute"}

    async def test_denied_without_grant(self):
        g = self._priv_parent()
        _seed(g, "msg", "hi")
        r = await FNDirector().run(g, RunContext(run_id="t"))
        assert r.converged is False
        assert "shell:execute" in (r.error or "")

    async def test_allowed_with_grant(self):
        g = self._priv_parent()
        _seed(g, "msg", "hi")
        r = await FNDirector().run(
            g,
            RunContext(run_id="t", granted_permissions={"shell:execute"}),
        )
        assert r.converged is True
        assert _out(g, "__out__", "out") == ["hi"]


class TestNestedTrace:
    async def test_child_spans_nested_under_subgraph(self):
        g = _parent_embedding(_child())
        _seed(g, "msg", "hello")
        tracer = InMemoryTracer()
        await FNDirector().run(g, RunContext(run_id="t", tracer=tracer))
        spans = {s.name: s for s in tracer.spans}
        assert "entity.fire.sub" in spans
        assert "entity.fire.inner" in spans
        # The child's inner span must descend from the subgraph span.
        assert spans["entity.fire.inner"].parent_span_id == spans["entity.fire.sub"].id


class TestNamespacedReplay:
    async def test_replay_reproduces_without_executing_child(self):
        # Record a live run.
        child = _child()
        g = _parent_embedding(child)
        _seed(g, "msg", "hello")
        rec = RunRecorder()
        await FNDirector().run(g, RunContext(run_id="t", recorder=rec))
        data = rec.to_dict()
        # Boundary capture + namespaced internal capture both present.
        assert "sub:0" in data
        assert "sub/inner:0" in data

        # Replay on a fresh graph with NO grant — child must not execute.
        child2 = _child()
        g2 = _parent_embedding(child2)
        _seed(g2, "msg", "hello")
        inner2 = child2.node("inner").instance
        await FNDirector().run(
            g2, RunContext(run_id="t2", replayer=RunReplayer(data))
        )
        assert inner2.calls == 0  # child never ran
        assert _out(g2, "__out__", "out") == ["out:hello"]


class TestIRSubgraphRef:
    def _write(self, tmp_path, name: str, text: str):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_load_subgraph_ref(self, tmp_path):
        from pharos.ir import load_graph

        self._write(
            tmp_path,
            "child.yaml",
            "name: child\ndirector: fn\n"
            "nodes:\n  - {id: a, type: faux, provider: faux, model: faux-fast}\n"
            "edges:\n"
            "  - { src: __in__.prompt, dst: a.prompt }\n"
            "  - { src: a.text, dst: __out__.result }\n",
        )
        parent = self._write(
            tmp_path,
            "parent.yaml",
            "name: parent\ndirector: fn\n"
            "nodes:\n  - {id: sub, type: subgraph, ref: child.yaml}\n"
            "edges:\n"
            "  - { src: __in__.prompt, dst: sub.prompt }\n"
            "  - { src: sub.result, dst: __out__.out }\n",
        )
        g, _ = load_graph(parent)
        from pharos.entities.subgraph import SubgraphEntity

        assert isinstance(g.node("sub").instance, SubgraphEntity)
        assert g.node("sub").instance.director_name == "fn"

    def test_cycle_detection(self, tmp_path):
        from pharos.ir import load_graph

        self._write(
            tmp_path,
            "a.yaml",
            "name: a\nnodes:\n  - {id: s, type: subgraph, ref: b.yaml}\n"
            "edges: []\n",
        )
        self._write(
            tmp_path,
            "b.yaml",
            "name: b\nnodes:\n  - {id: s, type: subgraph, ref: a.yaml}\n"
            "edges: []\n",
        )
        with pytest.raises(ValueError, match="circular subgraph reference"):
            load_graph(tmp_path / "a.yaml")

    def _child_and_parent(self, tmp_path, ref_line: str):
        from pharos.ir import _file_sha256, load_graph

        child = self._write(
            tmp_path,
            "child.yaml",
            "name: child\ndirector: fn\n"
            "nodes:\n  - {id: a, type: faux, provider: faux, model: faux-fast}\n"
            "edges:\n"
            "  - { src: __in__.prompt, dst: a.prompt }\n"
            "  - { src: a.text, dst: __out__.result }\n",
        )
        parent = self._write(
            tmp_path,
            "parent.yaml",
            "name: parent\ndirector: fn\n"
            f"nodes:\n  - {{id: sub, type: subgraph, ref: child.yaml{ref_line}}}\n"
            "edges:\n"
            "  - { src: __in__.prompt, dst: sub.prompt }\n"
            "  - { src: sub.result, dst: __out__.out }\n",
        )
        return _file_sha256(child), parent, load_graph

    def test_ref_sha_matches(self, tmp_path):
        sha, parent, load_graph = self._child_and_parent(
            tmp_path, ", ref_sha: __PLACEHOLDER__"
        )
        # Rewrite the parent with the real hash prefix now that we know it.
        parent.write_text(
            parent.read_text(encoding="utf-8").replace(
                "__PLACEHOLDER__", sha[:16]
            ),
            encoding="utf-8",
        )
        g, _ = load_graph(parent)
        from pharos.entities.subgraph import SubgraphEntity

        assert isinstance(g.node("sub").instance, SubgraphEntity)

    def test_ref_sha_mismatch_rejected(self, tmp_path):
        _sha, parent, load_graph = self._child_and_parent(
            tmp_path, ", ref_sha: deadbeefdeadbeef"
        )
        with pytest.raises(ValueError, match="does not match pinned ref_sha"):
            load_graph(parent)
