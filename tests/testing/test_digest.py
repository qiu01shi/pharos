"""Tests for pharos.testing.digest (chain_digest) + Phase 0 hash stability."""
from __future__ import annotations

import hashlib

from pharos.testing.digest import chain_digest


def _outputs() -> dict:
    return {
        "a:0": [{"port": "y", "type": "text", "payload": "hi"}],
        "b:0": [
            {"port": "json", "type": "json", "payload": {"line": 42, "ok": True}}
        ],
    }


class TestChainDigest:
    def test_deterministic(self):
        assert chain_digest(_outputs()) == chain_digest(_outputs())

    def test_order_independent(self):
        reordered = {
            "b:0": [
                {"port": "json", "type": "json", "payload": {"ok": True, "line": 42}}
            ],
            "a:0": [{"port": "y", "type": "text", "payload": "hi"}],
        }
        assert chain_digest(_outputs()) == chain_digest(reordered)

    def test_changes_with_payload(self):
        changed = {
            "a:0": [{"port": "y", "type": "text", "payload": "hi"}],
            "b:0": [
                {"port": "json", "type": "json", "payload": {"line": 43, "ok": True}}
            ],
        }
        assert chain_digest(_outputs()) != chain_digest(changed)

    def test_ignores_lineage_metadata(self):
        # A run recorded with self_hash/prev_hash/origin must digest the same
        # as the bare payload — chain_digest is content-based.
        with_meta = {
            "a:0": [
                {
                    "port": "y",
                    "type": "text",
                    "payload": "hi",
                    "self_hash": "deadbeef",
                    "prev_hash": None,
                    "origin": "a.y",
                }
            ],
            "b:0": [
                {"port": "json", "type": "json", "payload": {"line": 42, "ok": True}}
            ],
        }
        assert chain_digest(with_meta) == chain_digest(_outputs())

    def test_empty(self):
        assert chain_digest({}) == hashlib.sha256(b"").hexdigest()


class TestCrossRunHashStability:
    """Phase 0: the same graph + same input reproduces identical hashes."""

    async def _run_once(self) -> dict:
        from pharos.core.entity import Entity, entity
        from pharos.core.graph import CompositeGraph
        from pharos.core.port import InputPort, OutputPort
        from pharos.core.token import TypedValue
        from pharos.directors.base import RunContext
        from pharos.directors.fn import FNDirector
        from pharos.runtime import RunRecorder

        @entity
        class _Upper(Entity):
            ins = {"x": InputPort(name="x", accepted_types=["text"])}
            outs = {"y": OutputPort(name="y", accepted_types=["text"])}

            async def fire(self, ctx):  # type: ignore[override]
                for t in self.ins["x"].consume():
                    self.outs["y"].emit(
                        TypedValue(type="text", payload=t.value.payload.upper())
                    )

        g = CompositeGraph(name="det")
        g.add_entity("up", _Upper("up"))
        g.connect("__in__.p", "up.x")
        g.connect("up.y", "__out__.done")
        for e in g.edges:
            if e.src_node == "__in__" and e.src_port == "p":
                g.node(e.dst_node).instance.ins[e.dst_port].emit(  # type: ignore[union-attr]
                    TypedValue(type="text", payload="hello")
                )
        rec = RunRecorder()
        # Distinct run_id each call proves run_id does NOT enter the hash.
        import uuid

        ctx = RunContext(run_id=str(uuid.uuid4()), recorder=rec)
        await FNDirector().run(g, ctx)
        return rec.to_dict()

    async def test_self_hash_and_digest_stable_across_runs(self):
        o1 = await self._run_once()
        o2 = await self._run_once()
        # Payload is what we expect, and lineage was stamped.
        assert o1["up:0"][0]["payload"] == "HELLO"
        assert o1["up:0"][0]["origin"] == "up.y"
        # Same content -> same token hash across two independent runs.
        assert o1["up:0"][0]["self_hash"] == o2["up:0"][0]["self_hash"]
        assert chain_digest(o1) == chain_digest(o2)
