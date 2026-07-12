"""Fixture — a golden agent run captured to JSON and checked into the repo.

A fixture is the regression artifact behind ``pharos test``: it records a
graph run that a human has confirmed is correct, so later runs can be checked
against it — offline (replay the recorded outputs and re-assert) or live
(re-run the real model and diff against this golden).

Format (JSON, versioned)::

    {
      "version": 1,
      "name": "fix-typo",
      "graph_path": "graphs/coding-agent.yaml",
      "graph_sha256": "…",         # integrity: warn if the graph changed
      "director": "fn",
      "seed": {"prompt": "…"},      # __in__ seeds, replayed identically
      "grant": ["fs:read"],
      "var": {"model": "…"},
      "outputs": { "<node>:<fire>": [ {port,type,payload,self_hash,…}, … ] },
      "chain_digest": "…",          # fingerprint of outputs (offline gate)
      "assertions": [ {target, op, value, policy}, … ]
    }
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pharos.core.schema import validate as _validate_schema
from pharos.testing.digest import Outputs, canonical_payload, chain_digest

FIXTURE_VERSION = 2

# Field policies used by the live verdict (Phase 3). Declared here so a fixture
# can carry them, but the offline gate treats every declared assertion as hard.
POLICIES = ("pinned", "toleranced", "ignored")


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file's raw bytes (graph integrity pin)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Assertion:
    """A single declared expectation about a target port's output.

    ``target`` is ``"<node>.<port>"`` (fire 0) or ``"<node>:<fire>.<port>"``,
    resolved against the run's outputs by the runner. ``op`` is one of
    ``equals`` / ``contains`` / ``not_contains`` / ``regex`` / ``schema``.
    ``policy`` steers the live verdict (pinned/toleranced/ignored).
    """

    target: str
    op: str
    value: Any = None
    policy: str = "pinned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op": self.op,
            "value": self.value,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Assertion:
        return cls(
            target=d["target"],
            op=d["op"],
            value=d.get("value"),
            policy=d.get("policy", "pinned"),
        )

    def check(self, values: list[Any]) -> tuple[bool, str]:
        """Evaluate this assertion against the payloads emitted at ``target``.

        ``values`` is the list of payloads the target port emitted (usually
        one). Returns ``(ok, detail)`` where detail is a human-readable reason.
        """
        if not values:
            # A missing target is a pass only for not_contains (nothing there
            # certainly does not contain the needle); everything else fails.
            return (self.op == "not_contains", f"no value emitted at {self.target}")
        v = values[-1]
        if self.op == "equals":
            ok = canonical_payload(v) == canonical_payload(self.value)
            return ok, f"expected == {self.value!r}, got {v!r}"
        if self.op == "contains":
            return self._contains(v, self.value), (
                f"expected {v!r} to contain {self.value!r}"
            )
        if self.op == "not_contains":
            return (not self._contains(v, self.value)), (
                f"expected {v!r} NOT to contain {self.value!r}"
            )
        if self.op == "regex":
            ok = re.search(str(self.value), str(v)) is not None
            return ok, f"expected {v!r} to match /{self.value}/"
        if self.op == "schema":
            errors = _validate_schema(v, self.value or {})
            return (not errors), ("; ".join(errors) if errors else "schema ok")
        return False, f"unknown assertion op {self.op!r}"

    @staticmethod
    def _contains(v: Any, needle: Any) -> bool:
        if isinstance(v, (list, tuple)):
            return needle in v or any(
                isinstance(x, str) and str(needle) in x for x in v
            )
        if isinstance(v, dict):
            return needle in v
        return str(needle) in str(v)


@dataclass
class Fixture:
    """A recorded golden run plus its assertions."""

    name: str
    graph_path: str
    graph_sha256: str
    director: str
    seed: dict[str, str] = field(default_factory=dict)
    grant: list[str] = field(default_factory=list)
    var: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    chain_digest: str = ""
    assertions: list[Assertion] = field(default_factory=list)
    version: int = FIXTURE_VERSION

    @classmethod
    def build(
        cls,
        *,
        name: str,
        graph_path: str | Path,
        director: str,
        outputs: Outputs,
        seed: dict[str, str] | None = None,
        grant: list[str] | None = None,
        var: dict[str, str] | None = None,
        assertions: list[Assertion] | None = None,
    ) -> Fixture:
        """Construct a fixture from a run's outputs, computing hashes."""
        return cls(
            name=name,
            graph_path=str(graph_path),
            graph_sha256=file_sha256(graph_path),
            director=director,
            seed=dict(seed or {}),
            grant=list(grant or []),
            var=dict(var or {}),
            outputs={k: list(v) for k, v in outputs.items()},
            chain_digest=chain_digest(outputs, version=FIXTURE_VERSION),
            assertions=list(assertions or []),
        )

    def graph_matches(self, graph_path: str | Path | None = None) -> bool:
        """True if the graph file's bytes still match the pinned sha256."""
        path = graph_path or self.graph_path
        try:
            return file_sha256(path) == self.graph_sha256
        except OSError:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "graph_path": self.graph_path,
            "graph_sha256": self.graph_sha256,
            "director": self.director,
            "seed": self.seed,
            "grant": self.grant,
            "var": self.var,
            "outputs": self.outputs,
            "chain_digest": self.chain_digest,
            "assertions": [a.to_dict() for a in self.assertions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Fixture:
        return cls(
            name=d.get("name", ""),
            graph_path=d.get("graph_path", ""),
            graph_sha256=d.get("graph_sha256", ""),
            director=d.get("director", "fn"),
            seed=d.get("seed", {}),
            grant=d.get("grant", []),
            var=d.get("var", {}),
            outputs=d.get("outputs", {}),
            chain_digest=d.get("chain_digest", ""),
            assertions=[Assertion.from_dict(a) for a in d.get("assertions", [])],
            # Unversioned fixtures predate the ordered/lineage digest.
            version=d.get("version", 1),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> Fixture:
        # utf-8-sig tolerates a leading BOM some editors/tools add on re-save.
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(data)


__all__ = ["FIXTURE_VERSION", "POLICIES", "Assertion", "Fixture", "file_sha256"]
