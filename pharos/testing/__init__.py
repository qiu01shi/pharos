"""pharos.testing — Agent CI: make an agent run testable and diffable.

This package turns a recorded run into a regression artifact:

- ``digest``  — a canonical, cross-run-stable fingerprint of a run's
  per-node outputs (``chain_digest``). Same inputs + same graph reproduce
  the same digest, so a changed digest means something in the runtime or
  graph structure drifted.
- ``diff``    — a structured, node/port-aligned comparison of two runs'
  outputs, with downstream propagation, instead of an opaque text diff.
- ``fixture`` — a golden run captured to JSON (graph hash, seed, outputs,
  chain_digest, assertions) that lives in the repo.
- ``runner``  — the offline gate (replay + assert) and the live drift check.
"""

from __future__ import annotations

from pharos.testing.diff import (
    FieldChange,
    PortDiff,
    RunDiff,
    diff_json,
    diff_runs,
)
from pharos.testing.digest import canonical_payload, chain_digest
from pharos.testing.fixture import Assertion, Fixture, file_sha256

__all__ = [
    "Assertion",
    "FieldChange",
    "Fixture",
    "PortDiff",
    "RunDiff",
    "canonical_payload",
    "chain_digest",
    "diff_json",
    "diff_runs",
    "file_sha256",
]
