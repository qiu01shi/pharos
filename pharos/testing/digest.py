"""chain_digest — a cross-run-stable fingerprint of a run's outputs.

A recorded run's ``outputs`` map (from ``RunRecorder.to_dict()`` /
``runtime.get_run_outputs()``) looks like::

    {
        "<node_id>:<fire_index>": [
            {"port": "text", "type": "text", "payload": "...", ...},
            ...
        ],
        ...
    }

``chain_digest`` reduces that whole structure to a single sha256 hex
string over the *content* of every emitted token (node/fire, port, type,
canonical payload). Because Phase 0 made ``Token`` hashing exclude
wall-clock time, the same graph + same input reproduces the same digest.
A changed digest therefore means the runtime path or graph structure
produced different data — the offline regression signal in ``pharos test``.

The digest is deliberately payload-based (not the stored ``self_hash``) so
it is robust to lineage-stamping details evolving over time and depends
only on what each node actually output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pharos.core.token import _canonical

# An "outputs" map: "<node>:<fire>" -> list of emitted-token records.
Outputs = Mapping[str, list[dict[str, Any]]]


def canonical_payload(payload: Any) -> Any:
    """Deterministically normalize a payload for hashing/diffing.

    Thin re-export of the same canonicalizer ``Token`` uses, so digests,
    diffs, and token hashes all agree on what "the same value" means.
    """
    return _canonical(payload)


def _record_line(key: str, rec: dict[str, Any]) -> str:
    """Canonical one-line representation of a single emitted-token record."""
    entry = {
        "key": key,
        "port": rec.get("port"),
        "type": rec.get("type"),
        "payload": _canonical(rec.get("payload")),
    }
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)


def chain_digest(outputs: Outputs) -> str:
    """Return a stable sha256 hex fingerprint of a run's outputs.

    Order-independent: records are canonicalized and sorted, so neither the
    iteration order of nodes nor of ports within a node affects the result.
    An empty run digests to the sha256 of the empty string.
    """
    lines = [
        _record_line(key, rec)
        for key in sorted(outputs)
        for rec in outputs[key]
    ]
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


__all__ = ["Outputs", "canonical_payload", "chain_digest"]
