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

Version 2 fingerprints ordered output-boundary records, including lineage.
Version 1 remains available for existing fixtures and is payload-only and
order-independent.
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


def _record_line(key: str, index: int, rec: dict[str, Any], version: int) -> str:
    """Canonical one-line representation of a single emitted-token record."""
    entry = {
        "key": key,
        "index": index,
        "port": rec.get("port"),
        "type": rec.get("type"),
        "payload": _canonical(rec.get("payload")),
    }
    if version >= 2:
        entry.update(
            {
                "origin": rec.get("origin"),
                "prev_hash": rec.get("prev_hash"),
                "self_hash": rec.get("self_hash"),
            }
        )
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)


def chain_digest(outputs: Outputs, *, version: int = 2) -> str:
    """Return a stable sha256 hex fingerprint of a run's outputs.

    Mapping key order is irrelevant.  Version 2 preserves record order within
    each node fire and includes lineage.  Version 1 sorts records for backward
    compatibility with fixtures recorded by pharos <= 0.2.
    """
    lines = [
        _record_line(key, index, rec, version)
        for key in sorted(outputs)
        for index, rec in enumerate(outputs[key])
    ]
    if version <= 1:
        # Recreate the old representation exactly: it had no index/lineage.
        lines = [
            json.dumps(
                {
                    "key": key,
                    "payload": _canonical(rec.get("payload")),
                    "port": rec.get("port"),
                    "type": rec.get("type"),
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            for key in sorted(outputs)
            for rec in outputs[key]
        ]
        lines.sort()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


__all__ = ["Outputs", "canonical_payload", "chain_digest"]
