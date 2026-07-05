"""Token and TypedValue — the atoms flowing through pharos ports.

Design notes:
- TypedValue is the schema-tagged payload (text / json / image / code / ...).
- Token wraps a TypedValue with lineage: ts, origin, prev_hash, self_hash.
  The hash chain enables replay (re-execute the same lineage) and
  convergence detection (compare head tokens across iterations).
- Tokens are immutable; the only way to "mutate" is to produce a new one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TypedValue:
    """Schema-tagged value carried by a Token.

    `type` is a free-form string ("text", "json", "image", "code",
    "usage", "error", "tool_call", ...). Validation against an actual
    schema (Pydantic model) happens at the Port boundary; this class is
    just a payload + tag.
    """

    type: str
    payload: Any


@dataclass(frozen=True)
class Token:
    """Single unit of data flowing between entities.

    Hash chain:
        self_hash = sha256(serialize(value) + origin + prev_hash + iter)
    The chain lets a Director prove "this run produced the same outputs
    as that run" by comparing head hashes — without re-running the LLM.
    Wall-clock ``ts`` and ``run_id`` are deliberately excluded from the
    hash so the same payload produced by the same node hashes identically
    across runs; that cross-run stability is what makes drift detection
    and a byte-stable ``chain_digest`` possible.
    """

    value: TypedValue
    origin: str = ""  # "node_id.port_name" — who produced it
    ts: float = field(default_factory=time.time)
    prev_hash: str | None = None
    self_hash: str = ""
    run_id: str = ""
    iter: int = 0
    is_partial: bool = False
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Compute self_hash if not provided
        if not self.self_hash:
            object.__setattr__(self, "self_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        """Stable canonical hash. Order-independent for dicts in payload.

        Excludes ``ts`` and ``run_id`` on purpose: identity is content +
        lineage (``type``, ``payload``, ``origin``, ``prev_hash``, ``iter``),
        never wall-clock time, so re-running the same graph with the same
        input reproduces the same hashes.
        """
        canonical = {
            "type": self.value.type,
            "payload": _canonical(self.value.payload),
            "origin": self.origin,
            "prev_hash": self.prev_hash,
            "iter": self.iter,
        }
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def with_prev(self, prev: Token | None) -> Token:
        """Return a new Token with the previous token's hash set."""
        return Token(
            value=self.value,
            origin=self.origin,
            ts=self.ts,
            prev_hash=prev.self_hash if prev else None,
            run_id=self.run_id,
            iter=self.iter,
            is_partial=self.is_partial,
            cost_usd=self.cost_usd,
            metadata=self.metadata,
        )


def _canonical(obj: Any) -> Any:
    """Make obj JSON-serializable in a deterministic way."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    # Fall back to repr for unknown types (TypedValue, dataclasses, etc.)
    return repr(obj)


__all__ = ["Token", "TypedValue"]
