"""PermissionPolicy — one place that decides whether a capability is allowed.

Historically pharos checked permissions in two unrelated spots with two
different code paths:

- entity-level (e.g. ``ShellEntity`` needs ``shell:execute``) — checked by
  the Director before ``fire()``;
- tool-level (e.g. the ``bash`` tool needs ``bash:execute``) — checked by
  ``ToolRegistry`` inside ``LLMAgent.fire()``.

That meant two moments, two code paths, and overlapping capability names
(``shell:execute`` vs ``bash:execute`` are the same capability). This module
centralises the decision so every enforcement point routes through the same
policy and the same alias table.

A permission string is ``"<resource>:<action>"`` (e.g. ``fs:read``). Aliases
let differently-named-but-equivalent capabilities collapse onto one canonical
name, so granting the canonical name authorises all its aliases.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Canonical capability -> the alternative spellings that mean the same thing.
# Granting either side authorises both. Keep this list small and obvious.
_ALIASES: dict[str, str] = {
    # "bash:execute" and "shell:execute" both mean "run a shell command".
    "bash:execute": "shell:execute",
}


def canonical(permission: str) -> str:
    """Collapse a permission onto its canonical name (identity if none)."""
    return _ALIASES.get(permission, permission)


@dataclass(frozen=True)
class PermissionPolicy:
    """An immutable authorisation decision function for a single run.

    Build one from the run's granted permissions and ask it whether a
    required set/permission is allowed. Both the Director and the
    ToolRegistry use the same instance so enforcement is uniform.
    """

    granted: frozenset[str]

    @classmethod
    def from_grants(cls, grants: Iterable[str] | None) -> PermissionPolicy:
        """Build a policy from raw grant strings (canonicalising aliases)."""
        normalised = {canonical(g) for g in (grants or set())}
        return cls(granted=frozenset(normalised))

    def allows(self, permission: str) -> bool:
        """True if a single permission is authorised by this policy."""
        return canonical(permission) in self.granted

    def missing(self, required: Iterable[str] | None) -> set[str]:
        """Return the subset of `required` that is NOT granted.

        The returned names are the *original* (pre-canonical) spellings so
        error messages match what the caller declared.
        """
        out: set[str] = set()
        for perm in required or set():
            if canonical(perm) not in self.granted:
                out.add(perm)
        return out

    def check(self, required: Iterable[str] | None, *, subject: str) -> None:
        """Raise PermissionError if any `required` permission is missing.

        `subject` is a human-readable description of what is being gated
        (an entity id/class or a tool name) for the error message.
        """
        missing = self.missing(required)
        if missing:
            granted = sorted(self.granted) if self.granted else "no permissions"
            raise PermissionError(
                f"{subject} requires {sorted(missing)} but run only grants "
                f"{granted}"
            )


__all__ = ["PermissionPolicy", "canonical"]
