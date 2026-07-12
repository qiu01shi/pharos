"""First-class team specifications compiled into ordinary Pharos graphs."""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from pharos.ir.builder import GraphBuilder


class TeamMemberSpec(BaseModel):
    id: str
    model: str
    provider: str = "openai"
    system: str = ""
    tools: Literal["none", "coding", "builtin"] = "none"
    config: dict[str, Any] = Field(default_factory=dict)


class TerminationSpec(BaseModel):
    max_iterations: int = Field(default=20, ge=1)
    convergence_k: int = Field(default=2, ge=1)


class TeamSpec(BaseModel):
    """Reusable collaboration pattern that lowers to a typed graph.

    ``pipeline`` is an acyclic sequence. ``reflection`` uses exactly two
    members (producer, reviewer) and lowers to an SDF feedback graph.  More
    patterns can be added as macros without adding scheduling semantics.
    """

    name: str
    pattern: Literal["pipeline", "reflection"] = "pipeline"
    members: list[TeamMemberSpec]
    termination: TerminationSpec = Field(default_factory=TerminationSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_members(self) -> TeamSpec:
        if not self.members:
            raise ValueError("a team requires at least one member")
        ids = [member.id for member in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("team member ids must be unique")
        if self.pattern == "reflection" and len(self.members) != 2:
            raise ValueError("reflection teams require producer and reviewer")
        return self

    def to_builder(self) -> GraphBuilder:
        director: Literal["fn", "sdf"] = (
            "sdf" if self.pattern == "reflection" else "fn"
        )
        builder = GraphBuilder(
            self.name,
            director=director,
            metadata={**self.metadata, "team_pattern": self.pattern},
        ).execution(
            max_iterations=self.termination.max_iterations,
            convergence_k=self.termination.convergence_k,
        )
        refs = []
        for member in self.members:
            refs.append(
                builder.llm(
                    member.id,
                    model=member.model,
                    provider=member.provider,
                    system=member.system,
                    tools=member.tools,
                    **member.config,
                )
            )

        builder.connect(builder.input.output("prompt"), refs[0].input("prompt"))
        for left, right in pairwise(refs):
            builder.connect(left.output("text"), right.input("prompt"))
        if self.pattern == "reflection":
            builder.connect(refs[-1].output("text"), refs[0].input("prompt"))
        builder.connect(refs[-1].output("text"), builder.output.input("response"))
        return builder

    def to_dict(self) -> dict[str, Any]:
        return self.to_builder().to_dict()

    def build(self):
        return self.to_builder().build()


__all__ = ["TeamMemberSpec", "TeamSpec", "TerminationSpec"]
