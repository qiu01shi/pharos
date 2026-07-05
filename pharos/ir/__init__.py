"""pharos IR: YAML/JSON graph description + loader.

A pharos graph YAML has this shape:

    name: hello
    director: fn              # which Director to use
    nodes:
      - id: agent
        type: llm
        provider: glm
        model: glm-4.5-air
        system: "be brief"
      - id: shell_run
        type: shell
        timeout: 30
    edges:
      - { src: __in__.prompt, dst: agent.prompt }
      - { src: agent.text,   dst: shell_run.command }

The runtime treats `__in__.<port>` and `__out__.<port>` as virtual
endpoints. To inject input, call `director.run(graph, ctx, inputs=...)`
or set the seed via a real Entity in your test (see FNDirector docs).
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from pharos.core.entity import Entity
from pharos.core.graph import CompositeGraph
from pharos.entities.llm import LLMAgent, LLMEntityConfig
from pharos.entities.shell import ShellEntity
from pharos.llm.providers.faux import FauxConfig, FauxProvider  # noqa: F401
from pharos.llm.providers.glm import GLMProvider
from pharos.llm.providers.openai import OpenAIProvider

DirectorName = Literal["fn", "sdf", "de"]  # PN/CT added in later phases


# ---------- node specs ----------


class LLMNodeSpec(BaseModel):
    """Configuration for a `type: llm` node."""

    id: str
    type: Literal["llm", "faux"] = "llm"
    provider: Literal["openai", "glm", "faux"] = "openai"
    model: str
    system: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_level: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ShellNodeSpec(BaseModel):
    """Configuration for a `type: shell` node."""

    id: str
    type: Literal["shell"] = "shell"
    timeout: float = 300.0
    cwd: str | None = None


class PythonNodeSpec(BaseModel):
    """Configuration for a `type: python` node.

    Imports a user-defined Entity class from a module path. Format:
        type: python
        class: "my_pkg.my_module:WordCounter"
        params:
            threshold: 0.5
            mode: "strict"
    """

    id: str
    type: Literal["python"] = "python"
    class_: str = Field(alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_")
    @classmethod
    def _check_class_format(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError(
                f"python class spec must be 'module.path:ClassName', got {v!r}"
            )
        module, _, _name = v.partition(":")
        if not module or not _name:
            raise ValueError(
                f"python class spec must be 'module.path:ClassName', got {v!r}"
            )
        return v


# ---------- graph spec ----------


class EdgeSpec(BaseModel):
    src: str
    dst: str


class GraphSpec(BaseModel):
    name: str = "graph"
    director: DirectorName = "fn"
    nodes: list[dict[str, Any]]
    edges: list[EdgeSpec] = Field(default_factory=list)

    @field_validator("nodes")
    @classmethod
    def _ensure_ids_unique(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [n.get("id") for n in v]
        if len(set(ids)) != len(ids):
            dups = {x for x in ids if ids.count(x) > 1}
            raise ValueError(f"duplicate node ids: {dups}")
        return v


# ---------- loader ----------


_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "glm": GLMProvider,
    "faux": FauxProvider,
}


def _build_entity(node_raw: dict[str, Any]) -> Entity:
    """Construct a pharos Entity from an IR node dict."""
    ntype = node_raw.get("type")
    nid = node_raw.get("id")
    if not ntype or not nid:
        raise ValueError(f"node missing id/type: {node_raw}")

    if ntype in ("llm", "faux"):
        spec = LLMNodeSpec.model_validate(node_raw)
        provider_class = _PROVIDER_CLASSES[spec.provider]
        # For faux, the model_id is just a hint; FauxProvider ignores it
        cfg = LLMEntityConfig(
            provider_class=provider_class,
            provider_kwargs=dict(spec.params),
            model_id=spec.model,
            system_prompt=spec.system,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            thinking_level=spec.thinking_level,  # type: ignore[arg-type]
        )
        return LLMAgent(node_id=nid, config=cfg)

    if ntype == "shell":
        spec = ShellNodeSpec.model_validate(node_raw)
        return ShellEntity(
            node_id=nid,
            timeout=spec.timeout,
            cwd=spec.cwd,
        )

    if ntype == "python":
        spec = PythonNodeSpec.model_validate(node_raw)
        return _build_python_entity(nid, spec)

    raise ValueError(f"unknown node type: {ntype!r}")


def _build_python_entity(nid: str, spec: PythonNodeSpec) -> Entity:
    """Resolve a `type: python` class path and instantiate.

    Class path is "module.path:ClassName". The module is imported
    on demand; the class must inherit from `pharos.core.entity.Entity`
    and accept `node_id` as a constructor keyword.
    """
    module_path, _, class_name = spec.class_.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"python node {nid!r}: cannot import module {module_path!r}: {e}"
        ) from e
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(
            f"python node {nid!r}: module {module_path!r} has no class {class_name!r}"
        )
    if not isinstance(cls, type) or not issubclass(cls, Entity):
        raise ValueError(
            f"python node {nid!r}: class {class_name!r} is not an Entity subclass"
        )
    # Try the common signature: (node_id, **params). Fall back to
    # (node_id) and let setup() read from spec.params separately.
    try:
        return cls(node_id=nid, **spec.params)  # type: ignore[call-arg,abstract]
    except TypeError:
        return cls(node_id=nid)  # type: ignore[call-arg,abstract]


def load_graph(
    path: str | Path,
) -> tuple[CompositeGraph, dict[str, Any]]:
    """Load a YAML graph spec into a CompositeGraph.

    Returns (graph, raw_spec_dict). The raw dict is returned so the
    CLI can show validation messages with full context.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    spec = GraphSpec.model_validate(raw)

    g = CompositeGraph(name=spec.name)
    for node_raw in spec.nodes:
        entity = _build_entity(node_raw)
        g.add_entity(entity.node_id, entity)

    for e in spec.edges:
        g.connect(e.src, e.dst)

    errors = g.validate()
    if errors:
        # Allow cycles if the director supports them (SDF).
        # The Director itself decides whether to honor feedback.
        director = raw.get("director", "fn") if isinstance(raw, dict) else "fn"
        only_cycle = all("cycle" in e for e in errors)
        if not (only_cycle and director in ("sdf",)):
            raise ValueError(f"graph validation failed: {errors}")

    return g, raw


__all__ = [
    "EdgeSpec",
    "GraphSpec",
    "LLMNodeSpec",
    "ShellNodeSpec",
    "load_graph",
]
