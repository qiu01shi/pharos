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
    # Any name resolvable by `_resolve_provider_class` (static map or the
    # runtime registry). Kept as a free string so newly registered
    # providers work in YAML without editing this schema; unknown names
    # raise a clear error at load time in `_resolve_provider_class`.
    provider: str = "openai"
    model: str
    system: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_level: str | None = None
    # Tool preset: "coding" = bash/read/write/edit/delete/glob/grep
    #              "builtin" = echo/get_time/add_numbers
    #              "none" (default) = no tools
    tools: Literal["none", "coding", "builtin"] = "none"
    max_tool_iterations: int = 5
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
    # Anthropic / DeepSeek / MiniMax are loaded from the registry
    # at runtime, so we resolve them by string lookup below.
    # See `_resolve_provider_class`.
}


def _resolve_provider_class(name: str) -> type:
    """Resolve a provider name to a class.

    Try the static map first; fall back to the registry. This lets
    new providers (Anthropic / DeepSeek / MiniMax) work in YAML
    without modifying this module.
    """
    if name in _PROVIDER_CLASSES:
        return _PROVIDER_CLASSES[name]
    # Lazy import to avoid circular dependencies
    from pharos.llm.registry import get_provider_class, list_providers

    try:
        return get_provider_class(name)
    except KeyError:
        known = sorted(set(list(_PROVIDER_CLASSES) + list(list_providers())))
        raise ValueError(
            f"unknown LLM provider: {name!r}. Known: {known}"
        ) from None


def _build_entity(node_raw: dict[str, Any]) -> Entity:
    """Construct a pharos Entity from an IR node dict."""
    ntype = node_raw.get("type")
    nid = node_raw.get("id")
    if not ntype or not nid:
        raise ValueError(f"node missing id/type: {node_raw}")

    if ntype in ("llm", "faux"):
        spec = LLMNodeSpec.model_validate(node_raw)
        provider_class = _resolve_provider_class(spec.provider)

        # Auto-build ToolRegistry from the `tools` preset
        tool_registry = None
        if spec.tools == "coding":
            from pharos.entities.tools import ToolRegistry
            from pharos.entities.tools_coding import register_coding_tools

            tool_registry = ToolRegistry()
            register_coding_tools(tool_registry)
        elif spec.tools == "builtin":
            from pharos.entities.tools import ToolRegistry
            from pharos.entities.tools_builtins import register_builtins

            tool_registry = ToolRegistry()
            register_builtins(tool_registry)

        cfg = LLMEntityConfig(
            provider_class=provider_class,
            provider_kwargs=dict(spec.params),
            model_id=spec.model,
            system_prompt=spec.system,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            thinking_level=spec.thinking_level,  # type: ignore[arg-type]
            tool_registry=tool_registry,
            max_tool_iterations=spec.max_tool_iterations,
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


def load_graph_from_text(
    text: str,
) -> tuple[CompositeGraph, dict[str, Any]]:
    """Like `load_graph` but from a string (after var substitution).

    Returns (graph, raw_spec_dict).
    """
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("graph YAML must be a mapping")
    return load_graph_from_dict(raw)


def load_graph_from_dict(
    raw: dict[str, Any],
) -> tuple[CompositeGraph, dict[str, Any]]:
    """Like `load_graph` but from an already-parsed dict.

    Returns (graph, raw_spec_dict).
    """
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
        director = raw.get("director", "fn") if isinstance(raw, dict) else "fn"
        only_cycle = all("cycle" in e for e in errors)
        if not (only_cycle and director in ("sdf",)):
            raise ValueError(f"graph validation failed: {errors}")

    return g, raw


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
    return load_graph_from_dict(raw)


__all__ = [
    "EdgeSpec",
    "GraphSpec",
    "LLMNodeSpec",
    "PythonNodeSpec",
    "ShellNodeSpec",
    "load_graph",
    "load_graph_from_dict",
    "load_graph_from_text",
]
