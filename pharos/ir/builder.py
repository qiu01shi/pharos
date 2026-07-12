"""Ergonomic Python authoring API that compiles to the versioned Pharos IR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml


@dataclass(frozen=True)
class PortRef:
    node_id: str
    port: str

    def __str__(self) -> str:
        return f"{self.node_id}.{self.port}"


@dataclass(frozen=True)
class NodeRef:
    id: str

    def input(self, port: str) -> PortRef:
        return PortRef(self.id, port)

    def output(self, port: str) -> PortRef:
        return PortRef(self.id, port)


class GraphBuilder:
    """Build a graph in Python without creating a second runtime model.

    The builder only constructs the same versioned dictionary accepted by the
    YAML loader.  ``build()`` always passes through GraphSpec validation and the
    regular compiler, so Python, YAML, and future Studio authoring stay aligned.
    """

    def __init__(
        self,
        name: str,
        *,
        director: Literal["fn", "sdf", "de"] = "fn",
        metadata: dict[str, Any] | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        from pharos.ir import IR_VERSION

        self._raw: dict[str, Any] = {
            "apiVersion": IR_VERSION,
            "kind": "Graph",
            "metadata": dict(metadata or {}),
            "name": name,
            "director": director,
            "nodes": [],
            "edges": [],
        }
        self.base_dir = Path(base_dir) if base_dir is not None else None

    @property
    def input(self) -> NodeRef:
        return NodeRef("__in__")

    @property
    def output(self) -> NodeRef:
        return NodeRef("__out__")

    def add_node(self, node_id: str, node_type: str, **config: Any) -> NodeRef:
        if any(node.get("id") == node_id for node in self._raw["nodes"]):
            raise ValueError(f"duplicate node id: {node_id!r}")
        self._raw["nodes"].append({"id": node_id, "type": node_type, **config})
        return NodeRef(node_id)

    def llm(
        self,
        node_id: str,
        *,
        model: str,
        provider: str = "openai",
        system: str = "",
        **config: Any,
    ) -> NodeRef:
        return self.add_node(
            node_id,
            "llm",
            provider=provider,
            model=model,
            system=system,
            **config,
        )

    def shell(self, node_id: str, **config: Any) -> NodeRef:
        return self.add_node(node_id, "shell", **config)

    def tool(
        self,
        node_id: str,
        tool: str,
        *,
        preset: Literal["coding", "builtin"] = "coding",
        **config: Any,
    ) -> NodeRef:
        return self.add_node(
            node_id, "tool", tool=tool, preset=preset, **config
        )

    def remote(
        self,
        node_id: str,
        endpoint: str,
        *,
        timeout: float = 120.0,
        headers_env: dict[str, str] | None = None,
    ) -> NodeRef:
        return self.add_node(
            node_id,
            "remote",
            endpoint=endpoint,
            timeout=timeout,
            headers_env=dict(headers_env or {}),
        )

    def container(
        self,
        node_id: str,
        image: str,
        *,
        command: list[str] | None = None,
        timeout: float = 300.0,
        network: bool = False,
        allow_unpinned: bool = False,
        runtime: str = "docker",
    ) -> NodeRef:
        return self.add_node(
            node_id,
            "container",
            image=image,
            command=list(command or []),
            timeout=timeout,
            network=network,
            allow_unpinned=allow_unpinned,
            runtime=runtime,
        )

    def python(
        self, node_id: str, class_path: str, **params: Any
    ) -> NodeRef:
        return self.add_node(
            node_id, "python", **{"class": class_path, "params": params}
        )

    def connect(self, source: PortRef | str, target: PortRef | str) -> GraphBuilder:
        self._raw["edges"].append({"src": str(source), "dst": str(target)})
        return self

    def budget(
        self,
        *,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        mode: Literal["hard", "soft"] = "hard",
    ) -> GraphBuilder:
        self._raw["budget"] = {
            "max_tokens": max_tokens,
            "max_cost_usd": max_cost_usd,
            "mode": mode,
        }
        return self

    def execution(
        self, *, max_iterations: int = 20, convergence_k: int = 2
    ) -> GraphBuilder:
        self._raw["execution"] = {
            "max_iterations": max_iterations,
            "convergence_k": convergence_k,
        }
        return self

    def to_dict(self) -> dict[str, Any]:
        from pharos.ir import GraphSpec

        spec = GraphSpec.model_validate(self._raw)
        return spec.model_dump(by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return cast(
            str,
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
        )

    def build(self):
        from pharos.ir import load_graph_from_dict

        raw = self.to_dict()
        return load_graph_from_dict(raw, base_dir=self.base_dir)


__all__ = ["GraphBuilder", "NodeRef", "PortRef"]
