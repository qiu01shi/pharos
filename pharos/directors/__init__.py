"""pharos directors: schedule entity execution."""

from pharos.directors.base import (
    Director,
    FireContext,
    RunContext,
    RunResult,
    deliver_upstream,
    topo_layers,
)
from pharos.directors.de import DEDirector
from pharos.directors.fn import FNDirector
from pharos.directors.sdf import SDFDirector


def make_director(
    name: str,
    *,
    max_iters: int = 20,
    converge_k: int = 2,
) -> Director:
    """Build a Director by name. Shared by the CLI and SubgraphEntity so
    nested graphs can each pick their own scheduling semantics.
    """
    if name == "sdf":
        return SDFDirector(max_iterations=max_iters, convergence_k=converge_k)
    if name == "de":
        return DEDirector(max_iterations=max_iters)
    return FNDirector()


__all__ = [
    "DEDirector",
    "Director",
    "FNDirector",
    "FireContext",
    "RunContext",
    "RunResult",
    "SDFDirector",
    "deliver_upstream",
    "make_director",
    "topo_layers",
]
