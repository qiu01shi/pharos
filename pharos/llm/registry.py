"""Provider registry — name → class lookup.

Adding a new provider is a two-liner:
    1. implement the LLMProvider Protocol
    2. call `register_provider("myprov", MyProvider)`
"""

from __future__ import annotations

from pharos.llm.base import LLMProvider

_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """Register a provider class under a stable name."""
    if name in _REGISTRY:
        raise ValueError(f"provider already registered: {name}")
    _REGISTRY[name] = cls


def get_provider_class(name: str) -> type[LLMProvider]:
    """Look up a provider class by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown provider: {name!r}. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_providers() -> list[str]:
    """Return all registered provider names, sorted."""
    return sorted(_REGISTRY)


def create_provider(name: str, **kwargs: object) -> LLMProvider:
    """Instantiate a provider by name, forwarding kwargs to its constructor."""
    cls = get_provider_class(name)
    return cls(**kwargs)


# --- self-registration for built-in providers ---

def _register_builtins() -> None:
    # Imports here to avoid circular dependencies
    from pharos.llm.providers.anthropic import AnthropicProvider
    from pharos.llm.providers.deepseek import DeepSeekProvider
    from pharos.llm.providers.faux import FauxProvider
    from pharos.llm.providers.glm import GLMProvider
    from pharos.llm.providers.minimax import MiniMaxProvider
    from pharos.llm.providers.openai import OpenAIProvider

    register_provider("faux", FauxProvider)
    register_provider("openai", OpenAIProvider)
    register_provider("glm", GLMProvider)
    register_provider("deepseek", DeepSeekProvider)
    register_provider("anthropic", AnthropicProvider)
    register_provider("minimax", MiniMaxProvider)


_register_builtins()


__all__ = [
    "create_provider",
    "get_provider_class",
    "list_providers",
    "register_provider",
]
