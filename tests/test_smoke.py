"""Smoke test: ensure pharos package imports correctly."""


def test_imports():
    """All top-level modules should import without errors."""
    import pharos

    assert pharos.__version__ == "0.1.0"


def test_python_version():
    """Enforce minimum Python version."""
    import sys

    assert sys.version_info >= (3, 11), "pharos requires Python 3.11+"


def test_required_deps():
    """All required runtime dependencies must be available."""
    import importlib

    for dep in [
        "pydantic",
        "httpx",
        "anyio",
        "networkx",
        "typer",
        "rich",
        "yaml",
        "aiosqlite",
    ]:
        importlib.import_module(dep)
