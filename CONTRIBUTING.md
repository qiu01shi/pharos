# Contributing

## Development

```bash
uv sync --all-extras --frozen
uv run ruff check pharos tests
uv run mypy pharos
uv run pytest -q -m "not integration"
```

Changes to Director, replay, token, port, or IR semantics must include a
regression test that states the invariant being protected. Keep generated
artifacts, credentials, and local `.pharos` run data out of commits.

Open an issue before a large feature so its runtime and compatibility contract
can be agreed first.
