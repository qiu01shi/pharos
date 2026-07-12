# Changelog

All notable changes are documented here. The project follows Semantic
Versioning once a release is published.

## 0.3.0 - Unreleased

### Changed

- Use per-fire token and cost deltas for budgets and run summaries.
- Base SDF convergence on each fire's emitted outputs rather than port buffers.
- Add explicit graph and entity run-state reset APIs.
- Rename the Python distribution to `pharos-runtime`; the import package and
  CLI remain `pharos`.
- Clarify capability authorization and output-boundary replay guarantees.
- Upgrade run records and fixtures to v2, preserving complete token boundaries
  and using ordered, lineage-aware digests while retaining v1 compatibility.
- Validate port schemas as JSON Schema Draft 2020-12 and reject provably
  incompatible edges during graph compilation.

### Added

- Versioned `pharos.ai/v1` graph IR with execution metadata.
- Python `GraphBuilder` and reusable pipeline/reflection `TeamSpec` macros.
- Local Studio workbench for graph inspection and recorded-run trace overlays.
- `remote` and digest-pinned `container` entities using the language-neutral
  `pharos.worker/v1` request/response protocol.
- Sensitive-field and credential-pattern redaction in traces, plus restrictive
  permissions for local run artifacts and the SQLite index.
- Provider integration workflow for OpenAI, Anthropic, DeepSeek, GLM, and
  MiniMax, triggered manually to keep live API spend explicit.

### Fixed

- Prevent false SDF convergence when an unconnected output keeps changing.
- Include connected outputs in SDF convergence after delivery drains ports.
- Align the CLI/package version and structured-output documentation.
