# Pharos Studio

Studio is a local-first visual workbench over the same `pharos.ai/v1` IR used
by YAML and Python authoring. It is intentionally an inspector before it is an
editor: graph execution and validation semantics remain in the Python runtime.

## Current workflow

1. Start Studio with Node.js 22.13 or newer:

   ```bash
   cd studio
   npm install
   npm run dev
   ```

2. Load a YAML or JSON graph. Studio checks the IR version, duplicate node IDs,
   and missing edge references, then renders nodes, connections, configuration,
   and contracts.
3. Export a native run record and load it alongside the graph:

   ```bash
   uv run pharos trace <run_id> --export run.json
   ```

   The canvas and timeline then show fire state, duration, token counts, and
   trace details.

Files stay in the browser. Studio does not upload API keys, execute nodes, or
mutate the source graph.

## Evolution path

The next authoring increments should be schema-driven property forms,
connection validation via the runtime compiler, undo/redo, semantic diffs, and
round-trip export. A hosted collaborative control plane should come only after
identity, secret management, concurrency, and audit-log semantics are designed;
it should not be inferred from the local canvas.
