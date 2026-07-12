# Pharos Studio

Local-first graph and trace workbench for the `pharos.ai/v1` IR.

## Run locally

Node.js 22.13 or newer is required.

```bash
npm install
npm run dev
```

Open the displayed local URL, then load a Pharos YAML/JSON graph. A recorded
run can be loaded separately to overlay node status, duration, token counts,
and the fire timeline.

```bash
uv run pharos trace <run_id> --export run.json
```

Studio currently reads files entirely in the browser and does not execute the
graph or persist edits. The runtime IR remains the source of truth.
