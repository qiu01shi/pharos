# Worker protocol v1

`pharos.worker/v1` is the interoperability boundary for heterogeneous nodes.
It deliberately describes one entity fire, not an entire graph scheduler.

## Request

Remote workers accept an HTTP `POST`; container workers receive the same JSON
on standard input.

```json
{
  "protocol": "pharos.worker/v1",
  "run_id": "run-123",
  "node_id": "classify",
  "fire_id": "classify:0",
  "iteration": 0,
  "idempotency_key": "stable-key",
  "inputs": [
    {"port": "input", "type": "json", "payload": {"text": "hello"}, "self_hash": "..."}
  ]
}
```

Remote requests also carry an `Idempotency-Key` header and propagate a W3C
`traceparent` when a Pharos span is active. Workers should deduplicate side
effects by idempotency key.

## Response

```json
{
  "protocol": "pharos.worker/v1",
  "outputs": [
    {"port": "output", "type": "json", "payload": {"label": "greeting"}}
  ],
  "metadata": {"worker_version": "1.2.0"}
}
```

Unknown output ports are rejected. Transport errors and non-zero container
exits fail the fire; retry policy remains explicit in the graph.

## Security and operational boundary

- `remote` requires `net:connect`; authentication headers are read from named
  environment variables rather than stored in graph files.
- `container` requires `container:execute`, pins images by digest by default,
  uses a read-only filesystem, and disables networking unless explicitly
  enabled and granted.
- Workers are not trusted sandboxes. Operators must enforce OS/container
  isolation, egress rules, resource limits, image provenance, and secret scope.
- The local Director remains authoritative for scheduling, graph state,
  budgets, tracing, and authorization. Durable queues, leases, heartbeats,
  reconciliation, and cross-machine checkpoints are intentionally outside v1.
