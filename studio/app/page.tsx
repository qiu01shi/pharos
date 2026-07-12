"use client";

import { useMemo, useRef, useState } from "react";
import { parse as parseYaml } from "yaml";

type GraphNode = {
  id: string;
  type: string;
  provider?: string;
  model?: string;
  system?: string;
  [key: string]: unknown;
};

type GraphEdge = { src: string; dst: string };
type GraphIR = {
  apiVersion?: string;
  kind?: string;
  name?: string;
  director?: string;
  metadata?: Record<string, unknown>;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

type Span = {
  id?: string;
  name: string;
  duration_ms?: number;
  status?: string;
  attributes?: Record<string, unknown>;
  error?: string | null;
};

type RunRecord = {
  run_id?: string;
  director?: string;
  spans?: Span[];
  outputs?: Record<string, unknown[]>;
};

const sampleGraph: GraphIR = {
  apiVersion: "pharos.ai/v1",
  kind: "Graph",
  name: "code-review-team",
  director: "fn",
  metadata: { team_pattern: "pipeline", environment: "local" },
  nodes: [
    {
      id: "planner",
      type: "llm",
      provider: "openai",
      model: "gpt-4.1-mini",
      system: "Break the task into verifiable implementation steps.",
    },
    {
      id: "coder",
      type: "llm",
      provider: "minimax",
      model: "MiniMax-Text-01",
      tools: "coding",
      system: "Implement the plan and report changed files.",
    },
    {
      id: "reviewer",
      type: "llm",
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
      system: "Review correctness, security, and contract compatibility.",
    },
  ],
  edges: [
    { src: "__in__.prompt", dst: "planner.prompt" },
    { src: "planner.text", dst: "coder.prompt" },
    { src: "coder.text", dst: "reviewer.prompt" },
    { src: "reviewer.text", dst: "__out__.response" },
  ],
};

const sampleRun: RunRecord = {
  run_id: "run_demo_03f9",
  director: "fn",
  spans: [
    {
      id: "s1",
      name: "entity.fire.planner",
      duration_ms: 428,
      status: "ok",
      attributes: { entity: "planner", iter: 0, tokens: 312 },
    },
    {
      id: "s2",
      name: "entity.fire.coder",
      duration_ms: 861,
      status: "ok",
      attributes: { entity: "coder", iter: 1, tokens: 674 },
    },
    {
      id: "s3",
      name: "entity.fire.reviewer",
      duration_ms: 593,
      status: "ok",
      attributes: { entity: "reviewer", iter: 2, tokens: 401 },
    },
  ],
};

const NODE_W = 190;
const NODE_H = 106;

function entityFromSpan(span: Span) {
  return String(span.attributes?.entity ?? span.name.split(".").at(-1) ?? "");
}

function validateGraph(graph: GraphIR) {
  const issues: string[] = [];
  const ids = graph.nodes.map((node) => node.id);
  const seen = new Set<string>();
  ids.forEach((id) => {
    if (seen.has(id)) issues.push(`Duplicate node id: ${id}`);
    seen.add(id);
  });
  const known = new Set(["__in__", "__out__", ...ids]);
  graph.edges.forEach((edge) => {
    const source = edge.src.split(".")[0];
    const target = edge.dst.split(".")[0];
    if (!known.has(source)) issues.push(`Unknown edge source: ${edge.src}`);
    if (!known.has(target)) issues.push(`Unknown edge target: ${edge.dst}`);
    if (!edge.src.includes(".") || !edge.dst.includes(".")) {
      issues.push(`Malformed edge: ${edge.src} → ${edge.dst}`);
    }
  });
  if (graph.apiVersion && graph.apiVersion !== "pharos.ai/v1") {
    issues.push(`Unsupported apiVersion: ${graph.apiVersion}`);
  }
  return issues;
}

function readFile(file: File): Promise<unknown> {
  return file.text().then((text) => {
    if (file.name.endsWith(".json")) return JSON.parse(text);
    return parseYaml(text);
  });
}

export default function Home() {
  const [graph, setGraph] = useState<GraphIR>(sampleGraph);
  const [run, setRun] = useState<RunRecord>(sampleRun);
  const [selected, setSelected] = useState("coder");
  const [sourceOpen, setSourceOpen] = useState(false);
  const [notice, setNotice] = useState("Demo IR loaded");
  const graphInput = useRef<HTMLInputElement>(null);
  const runInput = useRef<HTMLInputElement>(null);

  const displayNodes = useMemo(
    () => [
      { id: "__in__", type: "input" },
      ...graph.nodes,
      { id: "__out__", type: "output" },
    ],
    [graph],
  );
  const positions = useMemo(() => {
    const result: Record<string, { x: number; y: number }> = {};
    displayNodes.forEach((node, index) => {
      const columns = Math.min(4, Math.max(2, displayNodes.length));
      result[node.id] = {
        x: 48 + (index % columns) * 238,
        y: 74 + Math.floor(index / columns) * 190,
      };
    });
    return result;
  }, [displayNodes]);
  const issues = useMemo(() => validateGraph(graph), [graph]);
  const spans = useMemo(() => run.spans ?? [], [run.spans]);
  const spanByEntity = useMemo(
    () => Object.fromEntries(spans.map((span) => [entityFromSpan(span), span])),
    [spans],
  );
  const selectedNode = displayNodes.find((node) => node.id === selected);
  const selectedSpan = spanByEntity[selected];
  const totalDuration = spans.reduce(
    (sum, span) => sum + Number(span.duration_ms ?? 0),
    0,
  );
  const totalTokens = spans.reduce(
    (sum, span) => sum + Number(span.attributes?.tokens ?? 0),
    0,
  );

  async function loadGraphFile(file?: File) {
    if (!file) return;
    try {
      const parsed = (await readFile(file)) as GraphIR;
      if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
        throw new Error("Expected nodes[] and edges[]");
      }
      setGraph(parsed);
      setSelected(parsed.nodes[0]?.id ?? "__in__");
      setNotice(`${file.name} loaded locally`);
    } catch (error) {
      setNotice(`Could not load graph: ${String(error)}`);
    }
  }

  async function loadRunFile(file?: File) {
    if (!file) return;
    try {
      const parsed = (await readFile(file)) as RunRecord;
      if (!Array.isArray(parsed.spans)) throw new Error("Expected spans[]");
      setRun(parsed);
      setNotice(`${file.name} trace attached`);
    } catch (error) {
      setNotice(`Could not load run: ${String(error)}`);
    }
  }

  return (
    <main className="studio-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">P</span>
          <div>
            <strong>Pharos Studio</strong>
            <span>Typed agent workbench</span>
          </div>
        </div>
        <div className="run-summary" aria-label="Run summary">
          <span><b>{graph.nodes.length}</b> nodes</span>
          <span><b>{graph.edges.length}</b> edges</span>
          <span><b>{totalTokens.toLocaleString()}</b> tokens</span>
          <span><b>{totalDuration.toLocaleString()} ms</b></span>
        </div>
        <div className="top-actions">
          <span className={issues.length ? "health warn" : "health"}>
            <i /> {issues.length ? `${issues.length} issues` : "Contracts clean"}
          </span>
          <button className="secondary" onClick={() => setSourceOpen(!sourceOpen)}>
            {sourceOpen ? "Canvas" : "IR source"}
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-heading">
            <span>Workspace</span>
            <small>LOCAL</small>
          </div>
          <button className="project-card active">
            <span className="project-icon">G</span>
            <span><b>{graph.name ?? "Untitled graph"}</b><small>{graph.director ?? "fn"} director</small></span>
          </button>

          <div className="section-heading spaced"><span>Inputs</span></div>
          <input
            ref={graphInput}
            type="file"
            accept=".yaml,.yml,.json"
            hidden
            onChange={(event) => loadGraphFile(event.target.files?.[0])}
          />
          <input
            ref={runInput}
            type="file"
            accept=".json"
            hidden
            onChange={(event) => loadRunFile(event.target.files?.[0])}
          />
          <button className="upload-card" onClick={() => graphInput.current?.click()}>
            <span className="upload-symbol">+</span>
            <span><b>Open graph</b><small>YAML or JSON stays on device</small></span>
          </button>
          <button className="upload-card" onClick={() => runInput.current?.click()}>
            <span className="upload-symbol trace-symbol">↗</span>
            <span><b>Attach run</b><small>Recorded run JSON</small></span>
          </button>

          <div className="section-heading spaced"><span>Diagnostics</span></div>
          <div className={issues.length ? "diagnostic bad" : "diagnostic"}>
            <span>{issues.length ? "!" : "✓"}</span>
            <div>
              <b>{issues.length ? "Compiler attention" : "Ready to compile"}</b>
              <small>{issues[0] ?? "IR shape and references are valid"}</small>
            </div>
          </div>
          <p className="notice">{notice}</p>
        </aside>

        <section className="stage">
          <div className="stage-toolbar">
            <div>
              <span className="eyebrow">GRAPH</span>
              <h1>{graph.name ?? "Untitled graph"}</h1>
            </div>
            <div className="chips">
              <span>{graph.apiVersion ?? "legacy IR"}</span>
              <span>{graph.director ?? "fn"}</span>
              <span>{run.run_id ?? "no run"}</span>
            </div>
          </div>

          {sourceOpen ? (
            <div className="source-panel">
              <div className="source-title"><span>Compiled semantic IR</span><small>Layout metadata excluded</small></div>
              <pre>{JSON.stringify(graph, null, 2)}</pre>
            </div>
          ) : (
            <div className="canvas-scroll">
              <div className="graph-canvas">
                {graph.edges.map((edge, index) => {
                  const from = positions[edge.src.split(".")[0]];
                  const to = positions[edge.dst.split(".")[0]];
                  if (!from || !to) return null;
                  const x1 = from.x + NODE_W;
                  const y1 = from.y + NODE_H / 2;
                  const x2 = to.x;
                  const y2 = to.y + NODE_H / 2;
                  const width = Math.hypot(x2 - x1, y2 - y1);
                  const angle = Math.atan2(y2 - y1, x2 - x1);
                  return (
                    <div
                      className="edge-line"
                      key={`${edge.src}-${edge.dst}-${index}`}
                      title={`${edge.src} → ${edge.dst}`}
                      style={{
                        width,
                        left: x1,
                        top: y1,
                        transform: `rotate(${angle}rad)`,
                      }}
                    ><span /></div>
                  );
                })}
                {displayNodes.map((node) => {
                  const position = positions[node.id];
                  const span = spanByEntity[node.id];
                  const virtual = node.id.startsWith("__");
                  return (
                    <button
                      key={node.id}
                      className={[
                        "graph-node",
                        selected === node.id ? "selected" : "",
                        virtual ? "virtual" : "",
                        span?.status === "error" ? "error" : "",
                      ].join(" ")}
                      style={{ left: position.x, top: position.y }}
                      onClick={() => setSelected(node.id)}
                    >
                      <span className="node-top">
                        <i className={`node-kind ${node.type}`}>{node.type.slice(0, 2).toUpperCase()}</i>
                        <small>{span ? `${span.duration_ms ?? 0} ms` : "not run"}</small>
                      </span>
                      <b>{node.id}</b>
                      <span className="node-subtitle">
                        {virtual
                          ? node.type
                          : [node.provider, node.model].filter(Boolean).join(" / ") || node.type}
                      </span>
                      <i className="port in-port" />
                      <i className="port out-port" />
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </section>

        <aside className="inspector">
          <div className="section-heading"><span>Inspector</span><small>{selectedSpan ? "TRACED" : "SPEC"}</small></div>
          {selectedNode ? (
            <>
              <div className="inspector-title">
                <span className={`large-kind ${selectedNode.type}`}>{selectedNode.type.slice(0, 2).toUpperCase()}</span>
                <div><h2>{selectedNode.id}</h2><p>{selectedNode.type} entity</p></div>
              </div>
              <div className="field-grid">
                <label>Type<span>{selectedNode.type}</span></label>
                <label>Provider<span>{String(selectedNode.provider ?? "—")}</span></label>
                <label>Model<span>{String(selectedNode.model ?? "—")}</span></label>
                <label>Director<span>{graph.director ?? "fn"}</span></label>
              </div>
              {selectedNode.system ? (
                <div className="prompt-block"><span>System prompt</span><p>{selectedNode.system}</p></div>
              ) : null}
              <div className="contract-block">
                <span>Connected contracts</span>
                {graph.edges
                  .filter((edge) => edge.src.startsWith(`${selectedNode.id}.`) || edge.dst.startsWith(`${selectedNode.id}.`))
                  .map((edge) => (
                    <div className="contract-row" key={`${edge.src}-${edge.dst}`}>
                      <code>{edge.src}</code><i>→</i><code>{edge.dst}</code>
                    </div>
                  ))}
                {!graph.edges.some((edge) => edge.src.startsWith(`${selectedNode.id}.`) || edge.dst.startsWith(`${selectedNode.id}.`)) ? (
                  <p className="muted">No connected ports</p>
                ) : null}
              </div>
              {selectedSpan ? (
                <div className="span-card">
                  <div><span>Latest fire</span><b className={selectedSpan.status === "error" ? "red" : ""}>{selectedSpan.status ?? "ok"}</b></div>
                  <div><span>Duration</span><b>{selectedSpan.duration_ms ?? 0} ms</b></div>
                  <div><span>Iteration</span><b>{String(selectedSpan.attributes?.iter ?? "—")}</b></div>
                  <div><span>Tokens</span><b>{String(selectedSpan.attributes?.tokens ?? "—")}</b></div>
                </div>
              ) : null}
              <details className="raw-details">
                <summary>Raw node config</summary>
                <pre>{JSON.stringify(selectedNode, null, 2)}</pre>
              </details>
            </>
          ) : null}
        </aside>
      </section>

      <section className="timeline">
        <div className="timeline-heading">
          <div><span className="eyebrow">RUN TIMELINE</span><b>{run.run_id ?? "No run attached"}</b></div>
          <span>{spans.length} fires · {run.director ?? graph.director ?? "fn"} director</span>
        </div>
        <div className="timeline-track">
          {spans.length ? spans.map((span, index) => {
            const duration = Number(span.duration_ms ?? 0);
            const fraction = totalDuration ? Math.max(8, (duration / totalDuration) * 72) : 12;
            return (
              <button
                key={span.id ?? `${span.name}-${index}`}
                className={`timeline-span ${span.status === "error" ? "error" : ""}`}
                style={{ width: `${fraction}%` }}
                onClick={() => setSelected(entityFromSpan(span))}
              >
                <span>{entityFromSpan(span)}</span>
                <small>{duration} ms</small>
              </button>
            );
          }) : <p className="empty-run">Attach a recorded run to overlay execution data.</p>}
        </div>
      </section>
    </main>
  );
}
