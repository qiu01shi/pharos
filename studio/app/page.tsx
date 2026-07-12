"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";

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
  status?: string;
  spans?: Span[];
  outputs?: Record<string, Record<string, unknown[]>>;
  result?: Record<string, unknown> | null;
};

type RuntimeEvent = {
  seq?: number;
  type: string;
  run_id?: string;
  node_id?: string;
  duration_ms?: number;
  tokens?: number;
  iteration?: number;
  fire_index?: number;
  status?: string;
  error?: string;
};

type RuntimeHealth = "checking" | "online" | "offline";
type NodeRunState = "idle" | "running" | "completed" | "failed";

const API_URL =
  process.env.NEXT_PUBLIC_PHAROS_API_URL ?? "http://127.0.0.1:8765";

const sampleGraph: GraphIR = {
  apiVersion: "pharos.ai/v1",
  kind: "Graph",
  name: "studio-live-demo",
  director: "fn",
  metadata: { environment: "local", description: "Zero-cost Studio demo" },
  nodes: [
    {
      id: "planner",
      type: "llm",
      provider: "faux",
      model: "faux-echo",
      system: "Break the request into a short plan.",
    },
    {
      id: "reviewer",
      type: "llm",
      provider: "faux",
      model: "faux-echo",
      system: "Review the plan and return the final response.",
    },
  ],
  edges: [
    { src: "__in__.prompt", dst: "planner.prompt" },
    { src: "planner.text", dst: "reviewer.prompt" },
    { src: "reviewer.text", dst: "__out__.response" },
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
    if (seen.has(id)) issues.push(`节点 ID 重复：${id}`);
    seen.add(id);
  });
  const known = new Set(["__in__", "__out__", ...ids]);
  graph.edges.forEach((edge) => {
    const source = edge.src.split(".")[0];
    const target = edge.dst.split(".")[0];
    if (!known.has(source)) issues.push(`连接起点不存在：${edge.src}`);
    if (!known.has(target)) issues.push(`连接终点不存在：${edge.dst}`);
    if (!edge.src.includes(".") || !edge.dst.includes(".")) {
      issues.push(`连接格式错误：${edge.src} → ${edge.dst}`);
    }
  });
  if (graph.apiVersion && graph.apiVersion !== "pharos.ai/v1") {
    issues.push(`不支持的 apiVersion：${graph.apiVersion}`);
  }
  return issues;
}

async function responseError(response: Response) {
  try {
    const body = (await response.json()) as { detail?: unknown };
    return typeof body.detail === "string"
      ? body.detail
      : JSON.stringify(body.detail ?? body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

function eventLabel(event: RuntimeEvent) {
  if (event.type === "run.queued") return "运行已排队";
  if (event.type === "run.started") return "运行开始";
  if (event.type === "run.completed") return "运行完成";
  if (event.type === "run.failed") return `运行失败：${event.error ?? "未知错误"}`;
  if (event.type === "run.cancelled") return "运行已取消";
  if (event.type === "fire.started") return `${event.node_id} 开始执行`;
  if (event.type === "fire.completed") {
    return `${event.node_id} 完成 · ${Number(event.duration_ms ?? 0).toFixed(1)} ms`;
  }
  if (event.type === "fire.failed") return `${event.node_id} 失败：${event.error ?? ""}`;
  return event.type;
}

export default function Home() {
  const [graph, setGraph] = useState<GraphIR>(sampleGraph);
  const [graphSource, setGraphSource] = useState(() => stringifyYaml(sampleGraph));
  const [graphFileName, setGraphFileName] = useState("studio-live-demo.yaml");
  const [run, setRun] = useState<RunRecord>({ spans: [] });
  const [selected, setSelected] = useState("planner");
  const [sourceOpen, setSourceOpen] = useState(false);
  const [notice, setNotice] = useState("已加载零费用演示工作流");
  const [runtimeHealth, setRuntimeHealth] = useState<RuntimeHealth>("checking");
  const [runtimeVersion, setRuntimeVersion] = useState("");
  const [prompt, setPrompt] = useState("请规划并检查一个简单任务");
  const [grantText, setGrantText] = useState("");
  const [baseDir, setBaseDir] = useState("");
  const [maxTokens, setMaxTokens] = useState("");
  const [maxCost, setMaxCost] = useState("");
  const [runStatus, setRunStatus] = useState("idle");
  const [runtimeEvents, setRuntimeEvents] = useState<RuntimeEvent[]>([]);
  const [nodeStates, setNodeStates] = useState<Record<string, NodeRunState>>({});
  const graphInput = useRef<HTMLInputElement>(null);
  const runInput = useRef<HTMLInputElement>(null);
  const eventSource = useRef<EventSource | null>(null);

  useEffect(() => {
    let active = true;
    async function checkRuntime() {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (!response.ok) throw new Error(await responseError(response));
        const health = (await response.json()) as { version?: string };
        if (!active) return;
        setRuntimeHealth("online");
        setRuntimeVersion(health.version ?? "");
      } catch {
        if (active) setRuntimeHealth("offline");
      }
    }
    void checkRuntime();
    const healthTimer = window.setInterval(() => void checkRuntime(), 3000);
    return () => {
      active = false;
      window.clearInterval(healthTimer);
      eventSource.current?.close();
    };
  }, []);

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
  const isRunning = runStatus === "queued" || runStatus === "running";

  async function validateWithRuntime(source: string, sourceName: string) {
    if (runtimeHealth !== "online") return;
    try {
      const response = await fetch(`${API_URL}/api/graphs/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          graph: source,
          source_name: sourceName,
          base_dir: baseDir || undefined,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setNotice(`${sourceName} 已通过 Runtime 编译校验`);
    } catch (error) {
      setNotice(`Runtime 校验失败：${String(error)}`);
    }
  }

  async function loadGraphFile(file?: File) {
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = (file.name.endsWith(".json")
        ? JSON.parse(text)
        : parseYaml(text)) as GraphIR;
      if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
        throw new Error("需要 nodes[] 和 edges[]");
      }
      setGraph(parsed);
      setGraphSource(text);
      setGraphFileName(file.name);
      setRun({ spans: [] });
      setRuntimeEvents([]);
      setNodeStates({});
      setRunStatus("idle");
      setSelected(parsed.nodes[0]?.id ?? "__in__");
      setNotice(`${file.name} 已在本地加载`);
      await validateWithRuntime(text, file.name);
    } catch (error) {
      setNotice(`无法加载工作流：${String(error)}`);
    } finally {
      if (graphInput.current) graphInput.current.value = "";
    }
  }

  async function loadRunFile(file?: File) {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as RunRecord;
      if (!Array.isArray(parsed.spans)) throw new Error("需要 spans[]");
      setRun(parsed);
      setRunStatus(parsed.status ?? "completed");
      setNotice(`${file.name} 运行记录已载入`);
    } catch (error) {
      setNotice(`无法加载运行记录：${String(error)}`);
    } finally {
      if (runInput.current) runInput.current.value = "";
    }
  }

  async function refreshRun(runId: string) {
    const response = await fetch(`${API_URL}/api/runs/${runId}`);
    if (!response.ok) throw new Error(await responseError(response));
    const snapshot = (await response.json()) as RunRecord & { status: string };
    setRun(snapshot);
    setRunStatus(snapshot.status);
  }

  function applyRuntimeEvent(event: RuntimeEvent, stream: EventSource) {
    setRuntimeEvents((current) => [...current, event].slice(-100));
    if (event.type === "run.started") setRunStatus("running");
    if (event.type === "fire.started" && event.node_id) {
      setNodeStates((current) => ({ ...current, [event.node_id!]: "running" }));
      setSelected(event.node_id);
    }
    if (event.type === "fire.completed" && event.node_id) {
      setNodeStates((current) => ({ ...current, [event.node_id!]: "completed" }));
      const liveSpan: Span = {
        id: `${event.node_id}:${event.fire_index ?? 0}`,
        name: `entity.fire.${event.node_id}`,
        duration_ms: event.duration_ms,
        status: "ok",
        attributes: {
          entity: event.node_id,
          iter: event.iteration,
          tokens: event.tokens,
          fire_index: event.fire_index,
        },
      };
      setRun((current) => ({
        ...current,
        spans: [
          ...(current.spans ?? []).filter((span) => span.id !== liveSpan.id),
          liveSpan,
        ],
      }));
    }
    if (event.type === "fire.failed" && event.node_id) {
      setNodeStates((current) => ({ ...current, [event.node_id!]: "failed" }));
    }
    if (["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
      const finalStatus = event.type.replace("run.", "");
      setRunStatus(finalStatus);
      stream.close();
      if (event.run_id) {
        void refreshRun(event.run_id).catch((error) => {
          setNotice(`读取最终运行结果失败：${String(error)}`);
        });
      }
    }
  }

  async function startRun() {
    if (runtimeHealth !== "online" || isRunning) return;
    eventSource.current?.close();
    setRuntimeEvents([]);
    setNodeStates({});
    setRun({ spans: [], director: graph.director });
    setRunStatus("queued");
    setNotice("正在提交工作流…");
    try {
      const grants = grantText.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean);
      const response = await fetch(`${API_URL}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          graph: graphSource,
          source_name: graphFileName,
          base_dir: baseDir || undefined,
          input: prompt,
          grants,
          max_tokens: maxTokens ? Number(maxTokens) : undefined,
          max_cost_usd: maxCost ? Number(maxCost) : undefined,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const created = (await response.json()) as { run_id: string; director?: string };
      setRun({ run_id: created.run_id, director: created.director, spans: [] });
      setNotice(`运行 ${created.run_id.slice(0, 8)} 已创建`);
      const stream = new EventSource(`${API_URL}/api/runs/${created.run_id}/events`);
      eventSource.current = stream;
      stream.onmessage = (message) => {
        applyRuntimeEvent(JSON.parse(message.data) as RuntimeEvent, stream);
      };
      stream.onerror = () => {
        if (isRunning) setNotice("实时连接中断，正在读取运行状态…");
        void refreshRun(created.run_id).catch(() => undefined);
      };
    } catch (error) {
      setRunStatus("failed");
      setNotice(`启动失败：${String(error)}`);
    }
  }

  async function cancelRun() {
    if (!run.run_id || !isRunning) return;
    await fetch(`${API_URL}/api/runs/${run.run_id}/cancel`, { method: "POST" });
    setNotice("已请求取消运行");
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
        <div className="run-summary" aria-label="运行摘要">
          <span><b>{graph.nodes.length}</b> 节点</span>
          <span><b>{graph.edges.length}</b> 连接</span>
          <span><b>{totalTokens.toLocaleString()}</b> tokens</span>
          <span><b>{totalDuration.toFixed(1)}</b> ms</span>
        </div>
        <div className="top-actions">
          <span className={`runtime-health ${runtimeHealth}`}>
            <i /> Runtime {runtimeHealth === "online" ? `v${runtimeVersion}` : runtimeHealth === "offline" ? "未连接" : "连接中"}
          </span>
          <span className={issues.length ? "health warn" : "health"}>
            <i /> {issues.length ? `${issues.length} 个问题` : "结构正常"}
          </span>
          <button className="secondary" onClick={() => setSourceOpen(!sourceOpen)}>
            {sourceOpen ? "画布" : "IR 源码"}
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-heading"><span>工作区</span><small>LOCAL</small></div>
          <button className="project-card active">
            <span className="project-icon">G</span>
            <span><b>{graph.name ?? "未命名工作流"}</b><small>{graph.director ?? "fn"} director</small></span>
          </button>

          <div className="section-heading spaced"><span>工作流文件</span></div>
          <input ref={graphInput} type="file" accept=".yaml,.yml,.json" hidden onChange={(event) => void loadGraphFile(event.target.files?.[0])} />
          <input ref={runInput} type="file" accept=".json" hidden onChange={(event) => void loadRunFile(event.target.files?.[0])} />
          <button className="upload-card" onClick={() => graphInput.current?.click()}>
            <span className="upload-symbol">+</span>
            <span><b>打开工作流</b><small>YAML 或 JSON</small></span>
          </button>
          <button className="upload-card" onClick={() => runInput.current?.click()}>
            <span className="upload-symbol trace-symbol">↗</span>
            <span><b>载入历史运行</b><small>原生 Run JSON</small></span>
          </button>

          <div className="section-heading spaced"><span>启动配置</span><small>{runStatus.toUpperCase()}</small></div>
          <div className="run-form">
            <label>输入 Prompt<textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={3} /></label>
            <label>权限授权<input value={grantText} onChange={(event) => setGrantText(event.target.value)} placeholder="fs:read, net:connect" /></label>
            <div className="run-form-row">
              <label>Token 上限<input inputMode="numeric" value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} placeholder="可选" /></label>
              <label>费用上限 $<input inputMode="decimal" value={maxCost} onChange={(event) => setMaxCost(event.target.value)} placeholder="可选" /></label>
            </div>
            <details>
              <summary>高级配置</summary>
              <label>子图基准目录<input value={baseDir} onChange={(event) => setBaseDir(event.target.value)} placeholder="/path/to/graphs" /></label>
            </details>
            <div className="run-buttons">
              <button className="run-button" disabled={runtimeHealth !== "online" || isRunning || issues.length > 0} onClick={() => void startRun()}>
                {isRunning ? "运行中…" : "▶ 一键启动"}
              </button>
              {isRunning ? <button className="cancel-button" onClick={() => void cancelRun()}>停止</button> : null}
            </div>
          </div>

          <div className="section-heading spaced"><span>诊断</span></div>
          <div className={issues.length ? "diagnostic bad" : "diagnostic"}>
            <span>{issues.length ? "!" : "✓"}</span>
            <div><b>{issues.length ? "需要处理" : "可以编译"}</b><small>{issues[0] ?? "IR 结构和引用有效"}</small></div>
          </div>
          <p className="notice">{notice}</p>
        </aside>

        <section className="stage">
          <div className="stage-toolbar">
            <div><span className="eyebrow">GRAPH</span><h1>{graph.name ?? "未命名工作流"}</h1></div>
            <div className="chips">
              <span>{graph.apiVersion ?? "legacy IR"}</span>
              <span>{graph.director ?? "fn"}</span>
              <span>{run.run_id?.slice(0, 8) ?? "尚未运行"}</span>
            </div>
          </div>

          {sourceOpen ? (
            <div className="source-panel">
              <div className="source-title"><span>语义 IR</span><small>{graphFileName}</small></div>
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
                  return <div className="edge-line" key={`${edge.src}-${edge.dst}-${index}`} title={`${edge.src} → ${edge.dst}`} style={{ width, left: x1, top: y1, transform: `rotate(${angle}rad)` }}><span /></div>;
                })}
                {displayNodes.map((node) => {
                  const position = positions[node.id];
                  const span = spanByEntity[node.id];
                  const state = nodeStates[node.id] ?? "idle";
                  const virtual = node.id.startsWith("__");
                  return (
                    <button key={node.id} className={["graph-node", selected === node.id ? "selected" : "", virtual ? "virtual" : "", span?.status === "error" || state === "failed" ? "error" : "", state].join(" ")} style={{ left: position.x, top: position.y }} onClick={() => setSelected(node.id)}>
                      <span className="node-top"><i className={`node-kind ${node.type}`}>{node.type.slice(0, 2).toUpperCase()}</i><small>{state === "running" ? "执行中" : span ? `${Number(span.duration_ms ?? 0).toFixed(1)} ms` : "未运行"}</small></span>
                      <b>{node.id}</b>
                      <span className="node-subtitle">{virtual ? node.type : [node.provider, node.model].filter(Boolean).join(" / ") || node.type}</span>
                      <i className="port in-port" /><i className="port out-port" />
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </section>

        <aside className="inspector">
          <div className="section-heading"><span>节点检查器</span><small>{nodeStates[selected]?.toUpperCase() ?? (selectedSpan ? "TRACED" : "SPEC")}</small></div>
          {selectedNode ? <>
            <div className="inspector-title"><span className={`large-kind ${selectedNode.type}`}>{selectedNode.type.slice(0, 2).toUpperCase()}</span><div><h2>{selectedNode.id}</h2><p>{selectedNode.type} entity</p></div></div>
            <div className="field-grid">
              <label>类型<span>{selectedNode.type}</span></label>
              <label>Provider<span>{String(selectedNode.provider ?? "—")}</span></label>
              <label>Model<span>{String(selectedNode.model ?? "—")}</span></label>
              <label>Director<span>{graph.director ?? "fn"}</span></label>
            </div>
            {selectedNode.system ? <div className="prompt-block"><span>System Prompt</span><p>{selectedNode.system}</p></div> : null}
            <div className="contract-block"><span>端口连接</span>
              {graph.edges.filter((edge) => edge.src.startsWith(`${selectedNode.id}.`) || edge.dst.startsWith(`${selectedNode.id}.`)).map((edge) => <div className="contract-row" key={`${edge.src}-${edge.dst}`}><code>{edge.src}</code><i>→</i><code>{edge.dst}</code></div>)}
              {!graph.edges.some((edge) => edge.src.startsWith(`${selectedNode.id}.`) || edge.dst.startsWith(`${selectedNode.id}.`)) ? <p className="muted">没有连接的端口</p> : null}
            </div>
            {selectedSpan ? <div className="span-card">
              <div><span>最近执行</span><b className={selectedSpan.status === "error" ? "red" : ""}>{selectedSpan.status ?? "ok"}</b></div>
              <div><span>耗时</span><b>{Number(selectedSpan.duration_ms ?? 0).toFixed(1)} ms</b></div>
              <div><span>迭代</span><b>{String(selectedSpan.attributes?.iter ?? "—")}</b></div>
              <div><span>Tokens</span><b>{String(selectedSpan.attributes?.tokens ?? "—")}</b></div>
            </div> : null}
            <details className="raw-details"><summary>原始节点配置</summary><pre>{JSON.stringify(selectedNode, null, 2)}</pre></details>
          </> : null}
        </aside>
      </section>

      <section className="timeline">
        <div className="timeline-heading">
          <div><span className="eyebrow">实时运行</span><b>{run.run_id?.slice(0, 18) ?? "尚未启动"}</b></div>
          <span className={`run-state ${runStatus}`}>{runStatus} · {spans.length} fires</span>
        </div>
        <div className="timeline-track">
          {spans.length ? spans.map((span, index) => {
            const duration = Number(span.duration_ms ?? 0);
            const fraction = totalDuration ? Math.max(8, (duration / totalDuration) * 72) : 12;
            return <button key={span.id ?? `${span.name}-${index}`} className={`timeline-span ${span.status === "error" ? "error" : ""}`} style={{ width: `${fraction}%` }} onClick={() => setSelected(entityFromSpan(span))}><span>{entityFromSpan(span)}</span><small>{duration.toFixed(1)} ms</small></button>;
          }) : <p className="empty-run">点击“一键启动”后，节点状态和时间线会实时更新。</p>}
        </div>
        <div className="run-console">
          <div className="run-console-title"><span>事件流</span><small>{runtimeEvents.length}</small></div>
          <div className="event-list">
            {runtimeEvents.slice(-6).map((event, index) => <div key={`${event.seq ?? index}-${event.type}`} className={event.type.includes("failed") ? "event error" : "event"}><i /> <span>{eventLabel(event)}</span></div>)}
            {!runtimeEvents.length ? <p>等待运行事件…</p> : null}
          </div>
          {run.outputs && Object.keys(run.outputs).length ? <details className="output-details"><summary>查看输出</summary><pre>{JSON.stringify(run.outputs, null, 2)}</pre></details> : null}
        </div>
      </section>
    </main>
  );
}
