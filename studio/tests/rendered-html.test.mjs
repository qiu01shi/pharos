import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Pharos Studio runtime workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Pharos Studio/);
  assert.match(html, /Pharos Studio/);
  assert.match(html, /Runtime/);
  assert.match(html, /一键启动/);
  assert.match(html, /实时运行/);
  assert.match(html, /studio-live-demo/);
});

test("wires Studio to the local Runtime API and SSE stream", async () => {
  const [page, css, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(page, /http:\/\/127\.0\.0\.1:8765/);
  assert.match(page, /\/api\/health/);
  assert.match(page, /\/api\/graphs\/validate/);
  assert.match(page, /\/api\/runs/);
  assert.match(page, /new EventSource/);
  assert.match(page, /fire\.started/);
  assert.match(page, /fire\.completed/);
  assert.match(css, /\.graph-node\.running/);
  assert.match(css, /\.run-console/);
  assert.match(layout, /lang="zh-CN"/);
});
