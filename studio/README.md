# Pharos Studio

Pharos Studio 是 `pharos.ai/v1` 工作流的本地可视化操作台。它通过本地 Runtime API
完成真实校验和执行，通过 SSE 实时展示节点状态、时间线和最终输出。

## 本地启动

需要 Node.js 22.13+。

终端一：

```bash
cd ..
uv sync --all-extras
uv run pharos serve
```

终端二：

```bash
cd studio
nvm use 24     # 如果已使用 Node 22.13+ 可省略
npm ci
npm run dev
```

打开 `http://localhost:3000`，顶部显示 `Runtime v...` 后即可一键启动内置的
零费用 Faux 演示工作流，也可打开自己的 YAML / JSON 工作流。

完整中文说明见 [`docs/studio.md`](../docs/studio.md)。
