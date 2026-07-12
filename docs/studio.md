# Pharos Studio 本地运行与实时执行

Pharos Studio 是 `pharos.ai/v1` 工作流的本地可视化操作台。Studio 负责展示、
运行配置和实时交互；Python Runtime 仍然是唯一的编译、调度、权限、预算和
记录语义。YAML、Python Builder、TeamSpec 和 Studio 都使用同一份版本化 IR。

## 当前已支持

- 在浏览器本地加载 YAML / JSON 工作流。
- 图形化查看节点、连接、Provider、Model、Prompt 和原始配置。
- 调用 Runtime 真实编译器完成端口、类型、子图和节点配置校验。
- 在 Studio 中输入 Prompt、授权、Token 上限和费用上限。
- 一键启动 FN / SDF / DE 工作流。
- 通过 SSE 实时接收 `run.*` 和 `fire.*` 生命周期事件。
- 在画布、时间线和事件流中展示执行中、成功和失败状态。
- 运行完成后自动显示图输出，并持久化到 `~/.pharos/runs` 和 SQLite 索引。
- 取消当前运行。
- 仍可手动加载 `pharos trace --export` 生成的历史 Run JSON。

Studio 目前还不支持拖拽新增/删除节点、修改连线、回写 YAML、工作流版本管理
和多用户协作。

## 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 22.13+
- npm

请在 `studio` 目录内再次检查 Node 版本。某些本机环境会根据目录切换 Node，如果实际
使用 Node 18，Vinext 会因缺少 `node:fs/promises.glob` 而无法启动。

```bash
cd studio
node --version
# 当前机器可使用：nvm use 24
```

## 完整本地启动

终端一：启动 Python Runtime API。

```bash
cd /path/to/pharos
uv sync --all-extras
uv run pharos serve
```

默认地址：

```text
http://127.0.0.1:8765
```

可以指定端口：

```bash
uv run pharos serve --port 9000
```

终端二：启动 Studio。

```bash
cd /path/to/pharos/studio
nvm use 24                    # 如果已是 Node 22.13+ 可省略
npm ci
npm run dev
```

打开：

```text
http://localhost:3000
```

如果 Runtime 使用非默认地址，启动 Studio 前设置：

```bash
NEXT_PUBLIC_PHAROS_API_URL=http://127.0.0.1:9000 npm run dev
```

## 在 Studio 中运行工作流

1. 页面顶部确认显示 `Runtime v...`。
2. 可直接使用内置的 Faux 演示图，它不需要 API Key；也可点击“打开工作流”。
3. 填写 Prompt。
4. 工作流需要文件、Shell、网络或容器时，填写它明确声明的授权，多个权限用逗号分隔。
5. 可选填写 Token 和费用上限。
6. 引用相对路径子图时，在“高级配置”中填写主图所在目录。
7. 点击“一键启动”。
8. 画布实时标记当前节点，页面底部同步显示时间线、事件和最终输出。

Provider 密钥仍由 Runtime 读取，不会传入 Studio。默认配置文件为：

```text
~/.pharos/.env
```

## Local Runtime API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | Runtime 版本和连接检查 |
| POST | `/api/graphs/validate` | 调用 Runtime 编译器校验工作流 |
| POST | `/api/runs` | 创建并启动运行 |
| GET | `/api/runs` | 当前进程内的运行列表 |
| GET | `/api/runs/{run_id}` | 运行快照、Trace 和输出 |
| GET | `/api/runs/{run_id}/events` | SSE 实时事件流 |
| POST | `/api/runs/{run_id}/cancel` | 取消运行 |

Runtime API 默认只绑定 `127.0.0.1`，CORS 只允许本机 `localhost` / `127.0.0.1`
页面。工作流可以执行经授权的文件、Shell、网络和容器操作，不建议将当前形态
直接暴露到公网。

## 手动载入历史运行

```bash
uv run pharos trace list
uv run pharos trace <run_id> --export run.json
```

在 Studio 中先打开原工作流，再点击“载入历史运行”选择 `run.json`。

## 开发验证

```bash
# Runtime
uv run ruff check pharos tests
uv run mypy pharos
uv run pytest -q -m "not integration"

# Studio
cd studio
npm run lint
npm test
```

## 下一步

下一个主要增量是可视化 Workflow Builder：基于 Runtime Schema 生成节点属性表单，
支持节点/连线编辑、Undo/Redo、类型安全连接、回写 YAML 与版本化保存。
