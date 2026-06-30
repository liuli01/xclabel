---
title: Workflow 可视化调试运行设计
date: 2026-06-29
status: draft
---

# Workflow 可视化调试运行设计

## 1. 动机

当前 xclabel-server 的 workflow 编辑器（`templates/workflow.html` + LiteGraph）支持拖拽编排 YOLO / VLLM / Condition / Output 节点，但运行工作流后结果仅在模态框中以文本+单张图片展示，缺乏直观的节点级反馈。

目标：在不改变整体架构的前提下，增强 workflow 编辑器的**可视化调试能力**，使用户能直观看到每个节点的运行结果，快速定位问题。

同时，deploy 端保留独立 YOLO 模型调试接口，清理已废弃的 nndeploy 残留文件。

## 2. 设计原则

- **增量增强**：不改动现有 LiteGraph 节点定义、后端 PipelineManager 逻辑、deploy 服务架构
- **前后端分离**：后端只增加 `exec_mode` 参数，不做结构性改动
- **保持兼容**：现有保存/加载/部署流程不变

## 3. 范围

### In Scope

| 模块 | 内容 |
|------|------|
| workflow.html 结果面板 | 右侧结果 tab，按节点分组展示运行输出 |
| 节点状态可视化 | 运行结果着色（成功/失败/跳过 + 耗时） |
| 双向联动 | 点击结果行 ↔ 选中画布节点 |
| `/api/wf/execute` 增强 | 增加 `exec_mode` 参数 |
| 清理 | 删除 nndeploy 日志，检查残留引用 |

### Out of Scope

- WebSocket 流式执行（当前一次 POST 返回全部结果，不改为流式）
- 两套 workflow API 合并（`/api/workflow/*` vs `/api/wf/*` 保留现状）
- deploy 服务端口合并（CPU:8000 / GPU:8001 各自保留）

## 4. 详细设计

### 4.1 结果面板（右侧 Tab）

**位置**：右侧属性面板区域，增加 Tab 切换 `[属性] [运行结果]`。

**Tab 切换实现**：在 `showProperties()` 的容器元素上增加 tab 按钮和面板容器。点击 tab 时切换显示内容，属性面板内容不变。

**结果面板渲染流程**：

```
用户点击 ▶ Run
  ↓
executeWorkflow() POST /api/wf/execute
  ↓
response 存入 lastRunResult
  ↓
打开结果 tab → renderResultPanel(data)
  ↓
遍历 data.node_outputs，按节点顺序渲染
```

**渲染模板**（纯 JS DOM 构建）：

每个节点项渲染为：
- 节点标题（类型 + ID）+ 状态图标 + 耗时
- 根据节点类型展示不同内容：
  - `input` → 图片缩略图（从 base64 或 URL 加载）
  - `yolo` → [查看标注图] [检测列表] 链接 + 检测统计（数量、最高置信度）
  - `condition` → 表达式原文 + 评估值（true/false）
  - `vllm` → 文本输出摘要（前 100 字，点击展开全文）
  - `output` → 合并节点列表 + 总检测数

**标注图预览**：点击「查看标注图」弹出模态框显示图片。图片来自 `node_outputs[node_id].annotated_image`（base64 JPEG）。

**原始 JSON 视图**：底部 `[📄 原始 JSON]` 按钮切换显示完整返回 JSON（`<pre>` 标签 + 语法高亮）。

### 4.2 双向联动

**结果行 → 选中节点**：点击结果面板中的节点行 → 调用 `graph.selectNode(node)` → LiteGraph 高亮该节点。

**选中节点 → 滚动结果**：LiteGraph `node.onSelected` / `node.onDeselected` 回调 → 查找结果面板中对应的节点行 → 调用 `.scrollIntoView()`。

### 4.3 节点状态可视化

在 `executeWorkflow()` 结果返回后，遍历 `lastRunResult.node_status` 和 `lastRunResult.node_timings`：

```javascript
const statusColors = {
  'ok':     '#27ae60',  // 绿色
  'error':  '#e74c3c',  // 红色
  'skipped':'#f39c12',  // 黄色
};

// 遍历节点设置颜色
graph._nodes.forEach(node => {
  const status = data.node_status[node.id];
  const time = data.node_timings[node.id];
  if (status) {
    node.boxcolor = statusColors[status] || '#555';
    node.title = `${node.originalTitle} ${status === 'ok' ? `✅ ${Math.round(time)}ms` : status === 'error' ? '❌' : '⏭'}`;
  }
});
```

保存 `node.originalTitle` 以便重置。

**重置**：用户点击「新建」或「打开」新工作流时，重置所有节点颜色和标题。

### 4.4 后端增强

`/api/wf/execute` 增加请求字段：

```python
class WfExecuteRequest:
    workflow_id: str
    image: str = None          # base64
    image_url: str = None      # 或 URL
    exec_mode: str = "auto"    # "auto" | "local" | "remote"
```

- `auto`（默认）：先尝试 `DEPLOY_SERVER_URL/pipeline/execute`，失败回退本地 PipelineManager
- `local`：直接本地 PipelineManager 执行，不走 HTTP
- `remote`：只走远程 deploy，失败返回错误

响应中增加 `mode` 字段标明实际执行方式：

```json
{
  "workflow_id": "my_workflow",
  "mode": "local_engine",
  "execution_time_ms": 320,
  "node_outputs": {...},
  "node_timings": {...},
  "node_status": {...},
  "errors": []
}
```

### 4.5 Deploy 端

保留现状，不做改动：

| 端点 | 保留原因 |
|------|----------|
| `GET /test` → `test.html` | YOLO 模型调试 UI，独立于主应用 |
| `POST /infer` | 推理 API |
| `POST /v1/predict` | 一键预测 |
| `POST /pipeline/execute` | 工作流执行 |
| `POST /pipeline/load` / `unload` | 工作流加载管理 |

### 4.6 清理

| 文件 | 操作 |
|------|------|
| `logs/nndeploy.log` | git rm |
| `logs/nndeploy_error.log` | git rm |
| `.vscode/launch.json` | 检查并移除 nndeploy 相关 launch 配置 |
| `docker-compose*.yml` | 检查并移除 nndeploy 服务引用 |
| `README.md` | 移除 nndeploy 章节（xclabel-nndeploy 镜像、端口 8002、目录结构等） |
| `.claude/settings.json` | 移除 nndeploy 相关的 allowlist 规则（历史残留） |

## 5. 实施步骤

### Step 1: 清理 nndeploy 残留
- 删除日志文件
- 检查 `.vscode/launch.json` 和 `docker-compose.yml` 的 nndeploy 引用

### Step 2: 后端增加 exec_mode 参数
- 修改 `/api/wf/execute` 的请求解析逻辑
- 增加 `exec_mode` 参数处理
- 强制 local 模式不走 HTTP 尝试

### Step 3: workflow.html 结果面板
- 属性面板容器增加 tab 按钮（`[属性]` / `[运行结果]`）
- 实现 `renderResultPanel(data)` 函数
- 按节点类型渲染差异化内容
- 实现标注图预览模态框

### Step 4: 节点状态 + 双向联动
- 结果返回后遍历 `node_status` 设置 `node.boxcolor` 和标题
- 结果行 `onClick` → `graph.selectNode()`
- 节点 `onSelected` → 结果面板滚动到对应行

## 6. 未解决的问题

- `annotated_image` 可能较大（base64），多个 YOLO 节点时结果体积需关注
- 本地执行模式需要 app.py 中 `_get_ml_engine_pool()` 等懒加载单例已初始化
