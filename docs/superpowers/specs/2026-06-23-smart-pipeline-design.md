# xclabel 智能流水线 (Smart Pipeline) 设计

## 1. 概述

xclabel 现有的 deploy 推理容器仅支持单模型/单工作流的串行推理。本项目将其升级为 **Smart Pipeline**：一个三层架构的智能编排系统，支持 YOLO 实时感知 + 条件分支 + VLLM 慢速核实的组合流水线。

### 核心概念

- **感知层** (YOLO)：毫秒级目标检测/分割/姿态估计
- **决策层** (PipelineManager)：解析 `workflow.yaml`，按条件调度节点
- **核实层** (VLLM)：仅对 YOLO 不确定的帧调用大模型进行二次判定

### 设计原则

- **配置即流程**：流水线拓扑由 YAML 定义，替换模型/参数无需改代码
- **懒核实**：VLLM 仅在高不确定区间触发，不阻塞实时检测
- **精度对齐**：YOLO 节点使用 ultralytics `predict()`，严禁手写后处理

## 2. 架构

```
┌─────────────────────────────────────────────────────────┐
│                   xclabel-server (Flask)                  │
│  ┌──────────────────────┐  ┌──────────────────────────┐  │
│  │  工作流编辑器          │  │  部署管理 API             │  │
│  │  (LiteGraph.js 画布)  │  │  - CRUD workflow.yaml    │  │
│  │  拖拽 YOLO/VLLM 节点  │  │  - 部署/取消部署          │  │
│  │  连线定义数据流        │  │  - 状态查询              │  │
│  └──────────┬───────────┘  └──────────┬───────────────┘  │
└─────────────┼─────────────────────────┼──────────────────┘
              │ POST /api/workflow/save  │ POST /api/workflow/deploy
              ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              xclabel-deploy (FastAPI)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │              PipelineManager                      │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────────┐ │   │
│  │  │ YOLO Node │──▶│ Branch   │──▶│ VLLM Node    │ │   │
│  │  │ (ultralytics)│  │ Judge   │   │ (OpenAI API)  │ │   │
│  │  └──────────┘   └──────────┘   └──────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │               Inference Engine Pool               │   │
│  │  (已有: YoloAdapter / NndeployAdapter)             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 所在服务 | 职责 |
|------|----------|------|
| 工作流编辑器 | server (Flask) | LiteGraph.js 可视化拖拽编排 |
| Workflow Config API | server (Flask) | 保存/读取/部署 workflow.yaml |
| PipelineManager | deploy (FastAPI) | 解析 YAML，按 DAG 调度节点执行 |
| YOLO Node | deploy | 通过 YoloAdapter 调用 ultralytics predict() |
| VLLM Node | deploy | 通过 OpenAI 兼容 API 调用大模型 |
| 引擎池 | deploy | 管理已加载的模型/工作流实例 |

## 3. 工作流 YAML 协议

### Schema

```yaml
# workflow.yaml
version: "1.0"
name: "ccrc_fall_detection"
project_id: "sv30_seg"

pipeline:
  - id: "detect_fall"
    type: "yolo"
    model: "sv30_seg/20260618_172731"    # 引用引擎池中已加载的模型
    task: "segment"
    params:
      conf: 0.25
      iou: 0.5

  - id: "confidence_check"
    type: "condition"
    source: "detect_fall"
    expression: "max_conf < 0.7 and max_conf > 0.3"
    # max_conf < 0.3 → 直接可信，走正常分支
    # 0.3 ≤ max_conf ≤ 0.7 → 不确定，触发 VLLM 核实
    # max_conf > 0.7 → 高置信，跳过核实

  - id: "vllm_verify"
    type: "vllm"
    enabled: true
    condition: "confidence_check"          # 仅当条件触发时执行
    api_url: "http://vllm-server:8000/v1"
    model: "qwen-vl-chat"
    prompt: |
      分析此图片中的老人是否存在跌倒或即将跌倒的危险。
      图片已裁剪为检测到的目标区域。
      请回答：是/否/不确定，并简要说明原因。
    params:
      temperature: 0.1
      max_tokens: 256
    extract_roi: true                     # 发 VLLM 前先裁剪 ROI

  - id: "merge_result"
    type: "output"
    source: ["detect_fall", "vllm_verify"]
    format: "json"
```

### 节点类型

| 类型 | 说明 | 输出 |
|------|------|------|
| `yolo` | 目标检测/分割/姿态 | detections + masks + conf |
| `condition` | 基于表达式做分支判断 | true/false |
| `vllm` | 调用大模型核实 | text + confidence |
| `output` | 汇总输出 | 合并 JSON |

### 数据传递

节点间通过上下文 `PipelineContext` 传递数据：

```python
class PipelineContext:
    image: np.ndarray              # 原始图片
    detections: list[Detection]    # YOLO 检测结果
    vllm_result: str | None        # VLLM 回复
    max_conf: float                # 最高置信度
    branch_conditions: dict        # 分支判断结果
```

## 4. PipelineManager 引擎

### 接口设计

```python
# deploy/pipeline_manager.py

class PipelineManager:
    """
    工作流编排引擎。
    - 接收 workflow.yaml → 构建 DAG
    - 按拓扑序执行节点
    - 支持条件分支跳转
    """

    def __init__(self, workflow_path: str):
        self.config = self._load_yaml(workflow_path)
        self.graph = self._build_dag(self.config.pipeline)

    async def execute(
        self,
        image: np.ndarray,
        engine_pool: EnginePool,
        metadata: dict | None = None,
    ) -> PipelineResult:
        """执行整个流水线，返回最终结果。"""
        ctx = PipelineContext(image=image)
        for node in self._topological_sort():
            result = await self._run_node(node, ctx)
            ctx.update(node.id, result)
        return ctx.to_result()
```

### 节点执行路由

```python
async def _run_node(self, node: NodeConfig, ctx: PipelineContext):
    if node.type == "yolo":
        return await self._exec_yolo(node, ctx)
    elif node.type == "condition":
        return self._eval_condition(node, ctx)
    elif node.type == "vllm":
        return await self._exec_vllm(node, ctx)
    elif node.type == "output":
        return self._merge_output(node, ctx)
```

### YOLO 节点

复用已有的 `YoloAdapter`，从引擎池中按 `model` 字段获取已加载的 YOLO 模型。

```python
async def _exec_yolo(self, node: NodeConfig, ctx: PipelineContext):
    engine = await self.engine_pool.get(node.model)
    if not engine:
        raise RuntimeError(f"Model not loaded: {node.model}")
    result = self.yolo_adapter.infer(
        engine.engine, ctx.image,
        confidence_threshold=node.params.conf,
        task_type=engine.metadata.get("task_type", "detect"),
    )
    ctx.max_conf = max(d["confidence"] for d in result["detections"]) if result["detections"] else 0
    return result
```

### Condition 节点

支持简单表达式求值（基于 `max_conf` 或 `detection_count` 等上下文变量）。

```python
def _eval_condition(self, node: NodeConfig, ctx: PipelineContext) -> bool:
    # 安全 eval，仅限预定义变量
    allowed = {"max_conf": ctx.max_conf, "detection_count": len(ctx.detections)}
    return bool(eval(node.expression, {"__builtins__": {}}, allowed))
```

### VLLM 节点

```python
async def _exec_vllm(self, node: NodeConfig, ctx: PipelineContext) -> dict:
    from openai import AsyncOpenAI

    # 裁剪 ROI（可选）
    image = self._crop_roi(ctx.image, ctx.detections) if node.extract_roi else ctx.image

    client = AsyncOpenAI(base_url=node.api_url, api_key="not-needed")
    resp = await client.chat.completions.create(
        model=node.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": node.prompt},
                {"type": "image_url", "image_url": {"url": self._image_to_b64(image)}},
            ],
        }],
        **node.params,
    )
    return {"vllm_result": resp.choices[0].message.content}
```

### 部署 API

在 deploy 中新增端点：

```http
POST /pipeline/execute
Content-Type: application/json

{
    "workflow_id": "ccrc_fall_detection",
    "image": "<base64>",
    "params": {}
}
```

## 5. Server 端 API

在 xclabel-server (Flask) 中新增：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/workflow/list` | GET | 列出项目下所有 workflow |
| `/api/workflow/get` | GET | 获取单个 workflow YAML |
| `/api/workflow/save` | POST | 保存/更新 workflow |
| `/api/workflow/delete` | POST | 删除 workflow |
| `/api/workflow/deploy` | POST | 部署 workflow 到 deploy 服务 |
| `/api/workflow/undeploy` | POST | 取消部署 |

### 数据存储

Workflow YAML 存储在 `projects/<project>/workflows/<name>.yaml`。

## 6. 前端工作流编辑器

### 页面位置

在 xclabel 导航栏新增 **"流程编排"** 按钮，打开新页面 `templates/workflow.html`。

### 技术选型

- **LiteGraph.js** — 纯 JS 节点编辑器，通过 `<script>` 标签引入
- 节点类型：YOLO Node、Condition Node、VLLM Node、Output Node
- 拖拽连线定义数据流

### 页面布局

```
┌─────────────────────────────────────────────────────────┐
│ [返回]  流程编排  ▸  sv30_seg    [保存] [部署]           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │   节点面板    │  │    画布 (LiteGraph)              │  │
│  │  ● YOLO      │  │                                 │  │
│  │  ● Condition │  │  [YOLO]──→[Condition]──→[VLLM] │  │
│  │  ● VLLM      │  │                     │          │  │
│  │  ● Output    │  │                     ▼          │  │
│  │             │  │                  [Output]       │  │
│  └─────────────┘  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │  节点属性面板 (选中节点时显示)                    │  │
│  │  model: sv30_seg/20260618_172731                │  │
│  │  conf: [0.25]  iou: [0.5]                       │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 与后端的数据交换

```javascript
// 保存 workflow → POST /api/workflow/save
const workflowJson = graph.serialize();  // LiteGraph 原生格式
fetch('/api/workflow/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        project: 'sv30_seg',
        name: 'ccrc_fall_detection',
        graph: workflowJson,  // 存为 JSON
    })
});
```

后端将 LiteGraph JSON 转换为 workflow.yaml 存盘。

## 7. 部署架构

### Docker Compose 更新

```yaml
services:
  xclabel-server:      # 已有，无需改动
    ...
  xclabel-deploy:      # 已有，增加 pipeline 端点
    build: .
    ports: ["8005:8000"]
    volumes:
      - ./projects:/app/projects    # 读取 workflow.yaml
    ...
  vllm-server:         # 新增，可选
    image: vllm/vllm-openai:latest
    command: --model Qwen/Qwen2-VL-7B-Instruct
    ports: ["8001:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 8. 实施计划

### 阶段一：PipelineManager 引擎 (deploy 端)

1. `deploy/pipeline_manager.py` — 核心编排引擎
   - workflow.yaml 加载与校验（Pydantic）
   - DAG 构建与拓扑排序
   - 节点执行路由
   - PipelineContext 数据传递
2. `deploy/vllm_client.py` — VLLM 异步客户端
   - OpenAI 兼容接口封装
   - ROI 裁剪（supervision）
   - 超时/重试逻辑
3. `deploy/main.py` — 新增 `/pipeline/execute` 端点
4. `deploy/requirements.txt` — 新增 `openai`, `supervision`, `pyyaml`

### 阶段二：Server 端 API + 前端编辑器

1. `app.py` — 新增 workflow CRUD API 端点
2. `templates/workflow.html` — LiteGraph.js 编辑器页面
3. `templates/projects.html` / 导航栏 — 添加入口
4. Workflow YAML ↔ JSON 转换逻辑

### 阶段三：端到端集成测试

1. 部署完整链路（server + deploy + vllm）
2. 创建 workflow → 保存 → 部署 → 执行 → 查看结果

## 9. 错误处理

| 场景 | 处理方式 |
|------|----------|
| VLLM 超时 | 跳过 VLLM 节点，返回 YOLO 结果 + warning |
| 模型未加载 | 自动尝试从 server 下载加载 |
| workflow.yaml 格式错误 | Pydantic 校验失败，返回具体错误信息 |
| 条件表达式异常 | 视为 false，继续执行后续节点 |
