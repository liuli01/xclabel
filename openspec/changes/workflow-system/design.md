## Context

当前 PipelineManager 支持单 YOLO 模型 + Condition + VLLM 的线性流水线。需扩展为支持多模型并行/串行组合、可视化编辑、一键部署到 deploy。

## Goals / Non-Goals

**Goals:**
- LiteGraph.js 编辑器支持 YOLO / Condition / VLLM / Output 节点完整参数配置
- Workflow 定义中可引用多个模型（如 YOLO detect + YOLO seg + VLLM）
- Server → Deploy 一键部署工作流
- Deploy 端返回标注结果图片（supervision 渲染）

**Non-Goals:**
- 不支持工作流版本管理（后续可加）
- 不支持定时调度（后续可加）

## Decisions

| 决策 | 选择 | 理由 |
|------|------|------|
| 工作流文件格式 | YAML（已有 schema） | 与 PipelineManager 兼容，可版本控制 |
| 多模型引用 | 用 `model` 字段引用引擎池中的模型 ID | 模型先通过 `/load/model` 加载到引擎池 |
| 编辑器通信 | LiteGraph JSON → Server 转换 YAML | 前端只负责可视化，后端负责格式转换 |
| 标注图 | deploy 端新增 `/infer/annotated` 端点 | 返回 supervision 渲染的 JPEG base64 |

## Risks / Trade-offs

- 多模型并行执行会增加内存占用 → 限制同时加载的模型数量（已有 MAX_ENGINES）
- VLLM 调用耗时可能阻塞 → 维持 30s 超时策略
