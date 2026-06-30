## Context

xclabel-deploy 当前支持单模型推理（通过 YoloAdapter/nndeploy）和单工作流执行。为实现多级智能流水线，需增加 PipelineManager 编排引擎。完整设计文档见 `docs/superpowers/specs/2026-06-23-smart-pipeline-design.md`。

## Goals / Non-Goals

**Goals:**
- 工作流由 `workflow.yaml` 定义，支持 YOLO + Condition + VLLM + Output 节点
- PipelineManager 按 DAG 拓扑序调度节点，节点间通过 PipelineContext 传递数据
- Condition 节点基于表达式（max_conf/detection_count）做分支判断
- VLLM 节点通过 OpenAI 兼容 API 调用，支持 ROI 裁剪
- Server 端提供 workflow CRUD API，前端提供 LiteGraph.js 可视化编辑器

**Non-Goals:**
- 不支持嵌套/子工作流
- 不支持实时视频流处理（仅单图）
- 不实现 VLLM 异步队列（同步调用+超时）

## Decisions

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|----------|------|
| 执行引擎位置 | deploy 服务 | 独立服务 | 复用现有 YoloAdapter 和引擎池，减少部署复杂度 |
| 工作流存储 | YAML 文件 | 数据库 | 文件可版本控制，编辑方便，与项目目录一致 |
| 条件表达式 | Python eval 沙箱 | 自定义 DSL | 简单场景够用，无需引入新 parser |
| 前端库 | LiteGraph.js | React Flow | 无需打包工具，直接 `<script>` 引入 Flask 模板 |
| VLLM 协议 | OpenAI 兼容 API | gRPC | 通用性强，支持 vLLM/LMDeploy/Ollama |

## Risks / Trade-offs

- **VLLM 超时阻塞** → deploy 设为 30s 超时，超时则跳过 VLLM 节点返回 YOLO 结果 + warning
- **eval 安全性** → 仅允许白名单变量（max_conf, detection_count），禁用内置函数
- **YAML 格式错误** → Pydantic 严格校验，返回具体错误位置
- **LiteGraph 兼容性** → 项目已有 jQuery，需测试 LiteGraph 与之共存
