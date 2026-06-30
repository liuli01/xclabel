## Why

xclabel 现有的推理服务仅支持单模型/单工作流的串行推理，缺乏条件分支和智能调度能力。在实际场景（如智慧养老跌倒检测）中，需要将 YOLO 的实时感知与大模型的深度理解结合，但现有架构无法支持这种多级流水线。本项目引入 Smart Pipeline，将系统升级为感知→决策→核实的智能编排架构。

## What Changes

1. **PipelineManager 编排引擎** — 在 deploy 服务中新增工作流执行引擎，解析 `workflow.yaml` 按 DAG 调度节点
2. **YOLO + VLLM 节点执行器** — 支持 YOLO 目标检测/分割、Condition 条件分支、VLLM 大模型核实三种节点类型
3. **workflow.yaml 配置协议** — 定义流水线拓扑的 YAML Schema，支持节点属性、条件表达式、数据传递
4. **Server 端 workflow CRUD API** — 6 个 REST 端点，管理工作流的保存/读取/部署/取消部署
5. **LiteGraph.js 可视化编辑器** — 在前端新增拖拽式工作流编辑页面，替换 nndeploy-app
6. **Docker Compose 扩展** — 可选增加 vllm-server 容器

## Capabilities

### New Capabilities

- `pipeline-execution-engine`: PipelineManager 工作流编排引擎，解析 YAML → 构建 DAG → 按拓扑序执行节点
- `pipeline-yolo-node`: YOLO 节点执行器，复用 YoloAdapter 调用 ultralytics predict()
- `pipeline-vllm-node`: VLLM 节点执行器，通过 OpenAI 兼容 API 调用大模型，支持 ROI 裁剪
- `pipeline-condition-node`: 条件分支节点，基于表达式（max_conf/detection_count）做流程分支
- `workflow-crud-api`: Server 端工作流 CRUD API（list/get/save/delete/deploy/undeploy）
- `workflow-editor-ui`: LiteGraph.js 可视化工作流编辑器页面

### Modified Capabilities

- *(无，不修改现有 spec 级别行为)*

## Impact

- **deploy 服务**: 新增 `pipeline_manager.py`, `vllm_client.py`，`main.py` 加 `/pipeline/execute` 端点，`requirements.txt` 加 `openai` / `supervision` / `pyyaml`
- **server 服务**: `app.py` 加 workflow CRUD 端点，`templates/workflow.html` 新增页面
- **依赖**: 新增 openai, supervision, pyyaml 依赖；VLLM 容器可选
- **文档**: docs/superpowers/specs/2026-06-23-smart-pipeline-design.md 已有设计文档
