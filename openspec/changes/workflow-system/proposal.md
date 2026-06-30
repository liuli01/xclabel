## Why

当前已实现 PipelineManager 基础编排引擎（YOLO→Condition→VLLM→Output），但距离完整的可视化工作流系统还有差距。目标是打造类似 Roboflow 的工作流体验：在浏览器中拖拽配置多模型流水线（YOLO + VLLM + 自定义节点），保存后在 deploy 服务中一键部署执行。

## What Changes

1. **工作流可视化编辑器完善** — 替代 nndeploy-app，支持多节点类型，拖拽连线，参数面板
2. **多模型管理** — workflow 可引用多个已加载模型（YOLO detect/segment/pose + VLLM）
3. **工作流部署到 deploy** — 从 server 端将 workflow.yaml 推送到 deploy 并加载为可执行 Pipeline
4. **推理结果标注图生成** — deploy 端使用 supervision 生成带标注的结果图片
5. **工作流执行历史与监控** — 记录每次执行结果，可视化查看

## Capabilities

### New Capabilities
- `workflow-visual-editor`: LiteGraph.js 拖拽编辑器，支持 YOLO/Condition/VLLM/Output 节点
- `workflow-deploy-engine`: Server 端将 workflow 部署到 deploy 服务的完整链路
- `multi-model-registry`: 工作流中引用多个模型，自动管理模型加载依赖
- `annotated-result-image`: Deploy 端使用 supervision 生成标注结果图

### Modified Capabilities
- *(无，本次新增功能不修改现有 spec)*

## Impact

- **deploy**: `pipeline_manager.py` 增强多模型支持，新增标注图生成端点
- **server**: `app.py` 新增 workflow 部署/执行 API，`templates/workflow.html` 完善编辑器
- **依赖**: supervision 已安装，opencv-python 已安装
