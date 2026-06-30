## ADDED Requirements

### Requirement: 工作流编辑器支持多节点类型
编辑器 SHALL 支持 YOLO（detect/segment/pose）、Condition、VLLM、Output 四种节点类型的拖拽添加和参数配置。

#### Scenario: 拖拽添加 YOLO 节点
- **WHEN** 用户从节点面板拖拽 YOLO Node 到画布
- **THEN** 画布上出现 YOLO 节点，可配置 model、task、conf、iou 参数

### Requirement: 工作流一键部署到 deploy
系统 SHALL 支持从 server 页面将 workflow.yaml 一键推送到 deploy 服务并加载为可执行 Pipeline。

#### Scenario: 部署工作流
- **WHEN** 用户点击"部署"按钮
- **THEN** server 调用 deploy 的 /pipeline/load 端点，返回部署状态

### Requirement: Deploy 返回标注结果图
Deploy SHALL 提供 `/infer/annotated` 端点，返回 supervision 渲染的标注结果图片（base64 JPEG）。

#### Scenario: 获取标注图
- **WHEN** 用户请求 /infer/annotated 带 engine_id 和 image
- **THEN** 返回包含标注框/多边形和标签的 JPEG 图片 base64
