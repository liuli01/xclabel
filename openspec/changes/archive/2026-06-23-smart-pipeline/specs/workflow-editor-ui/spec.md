## ADDED Requirements

### Requirement: 导航栏添加"流程编排"入口
xclabel 导航栏 SHALL 添加"流程编排"按钮，点击后跳转到 workflow.html 页面。

#### Scenario: 点击导航按钮
- **WHEN** 用户点击导航栏的"流程编排"
- **THEN** 跳转到 /workflow?project=<current>

### Requirement: LiteGraph.js 可视化编辑器
workflow.html SHALL 使用 LiteGraph.js 提供拖拽式节点编辑器，支持以下节点类型：
- YOLO Node（模型选择、conf/iou 参数）
- Condition Node（表达式输入）
- VLLM Node（API URL、模型名、prompt 编辑）
- Output Node

#### Scenario: 拖拽节点
- **WHEN** 用户从节点面板拖拽 YOLO Node 到画布
- **THEN** 画布上出现 YOLO 节点，可编辑参数

#### Scenario: 连线节点
- **WHEN** 用户从一个节点的输出拖线到另一个节点的输入
- **THEN** 两节点间出现连线，表示数据流方向

### Requirement: 保存/部署按钮
编辑器 SHALL 提供"保存"和"部署"按钮。
保存将画布内容序列化为 JSON 并发送到 POST /api/workflow/save。
部署调用 POST /api/workflow/deploy。

#### Scenario: 保存工作流
- **WHEN** 用户点击"保存"按钮
- **THEN** 画布序列化为 JSON，发送到后端保存

#### Scenario: 部署工作流
- **WHEN** 用户点击"部署"按钮
- **THEN** 先保存，再调用部署 API 推送到 deploy 服务
