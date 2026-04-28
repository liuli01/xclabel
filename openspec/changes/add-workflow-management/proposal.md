## Why

当前 xclabel 仅支持单步的模型训练和推理流程，用户无法将标注、训练、推理、后处理等多个步骤串联成可复用的自动化工作流。参考 Roboflow Workflow 的设计，引入工作流机制可以让用户按业务场景编排处理链路（如「自动标注 → 人工审核 → 模型训练 → 批量推理」），提升数据闭环效率。

## What Changes

- 在工程界面新增「工作流」标签页，展示当前工程下的工作流列表
- 支持新建工作流：提供可视化或表单式的工作流编辑器，允许用户添加/连接/配置节点
- 工作流节点类型至少包括：数据集导入、AI 自动标注、人工审核、模型训练、模型推理、结果导出
- 支持保存/编辑/删除/重命名工作流配置
- 支持运行工作流，实时显示各节点执行状态和日志
- 工作流配置以 JSON 格式持久化到工程目录

## Capabilities

### New Capabilities
- `workflow-management`: 工作流的创建、编辑、列表展示、删除和运行控制

### Modified Capabilities
- `project-management`: 工程界面新增「工作流」入口标签页，扩展工程管理页面的导航结构

## Impact

- 前端：`templates/projects.html` 新增工作流标签页和列表渲染逻辑；新增 `templates/workflow-editor.html` 工作流编辑页面
- 后端：`app.py` 新增 `/api/workflows/*` 系列 API（CRUD + 运行控制）
- 数据存储：工程目录下新增 `workflows/` 文件夹，每个工作流保存为独立 JSON 文件
- 依赖：无新增第三方依赖
