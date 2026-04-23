## Why

目前 xclabel 仅支持图片标注和 AI 自动标注，缺少模型训练闭环。用户完成数据标注后，需要离开工具去手动配置训练环境、编写训练脚本，流程断裂。在工程管理页面直接提供 YOLO 模型训练入口，可以让用户基于当前工程的标注数据快速启动训练，形成"标注-训练-迭代"的完整工作流。

## What Changes

- 新增 YOLO 模型训练页面 `/train`，支持选择工程、配置训练参数、启动训练任务
- 后端新增 `/api/train` 系列 API，基于 Ultralytics YOLO 框架执行训练
- 新增训练环境安装逻辑：在 `plugins/` 目录下创建独立 venv 并安装 `ultralytics`
- 工程管理页面 (`/projects`) 每个工程卡片新增"进入训练"按钮，位于"进入标注"旁边
- 训练过程实时推送进度（SocketIO），页面展示 loss、mAP、迭代轮数等关键指标
- 训练完成后自动保存最佳权重到工程目录下的 `models/` 文件夹

## Capabilities

### New Capabilities

- `yolo-model-training`: YOLO 模型训练页面与后端训练任务管理，包含训练参数配置、任务启动、进度监控、结果保存

### Modified Capabilities

- （无 spec 级别需求变更，工程管理页面的按钮添加属于实现细节）

## Impact

- **前端**: 新增 `templates/train.html`，修改 `templates/projects.html` 增加训练入口按钮
- **后端**: 新增训练 API 路由、Ultralytics 环境安装检测、训练任务封装（进程/线程管理）
- **依赖**: 可选依赖 `ultralytics`（按需安装到独立 venv，不污染主环境）
- **文件系统**: 工程目录下新增 `models/` 子目录用于保存训练权重
