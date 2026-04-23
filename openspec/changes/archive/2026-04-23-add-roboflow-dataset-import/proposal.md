## Why

Roboflow 是目前最主流的计算机视觉数据集托管与管理平台之一。用户在 Roboflow 上导出的数据集通常以 `.zip` 格式（如 `river_color.v4i.yolov8.zip`）提供，包含图像和 YOLO 格式标注文件。当前 xclabel 仅支持图片上传、视频抽帧和 LabelMe 数据集导入，不支持直接导入 Roboflow 格式的 ZIP 数据集，导致用户需要手动解压、整理文件后才能使用。添加 Roboflow 数据集导入选项可以大幅降低用户迁移成本，提升工具可用性。

## What Changes

- 在"添加数据集"弹窗中新增"Roboflow 数据集"选项卡
- 支持上传 `.zip` 格式的 Roboflow 数据集文件（如 YOLOv8 格式）
- 后端新增 `/api/upload/roboflow` API 端点，用于解析 ZIP 文件
- 解压 ZIP 并提取图片文件到 `uploads/` 目录
- 解析 `data.yaml` 获取类别信息，并同步到 `classes.json`
- 解析 YOLO `.txt` 标注文件，转换为内部标注格式并保存到 `annotations.json`
- 若 ZIP 中不含标注文件，则仅导入图片作为未标注数据集
- 修改 YOLO 导出功能 (`/api/export`)，多边形标注导出为 YOLO 分割格式（`class_id x1 y1 x2 y2 ...`），矩形标注仍导出为边界框格式（`class_id cx cy w h`）

## Capabilities

### New Capabilities
- `roboflow-dataset-import`: 支持上传并解析 Roboflow 格式的 ZIP 数据集文件，提取图片、类别和标注信息到 xclabel 内部存储格式
- `yolo-segmentation-export`: 修改 YOLO 导出逻辑，多边形标注按 YOLO 分割格式导出（归一化多边形点），矩形标注仍按边界框格式导出

### Modified Capabilities
- 无现有能力需要修改（本功能为新增能力，不涉及已有能力的需求变更）

## Impact

- **前端 UI**: `templates/index.html`（新增选项卡和上传区域）、`static/script.js`（新增事件处理）
- **后端 API**: `app.py`（新增 `/api/upload/roboflow` 路由及解析逻辑；修改 `/api/export` 导出多边形标注逻辑）
- **数据存储**: 复用现有的 `uploads/`、`annotations.json`、`classes.json` 存储机制
- **依赖**: 复用 Python 标准库 `zipfile`、`yaml`（PyYAML）进行 ZIP 解压和 YAML 解析
