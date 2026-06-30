## Why

当前系统支持通过 VLM API 进行自动标注和测试，但仅支持矩形框（bbox）格式。用户需要在线 AI 大模型返回旋转框（OBB）坐标，并在全链路中正确处理：从配置页面的 API 测试，到标注页面的 AI 标注结果写入，再到 Canvas 渲染展示。ai-config 页面已有 OBB 测试需求，且标注页面的画布已支持 `type: "obb"` 标注的显示，但中间的数据解析和写入环节缺失。

## What Changes

- **VLM 响应解析增强**：`AiUtils.py` 的 `analyze_image()` 同时解析 `bbox`（矩形框）和 `points`（OBB 四点坐标）两种格式
- **OBB 格式坐标缩放**：`analyze_image()` 中的坐标缩放逻辑支持 `points` 的 4 角点等比例转换
- **服务端渲染增强**：`render_detections()` 支持 OBB 4 点多边形绘制（`cv2.polylines`）
- **AI 标注结果写入 OBB**：`app.py` 的 `/api/ai-label` 端点在检测到 VLM 返回 `points` 时，创建 `type: "obb"` 的标注
- **ai-config OBB 测试模式**：在测试标签页添加 OBB 模式切换，使用 OBB 默认提示词模板，渲染结果显示 4 点多边形

## Capabilities

### New Capabilities

- `vlm-obb-backend`: VLM 响应的 OBB 格式解析与坐标转换，以及 `/api/ai-label` 的 OBB 标注创建
- `obb-testing-ui`: ai-config 页面的 OBB 模式切换、提示词模板切换、以及 OBB 检测结果的多边形渲染

### Modified Capabilities

- （无现有规格变更）

## Impact

- `AiUtils.py` — `analyze_image()` 解析 `points` 字段 + 坐标缩放；`render_detections()` 支持多边形渲染
- `app.py` — `/api/ai-label` 识别 OBB 结果并写入 `type: "obb"` 标注
- `templates/ai_config.html` — OBB 模式开关 + 提示词模板 + 渲染逻辑
- `templates/index.html` — 无改动（Canvas 已支持 `type: "obb"`）
- `static/script.js` — 无改动（Canvas 渲染已兼容 OBB）
- 无新增依赖
