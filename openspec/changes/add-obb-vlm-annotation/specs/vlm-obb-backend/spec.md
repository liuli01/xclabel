## ADDED Requirements

### Requirement: VLM 响应支持 OBB 格式解析

系统 SHALL 在 `AiUtils.py` 的 `analyze_image()` 方法中，支持从 VLM 响应中解析 OBB 4 角点坐标。

#### Scenario: 解析 OBB points 格式
- **WHEN** VLM 返回的 JSON 中 `detections` 数组某个元素包含 `points` 键（值为 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`）
- **THEN** 系统 SHALL 识别该检测为 OBB 格式
- **THEN** 系统 SHALL 对每个角点应用坐标缩放（`x * upscale, y * upscale`）
- **THEN** 返回结果中 SHALL 保留 `points` 字段和原始 `label`、`confidence` 字段

#### Scenario: 兼容 bbox 格式
- **WHEN** VLM 返回的 JSON 中 `detections` 数组元素包含 `bbox` 但不含 `points`
- **THEN** 系统 SHALL 按现有逻辑处理（bbox 坐标缩放，不产生 points 字段）

#### Scenario: points 和 bbox 同时存在
- **WHEN** VLM 返回的检测元素同时包含 `points` 和 `bbox`
- **THEN** 系统 SHALL 优先使用 `points` 作为 OBB 格式

#### Scenario: 坐标缩放一致性
- **WHEN** OBB 的 4 个角点经过缩放后
- **THEN** 每个点的 x/y 值 SHALL 为 `int(x * upscale)` / `int(y * upscale)`
- **THEN** 4 个点的顺序 SHALL 保持与 VLM 返回一致，不做重排

### Requirement: 服务端渲染支持 OBB 多边形

系统 SHALL 在 `AiUtils.py` 的 `render_detections()` 方法中支持绘制 OBB 4 点多边形。

#### Scenario: 渲染 OBB 多边形
- **WHEN** 检测元素包含 `points` 字段（4 个角点）
- **THEN** 系统 SHALL 使用 `cv2.polylines()` 绘制闭合 4 点多边形
- **THEN** 线宽 SHALL 为 2px
- **THEN** 颜色 SHALL 使用当前检测的 label 对应颜色

#### Scenario: 渲染标签文字
- **WHEN** 检测元素有 `points` 和 `label`、`confidence`
- **THEN** 标签文字 SHALL 显示在第一个角点上方 (`points[0].x, points[0].y - 20`)
- **THEN** 标签格式 SHALL 为 `{label}: {confidence:.2f}`

#### Scenario: 坐标边界限制
- **WHEN** OBB 点坐标超出图像边界
- **THEN** 系统 SHALL 将坐标 clamp 到图像尺寸范围内（`[0, width)`、`[0, height)`）

#### Scenario: 渲染矩形框
- **WHEN** 检测元素包含 `bbox` 但不含 `points`
- **THEN** 系统 SHALL 使用 `cv2.rectangle()` 绘制（现有逻辑不变）

### Requirement: AI 标注支持 OBB 写入

系统 SHALL 在 `app.py` 的 `/api/ai-label` 端点中，根据 VLM 返回格式动态创建对应类型的标注。

#### Scenario: VLM 返回 OBB 格式
- **WHEN** VLM API 返回的检测结果包含 `points` 字段
- **THEN** 创建的标注 SHALL 设置 `type: "obb"`
- **THEN** 标注的 `points` SHALL 直接使用 4 角点 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`

#### Scenario: VLM 返回 bbox 格式
- **WHEN** VLM API 返回的检测结果包含 `bbox` 但不含 `points`
- **THEN** 创建的标注 SHALL 设置 `type: "rectangle"`（现有逻辑不变）
- **THEN** `points` SHALL 从 bbox 四边派生 `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]`
