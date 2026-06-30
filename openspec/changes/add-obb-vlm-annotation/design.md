## Context

当前 VLM 标注全链路（ai-config 测试 → /api/ai-label 标注 → Canvas 展示）仅支持矩形框（bbox）格式。Canvas 已支持 `type: "obb"` 标注的渲染（带 `[obb]` 标签前缀），但数据源端的解析和写入环节不支持 OBB 格式。

数据流现状：

```
VLM API 返回 JSON
  ↓
AiUtils.py analyze_image()  只解析 bbox，不解析 points
  ↓
render_detections()         只画矩形（cv2.rectangle）
  ↓
/api/ai-label                硬编码 type: "rectangle"
  ↓
Canvas drawAnnotation()      ✅ 已支持 type: "obb"（无需修改）
```

## Goals / Non-Goals

**Goals:**
- VLM 返回 OBB 格式（4 角点）时，全链路正确处理
- ai-config 测试页显示 OBB 多边形渲染结果
- 标注页面的 AI 标注（VLM API 模式）写入 `type: "obb"` 标注
- 兼容原有 bbox 格式，不破坏现有矩形框流程
- OBB 坐标按 DOTA 标准（顺时针 4 角点）处理

**Non-Goals:**
- 不修改 Canvas 的 OBB 工具（鼠标绘制仍是轴对齐矩形——VLM 驱动的 OBB 无需此功能）
- 不改动 YOLO 自动检测和智能分割模式（已有各自的处理路径）
- 不新增第三方依赖
- 不做 OBB 标注的手动旋转编辑（后续版本可考虑）

## Decisions

### 1. VLM 响应格式检测策略

**决策：每项检测动态检测是 `bbox` 还是 `points` 格式**

在 `analyze_image()` 中：
- 检测项包含 `points` 键 → 视为 OBB（4 角点）
- 检测项包含 `bbox` 键但不含 `points` → 视为矩形框
- 两者都包含 → 优先使用 `points`

这样 VLM 可以根据提示词灵活返回任意格式，系统自动适配。

### 2. OBB 坐标缩放

**决策：与 bbox 相同的 upscale 缩放逻辑**

现有代码在缩放图片后，通过 `upscale = 1.0 / scale` 将检测坐标从缩放尺寸转换回原始尺寸。OBB 的 4 个角点每个点独立应用相同缩放：
```
scaled_points = [[int(x * upscale), int(y * upscale)] for x, y in points]
```
点序保持不变（顺时针）。

### 3. 服务端多边形渲染

**决策：`render_detections()` 中 OBB 用 `cv2.polylines()` 渲染**

- `bbox` 格式 → `cv2.rectangle()`（不变）
- `points` 格式 → `cv2.polylines()`（闭合 4 点多边形）
- 多边形填充半透明色，线框 2px，与矩形框视觉风格一致
- 标签文字位置取第 1 个角点上方

### 4. AI 标注结果 OBB 写入

**决策：`/api/ai-label` 根据检测格式动态选择 annotation type**

- VLM 返回 `points`（OBB）→ 创建 `type: "obb"`，points 直接使用 4 角点
- VLM 返回 `bbox`（矩形）→ 创建 `type: "rectangle"`（现有逻辑不变）
- 标注 ID、class、confidence 字段不变

### 5. ai-config OBB 模式 UI

**决策：在现有「图片标注」标签页内添加 OBB 模式切换，不新增标签页**

- 在提示词上方增加一个 `checkbox` 或 `radio`："OBB 模式"
- 勾选后：
  - 提示词文本域切换为 OBB 模板（含顺时针 4 角点说明）
  - 不影响请求流程（仍调用 `/api/auto-label/image`）
- 渲染部分由服务端 `render_detections()` 自动适配，前端无感

## 数据流（修改后）

```
VLM API 返回 JSON（bbox 或 points）
  ↓
AiUtils.py analyze_image()
  ├─ 解析 bbox → 矩形格式（已有）
  └─ 解析 points → OBB 格式（新增）
  ↓
render_detections()
  ├─ bbox → cv2.rectangle()（已有）
  └─ points → cv2.polylines()（新增）
  ↓
/api/ai-label 创建标注
  ├─ bbox → type: "rectangle"（已有）
  └─ points → type: "obb"（新增）
  ↓
Canvas drawAnnotation()
  ├─ type: "rectangle" → 绘制 4 点矩形（已有）
  └─ type: "obb" → 绘制 4 点多边形 + [obb] 前缀（已有）
```

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| VLM 返回的点序非标准（非顺时针） | 提示词中明确要求顺时针顺序；前端不额外做点序重排（避免引入新的错误源） |
| OBB 点坐标超出图像边界 | `render_detections()` 中 clamp 到图像尺寸内 |
| 与现有 bbox 流程兼容性 | 检测格式动态判断，`points` 优先级高于 `bbox`，无 bbox 字段时完全走 OBB 路径 |
| Canvas 中 OBB 标签位置可能遮挡其他检测 | 标签放在第一个角点外侧，如果位置靠近边缘则自动上移/下移（复用现有逻辑） |
