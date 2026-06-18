## Why

当前 xclabel 的 VLM AI 标注只能产生矩形框，缺乏精细化分割能力；而 SAM 2 能精确分割轮廓但无法识别物体类别。将两者串联可以在批量标注时同时获得「正确的类别标签」和「精确的分割轮廓」，大幅减少人工修正工作量。

## What Changes

- 新增 VLM→SAM 精细化标注管道：VLM 检测物体（label + bbox）→ SAM 2 用 bbox 作为 Box Prompt 生成分割 Mask → 自动保存为多边形标注
- AI 标注模态框新增「VLM+SAM 精细化」模式
- 新 API 路由 `POST /api/auto-label/vlm-sam`
- SocketIO 事件 `vlm_sam_progress` 推送每张图的进度

## Capabilities

### New Capabilities
- `vlm-sam-pipeline`: VLM 视觉语言模型检测 + SAM 2 精细化分割的串联管道，支持批量自动标注

### Modified Capabilities

无（不修改现有规范，仅新增管道）

## Impact

- `app.py`：新增 API 路由 `/api/auto-label/vlm-sam`
- `templates/index.html`：AI 模态框新增第三种模式
- `static/script.js`：新增 `startVlmSamLabel()` 函数
- 无新增依赖（复用 `AIAutoLabeler` + `SAM2Engine`）
