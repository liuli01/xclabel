## 1. 后端：VLM 响应解析支持 OBB

- [ ] 1.1 修改 `AiUtils.py` 的 `analyze_image()`：在遍历 `detections` 时，检测每个元素是否包含 `points` 字段
- [ ] 1.2 实现 `points` 坐标缩放：对 4 个角点分别应用 `x * upscale, y * upscale`
- [ ] 1.3 确保 `points` 和 `bbox` 同时存在时优先使用 `points`
- [ ] 1.4 添加坐标边界 clamp（超出图像尺寸时切齐到边界）

## 2. 后端：服务端渲染支持 OBB 多边形

- [ ] 2.1 修改 `AiUtils.py` 的 `render_detections()`：检测 `points` 字段
- [ ] 2.2 实现 `cv2.polylines()` 绘制闭合 4 点多边形（OBB 路径）
- [ ] 2.3 调整标签文字位置：当 OBB 时使用 `points[0]` 作为锚点
- [ ] 2.4 保持矩形框（bbox）渲染逻辑不变

## 3. 后端：AI 标注写入支持 OBB

- [ ] 3.1 修改 `app.py` 的 `/api/ai-label` 端点的检测结果处理逻辑
- [ ] 3.2 VLM 返回 `points` 时设置 `type: "obb"` 并使用 4 角点
- [ ] 3.3 VLM 返回 `bbox` 时保持 `type: "rectangle"` 现有逻辑

## 4. 前端：ai-config OBB 模式

- [ ] 4.1 在 `ai_config.html` 的「图片标注」标签页提示词上方添加「OBB 模式」复选框
- [ ] 4.2 编写 OBB 默认提示词模板（含顺时针 4 角点说明）
- [ ] 4.3 实现复选框切换逻辑：选中/取消时切换到对应提示词模板
- [ ] 4.4 确保请求参数中携带 OBB 模式标识（可选，渲染由后端自动适配）
