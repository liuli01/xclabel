# OBB VLM 全链路标注支持 实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 VLM API 返回 OBB（旋转框）格式时全链路正确处理：从 ai-config 测试页渲染到标注页的 AI 标注写入。

**Architecture:** 后端 3 处修改 + 前端 1 处修改。AiUtils.py 的 `analyze_image()` 新增 `points` 字段解析与坐标缩放，`render_detections()` 新增多边形渲染；app.py 的 `/api/ai-label` 动态判断 OBB 格式并写入 `type: "obb"` 标注；ai_config.html 增加 OBB 模式切换与提示词模板。

**Tech Stack:** Python / Flask / OpenCV / JavaScript / Canvas 2D

---

### Task 1: AiUtils.py `analyze_image()` — OBB 坐标解析与缩放

**Files:**
- Modify: `AiUtils.py:553-567`

在坐标缩放循环中，增加对 `points` 字段的处理（4 角点缩放），同时兼容现有的 `bbox` 逻辑。`points` 优先于 `bbox`。

- [ ] **Step 1: 修改坐标缩放循环，增加 OBB points 解析**

将 `AiUtils.py` 第 553-567 行的代码替换为：

```python
                    # 将检测到的坐标从缩放后的尺寸转换回原始图片尺寸
                    for detection in result_json["detections"]:
                        # OBB 格式：points 4 角点（顺时针）
                        if "points" in detection:
                            pts = detection["points"]
                            if len(pts) == 4:
                                scaled = []
                                for pt in pts:
                                    x = int(float(pt[0]) * upscale)
                                    y = int(float(pt[1]) * upscale)
                                    # clamp 到图像边界
                                    x = max(0, min(x, original_w - 1))
                                    y = max(0, min(y, original_h - 1))
                                    scaled.append([x, y])
                                detection["points"] = scaled
                        # 矩形框格式：bbox [x1,y1,x2,y2]
                        elif "bbox" in detection:
                            bbox = detection["bbox"]
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = map(float, bbox)

                                # 如果图片被缩小了，则检测框需要等比放大
                                # upscale = 1.0 / scale
                                x1 = int(x1 * upscale)
                                y1 = int(y1 * upscale)
                                x2 = int(x2 * upscale)
                                y2 = int(y2 * upscale)

                                detection["bbox"] = [x1, y1, x2, y2]
```

关键点：
- `points` 存在时优先处理 OBB，跳过 `bbox`
- `points` 不存在时才检查 `bbox`（保持原有逻辑）
- 4 个角点独立缩放，点序不变
- 坐标 clamp 到 `[0, width-1]` `[0, height-1]`

- [ ] **Step 2: 验证修改**

启动 app：
```bash
cd e:/coding/project/xclabel && python app.py
```

构造一个带 `points` 的测试返回。由于这是后端修改，测试可以配合 ai-config 页面进行端到端验证（Task 4 完成后），或临时写一个简单的 Python 测试脚本验证坐标缩放逻辑。

---

### Task 2: AiUtils.py `render_detections()` — OBB 多边形渲染

**Files:**
- Modify: `AiUtils.py:609-659`

将检测渲染逻辑从"只画矩形框"改为"检测到 `points` 时画多边形，否则画矩形框"。标签文字位置跟随多边形第一个顶点。

- [ ] **Step 1: 修改渲染循环，增加 OBB 多边形绘制**

将 `AiUtils.py` 第 609-659 行替换为：

```python
        # 渲染检测框和标签
        for detection in detections:
            # 解析检测结果
            if isinstance(detection, dict):
                label = detection.get("label", "unknown")
                confidence = detection.get("confidence", 0.0)
                points = detection.get("points", None)
                bbox = detection.get("bbox", [0, 0, 0, 0])
            else:
                continue

            # 获取颜色
            color = self.colors.get(label, self.colors["default"])

            # 判断是否为 OBB（有 points 且至少 3 个点）
            is_obb = points is not None and len(points) >= 3

            if is_obb:
                # OBB: 绘制 4 点多边形
                pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts_array], isClosed=True, color=color, thickness=2)
                anchor_x, anchor_y = int(points[0][0]), int(points[0][1])
            else:
                # 矩形框：绘制矩形
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                anchor_x, anchor_y = x1, y1

            # 绘制标签和置信度（支持中文）
            label_text = f"{label}: {confidence:.2f}"

            # 尝试使用PIL库渲染中文
            try:
                import numpy as np
                from PIL import Image, ImageDraw, ImageFont

                # 转换为PIL图像
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                # 加载默认中文字体或指定字体文件
                try:
                    # 尝试使用系统默认中文字体
                    font = ImageFont.truetype("simhei.ttf", 16)
                except IOError:
                    # 如果没有找到，使用PIL默认字体
                    font = ImageFont.load_default()

                # 绘制文本（使用 anchor 作为标签位置）
                text_x = anchor_x
                text_y = anchor_y - 20 if anchor_y > 20 else anchor_y + 20
                draw.text((text_x, text_y), label_text, font=font, fill=tuple(color[::-1]))

                # 转换回OpenCV图像
                image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 如果PIL渲染失败，使用OpenCV默认渲染（可能会有乱码）
                logging.warning(f"中文渲染失败，使用默认渲染: {e}")
                cv2.putText(image, label_text, (anchor_x, anchor_y - 10 if anchor_y > 10 else anchor_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```

关键点：
- `points` 存在且 ≥3 个点 → 视为 OBB，用 `cv2.polylines()` 画闭合多边形
- 否则 → `bbox` 矩形框（兼容原有逻辑）
- 标签位置统一用 `(anchor_x, anchor_y)`——OBB 时用第一个角点，bbox 时用左上角
- `cv2.polylines` 的 `isClosed=True` 确保首尾相连

- [ ] **Step 2: 验证渲染**

重启 app，打开 `http://127.0.0.1:9924/ai-config`，在"图片标注"标签页测试一张图片（此时即使没用 OBB 模式，现有的 bbox 格式仍然正常渲染）。

---

### Task 3: app.py `/api/ai-label` — OBB 标注写入

**Files:**
- Modify: `app.py:1591-1617`

在 `/api/ai-label` 的检测结果处理循环中，增加对 `points` 格式的检测，OBB 时创建 `type: "obb"` 标注。

- [ ] **Step 1: 修改标注创建逻辑**

将 `app.py` 第 1591-1617 行替换为：

```python
                    for detection in detections:
                        # 确保detection是字典
                        if isinstance(detection, dict):
                            label = selected_label  # 使用选中的标签
                            confidence = detection.get("confidence", 0.0)
                            obb_points = detection.get("points", None)

                            if obb_points and len(obb_points) >= 3:
                                # OBB 格式：直接使用 4 角点
                                points = [[float(p[0]), float(p[1])] for p in obb_points[:4]]
                                annotation = {
                                    "id": str(uuid.uuid4()),
                                    "class": label,
                                    "type": "obb",
                                    "points": points,
                                    "confidence": confidence
                                }
                            else:
                                # 矩形框格式（现有逻辑）
                                bbox = detection.get("bbox", [0, 0, 0, 0])
                                bbox = list(map(float, bbox)) if isinstance(bbox, (list, tuple)) else [0, 0, 0, 0]
                                if len(bbox) < 4:
                                    bbox = bbox + [0] * (4 - len(bbox))
                                x1, y1, x2, y2 = bbox[:4]
                                annotation = {
                                    "id": str(uuid.uuid4()),
                                    "class": label,
                                    "type": "rectangle",
                                    "points": [
                                        [x1, y1],
                                        [x2, y1],
                                        [x2, y2],
                                        [x1, y2]
                                    ],
                                    "confidence": confidence
                                }
```

关键点：
- `points` 存在且 ≥3 个点 → OBB：`type: "obb"`，直接使用 4 角点
- 否则 → 矩形框：`type: "rectangle"`，从 bbox 派生 4 点（现有逻辑保持）
- Canvas 侧已支持 `type: "obb"`——带 `[obb]` 前缀渲染多边形

- [ ] **Step 2: 验证标注写入**

重启 app，打开一个 OBB 项目（如 `ocr_table`），点击「AI标注」→ 选择「VLM API 标注」模式 → 使用 OBB 提示词让 VLM 返回 `points` → 点击「开始执行」→ 检查标注结果是否为 `type: "obb"` 并正确渲染为旋转框。

---

### Task 4: ai_config.html — OBB 模式切换与提示词模板

**Files:**
- Modify: `templates/ai_config.html`

在「图片标注」标签页的提示词区域上方添加 OBB 模式复选框，勾选时切换提示词模板为 OBB 格式。

- [ ] **Step 1: 添加 OBB 模式复选框 HTML**

在 `ai_config.html` 第 617 行（`<div class="tool-section">` 即提示词 section 的开头）之前，插入 OBB 模式开关：

```html
                    <div class="tool-section">
                        <h4>标注模式</h4>
                        <div class="form-group">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                    <input type="checkbox" id="obbMode">
                                    <i class="fas fa-sync-alt"></i> OBB 模式（旋转框）
                                </label>
                                <span style="font-size: 12px; color: #999;">启用后将使用 4 角点格式提示词</span>
                            </div>
                        </div>
                    </div>
```

- [ ] **Step 2: 添加 OBB 提示词模板和切换逻辑**

在 `ai_config.html` 的 `<script>` 区域（约第 1073 行的 `getApiConfig()` 函数之后），添加 OBB 提示词常量和切换监听：

```javascript
        // OBB 默认提示词模板
        const OBB_PROMPT_TEMPLATE = '检测图中的旋转框（OBB），返回JSON格式：\\n' +
            '{"detections":[\\n' +
            '  {"label":"类别","confidence":0.9,"points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}\\n' +
            ']}\\n' +
            '要求：4个角点按顺时针排列（左上→右上→右下→左下），坐标为原始图片像素值。';

        // 普通检测默认提示词模板
        const NORMAL_PROMPT_TEMPLATE = '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}';

        // OBB 模式切换
        document.addEventListener('DOMContentLoaded', function() {
            const obbCheckbox = document.getElementById('obbMode');
            const promptTextarea = document.getElementById('prompt');
            let previousPrompt = promptTextarea.value;

            obbCheckbox.addEventListener('change', function() {
                if (this.checked) {
                    // 保存当前提示词，切换到 OBB 模板
                    previousPrompt = promptTextarea.value;
                    promptTextarea.value = OBB_PROMPT_TEMPLATE;
                } else {
                    // 恢复之前保存的提示词
                    promptTextarea.value = previousPrompt;
                }
            });
        });
```

- [ ] **Step 3: 验证 OBB 模式切换**

重启 app，打开 `http://127.0.0.1:9924/ai-config` →「图片标注」标签页：
1. 勾选「OBB 模式」→ 提示词应切换为 OBB 模板
2. 取消勾选 → 提示词应恢复为原来的内容
3. 在 OBB 模式下测试一张图片 → 渲染结果应显示 4 点多边形（由 Task 2 的服务端渲染支撑）
