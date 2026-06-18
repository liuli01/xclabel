# VLM → SAM 精细化标注管道设计文档

> 日期：2026-06-17
> 状态：设计完成，待实施

## 1. 概述

将现有的 VLM 视觉语言模型标注功能与 SAM 2 分割能力串联，形成"VLM 识别类别 → SAM 精细化分割"的自动化标注管道。

### 目标

一键批量：选中图片 → VLM 检测物体（返回 label + bbox） → SAM 2 对每个 bbox 做精细化分割 → 自动保存带 Mask 的多边形标注

---

## 2. 架构

```
┌─────────────────────────────────────────────────────────────┐
│  AI 模态框「VLM+SAM 精细化」模式                               │
│  选中图片 → 点击开始执行                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ POST /api/auto-label/vlm-sam
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Flask 后端                                                  │
│                                                             │
│  for each image:                                            │
│    ┌─────────────────┐    ┌──────────────┐    ┌──────────┐  │
│    │ AIAutoLabeler    │ →  │ SAM2Engine   │ →  │ 保存标注  │  │
│    │ (VLM 检测)       │    │ (Box Prompt) │    │ .json    │  │
│    │ → {label, bbox}  │    │ → mask多边形  │    └──────────┘  │
│    └─────────────────┘    └──────────────┘                  │
│         ↕ SocketIO 进度推送                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. API 设计

### `POST /api/auto-label/vlm-sam`

**请求：**
```json
{
  "images": ["img001.jpg", "img002.jpg"],
  "prompt": "检测图中所有物体，返回JSON格式...",
  "label_filter": null
}
```

- `prompt`：可覆盖 AI 配置中的默认提示词
- `label_filter`：可选，只保留指定类别的检测结果

**响应：**
```json
{
  "success": true,
  "total": 20,
  "labeled": 18,
  "annotations": {
    "img001.jpg": [
      {"class": "cat", "type": "polygon", "points": [...], "source": "vlm_sam", "confidence": 0.95}
    ]
  }
}
```

### SocketIO 事件：`vlm_sam_progress`

```json
{
  "current": 3,
  "total": 20,
  "image": "img003.jpg",
  "status": "sam",
  "found": 2
}
```

- `status`: `"vlm"`（正在 VLM 检测）| `"sam"`（正在 SAM 精化）| `"done"`（完成）

---

## 4. 后端实现

### 新路由：`/api/auto-label/vlm-sam`

在 `app.py` 中新增路由，处理流程：

```python
@app.route('/api/auto-label/vlm-sam', methods=['POST'])
def auto_label_vlm_sam():
    data = request.get_json()
    images = data.get('images', [])
    prompt = data.get('prompt')
    
    # 1. 加载 AI 配置
    api_config = load_ai_config()
    
    # 2. 获取 GPU 锁（VLM + SAM 连续占用）
    if not acquire_gpu("vlm_sam", timeout=30):
        return jsonify({'error': 'GPU 正忙'}), 503
    
    try:
        for i, image_name in enumerate(images):
            # 2a. VLM 检测
            socketio.emit('vlm_sam_progress', {
                'current': i+1, 'total': len(images),
                'image': image_name, 'status': 'vlm'
            })
            
            labeler = AIAutoLabeler(**api_config, prompt=prompt)
            result = labeler.analyze_image(image_path)
            detections = result.get('detections', [])
            
            # 2b. 对每个检测做 SAM 精化
            annotations = []
            for det in detections:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                label = det['label']
                confidence = det.get('confidence', 0.0)
                
                # SAM 2 Box Prompt
                sam_engine = get_sam2_engine()
                sam_result = sam_engine.predict(image_path, [
                    {'type': 'box', 'x1': bbox[0], 'y1': bbox[1], 
                     'x2': bbox[2], 'y2': bbox[3]}
                ])
                
                polygons = sam_result.get('mask_polygons', [])
                if polygons:
                    # 取最大多边形
                    annotations.append({
                        'class': label,
                        'type': 'polygon',
                        'points': polygons[0],
                        'confidence': confidence,
                        'source': 'vlm_sam',
                    })
                else:
                    # SAM 失败，降级为矩形框
                    annotations.append({
                        'class': label,
                        'type': 'rectangle',
                        'points': rect_to_points(bbox),
                        'confidence': confidence,
                        'source': 'vlm_sam_fallback',
                    })
            
            # 2c. 保存
            save_annotations(image_name, annotations)
            
            socketio.emit('vlm_sam_progress', {
                'current': i+1, 'total': len(images),
                'image': image_name, 'status': 'done',
                'found': len(annotations)
            })
    finally:
        release_gpu("vlm_sam")
```

### SAM 降级策略

如果 SAM 2 对某个 bbox 推理失败（空 mask），自动降级为保存 VLM 的矩形框，不中断流程。

---

## 5. 前端实现

### AI 模态框新增模式

在现有模式切换中添加第三个选项：

```html
<label>
  <input type="radio" name="aiMode" value="vlm_sam">
  <span>VLM+SAM 精细化</span>
</label>
```

选中时：
- 显示 VLM 配置区域（复用，只读展示）
- 显示提示词输入框（复用）
- 不显示 YOLO 配置
- 点击"开始执行" → `startVlmSamLabel()`

### 进度显示

复用现有的进度条 UI，进度信息更丰富：

```
图片 3/20: img003.jpg
状态: SAM 精细化 (已检测 2 个目标)
[████████░░░░░░░░░░░░] 45%
```

---

## 6. 无新增依赖

| 组件 | 来源 | 状态 |
|------|------|------|
| VLM 推理 | `AiUtils.py` → `AIAutoLabeler` | 已有 |
| SAM 2 推理 | `ai_manager.py` → `SAM2Engine` | 已有 |
| GPU 锁 | `app.py` → `acquire_gpu/release_gpu` | 已有 |
| SocketIO 推送 | `app.py` → `socketio.emit` | 已有 |
| AI 配置读写 | `/api/load-api-config` | 已有 |
| 前端模式切换 | AI 模态框 radio group | 已有 |
| 进度显示 | 现有进度条组件 | 已有 |

---

## 7. 边界情况

| 场景 | 处理 |
|------|------|
| VLM 无检测结果 | 跳过 SAM，记录日志，继续下一张 |
| SAM 推理失败 | 降级保存 VLM 矩形框 |
| GPU 锁等待超时 | 提示用户稍后重试 |
| 中途中断 | 已处理的图片标注已保存，不丢失 |
