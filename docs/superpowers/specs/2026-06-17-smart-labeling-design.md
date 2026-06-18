# 智能标注功能设计文档（SAM 2 + YOLO）

> 日期：2026-06-17
> 状态：设计完成，待实施

---

## 1. 概述

为 xclabel 标注工具增加智能标注能力，在现有 Web 标注界面中集成交互式分割（SAM 2）和自动检测（YOLO），提供类似 [anylabeling](https://github.com/vietanhdev/anylabeling) 的 AI 辅助标注体验。

### 目标

1. **交互式分割**：用户在画布上点击/画框，SAM 2 实时生成分割 Mask，转为多边形标注
2. **自动检测预标注**：用 YOLO 模型对图片批量检测，生成初始矩形标注
3. **与现有流程无缝集成**：所有生成的标注存入 `annotations.json`，可手动修改

---

## 2. 架构设计

```
┌──────────────────────────────────────────────────────────┐
│                    前端（浏览器）                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ 现有标注工具    │  │ 智能标注(SAM) │  │ AI标注模态框     │  │
│  │ (rect/polygon │  │ 点/框 Prompt  │  │ └ YOLO自动检测  │  │
│  │  /obb/pose等) │  │ 实时Mask预览  │  │ └ 现有VLM标注   │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
│                          ↕ HTTP                            │
├──────────────────────────────────────────────────────────┤
│                    Flask 后端 (app.py)                      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              ai_manager.py (新增)                     │  │
│  │  ┌────────────────┐  ┌─────────────────────────┐    │  │
│  │  │  SAM2Engine     │  │  YOLOAutoLabeler        │    │  │
│  │  │  · sam2 Hiera   │  │  · 加载 plugins/yolo*   │    │  │
│  │  │  · 点/Box 推理   │  │  · 批量检测→矩形标注     │    │  │
│  │  │  · 多Mask输出   │  │  · subprocess 调用      │    │  │
│  │  └────────────────┘  └─────────────────────────┘    │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  MaskProcessor                                │   │  │
│  │  │  · findContours → approxPolyDP → 多边形     │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  现有 GPU 锁扩展：gpu_task_lock + gpu_task_owner          │
│  ("train" | "sam" | "yolo_label" | None)                 │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 后端实现

### 3.1 新增文件：`ai_manager.py`

三个主要类：

#### SAM2Engine

| 方法 | 说明 |
|------|------|
| `__init__(model_type="tiny")` | 初始化，加载 SAM 2 模型和处理器 |
| `predict(image_path, prompts)` | 主推理接口，接受图片路径和提示词列表 |
| `reset()` | 清除当前图片的推理缓存 |
| `release()` | 释放模型显存 |

SAM 2 加载逻辑（应用启动时预加载）：

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Engine:
    def __init__(self, model_type="tiny"):
        model_cfg = {
            "tiny": "sam2_hiera_t.yaml",
            "small": "sam2_hiera_s.yaml",
            "base_plus": "sam2_hiera_b+.yaml",
        }
        sam2 = build_sam2(model_cfg[model_type], f"models/sam2_hiera_{model_type}.pt")
        self.predictor = SAM2ImagePredictor(sam2)
        self._current_image = None
```

#### YOLOAutoLabeler

复用 `plugins/yolo*/` 虚拟环境，通过 `subprocess` 调用 ultralytics 推理：

| 方法 | 说明 |
|------|------|
| `load_model(version, model_path)` | 指定 YOLO 版本和模型权重路径 |
| `label_image(image_path)` | 单张图片检测，返回 bbox 列表 |
| `label_batch(image_paths)` | 批量检测，返回标注数据 |

#### MaskProcessor

```python
class MaskProcessor:
    @staticmethod
    def mask_to_polygons(mask, min_area=100, simplify_tolerance=2.0):
        """
        将 SAM 输出的二值 Mask 转为多边形点集。
        使用 OpenCV findContours + approxPolyDP 提取轮廓。
        """
```

### 3.2 新增 API 路由

#### `POST /api/sam/predict`

**请求：**
```json
{
  "image": "image001.jpg",
  "prompts": [
    {"type": "point", "x": 100, "y": 200, "label": 1},
    {"type": "point", "x": 300, "y": 400, "label": 0},
    {"type": "box", "x1": 50, "y1": 60, "x2": 200, "y2": 300}
  ]
}
```

**响应：**
```json
{
  "success": true,
  "mask_polygons": [
    [[x1,y1], [x2,y2], ..., [xn,yn]]
  ],
  "area": 15000,
  "logits": null
}
```

**处理流程：**
1. 获取 GPU 锁 (owner="sam")
2. 读取图片 → 设置到 predictor
3. 按顺序应用 prompts（支持累积提示）
4. 推理 → 获取 mask 和分数
5. MaskProcessor 转多边形
6. 释放 GPU 锁
7. 返回多边形坐标

#### `POST /api/sam/reset`

清除当前图片的 SAM 缓存状态，开始新标注。

#### `GET /api/sam/status`

```json
{
  "loaded": true,
  "model": "sam2_hiera_tiny",
  "gpu_available": true
}
```

#### `POST /api/auto-label/yolo`

YOLO 批量自动检测，在后台线程中运行，通过 SocketIO 推进度。

**请求：**
```json
{
  "images": ["img1.jpg", "img2.jpg", ...],
  "yolo_version": "yolo11",
  "model_path": "plugins/yolo11/runs/detect/train/weights/best.pt",
  "confidence": 0.25,
  "selected_classes": ["person", "car"]
}
```

**响应：**
```json
{
  "success": true,
  "total": 20,
  "labeled": 18,
  "annotations": {
    "img1.jpg": [...],
    "img2.jpg": [...]
  }
}
```

### 3.3 GPU 锁管理

在 `app.py` 中扩展现有锁机制：

```python
gpu_task_lock = threading.Lock()
gpu_task_owner = None  # None | "train" | "sam" | "yolo_label"
gpu_task_timestamp = 0  # 最后活动时间，用于超时检测

def acquire_gpu(owner, timeout=300):
    """获取 GPU 锁，timeout 秒后超时"""
    
def release_gpu(owner):
    """释放 GPU 锁，校验 owner"""
```

SAM 推理和 YOLO 训练互斥：
- SAM 推理时：YOLO 训练等待或提示"GPU 正忙"
- YOLO 训练时：SAM 推理等待
- 自动超时释放（5 分钟无活动）

---

## 4. 前端实现

### 4.1 新增智能标注工具

在工具栏新增 `smartTool` 按钮：

```html
<button id="smartTool" class="tool-btn" title="智能标注 (SAM)">
  <i class="fas fa-magic"></i> 智能
</button>
```

### 4.2 SAM 交互状态机

```
IDLE ──→ 选智能工具 ──→ PROMPTING
         ↑                  │
         │    点/框 Prompt  │ 调用 /api/sam/predict
         │        + 实时Mask预览
         │                  ↓
         │           MASK_PREVIEW
         │          ┌───────┴───────┐
         │          │               │
         │     按 Enter       按 ESC
         │     确认 Mask      取消
         │          │               │
         │          ↓               ↓
         │    ANNOTATION      PROMPTING
         │    生成多边形      清除提示词
         │    (回到 IDLE)     (回到 PROMPTING)
         └─────────────────────────┘
```

#### 关键实现细节

**canvas 交互层叠加：**
- 正样本点：绿色实心圆（`fillStyle: #00ff00`）
- 负样本点：红色实心圆（`fillStyle: #ff0000`）
- Box Prompt：蓝色虚线框
- SAM Mask：半透明彩色覆盖层（`globalAlpha: 0.4`）
- 多边形确认后：正常多边形样式（可选中编辑）

**通信逻辑：**

```javascript
// 每次用户操作后 300ms 防抖再调 API
let samDebounceTimer = null;

function onSamCanvasClick(e) {
    clearTimeout(samDebounceTimer);
    samDebounceTimer = setTimeout(() => {
        fetch('/api/sam/predict', {
            method: 'POST',
            body: JSON.stringify({
                image: currentImage,
                prompts: samPrompts
            })
        })
        .then(r => r.json())
        .then(data => {
            samMaskPolygons = data.mask_polygons;
            redrawCanvas(); // 触发重绘，显示 Mask
        });
    }, 300);
}
```

**确认时 Mask 转标注：**

```javascript
function confirmSamMask() {
    // 当前 Mask 多边形转为 annotation
    const annotation = {
        id: generateId(),
        class: selectedClassName,
        type: "polygon",
        points: samMaskPolygons[0],  // 取最大区域
        source: "sam2"
    };
    currentAnnotations.push(annotation);
    samMaskPolygons = null;
    samPrompts = [];
    saveAnnotations();
    redrawCanvas();
}
```

### 4.3 YOLO 自动标注入口

在现有 `aiLabelModal`（AI标注模态框）中添加选项：

```
┌─────────────────────────────────┐
│  AI 标注                        │
│                                 │
│  ○ VLM API 标注（现有）          │
│  ● YOLO 自动检测（新增）          │
│                                 │
│  YOLO 版本: [yolo11 ▼]          │
│  模型文件: [best.pt  ▼]          │
│  置信度: [0.25]                 │
│                                 │
│  [开始执行]                      │
└─────────────────────────────────┘
```

### 4.4 Canvas 重绘扩展

`redrawCanvas()` 增加新渲染层：

```javascript
function drawSamOverlay() {
    if (!samMaskPolygons || samMaskPolygons.length === 0) return;
    
    ctx.save();
    ctx.globalAlpha = 0.4;
    // 绘制每个 Mask 多边形（填充半透明色）
    samMaskPolygons.forEach(polygon => {
        ctx.beginPath();
        ctx.moveTo(polygon[0][0], polygon[0][1]);
        polygon.forEach(p => ctx.lineTo(p[0], p[1]));
        ctx.closePath();
        ctx.fillStyle = '#4a9eff';
        ctx.fill();
        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
    ctx.restore();
    
    // 绘制 Prompt 点
    samPrompts.forEach(p => {
        if (p.type === 'point') {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = p.label === 1 ? '#00ff00' : '#ff0000';
            ctx.fill();
        }
    });
}
```

---

## 5. 依赖项

### 新增依赖（主环境 .venv）

```
sam2                       # SAM 2 模型
opencv-python>=4.8.0       # Mask → 多边形（已有，确认版本）
```

### 模型权重文件

首次启动自动从 HuggingFace 下载到 `models/` 目录：

```
models/
└── sam2_hiera_tiny.pt      # 首选，~700MB
```

如网络受限，可手动下载后放入。

### YOLO 自动标注

复用 `plugins/yolo*/` 已有环境和模型，无需新增依赖。

---

## 6. 部署变更

### Docker 镜像调整

`Dockerfile.server.cpu/gpu` 中新增：

```dockerfile
# 下载 SAM 2 模型权重
RUN python -c "from sam2.build_sam import build_sam2; build_sam2('sam2_hiera_t.yaml', 'models/sam2_hiera_tiny.pt')"
```

### 启动脚本调整

- 添加启动参数控制 SAM 模型加载：`--sam-model tiny|small|base`
- 首次启动时自动下载模型

### 资源需求

| 模型 | 显存需求 | 推荐场景 |
|------|---------|---------|
| SAM 2 Hiera Tiny | ~1.5GB | CPU/GPU 均可，推荐 |
| SAM 2 Hiera Small | ~2.5GB | GPU 推荐 |
| SAM 2 Hiera Base+ | ~4GB | 高精度需求 |

建议生产环境至少 4GB 显存可用。

---

## 7. 升级路径（SAM 3）

当前设计实现了 `SAM2Engine` 类，未来升级到 SAM 3 只需：

1. 新建 `SAM3Engine` 类，实现相同接口
2. 在 `ai_manager.py` 中根据配置选择引擎
3. 前端无需任何改动

```python
class SAM3Engine:
    """与 SAM2Engine 保持相同的 predict/reset/release 接口"""
```

---

## 8. 边界情况处理

| 场景 | 处理方式 |
|------|---------|
| 无 GPU | 自动降级到 CPU 模式（速度较慢），或提示用户 |
| SAM 模型加载失败 | 返回错误信息，智能标注按钮禁用，不影响其他功能 |
| Mask 面积过大 | `approxPolyDP` 简化到最多 128 个顶点 |
| 无检测目标 | YOLO 返回空列表，不生成标注 |
| GPU 锁等待超时 | 提示用户"GPU 正被其他任务占用，请稍后重试" |
| 图片过大 | 后端自动缩放到 1024px 以内再推理，坐标等比还原 |
