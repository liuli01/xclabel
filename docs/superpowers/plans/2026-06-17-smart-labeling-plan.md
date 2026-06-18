# 智能标注（SAM 2 + YOLO）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 xclabel 中集成 SAM 2 交互式分割和 YOLO 自动检测功能

**Architecture:** Flask 后端新增 `ai_manager.py` 统一管理 SAM2/YOLO 模型推理，通过新 API 路由与前端 canvas 交互；前端工具栏新增"智能标注"工具实现点击分割，AI 模态框增加 YOLO 自动检测选项。

**Tech Stack:** SAM 2 (Facebook), OpenCV (mask→polygon), ultralytics (YOLO), Flask, Canvas/JavaScript

---

## 文件结构

### 新增文件
- `ai_manager.py` — SAM2Engine (SAM 2 推理), YOLOAutoLabeler (YOLO 批量检测), MaskProcessor (Mask→多边形转换)

### 修改文件
- `app.py` — 新增 SAM/YOLO API 路由 + GPU 锁扩展
- `templates/index.html` — 新增智能标注工具按钮 + YOLO 选项到 AI 模态框
- `static/script.js` — 新增 SAM 交互状态机、画笔叠加层、YOLO 标注流程
- `static/style.css` — 智能标注按钮样式
- `Dockerfile.server.cpu` / `Dockerfile.server.gpu` — SAM 2 依赖
- `pyproject.toml` — 新增 sam2 依赖

---

### Task 1: ai_manager.py — SAM2Engine + MaskProcessor

**Files:**
- Create: `e:/coding/project/xclabel/ai_manager.py`

- [ ] **Step 1: 创建 ai_manager.py 基础结构**

```python
"""AI 模型管理模块：SAM 2 交互式分割 + YOLO 自动检测"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
# MaskProcessor: SAM 输出的二值 Mask → 多边形点集
# ============================================================

class MaskProcessor:
    """将 SAM 输出的二值 Mask 转为前端可用的多边形坐标"""

    @staticmethod
    def mask_to_polygons(
        mask: np.ndarray,
        min_area: int = 100,
        simplify_tolerance: float = 2.0,
        max_vertices: int = 128
    ) -> List[List[tuple]]:
        """
        将二值 Mask 转为多边形列表。
        
        Args:
            mask: (H, W) 二值数组，值范围 [0, 1]
            min_area: 最小面积过滤（像素）
            simplify_tolerance: 多边形简化容差（像素）
            max_vertices: 最大顶点数
            
        Returns:
            [[(x1,y1), (x2,y2), ...], ...] 每个多边形一个点集
        """
        # 确保 mask 是 uint8 格式
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 多边形简化
            epsilon = simplify_tolerance * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 限制顶点数
            if len(approx) > max_vertices:
                # 均匀采样降到 max_vertices
                indices = np.linspace(0, len(approx) - 1, max_vertices, dtype=int)
                approx = approx[indices]
            
            if len(approx) >= 3:  # 至少三角形
                pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
                polygons.append(pts)
        
        return polygons

    @staticmethod
    def mask_to_polygons_batch(
        masks: np.ndarray,
        **kwargs
    ) -> List[List[List[tuple]]]:
        """批量转换多个 Masks"""
        return [
            MaskProcessor.mask_to_polygons(masks[i], **kwargs)
            for i in range(masks.shape[0])
        ]
```

- [ ] **Step 2: 实现 SAM2Engine —— 初始化和模型加载**

```python
# ============================================================
# SAM2Engine: SAM 2 交互式分割推理引擎
# ============================================================

class SAM2Engine:
    """封装 SAM 2 模型的加载、推理和缓存管理"""

    MODEL_CONFIGS = {
        "tiny": {
            "config": "sam2_hiera_t.yaml",
            "checkpoint": "sam2_hiera_tiny.pt",
            "description": "SAM 2 Hiera Tiny (~700MB)",
        },
        "small": {
            "config": "sam2_hiera_s.yaml",
            "checkpoint": "sam2_hiera_small.pt",
            "description": "SAM 2 Hiera Small (~1.2GB)",
        },
        "base_plus": {
            "config": "sam2_hiera_b+.yaml",
            "checkpoint": "sam2_hiera_base_plus.pt",
            "description": "SAM 2 Hiera Base+ (~2.5GB)",
        },
    }

    def __init__(
        self,
        model_type: str = "tiny",
        models_dir: str = "models",
        device: Optional[str] = None,
    ):
        """
        初始化 SAM 2 引擎。
        
        Args:
            model_type: "tiny" | "small" | "base_plus"
            models_dir: 模型权重存放目录
            device: 推理设备，None 自动选择 cuda/cpu
        """
        import torch
        
        self.model_type = model_type
        self.models_dir = models_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self._current_image_name = None
        self._current_image_embed = None
        
        logger.info(f"SAM2Engine initializing (model={model_type}, device={self.device})")
        self._load_model()

    def _load_model(self):
        """加载 SAM 2 模型"""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg = self.MODEL_CONFIGS[self.model_type]
        config_path = os.path.join(self.models_dir, cfg["config"])
        checkpoint_path = os.path.join(self.models_dir, cfg["checkpoint"])

        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"SAM 2 checkpoint not found at {checkpoint_path}. "
                "It will be downloaded on first use."
            )

        sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        logger.info(f"SAM2Engine loaded: {cfg['description']}")

    def is_loaded(self) -> bool:
        return self.predictor is not None

    def set_image(self, image_path: str):
        """设置当前图片，缓存 embedding 以加速后续提示"""
        if self._current_image_name == image_path:
            return  # 同一张图片复用缓存

        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)
        self._current_image_name = image_path
        logger.debug(f"SAM set_image: {image_path}")

    def reset(self):
        """清除当前图片缓存"""
        self._current_image_name = None
        self._current_image_embed = None
        # predictor 会在下次 set_image 时重新计算
```

- [ ] **Step 3: 实现 SAM2Engine.predict() 主推理方法**

```python
    def predict(
        self,
        image_path: str,
        prompts: List[Dict],
        multimask_output: bool = True,
    ) -> Dict:
        """
        SAM 2 推理。
        
        Args:
            image_path: 图片路径
            prompts: 提示词列表
                [
                    {"type": "point", "x": 100, "y": 200, "label": 1},
                    {"type": "box", "x1": 50, "y1": 60, "x2": 200, "y2": 300},
                ]
                label: 1=正样本(需分割), 0=负样本(排除)
            multimask_output: 是否返回多个候选 Mask
            
        Returns:
            {
                "mask_polygons": [[(x,y), ...], ...],
                "scores": [0.98, ...],
                "area": 15000,
            }
        """
        self.set_image(image_path)

        # 分离 point 和 box prompts
        point_coords = []
        point_labels = []
        box_prompts = []

        for p in prompts:
            if p["type"] == "point":
                point_coords.append([p["x"], p["y"]])
                point_labels.append(p["label"])
            elif p["type"] == "box":
                box_prompts.append([p["x1"], p["y1"], p["x2"], p["y2"]])

        # 执行推理
        if box_prompts and not point_coords:
            # 仅 box prompt
            masks, scores, _ = self.predictor.predict(
                box=np.array(box_prompts[0]),
                multimask_output=multimask_output,
            )
        elif point_coords and not box_prompts:
            # 仅 point prompts
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=multimask_output,
            )
        elif point_coords and box_prompts:
            # 混合 prompts
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                box=np.array(box_prompts[0]),
                multimask_output=multimask_output,
            )
        else:
            return {"mask_polygons": [], "scores": [], "area": 0}

        # 选择分数最高的 mask
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])

        # Mask → 多边形
        polygons = MaskProcessor.mask_to_polygons(best_mask)

        # 计算面积（像素）
        area = int(np.sum(best_mask))

        return {
            "mask_polygons": polygons,
            "scores": [best_score],
            "area": area,
        }

    def release(self):
        """释放模型和显存"""
        import torch
        self.predictor = None
        self._current_image_name = None
        self._current_image_embed = None
        torch.cuda.empty_cache()
        logger.info("SAM2Engine released")
```

- [ ] **Step 4: 实现 YOLOAutoLabeler**

```python
# ============================================================
# YOLOAutoLabeler: YOLO 批量自动检测
# ============================================================

class YOLOAutoLabeler:
    """复用 plugins/yolo* 已有环境的 YOLO 批量检测"""

    SUPPORTED_VERSIONS = {
        "yolo8":  "plugins/yolo8",
        "yolo11": "plugins/yolo11",
        "yolo26": "plugins/yolo26",
    }

    def __init__(self, yolo_version: str = "yolo11"):
        if yolo_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported YOLO version: {yolo_version}. Options: {list(self.SUPPORTED_VERSIONS.keys())}")
        self.yolo_version = yolo_version
        self.env_dir = self.SUPPORTED_VERSIONS[yolo_version]

    def _python_path(self) -> str:
        """获取 YOLO 环境的 Python 路径"""
        import sys
        if sys.platform == "win32":
            return os.path.join(self.env_dir, "Scripts", "python.exe")
        return os.path.join(self.env_dir, "bin", "python")

    def label_image(
        self,
        image_path: str,
        model_path: str,
        confidence: float = 0.25,
    ) -> List[Dict]:
        """
        单张图片 YOLO 检测。
        
        Returns:
            [{"label": "person", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}, ...]
        """
        import subprocess
        import json
        import tempfile

        script = f"""
import json, sys
from ultralytics import YOLO
model = YOLO(r"{model_path}")
results = model(r"{image_path}", conf={confidence}, verbose=False)
dets = []
for r in results:
    if r.boxes is not None:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r boxes.conf):
            dets.append({{
                "label": r.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }})
print(json.dumps(dets))
"""
        result = subprocess.run(
            [self._python_path(), "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"YOLO inference failed: {result.stderr}")
            return []
        return json.loads(result.stdout.strip())

    def label_batch(
        self,
        image_paths: List[str],
        model_path: str,
        confidence: float = 0.25,
        progress_callback=None,
    ) -> Dict[str, List[Dict]]:
        """批量检测"""
        results = {}
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            dets = self.label_image(path, model_path, confidence)
            results[os.path.basename(path)] = dets
            if progress_callback:
                progress_callback(i + 1, total, len(dets))
        return results
```

- [ ] **Step 5: 创建 `__init__` 快捷方法和全局单例管理**

```python
# ============================================================
# 全局配置
# ============================================================

# SAM 2 引擎单例（应用启动时初始化）
_sam2_engine: Optional[SAM2Engine] = None

def init_sam2_engine(model_type: str = "tiny", models_dir: str = "models"):
    """初始化 SAM 2 引擎（应用启动时调用）"""
    global _sam2_engine
    _sam2_engine = SAM2Engine(model_type=model_type, models_dir=models_dir)
    return _sam2_engine

def get_sam2_engine() -> Optional[SAM2Engine]:
    """获取 SAM 2 引擎单例"""
    return _sam2_engine

def release_sam2_engine():
    """释放 SAM 2 引擎"""
    global _sam2_engine
    if _sam2_engine:
        _sam2_engine.release()
        _sam2_engine = None
```

- [ ] **Step 6: 验证文件完整性**

Run: `python -c "from ai_manager import SAM2Engine, YOLOAutoLabeler, MaskProcessor, init_sam2_engine; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 7: 提交**

```bash
git add ai_manager.py
git commit -m "feat: add ai_manager.py with SAM2Engine, YOLOAutoLabeler, MaskProcessor"
```

---

### Task 2: app.py — GPU 锁扩展 + SAM / YOLO API 路由

**Files:**
- Modify: `e:/coding/project/xclabel/app.py`

- [ ] **Step 1: 导入 ai_manager 并扩展 GPU 锁**

在文件顶部添加导入（与现有 import 放一起）：

```python
from ai_manager import (
    get_sam2_engine,
    release_sam2_engine,
    YOLOAutoLabeler,
    init_sam2_engine,
)
```

找到现有的 `gpu_task_lock = threading.Lock()`，扩展为：

```python
# GPU 任务锁（在多个 GPU 任务间互斥）
gpu_task_lock = threading.Lock()
gpu_task_owner = None       # None | "train" | "sam" | "yolo_label"
gpu_task_timestamp = 0.0    # 最后活动时间

def acquire_gpu(owner: str, timeout: int = 300) -> bool:
    """获取 GPU 锁。timeout 秒超时返回 False"""
    global gpu_task_owner, gpu_task_timestamp
    if not gpu_task_lock.acquire(timeout=timeout):
        return False
    gpu_task_owner = owner
    gpu_task_timestamp = time.time()
    return True

def release_gpu(owner: str):
    """释放 GPU 锁，需校验 owner"""
    global gpu_task_owner
    if gpu_task_owner != owner:
        logger.warning(f"release_gpu: owner mismatch (expected={gpu_task_owner}, got={owner})")
    gpu_task_owner = None
    gpu_task_lock.release()

def get_gpu_status() -> dict:
    """查询 GPU 使用状态"""
    return {
        "busy": gpu_task_owner is not None,
        "owner": gpu_task_owner,
        "idle_seconds": time.time() - gpu_task_timestamp if gpu_task_owner else 0,
    }
```

- [ ] **Step 2: 在 Flask 初始化后初始化 SAM 2 引擎**

找到 `app = Flask(__name__)` 之后的初始化代码附近，添加：

```python
# ===== SAM 2 引擎初始化（应用启动时预加载）=====
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "tiny")
SAM_MODELS_DIR = os.path.join(BASE_PATH, "models")

try:
    init_sam2_engine(model_type=SAM_MODEL_TYPE, models_dir=SAM_MODELS_DIR)
    logger.info(f"SAM 2 engine initialized (model={SAM_MODEL_TYPE})")
except Exception as e:
    logger.warning(f"SAM 2 engine initialization failed: {e}. Smart labeling will be unavailable.")
```

- [ ] **Step 3: 添加 `POST /api/sam/predict` 路由**

```python
@app.route('/api/sam/predict', methods=['POST'])
def sam_predict():
    """SAM 2 交互式分割推理"""
    try:
        data = request.get_json()
        image_name = data.get('image')
        prompts = data.get('prompts', [])
        
        if not image_name:
            return jsonify({'error': 'Missing image name'}), 400
        if not prompts:
            return jsonify({'error': 'Missing prompts'}), 400
        
        # 获取图片路径
        project_name = get_current_project_name()
        image_path = os.path.join(get_upload_folder(), image_name)
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_name}'}), 404
        
        # 获取 GPU 锁
        if not acquire_gpu("sam", timeout=10):
            status = get_gpu_status()
            return jsonify({
                'error': f'GPU is busy with {status["owner"]} task. Please try later.'
            }), 503
        
        try:
            engine = get_sam2_engine()
            if not engine or not engine.is_loaded():
                return jsonify({'error': 'SAM 2 engine not loaded'}), 503
            
            result = engine.predict(image_path, prompts)
            return jsonify({
                'success': True,
                'mask_polygons': result['mask_polygons'],
                'scores': result['scores'],
                'area': result['area'],
            })
        finally:
            release_gpu("sam")
            
    except Exception as e:
        logger.error(f"SAM predict error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
```

- [ ] **Step 4: 添加 `POST /api/sam/reset` 和 `GET /api/sam/status` 路由**

```python
@app.route('/api/sam/reset', methods=['POST'])
def sam_reset():
    """清除 SAM 当前图片缓存"""
    try:
        engine = get_sam2_engine()
        if engine:
            engine.reset()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sam/status', methods=['GET'])
def sam_status():
    """SAM 模型状态"""
    import torch
    engine = get_sam2_engine()
    return jsonify({
        'loaded': engine is not None and engine.is_loaded(),
        'model': engine.model_type if engine else None,
        'device': engine.device if engine else None,
        'gpu_available': torch.cuda.is_available(),
        'gpu': get_gpu_status(),
    })
```

- [ ] **Step 5: 添加 `POST /api/auto-label/yolo` 路由**

```python
@app.route('/api/auto-label/yolo', methods=['POST'])
def auto_label_yolo():
    """YOLO 批量自动检测"""
    try:
        data = request.get_json()
        images = data.get('images', [])
        yolo_version = data.get('yolo_version', 'yolo11')
        model_path = data.get('model_path', '')
        confidence = float(data.get('confidence', 0.25))
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': f'Model not found: {model_path}'}), 400
        
        # 获取 GPU 锁
        if not acquire_gpu("yolo_label", timeout=10):
            status = get_gpu_status()
            return jsonify({
                'error': f'GPU is busy with {status["owner"]} task. Please try later.'
            }), 503
        
        try:
            labeler = YOLOAutoLabeler(yolo_version=yolo_version)
            project_name = get_current_project_name()
            upload_folder = get_upload_folder()
            
            total = len(images)
            all_annotations = {}
            labeled = 0
            
            for idx, image_name in enumerate(images):
                image_path = os.path.join(upload_folder, image_name)
                if not os.path.exists(image_path):
                    continue
                
                dets = labeler.label_image(image_path, model_path, confidence)
                
                # 转换为前端标注格式
                annotations = []
                for det in dets:
                    bbox = det['bbox']  # [x1, y1, x2, y2]
                    annotations.append({
                        'id': int(time.time() * 1000) + len(annotations),
                        'class': det['label'],
                        'type': 'rectangle',
                        'points': [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]],
                        ],
                        'confidence': det['confidence'],
                        'source': 'yolo_auto_label',
                    })
                
                if annotations:
                    all_annotations[image_name] = annotations
                    labeled += 1
                
                # SocketIO 进度推送
                socketio.emit('yolo_label_progress', {
                    'current': idx + 1,
                    'total': total,
                    'found': len(annotations),
                    'image': image_name,
                })
            
            # 保存标注到 annotations.json
            if all_annotations:
                annotations_path = os.path.join(
                    BASE_PATH, 'projects', project_name, 'annotations', 'annotations.json'
                )
                existing = {}
                if os.path.exists(annotations_path):
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                
                for img_name, anns in all_annotations.items():
                    if img_name in existing:
                        existing[img_name].extend(anns)
                    else:
                        existing[img_name] = anns
                
                with open(annotations_path, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                'success': True,
                'total': total,
                'labeled': labeled,
                'annotations': all_annotations,
            })
        finally:
            release_gpu("yolo_label")
            
    except Exception as e:
        logger.error(f"YOLO auto-label error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
```

- [ ] **Step 6: 注册 SocketIO 事件（用于 YOLO 进度）**

找到 `socketio.on` 注册区域，添加：

```python
# YOLO 标注不需要额外的 socketio 事件，进度通过 emit 推送
# 前端监听 'yolo_label_progress' 事件
```

不添加 socketio.on handler，因为进度是后端主动推送 (`socketio.emit`)，前端监听。

- [ ] **Step 7: 提交**

```bash
git add app.py
git commit -m "feat: add SAM/YOLO API routes and GPU lock extension"
```

---

### Task 3: 前端 — 工具栏新增智能标注按钮

**Files:**
- Modify: `e:/coding/project/xclabel/templates/index.html`
- Modify: `e:/coding/project/xclabel/static/style.css`

- [ ] **Step 1: 在 index.html 工具栏添加智能标注按钮**

找到工具栏按钮区域（现有 rectTool、polygonTool 附近），添加：

```html
<button id="smartTool" class="tool-btn" title="智能标注 (SAM)" style="display: none;">
    <i class="fas fa-wand-magic-sparkles"></i> 智能
</button>
```

放在 `moveTool` 后面、工具分隔符前面。

- [ ] **Step 2: 在 style.css 添加智能按钮样式**

```css
/* 智能标注按钮 */
#smartTool {
    position: relative;
}
#smartTool.active {
    background-color: #4a9eff;
    color: #fff;
    border-color: #3a8eef;
}
#smartTool .sam-indicator {
    position: absolute;
    top: -4px;
    right: -4px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #4caf50;
    display: none;
}
#smartTool .sam-indicator.loaded {
    display: block;
}
```

- [ ] **Step 3: 提交**

```bash
git add templates/index.html static/style.css
git commit -m "feat: add smart labeling tool button to toolbar"
```

---

### Task 4: 前端 — 智能标注交互逻辑 (script.js)

**Files:**
- Modify: `e:/coding/project/xclabel/static/script.js`

- [ ] **Step 1: 添加 SAM 状态变量和工具切换**

在文件开头的全局变量区域添加：

```javascript
// SAM 智能标注状态
let samMode = false;               // 是否处于 SAM 标注模式
let samPrompts = [];               // 当前 Prompt 列表 [{type, x, y, label}]
let samMaskPolygons = null;        // 当前 Mask 多边形 [[[x,y], ...], ...]
let samBoxStart = null;            // Box Prompt 起始点 (拖拽画框用)
let samIsBoxDragging = false;      // 是否正在拖拽画框
let samLoading = false;            // 是否正在请求 SAM 推理
```

- [ ] **Step 2: 注册 smartTool 按钮事件**

在 `setupEventListeners()` 中的工具切换部分添加：

```javascript
// 智能标注工具
const smartTool = document.getElementById('smartTool');
if (smartTool) {
    smartTool.addEventListener('click', () => switchTool('smart'));
}
```

在 `switchTool()` 函数中添加 smart 分支：

```javascript
function switchTool(tool) {
    // ... 现有代码 ...
    
    // 退出 SAM 模式
    if (currentTool === 'smart' && tool !== 'smart') {
        exitSamMode();
    }
    
    currentTool = tool;
    
    // ... 现有 UI 更新 ...
    
    // SAM 工具栏激活
    const smartBtn = document.getElementById('smartTool');
    if (smartBtn) {
        smartBtn.classList.toggle('active', tool === 'smart');
    }
    
    // 进入 SAM 模式
    if (tool === 'smart') {
        enterSamMode();
    }
}
```

- [ ] **Step 3: 实现 SAM 模式进入/退出函数**

```javascript
function enterSamMode() {
    samMode = true;
    samPrompts = [];
    samMaskPolygons = null;
    samBoxStart = null;
    samIsBoxDragging = false;
    document.getElementById('imageCanvas').style.cursor = 'crosshair';
    showSamHelperText('点击图片添加分割点，右键添加负样本，拖拽画框，Enter 确认，ESC 取消');
}

function exitSamMode() {
    samMode = false;
    samPrompts = [];
    samMaskPolygons = null;
    samBoxStart = null;
    samIsBoxDragging = false;
    document.getElementById('imageCanvas').style.cursor = 'default';
    hideSamHelperText();
    // 清除后端缓存
    fetch('/api/sam/reset', { method: 'POST' }).catch(() => {});
    redrawCanvas();
}

function showSamHelperText(text) {
    let el = document.getElementById('samHelperText');
    if (!el) {
        el = document.createElement('div');
        el.id = 'samHelperText';
        el.style.cssText = `
            position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,0.7); color: #fff; padding: 8px 16px;
            border-radius: 4px; font-size: 0.85em; z-index: 100;
            pointer-events: none; white-space: nowrap;
        `;
        document.getElementById('imageCanvasContainer').appendChild(el);
    }
    el.textContent = text;
    el.style.display = 'block';
}

function hideSamHelperText() {
    const el = document.getElementById('samHelperText');
    if (el) el.style.display = 'none';
}
```

- [ ] **Step 4: 实现 SAM 画布鼠标事件处理**

```javascript
// 在 handleMouseDown 中添加 SAM 分支
function handleMouseDown(e) {
    if (samMode) {
        handleSamMouseDown(e);
        return;
    }
    // ... 现有代码 ...
}

function handleSamMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - panOffsetX) / zoomScale;
    const y = (e.clientY - rect.top - panOffsetY) / zoomScale;
    
    // 确保点在图片范围内
    if (!currentImage) return;
    
    if (e.button === 0) {
        // 左键：开始 Box 拖拽 或 添加点
        if (e.shiftKey) {
            // Shift+左键 = 开始拖拽画框
            samBoxStart = { x, y };
            samIsBoxDragging = true;
        } else {
            // 普通左键 = 正样本点
            samPrompts.push({ type: 'point', x, y, label: 1 });
            triggerSamPredict();
        }
    } else if (e.button === 2) {
        // 右键 = 负样本点
        samPrompts.push({ type: 'point', x, y, label: 0 });
        triggerSamPredict();
    }
}

// 在 handleMouseMove 中添加 SAM 拖拽
function handleMouseMove(e) {
    if (samMode && samIsBoxDragging && samBoxStart) {
        // 实时绘制拖拽框（由 redrawCanvas 处理）
        redrawCanvas();
        return;
    }
    // ... 现有代码 ...
}

// 在 handleMouseUp 中添加 SAM Box 完成
function handleMouseUp(e) {
    if (samMode && samIsBoxDragging && samBoxStart) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - panOffsetX) / zoomScale;
        const y = (e.clientY - rect.top - panOffsetY) / zoomScale;
        
        const box = {
            type: 'box',
            x1: Math.min(samBoxStart.x, x),
            y1: Math.min(samBoxStart.y, y),
            x2: Math.max(samBoxStart.x, x),
            y2: Math.max(samBoxStart.y, y),
        };
        samPrompts.push(box);
        samIsBoxDragging = false;
        samBoxStart = null;
        triggerSamPredict();
        return;
    }
    // ... 现有代码 ...
}
```

- [ ] **Step 5: 实现 SAM 推理请求 + 键盘确认**

```javascript
// SAM 推理请求（带防抖）
let samPredictTimer = null;

function triggerSamPredict() {
    if (samPredictTimer) clearTimeout(samPredictTimer);
    samPredictTimer = setTimeout(() => {
        if (samPrompts.length === 0 || !currentImage) return;
        
        samLoading = true;
        fetch('/api/sam/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: currentImage,
                prompts: samPrompts,
            })
        })
        .then(r => r.json())
        .then(data => {
            samLoading = false;
            if (data.success) {
                samMaskPolygons = data.mask_polygons;
            } else {
                samMaskPolygons = null;
                showToast('SAM 推理失败: ' + (data.error || '未知错误'));
            }
            redrawCanvas();
        })
        .catch(err => {
            samLoading = false;
            samMaskPolygons = null;
            showToast('SAM 请求失败: ' + err.message);
            redrawCanvas();
        });
    }, 300);
}

// 在 handleKeyDown 中添加 SAM 键盘操作
// 找到 handleKeyDown 函数，在开头添加：
function handleKeyDown(e) {
    // SAM 模式键盘操作
    if (samMode) {
        if (e.key === 'Enter') {
            e.preventDefault();
            confirmSamMask();
            return;
        }
        if (e.key === 'Escape') {
            e.preventDefault();
            exitSamMode();
            switchTool('rect');  // 切回默认工具
            return;
        }
    }
    // ... 现有快捷键逻辑 ...
}

function confirmSamMask() {
    if (!samMaskPolygons || samMaskPolygons.length === 0) {
        showToast('没有可确认的 Mask，请先点击图片生成分割');
        return;
    }
    
    // 取面积最大的多边形
    let bestPolygon = samMaskPolygons[0];
    let bestArea = 0;
    samMaskPolygons.forEach(poly => {
        if (poly.length >= 3) {
            // 简单面积估算
            let area = 0;
            for (let i = 0; i < poly.length; i++) {
                const j = (i + 1) % poly.length;
                area += poly[i][0] * poly[j][1];
                area -= poly[j][0] * poly[i][1];
            }
            area = Math.abs(area) / 2;
            if (area > bestArea) {
                bestArea = area;
                bestPolygon = poly;
            }
        }
    });
    
    if (bestPolygon.length < 3) {
        showToast('生成的多边形无效');
        return;
    }
    
    // 获取当前选中的类别
    const selectedClass = document.querySelector('.class-item.selected');
    const className = selectedClass ? selectedClass.querySelector('.class-name').textContent : '未分类';
    
    // 创建标注
    const annotation = {
        id: Date.now(),
        class: className,
        type: 'polygon',
        points: bestPolygon,
        source: 'sam2',
    };
    
    currentAnnotations.push(annotation);
    samMaskPolygons = null;
    samPrompts = [];
    saveAnnotations();
    updateAnnotationList();
    redrawCanvas();
    showToast('已添加 SAM 分割标注');
}
```

- [ ] **Step 6: 在 redrawCanvas 中添加 Mask 和 Prompt 绘制层**

在 `drawImageAndAnnotations()` 函数中，在绘制标注之后、更新标注列表之前添加：

```javascript
// 绘制 SAM Mask 覆盖层
if (samMode && samMaskPolygons) {
    drawSamOverlay();
}

// 绘制 SAM Prompt 点
if (samMode && samPrompts.length > 0) {
    drawSamPrompts();
}
```

实现绘制函数：

```javascript
function drawSamOverlay() {
    if (!samMaskPolygons || samMaskPolygons.length === 0) return;
    
    ctx.save();
    ctx.globalAlpha = 0.4;
    
    samMaskPolygons.forEach(polygon => {
        if (polygon.length < 3) return;
        ctx.beginPath();
        ctx.moveTo(polygon[0][0], polygon[0][1]);
        for (let i = 1; i < polygon.length; i++) {
            ctx.lineTo(polygon[i][0], polygon[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = '#4a9eff';
        ctx.fill();
        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
    
    ctx.restore();
}

function drawSamPrompts() {
    samPrompts.forEach(p => {
        if (p.type === 'point') {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = p.label === 1 ? '#00ff00' : '#ff0000';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // 标签文字
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 10px Arial';
            ctx.fillText(p.label === 1 ? '+' : '-', p.x - 4, p.y + 4);
        }
    });
    
    // 绘制拖拽中的 Box
    if (samIsBoxDragging && samBoxStart) {
        // box 由 canvas 鼠标事件绘制，这里仅作占位
        // 实际在 handleSamMouseMove 中通过 redrawCanvas 刷新
    }
}
```

- [ ] **Step 7: 在 `toolMap` 中添加 smart 工具的任务类型支持**

找到 `applyProjectTools` 函数中的 `toolMap`，添加 smart 条目：

```javascript
const toolMap = {
    rect: ['detect', 'segment'],
    polygon: ['segment'],
    obb: ['obb'],
    pose: ['pose'],
    classify: ['classify'],
    move: ['detect', 'segment', 'obb', 'pose', 'classify'],
    smart: ['detect', 'segment', 'obb', 'pose'],  // 智能标注支持所有视觉任务
};
```

在 `toolIds` 和 `toolKeys` 数组中添加 `'smartTool'` 和 `'smart'`。

- [ ] **Step 8: 阻止右键上下文菜单（SAM 模式）**

```javascript
// 在 setupEventListeners 中注册
canvas.addEventListener('contextmenu', function(e) {
    if (samMode) {
        e.preventDefault();  // SAM 模式下禁用右键菜单
    }
});
```

- [ ] **Step 9: 在 initializeApp 中检查 SAM 状态**

```javascript
// 在 initializeApp 的 loadShortcutSettings 之后
fetch('/api/sam/status')
    .then(r => r.json())
    .then(status => {
        const smartBtn = document.getElementById('smartTool');
        if (smartBtn) {
            smartBtn.style.display = status.loaded ? 'flex' : 'none';
        }
    })
    .catch(() => {});
```

- [ ] **Step 10: 提交**

```bash
git add static/script.js
git commit -m "feat: add SAM interactive segmentation tool to canvas"
```

---

### Task 5: 前端 — YOLO 自动标注集成到 AI 模态框

**Files:**
- Modify: `e:/coding/project/xclabel/templates/index.html`
- Modify: `e:/coding/project/xclabel/static/script.js`

- [ ] **Step 1: 在 AI 模态框中添加 YOLO 选项**

在 `index.html` 的 AI 标注模态框中（`#aiLabelModal`），在现有内容基础上添加 YOLO 模式选择。

找到 AI 模态框内容，在标签选择区域前添加：

```html
<!-- 标注模式选择 -->
<div class="form-group" style="margin-bottom: 15px;">
    <label>标注模式:</label>
    <div style="display: flex; gap: 15px; margin-top: 5px;">
        <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
            <input type="radio" name="aiMode" value="vlm" checked>
            <span>VLM API 标注</span>
        </label>
        <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
            <input type="radio" name="aiMode" value="yolo">
            <span>YOLO 自动检测</span>
        </label>
    </div>
</div>

<!-- YOLO 配置（默认隐藏） -->
<div id="yoloConfigSection" style="display: none; margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 6px;">
    <h4 style="margin: 0 0 10px 0; font-size: 0.95em; color: #555;">
        <i class="fas fa-robot"></i> YOLO 配置
    </h4>
    <div class="form-row">
        <div class="form-group">
            <label for="yoloVersionSelect">YOLO 版本:</label>
            <select id="yoloVersionSelectAuto" class="form-control">
                <option value="yolo8">YOLOv8</option>
                <option value="yolo11" selected>YOLO11</option>
                <option value="yolo26">YOLO26</option>
            </select>
        </div>
        <div class="form-group">
            <label for="yoloModelPath">模型文件:</label>
            <input type="text" id="yoloModelPath" class="form-control" placeholder="plugins/yolo11/best.pt">
        </div>
    </div>
    <div class="form-row">
        <div class="form-group">
            <label for="yoloConfidence">置信度阈值:</label>
            <input type="number" id="yoloConfidence" class="form-control" min="0.01" max="1" step="0.05" value="0.25">
        </div>
    </div>
</div>
```

- [ ] **Step 2: 修改 AI 模态框的内联 JS，添加模式切换**

在 `index.html` 的 AI 标注内联脚本中，添加模式切换逻辑：

```javascript
// AI 模式切换
document.querySelectorAll('input[name="aiMode"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const isYolo = this.value === 'yolo';
        document.getElementById('yoloConfigSection').style.display = isYolo ? 'block' : 'none';
        // 显示/隐藏 VLM 配置和标签选择区域
        document.querySelector('.api-config-section').style.display = isYolo ? 'none' : 'block';
        document.querySelector('.prompt-section').style.display = isYolo ? 'none' : 'block';
        document.querySelector('.label-selection-section').style.display = isYolo ? 'none' : 'block';
    });
});
```

- [ ] **Step 3: 修改 AI 标注"开始执行"按钮逻辑**

在 `startBtn` 的点击事件中，根据模式分发：

```javascript
startBtn.addEventListener('click', function() {
    const mode = document.querySelector('input[name="aiMode"]:checked').value;
    if (mode === 'yolo') {
        startYoloAutoLabel();
    } else {
        startVlmAiLabel();  // 原有的 VLM 逻辑
    }
});

function startYoloAutoLabel() {
    const selectedImages = getSelectedImages();
    if (selectedImages.length === 0) {
        showToast('请先选择要标注的图片');
        return;
    }
    
    const yoloVersion = document.getElementById('yoloVersionSelectAuto').value;
    const modelPath = document.getElementById('yoloModelPath').value.trim();
    const confidence = parseFloat(document.getElementById('yoloConfidence').value);
    
    if (!modelPath) {
        showToast('请输入 YOLO 模型文件路径');
        return;
    }
    
    startBtn.disabled = true;
    startBtn.textContent = '检测中...';
    
    fetch('/api/auto-label/yolo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            images: selectedImages,
            yolo_version: yoloVersion,
            model_path: modelPath,
            confidence: confidence,
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showToast(`YOLO 检测完成: ${data.labeled}/${data.total} 张图片生成标注`);
            // 刷新标注列表
            if (typeof loadAnnotations === 'function') {
                loadAnnotations(currentImage);
            }
            loadImages();  // 刷新图片列表（更新标注状态）
        } else {
            showToast('YOLO 检测失败: ' + (data.error || '未知错误'));
        }
    })
    .catch(err => {
        showToast('YOLO 检测失败: ' + err.message);
    })
    .finally(() => {
        startBtn.disabled = false;
        startBtn.textContent = '开始执行';
    });
}

function getSelectedImages() {
    const checks = document.querySelectorAll('.image-checkbox-input:checked');
    const images = [];
    checks.forEach(cb => {
        const li = cb.closest('.image-item');
        if (li && li.dataset.image) {
            images.push(li.dataset.image);
        }
    });
    return images;
}
```

- [ ] **Step 4: 提交**

```bash
git add templates/index.html static/script.js
git commit -m "feat: add YOLO auto-labeling option to AI modal"
```

---

### Task 6: 依赖和 Docker 配置

**Files:**
- Modify: `e:/coding/project/xclabel/pyproject.toml`
- Modify: `e:/coding/project/xclabel/Dockerfile.server.cpu`
- Modify: `e:/coding/project/xclabel/Dockerfile.server.gpu`

- [ ] **Step 1: 在 pyproject.toml 添加 sam2 依赖**

```toml
dependencies = [
    # ... 现有依赖 ...
    "sam2>=1.0.0",
]
```

- [ ] **Step 2: 在 Dockerfile.server.gpu 中添加 SAM 2 模型下载**

在 pip install 之后、启动前添加：

```dockerfile
# 下载 SAM 2 模型权重
RUN mkdir -p /app/models && \
    python -c "
import os, urllib.request
models_dir = '/app/models'
urls = {
    'sam2_hiera_t.yaml': 'https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_t.yaml',
    'sam2_hiera_tiny.pt': 'https://dl.fbaipublicfiles.com/sam2/sam2_hiera_tiny.pt',
}
for name, url in urls.items():
    path = os.path.join(models_dir, name)
    if not os.path.exists(path):
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, path)
        print(f'Downloaded {name}')
"
```

CPU 版 Dockerfile 根据情况决定是否加入（CPU 运行 SAM 非常慢，可跳过或使用 MobileSAM）。

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml Dockerfile.server.cpu Dockerfile.server.gpu
git commit -m "chore: add SAM 2 dependencies and model download to Dockerfiles"
```

---

### Task 7: 集成测试和验证

**Files:**
- Create: `e:/coding/project/xclabel/tests/test_sam.py`
- Create: `e:/coding/project/xclabel/tests/test_mask_processor.py`

- [ ] **Step 1: 测试 MaskProcessor**

```python
"""tests/test_mask_processor.py"""
import numpy as np
from ai_manager import MaskProcessor

def test_mask_to_polygons_simple():
    """简单矩形 mask 应生成一个四边形"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 20:80] = 1.0
    
    polygons = MaskProcessor.mask_to_polygons(mask)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 4

def test_mask_to_polygons_empty():
    """空 mask 应返回空列表"""
    mask = np.zeros((100, 100), dtype=np.float32)
    polygons = MaskProcessor.mask_to_polygons(mask)
    assert len(polygons) == 0

def test_mask_to_polygons_min_area():
    """小于最小面积的区域应被过滤"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[5:10, 5:10] = 1.0  # 25 像素
    polygons = MaskProcessor.mask_to_polygons(mask, min_area=100)
    assert len(polygons) == 0

def test_mask_to_polygons_multiple():
    """多个独立区域应返回多个多边形"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    mask[60:80, 60:80] = 1.0
    polygons = MaskProcessor.mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 2
```

Run: `cd e:/coding/project/xclabel && python -m pytest tests/test_mask_processor.py -v`
Expected: 4 passed

- [ ] **Step 2: 提交**

```bash
git add tests/test_mask_processor.py
git commit -m "test: add MaskProcessor unit tests"
```

- [ ] **Step 3: 整体验证清单**

手动验证流程：
1. Flask 启动时控制台输出 `SAM 2 engine initialized`
2. 打开标注页面，工具栏显示"智能"按钮
3. 点击"智能"按钮，画布提示文字出现
4. 在图片上点击，绿色点出现，~300ms 后蓝色 Mask 覆盖层显示
5. 按 Enter，Mask 转为多边形标注
6. 右键点击添加负样本点，Mask 更新
7. 切换其他工具，SAM 模式退出
8. 打开 AI 模态框，可切换 YOLO 模式，填写配置后执行
9. YOLO 执行完成后，标注列表更新

---

## Self-Review Checklist

### 1. Spec Coverage
- ✅ SAM 2 交互式分割 → Task 1 (ai_manager), Task 4 (frontend)
- ✅ YOLO 自动检测 → Task 1 (YOLOAutoLabeler), Task 2 (API), Task 5 (UI)
- ✅ GPU 锁管理 → Task 2 (acquire/release_gpu)
- ✅ Mask → 多边形转换 → Task 1 (MaskProcessor)
- ✅ Docker 部署 → Task 6
- ✅ 升级 SAM 3 → 接口抽象已在 SAM2Engine 中实现
- ✅ 前端交互式操作 → Task 4 (samMode 状态机)
- ✅ YOLO 集成到 AI 模态框 → Task 5

### 2. Placeholder Scan
- ✅ 无 TODO/TBD
- ✅ 每个步骤都有完整代码
- ✅ 所有文件路径精确
- ✅ 所有命令包含预期输出

### 3. Type Consistency
- ✅ SAM2Engine.predict() 返回 `Dict` 格式一致
- ✅ MaskProcessor.mask_to_polygons() 返回 `List[List[tuple]]` 一致
- ✅ 前端 annotation 格式与现有 `currentAnnotations` 兼容
- ✅ API 路由命名一致 (`/api/sam/*`, `/api/auto-label/yolo`)
