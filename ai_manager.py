"""AI Model Manager - 封装 SAM2、YOLO 等 AI 模型的标注功能。

提供 MaskProcessor (掩码转多边形)、SAM2Engine (SAM2 推理引擎)、
YOLOAutoLabeler (YOLO 批量检测) 等核心类。
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MaskProcessor:
    """将 SAM 2 输出的二值 Mask 转为多边形点集。"""

    @staticmethod
    def mask_to_polygons(
        mask: np.ndarray,
        min_area: int = 100,
        simplify_tolerance: float = 2.0,
        max_vertices: int = 128,
    ) -> List[List[Tuple[int, int]]]:
        """将单个二值掩码转换为多边形点集。

        使用 OpenCV 的 findContours 查找轮廓，再通过 approxPolyDP 简化多边形，
        最终限制最大顶点数不超过 max_vertices。

        Args:
            mask: 二值掩码数组，形状 (H, W)，值范围为 0-1 或 0-255。
            min_area: 最小轮廓面积过滤阈值（像素）。
            simplify_tolerance: approxPolyDP 简化容差（像素）。
            max_vertices: 多边形最大顶点数限制。

        Returns:
            多边形列表，每个多边形由 (x, y) 坐标元组列表表示。
            若未检测到有效多边形，返回空列表。
        """
        # 确保 mask 为 uint8 类型且值为 0 或 255
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # 查找外轮廓（使用 copy 避免 findContours 修改原数组）
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons: List[List[Tuple[int, int]]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 多边形简化（simplify_tolerance 为像素单位，直接作为 approxPolyDP 的 epsilon）
            simplified = cv2.approxPolyDP(contour, simplify_tolerance, closed=True)

            # 提取顶点坐标并限制顶点数
            points = simplified.squeeze(axis=1).tolist()
            if len(points) < 3:
                continue  # 多边形至少需要 3 个顶点
            if len(points) > max_vertices:
                indices = np.linspace(0, len(points) - 1, max_vertices, dtype=int)
                points = [points[i] for i in indices]

            polygon = [(int(x), int(y)) for x, y in points]
            polygons.append(polygon)

        return polygons

    @staticmethod
    def mask_to_polygons_batch(
        masks: np.ndarray,
        **kwargs: Any,
    ) -> List[List[List[Tuple[int, int]]]]:
        """批量将二值掩码转换为多边形点集。

        Args:
            masks: 批量二值掩码数组，形状 (B, H, W)。
            **kwargs: 传递给 mask_to_polygons 的额外参数，
                如 min_area、simplify_tolerance、max_vertices。

        Returns:
            嵌套列表，外层对应每个掩码，内层为每个掩码的多边形列表。
        """
        results: List[List[List[Tuple[int, int]]]] = []
        for i in range(masks.shape[0]):
            polygons = MaskProcessor.mask_to_polygons(masks[i], **kwargs)
            results.append(polygons)
        return results


class SAM2Engine:
    """封装 SAM 2 模型的加载、推理和缓存管理。"""

    MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
        "tiny": {"config": "sam2_hiera_t", "checkpoint": "sam2_hiera_tiny.pt"},
        "small": {"config": "sam2_hiera_s", "checkpoint": "sam2_hiera_small.pt"},
        "base_plus": {"config": "sam2_hiera_b+", "checkpoint": "sam2_hiera_base_plus.pt"},
    }

    def __init__(
        self,
        model_type: str = "tiny",
        models_dir: str = "models",
        device: Optional[str] = None,
    ):
        """初始化 SAM2Engine。

        Args:
            model_type: 模型类型，可选 "tiny"、"small"、"base_plus"。
            models_dir: 模型文件存放目录（config 和 checkpoint 所在目录）。
            device: 推理设备，如 "cuda:0"、"cpu"。若为 None 则自动选择。

        Raises:
            ValueError: 不支持的 model_type。
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(
                f"不支持的模型类型: {model_type}，可选: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_type = model_type
        self.models_dir = models_dir
        self.device = device or self._auto_device()

        self._model: Any = None
        self._predictor: Any = None
        self._current_image_path: Optional[str] = None

        # 启动时预加载模型（文件不存在时，等待首次调用 set_image 时加载）
        try:
            self._load_model()
            logger.info(
                "SAM2Engine 模型加载完成 (model_type=%s, device=%s)",
                model_type,
                self.device,
            )
        except (FileNotFoundError, ImportError, Exception) as e:
            logger.warning(
                "SAM2Engine 模型预加载失败，将在首次使用时加载: %s", e,
            )

    @staticmethod
    def _auto_device() -> str:
        """自动选择可用的推理设备。

        Returns:
            设备字符串，"cuda:0"（如有 GPU）或 "cpu"。
        """
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> None:
        """加载 SAM2 模型和图像预测器。

        使用 sam2.build_sam 构建模型，通过 SAM2ImagePredictor 进行图像推理。

        Raises:
            ImportError: 未安装 sam2 库。
            FileNotFoundError: 模型文件不存在。
        """
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "需要安装 SAM2 库，请执行: pip install sam2"
            ) from e

        model_cfg = self.MODEL_CONFIGS[self.model_type]
        config_name = model_cfg["config"]  # Hydra 配置名，不含路径和扩展名
        checkpoint_path = os.path.join(self.models_dir, model_cfg["checkpoint"])

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM2 权重文件不存在: {checkpoint_path}")

        import torch

        device = torch.device(self.device)

        self._model = build_sam2(
            config_file=config_name,
            ckpt_path=checkpoint_path,
            device=device,
        )
        self._predictor = SAM2ImagePredictor(self._model)

        logger.info("SAM2 模型加载完成: %s", self.model_type)

    def is_loaded(self) -> bool:
        """检查模型是否已加载。

        Returns:
            模型已加载返回 True，否则返回 False。
        """
        return self._model is not None and self._predictor is not None

    def set_image(self, image_path: str) -> None:
        """设置当前图片并缓存其 embedding。

        若图片路径与缓存相同，则跳过加载以提升性能。

        Args:
            image_path: 图片文件路径。

        Raises:
            FileNotFoundError: 图片文件不存在。
            ValueError: 图片无法读取。
        """
        if image_path == self._current_image_path:
            return

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        if not self.is_loaded():
            self._load_model()

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self._predictor.set_image(image)
        self._current_image_path = image_path

        logger.debug("图片已设置: %s", image_path)

    def reset(self) -> None:
        """清除当前图片和 embedding 缓存。"""
        self._current_image_path = None
        if self._predictor is not None:
            self._predictor.reset_image()
        logger.debug("SAM2Engine 缓存已清除")

    def predict(
        self,
        image_path: str,
        prompts: List[Dict[str, Any]],
        multimask_output: bool = True,
        mask_mode: str = "best",
    ) -> Dict[str, Any]:
        """执行 SAM2 推理，返回掩码多边形。

        支持三种提示模式：
        - 仅点提示 (point)
        - 仅框提示 (box)
        - 混合提示 (point + box)

        Args:
            image_path: 图片文件路径。
            prompts: 提示列表，每项格式：
                - 点: {"type": "point", "x": int, "y": int, "label": 1}
                  其中 label=1 为前景，label=0 为背景。
                - 框: {"type": "box", "x1": int, "y1": int, "x2": int, "y2": int}
            multimask_output: 是否返回多个掩码结果（含备选）。

        Returns:
            包含以下键的字典：
                - "mask_polygons": 多边形列表
                - "scores": 置信度列表
                - "area": 多边形总面积（像素）

        Raises:
            ValueError: 未提供任何有效提示。
        """
        # 1. 设置图片
        self.set_image(image_path)

        # 2. 分离点和框提示
        input_points: List[List[int]] = []
        input_labels: List[int] = []
        input_boxes: List[List[int]] = []

        for prompt in prompts:
            ptype = prompt.get("type", "")
            if ptype == "point":
                input_points.append([prompt["x"], prompt["y"]])
                input_labels.append(prompt.get("label", 1))
            elif ptype == "box":
                input_boxes.append([
                    prompt["x1"],
                    prompt["y1"],
                    prompt["x2"],
                    prompt["y2"],
                ])

        # 3. 转换为 numpy 数组
        point_coords = np.array(input_points, dtype=np.float32) if input_points else None
        point_labels = np.array(input_labels, dtype=np.int32) if input_labels else None
        box_coords = np.array(input_boxes, dtype=np.float32) if input_boxes else None

        # 4. 执行推理
        if point_coords is not None and box_coords is not None:
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=multimask_output,
            )
        elif point_coords is not None:
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
        elif box_coords is not None:
            masks, scores, _ = self._predictor.predict(
                box=box_coords,
                multimask_output=multimask_output,
            )
        else:
            raise ValueError("至少需要提供一个点提示或框提示")

        # 5. 处理可能的多 mask 维度
        # SAM2 在 multimask_output=True 时返回 (N, M, H, W)
        if masks.ndim == 4:
            n, m, h, w = masks.shape
            masks = masks.reshape(-1, h, w)
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            scores = scores.flatten()

        # 6. 根据 mask_mode 选择掩码
        if mask_mode == "best" and masks.shape[0] > 1:
            # 只取评分最高的 mask
            best_idx = int(np.argmax(scores))
            masks = masks[best_idx:best_idx + 1]
            if isinstance(scores, np.ndarray):
                scores = scores[best_idx:best_idx + 1]

        # 7. 掩码转多边形
        mask_polygons_nested = MaskProcessor.mask_to_polygons_batch(masks)
        # 展平：[[poly1, poly2], [poly3]] -> [poly1, poly2, poly3]
        mask_polygons = []
        for polygons in mask_polygons_nested:
            mask_polygons.extend(polygons)

        # 7. 计算总面积（所有多边形的面积之和）
        total_area = 0
        for polygon in mask_polygons:
            if len(polygon) >= 3:
                contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                total_area += int(cv2.contourArea(contour))

        # 安全转换 scores 为列表
        if scores is None:
            scores_list: List[float] = []
        elif isinstance(scores, np.ndarray):
            scores_list = scores.tolist()
        elif isinstance(scores, (list, tuple)):
            scores_list = list(scores)
        else:
            scores_list = [float(scores)]

        return {
            "mask_polygons": mask_polygons,
            "scores": scores_list,
            "area": total_area,
        }

    def release(self) -> None:
        """释放模型和显存。"""
        self.reset()
        self._predictor = None
        if self._model is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    self._model.cpu()
                del self._model
            except ImportError:
                del self._model
            self._model = None

        # 尝试清理 CUDA 缓存
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("SAM2Engine 已释放")


class YOLOAutoLabeler:
    """复用 plugins/yolo* 环境的 YOLO 批量检测器。

    通过 subprocess 调用指定 YOLO 版本的 Python 环境执行推理，
    避免与主项目的依赖发生版本冲突。
    """

    SUPPORTED_VERSIONS: Dict[str, str] = {
        "yolo8": "plugins/yolo8",
        "yolo11": "plugins/yolo11",
        "yolo26": "plugins/yolo26",
    }

    def __init__(self, yolo_version: str = "yolo11"):
        """初始化 YOLOAutoLabeler。

        Args:
            yolo_version: YOLO 版本标识，可选 "yolo8"、"yolo11"、"yolo26"。

        Raises:
            ValueError: 不支持的 YOLO 版本。
            FileNotFoundError: 对应版本的 Python 环境不存在。
        """
        if yolo_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"不支持的 YOLO 版本: {yolo_version}，"
                f"可选: {list(self.SUPPORTED_VERSIONS.keys())}"
            )

        self.yolo_version = yolo_version
        self._plugin_dir = self.SUPPORTED_VERSIONS[yolo_version]
        self._python = self._python_path()

    def _python_path(self) -> str:
        """获取当前 YOLO 版本的 Python 可执行文件路径。

        Returns:
            Python 可执行文件的绝对路径。

        Raises:
            FileNotFoundError: 未找到对应的 Python 环境。
        """
        project_root = os.path.dirname(os.path.abspath(__file__))
        plugin_dir = os.path.join(project_root, self._plugin_dir)

        if sys.platform == "win32":
            python_path = os.path.join(plugin_dir, "venv", "Scripts", "python.exe")
        else:
            python_path = os.path.join(plugin_dir, "venv", "bin", "python")

        if not os.path.exists(python_path):
            raise FileNotFoundError(
                f"YOLO {self.yolo_version} 的 Python 环境不存在: {python_path}"
            )

        return python_path

    def label_image(
        self,
        image_path: str,
        model_path: str,
        confidence: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """对单张图片执行 YOLO 检测。

        通过 subprocess 调用 YOLO 独立环境的 Python，运行 ultralytics.YOLO 推理。

        Args:
            image_path: 图片文件路径。
            model_path: YOLO 模型权重文件路径（.pt）。
            confidence: 置信度阈值，默认 0.25。

        Returns:
            检测结果列表，每项包含 label、confidence、bbox、class_id 字段。

        Raises:
            RuntimeError: 子进程执行失败或输出解析失败。
        """
        # 内联 Python 脚本：加载模型、推理、输出 JSON
        # 注意：r.boxes.conf 中的点号必不可少，原 bug 写法 r boxes.conf 缺少点号
        script = (
            "import json, sys\n"
            "from ultralytics import YOLO\n"
            "model = YOLO(sys.argv[1])\n"
            "results = model(sys.argv[2], conf=float(sys.argv[3]))\n"
            "output = []\n"
            "for r in results:\n"
            "    if r.boxes is not None:\n"
            "        for i in range(len(r.boxes)):\n"
            "            box = r.boxes.xyxy[i].tolist()\n"
            "            conf = float(r.boxes.conf[i])\n"
            "            cls_id = int(r.boxes.cls[i])\n"
            "            label = r.names[cls_id] if hasattr(r, 'names') and r.names else str(cls_id)\n"
            "            output.append({\n"
            '                "label": label,\n'
            '                "confidence": conf,\n'
            '                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],\n'
            '                "class_id": cls_id,\n'
            "            })\n"
            "print(json.dumps(output))\n"
        )

        result = subprocess.run(
            [self._python, "-c", script, model_path, image_path, str(confidence)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"YOLO 推理失败 (returncode={result.returncode}):\n"
                f"stderr: {result.stderr}"
            )

        try:
            detections = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"YOLO 输出 JSON 解析失败: {e}\nstdout: {result.stdout}"
            ) from e

        return detections

    def label_batch(
        self,
        image_paths: List[str],
        model_path: str,
        confidence: float = 0.25,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """批量执行 YOLO 检测。

        Args:
            image_paths: 图片文件路径列表。
            model_path: YOLO 模型权重文件路径。
            confidence: 置信度阈值。
            progress_callback: 进度回调函数，原型为 callback(current, total)。

        Returns:
            字典，键为图片路径，值为该图片的检测结果列表。
        """
        total = len(image_paths)
        results: Dict[str, List[Dict[str, Any]]] = {}

        for idx, image_path in enumerate(image_paths):
            try:
                detections = self.label_image(image_path, model_path, confidence)
                results[image_path] = detections
            except Exception as e:
                logger.error("YOLO 检测失败 [%s]: %s", image_path, e)
                results[image_path] = []

            if progress_callback is not None:
                progress_callback(idx + 1, total)

        return results


# ---------------------------------------------------------------------------
# 全局单例管理
# ---------------------------------------------------------------------------

_sam2_engine: Optional[SAM2Engine] = None
"""全局 SAM2Engine 单例。"""


def init_sam2_engine(
    model_type: str = "tiny",
    models_dir: str = "models",
) -> SAM2Engine:
    """初始化全局 SAM2Engine 单例。

    若已有实例则先释放再重新创建。

    Args:
        model_type: 模型类型，可选 "tiny"、"small"、"base_plus"。
        models_dir: 模型文件存放目录。

    Returns:
        已初始化的 SAM2Engine 实例。
    """
    global _sam2_engine
    if _sam2_engine is not None:
        logger.warning("SAM2Engine 已存在，重新创建前先释放旧实例")
        _sam2_engine.release()

    _sam2_engine = SAM2Engine(model_type=model_type, models_dir=models_dir)
    return _sam2_engine


def get_sam2_engine() -> Optional[SAM2Engine]:
    """获取全局 SAM2Engine 单例。

    Returns:
        SAM2Engine 实例，若未初始化则返回 None。
    """
    return _sam2_engine


def release_sam2_engine() -> None:
    """释放全局 SAM2Engine 单例。"""
    global _sam2_engine
    if _sam2_engine is not None:
        _sam2_engine.release()
        _sam2_engine = None
        logger.info("全局 SAM2Engine 已释放")
