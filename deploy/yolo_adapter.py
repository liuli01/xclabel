"""Ultralytics YOLO adapter — loads .pt/.engine models and runs predict()."""
import json
import io
import os
from typing import Any, Dict, Optional

from PIL import Image


class YoloAdapter:
    """Adapter for Ultralytics YOLO model loading and inference."""

    def __init__(self):
        self._yolo = None
        self._available = False
        self._init_yolo()

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
            self._yolo = YOLO
            self._available = True
        except ImportError as e:
            print(f"ultralytics import error: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load_model(self, model_path: str) -> Any:
        """Load a .pt or .engine model via Ultralytics YOLO."""
        if not self._available:
            raise RuntimeError("ultralytics is not installed")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        supported = ['.pt', '.engine', '.onnx']
        ext = os.path.splitext(model_path)[1].lower()
        if ext not in supported:
            raise ValueError(f"Unsupported model format: {ext} (supported: {supported})")

        print(f"Loading YOLO model: {model_path}")
        model = self._yolo(model_path)
        return model

    def infer(self, model: Any, image: Any,
              confidence_threshold: float = 0.25,
              task_type: str = "detect",
              metadata: Dict = None) -> Dict:
        """Run model.predict() and return structured JSON."""
        if not self._available:
            raise RuntimeError("ultralytics is not installed")

        if hasattr(image, 'read'):
            if hasattr(image, 'seek'):
                image.seek(0)
            img = Image.open(image).convert('RGB')
        else:
            img = image

        print(f'[yolo_debug] img size: {img.size}, mode: {img.mode}, task: {task_type}')
        results = model.predict(img, task=task_type, conf=confidence_threshold, verbose=False)
        print(f'[yolo_debug] results: {len(results) if results else 0} boxes')

        if not results:
            return {"detections": []}

        result = results[0]
        json_str = result.to_json()
        predictions = json.loads(json_str) if json_str else []

        detections = []
        class_names = []
        if metadata and isinstance(metadata.get("classes"), list):
            class_names = metadata["classes"]

        for pred in predictions:
            class_id = pred.get("class", pred.get("class_id", 0))
            bbox = pred.get("box", {})
            if isinstance(bbox, dict):
                bbox = [bbox.get("x1", 0), bbox.get("y1", 0),
                        bbox.get("x2", 0), bbox.get("y2", 0)]

            det = {
                "class_id": class_id,
                "class_name": pred.get("name", ""),
                "confidence": round(pred.get("confidence", 0), 4),
                "bbox": bbox,
            }

            seg = pred.get("segments", {})
            if isinstance(seg, dict) and "x" in seg and "y" in seg:
                xs, ys = seg["x"], seg["y"]
                if xs and ys and len(xs) == len(ys):
                    det["points"] = [
                        {"x": round(float(x), 2), "y": round(float(y), 2)}
                        for x, y in zip(xs, ys)
                    ]

            if "keypoints" in pred:
                det["keypoints"] = pred["keypoints"]

            if not det["class_name"] and class_names:
                if class_id < len(class_names):
                    det["class_name"] = class_names[class_id]

            detections.append(det)

        return {"detections": detections, "raw_tojson": json_str}

    def infer_annotated(self, model: Any, image: Any,
                        confidence_threshold: float = 0.25,
                        task_type: str = "detect",
                        metadata: Dict = None,
                        existing_detections: list = None) -> bytes:
        """Run inference and return annotated JPEG bytes.

        For segment tasks uses Ultralytics' built-in plot() which draws masks
        and polygons natively.  For detect/pose falls back to supervision.
        """
        import cv2
        pil_img = None
        if hasattr(image, 'seek'):
            image.seek(0)
        if hasattr(image, 'read'):
            pil_img = Image.open(image).convert('RGB')
        else:
            pil_img = image

        # ── Segmentation: use Ultralytics' plot() — handles masks correctly ──
        if task_type == 'segment':
            results = model.predict(pil_img, task=task_type, conf=confidence_threshold, verbose=False)
            if results:
                annotated_bgr = results[0].plot(boxes=True, masks=True, labels=True, conf=True)
                _, buf = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                return buf.tobytes()

        # ── Detect / Pose: use supervision ──
        import numpy as np
        import supervision as sv

        if existing_detections is not None:
            dets = existing_detections
        else:
            result = self.infer(model, image, confidence_threshold, task_type, metadata)
            dets = result.get("detections", [])

        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if not dets:
            _, buf = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return buf.tobytes()

        xyxy = np.array([d["bbox"] for d in dets if len(d.get("bbox", [])) == 4])
        conf = np.array([d["confidence"] for d in dets])
        class_id = np.array([d["class_id"] for d in dets])

        if len(xyxy) == 0:
            _, buf = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return buf.tobytes()

        sv_detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        annotated = img_cv.copy()
        annotated = sv.BoxAnnotator().annotate(annotated, sv_detections)
        annotated = sv.LabelAnnotator().annotate(annotated, sv_detections)

        if any(d.get("points") for d in dets):
            annotated = sv.MaskAnnotator().annotate(annotated, sv_detections)

        _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes()
