import json
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image


class NndeployAdapter:
    """Adapter for nndeploy model loading and inference."""

    def __init__(self):
        self._nndeploy = None
        self._base = None
        self._available = False
        self._init_nndeploy()

    def _init_nndeploy(self):
        try:
            import nndeploy
            import nndeploy.base as base
            import nndeploy.inference as inference
            import nndeploy.dag as dag
            self._nndeploy = nndeploy
            self._base = base
            self._inference = inference
            self._dag = dag
            self._available = True
        except ImportError as e:
            print(f"nndeploy import error: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load_model(self, model_dir: str) -> tuple[Any, Dict]:
        if not self._available:
            raise RuntimeError("nndeploy is not installed")

        # Find ONNX model in directory
        onnx_path = None
        for f in os.listdir(model_dir):
            if f.endswith(".onnx"):
                onnx_path = os.path.join(model_dir, f)
                break

        if not onnx_path:
            raise ValueError(f"No ONNX model found in {model_dir}")

        # Load metadata if exists
        metadata = {}
        metadata_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Load model with nndeploy
        inference = self._inference.create_inference(
            self._base.InferenceType.OnnxRuntime
        )
        param = self._inference.create_inference_param(
            self._base.InferenceType.OnnxRuntime
        )
        param.set_model_value([onnx_path])
        param.set_is_path(True)

        inference.set_param(param)
        inference.init()

        metadata["onnx_path"] = onnx_path
        return inference, metadata

    def load_workflow(self, workflow_path: str) -> tuple[Any, Dict]:
        if not self._available:
            raise RuntimeError("nndeploy is not installed")

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_json = json.load(f)

        # Use dag.run_json for workflow execution
        # Note: dag.run_json executes the workflow directly
        # The pipeline object here is the workflow definition itself
        return workflow_json, workflow_json

    def infer_model(self, model: Any, image: Any,
                    confidence_threshold: float = 0.5,
                    metadata: Dict = None) -> Dict:
        if not self._available:
            raise RuntimeError("nndeploy is not installed")

        # Convert image bytes to PIL Image
        if hasattr(image, 'read'):
            img = Image.open(image).convert('RGB')
        else:
            img = image

        # Get tensor names
        input_name = model.get_input_name()
        output_name = model.get_output_name()

        if not input_name or not output_name:
            raise RuntimeError("Failed to get model tensor names")

        # Get input shape from model
        all_shapes = model.get_all_input_shape()
        input_shape = all_shapes.get(input_name, [])
        if len(input_shape) < 4:
            raise RuntimeError(f"Unexpected input shape: {input_shape}")
        input_h, input_w = input_shape[2], input_shape[3]

        # Resize and preprocess
        img_resized = img.resize((input_w, input_h))
        img_array = np.array(img_resized).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
        img_array = img_array / 255.0  # Normalize
        # Ensure C-contiguous memory layout for nndeploy
        img_array = np.ascontiguousarray(img_array)

        # Set input tensor (from_numpy returns a new tensor, copy to original)
        input_tensor = model.get_input_tensor(input_name)
        device_type = input_tensor.get_device_type()
        new_tensor = input_tensor.from_numpy(img_array, device_type)
        new_tensor.copy_to(input_tensor)

        # Run inference
        model.run()

        # Get output and parse detections
        output_tensor = model.get_output_tensor_after_run(output_name, device_type, False)
        output_data = output_tensor.to_numpy()

        # Build class name mapping from metadata
        class_names = []
        if metadata and isinstance(metadata.get("classes"), list):
            class_names = metadata["classes"]

        detections = []
        if output_data is None:
            return {"detections": detections}

        shape = output_data.shape
        if len(shape) == 3:
            # Format 1: [batch, num_dets, 6] — Ultralytics with NMS
            # Each row: [x1, y1, x2, y2, confidence, class_id]
            if shape[2] == 6:
                predictions = output_data[0]
                for pred in predictions:
                    confidence = float(pred[4])
                    if confidence >= confidence_threshold:
                        class_id = int(pred[5])
                        class_name = class_names[class_id] if class_id < len(class_names) else ""
                        detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "bbox": [round(float(pred[0]), 2), round(float(pred[1]), 2),
                                     round(float(pred[2]), 2), round(float(pred[3]), 2)],
                        })
            # Format 2: [batch, 4 + num_classes, num_anchors] — raw YOLO output
            elif shape[1] > shape[2]:
                predictions = output_data[0]  # Shape: [4 + nc, 8400]
                num_classes = predictions.shape[0] - 4
                for i in range(predictions.shape[1]):
                    x_center = float(predictions[0, i])
                    y_center = float(predictions[1, i])
                    w = float(predictions[2, i])
                    h = float(predictions[3, i])
                    class_scores = predictions[4:, i]
                    class_id = int(np.argmax(class_scores))
                    score = float(class_scores[class_id])
                    if score >= confidence_threshold:
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        detections.append({
                            "class_id": class_id,
                            "class_name": "",
                            "confidence": round(score, 4),
                            "bbox": [round(x1, 2), round(y1, 2),
                                     round(x2, 2), round(y2, 2)],
                        })
            # Format 3: [batch, num_anchors, 4 + num_classes] — transposed raw YOLO
            elif shape[2] > shape[1]:
                predictions = output_data[0]  # Shape: [8400, 4 + nc]
                for pred in predictions:
                    x_center, y_center, w, h = pred[0:4]
                    class_scores = pred[4:]
                    class_id = int(np.argmax(class_scores))
                    score = float(class_scores[class_id])
                    if score >= confidence_threshold:
                        x1 = float(x_center - w / 2)
                        y1 = float(y_center - h / 2)
                        x2 = float(x_center + w / 2)
                        y2 = float(y_center + h / 2)
                        detections.append({
                            "class_id": class_id,
                            "class_name": "",
                            "confidence": round(score, 4),
                            "bbox": [round(x1, 2), round(y1, 2),
                                     round(x2, 2), round(y2, 2)],
                        })

        return {"detections": detections}

    def infer_workflow(self, workflow_json: Dict, image: Any,
                       project_id: str = None,
                       server_client=None,
                       cache_dir: str = None) -> Dict:
        if not self._available:
            raise RuntimeError("nndeploy is not installed")

        # Parse workflow JSON to extract model path and inference parameters
        model_path = None
        score_threshold = 0.5
        metadata = {}

        for node in workflow_json.get("node_repository_", []):
            key = node.get("key_", "")
            param = node.get("param_", {})

            if "Infer" in key:
                model_value = param.get("model_value_", [])
                if model_value and model_value[0]:
                    model_path = model_value[0]
                # Extract device type
                device_type_str = param.get("device_type_", "kDeviceTypeCodeCpu:0")

            if "YoloPostProcess" in key:
                score_threshold = param.get("score_threshold_", 0.5)
                metadata["nms_threshold"] = param.get("nms_threshold_", 0.45)
                metadata["num_classes"] = param.get("num_classes_", 80)
                metadata["model_h"] = param.get("model_h_", 640)
                metadata["model_w"] = param.get("model_w_", 640)
                metadata["version"] = param.get("version_", 11)

        if not model_path:
            return {
                "detections": [],
                "error": "No model path found in workflow",
                "workflow_name": workflow_json.get("name_"),
            }

        # Handle relative paths from workflow JSON
        if not os.path.isabs(model_path):
            candidates = [
                model_path,
                os.path.join("/app", model_path),
                os.path.join(os.getcwd(), model_path),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        # Auto-download model from server if not found locally
        if not os.path.exists(model_path) and project_id and server_client and cache_dir:
            filename = os.path.basename(model_path)
            if filename.startswith(project_id + "_") and filename.endswith(".onnx"):
                version = filename[len(project_id) + 1:-5]
                try:
                    model_dir = server_client.download_model(
                        project_id, version, cache_dir
                    )
                    downloaded_onnx = os.path.join(model_dir, "best.onnx")
                    if os.path.exists(downloaded_onnx):
                        model_path = downloaded_onnx
                except Exception as e:
                    print(f"Failed to download model {project_id}/{version}: {e}")

        if not os.path.exists(model_path):
            return {
                "detections": [],
                "error": f"Model file not found: {model_path}",
                "workflow_name": workflow_json.get("name_"),
            }

        # Load model and run inference using existing infer_model logic
        inference = self._inference.create_inference(
            self._base.InferenceType.OnnxRuntime
        )
        param = self._inference.create_inference_param(
            self._base.InferenceType.OnnxRuntime
        )
        param.set_model_value([model_path])
        param.set_is_path(True)
        inference.set_param(param)
        inference.init()

        try:
            result = self.infer_model(
                inference, image,
                confidence_threshold=score_threshold,
                metadata=metadata
            )
            result["workflow_name"] = workflow_json.get("name_")
            return result
        finally:
            if hasattr(inference, 'deinit'):
                inference.deinit()
