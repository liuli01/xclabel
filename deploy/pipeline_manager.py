"""
PipelineManager — 工作流编排引擎。

解析 workflow.yaml → 构建 DAG → 按拓扑序执行节点。
节点类型: yolo, condition, vllm, output
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


# ──────────────────────────────────────────────
# Pydantic models for workflow.yaml schema
# ──────────────────────────────────────────────

class NodeType(str, Enum):
    INPUT = "input"
    YOLO = "yolo"
    CONDITION = "condition"
    VLLM = "vllm"
    OUTPUT = "output"
    CALC = "calc"


class YoloParams(BaseModel):
    conf: float = 0.25
    iou: float = 0.5


class VllmParams(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: int = 30


class NodeConfig(BaseModel):
    id: str
    type: NodeType
    model: Optional[str] = None          # for yolo nodes
    source: Optional[List[str]] = None   # for output nodes
    condition: Optional[str] = None      # for vllm nodes (condition id)
    task: Optional[str] = "detect"
    api_url: Optional[str] = None        # for vllm nodes
    api_key: Optional[str] = None        # for vllm nodes
    model_name: Optional[str] = None     # for vllm nodes (the LLM model)
    prompt: Optional[str] = None         # for vllm nodes
    expression: Optional[str] = None     # for condition nodes
    extract_roi: bool = False
    input_type: Optional[str] = "upload"  # for input nodes: upload|url|stream
    url: Optional[str] = None            # for input nodes (url/stream source)
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Root model for workflow.yaml."""
    version: str = "1.0"
    name: str = "untitled"
    project_id: Optional[str] = None
    pipeline: List[NodeConfig]


# ──────────────────────────────────────────────
# PipelineContext — data passed between nodes
# ──────────────────────────────────────────────

@dataclass
class PipelineContext:
    image: Any = None                    # Original image (numpy array / PIL Image)
    detections: List[Dict] = field(default_factory=list)
    vllm_result: Optional[str] = None
    max_conf: float = 0.0
    detection_count: int = 0
    branch_conditions: Dict[str, bool] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def update(self, node_id: str, output: Any):
        self.node_outputs[node_id] = output
        if isinstance(output, dict):
            if "detections" in output:
                self.detections = output["detections"]
                self.detection_count = len(output["detections"])
            if "max_conf" in output:
                self.max_conf = output["max_conf"]
            if "vllm_result" in output:
                self.vllm_result = output["vllm_result"]
            if "condition_result" in output:
                self.branch_conditions[node_id] = output["condition_result"]

    def to_result(self) -> Dict:
        return {
            "detections": self.detections,
            "max_conf": self.max_conf,
            "detection_count": self.detection_count,
            "vllm_result": self.vllm_result,
            "branch_conditions": self.branch_conditions,
            "node_outputs": {k: v for k, v in self.node_outputs.items()
                             if k != "raw_image"},
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ──────────────────────────────────────────────
# PipelineManager — orchestrator
# ──────────────────────────────────────────────

class PipelineManager:
    """Workflow orchestration engine.

    Usage:
        mgr = PipelineManager(workflow_path)
        result = await mgr.execute(image, engine_pool)
    """

    def __init__(self, workflow_path: str):
        self.config = self._load_yaml(workflow_path)
        self.graph = self._build_dag(self.config.pipeline)
        self._topo_order = self._topological_sort()

    @staticmethod
    def from_config(config: PipelineConfig) -> "PipelineManager":
        """Create manager from an already-parsed config (no file needed)."""
        mgr = object.__new__(PipelineManager)
        mgr.config = config
        mgr.graph = mgr._build_dag(config.pipeline)
        mgr._topo_order = mgr._topological_sort()
        return mgr

    # ── YAML loading ──

    def _load_yaml(self, path: str) -> PipelineConfig:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Workflow file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            raise ValueError(f"Empty or invalid YAML: {path}")
        return PipelineConfig(**raw)

    # ── DAG construction ──

    def _build_dag(self, nodes: List[NodeConfig]) -> Dict[str, List[str]]:
        """Build adjacency list: node_id -> [dependent node ids]."""
        dag: Dict[str, List[str]] = {n.id: [] for n in nodes}
        node_map = {n.id: n for n in nodes}

        for node in nodes:
            if node.type in (NodeType.VLLM, NodeType.CALC) and node.condition:
                dag[node.id].append(node.condition)
            if node.source:
                for src in node.source:
                    if src in node_map:
                        dag[node.id].append(src)
        return dag

    def _topological_sort(self) -> List[str]:
        """Topological sort with cycle detection (Kahn's algorithm)."""
        in_degree = {nid: 0 for nid in self.graph}
        # Build reverse map: dependents of each node
        dependents = {nid: [] for nid in self.graph}
        for nid, deps in self.graph.items():
            for dep in deps:
                in_degree[nid] = in_degree.get(nid, 0) + 1
                dependents.setdefault(dep, []).append(nid)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for dependent in dependents.get(nid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self.graph):
            raise ValueError("Circular dependency detected in pipeline graph")

        return order

    # ── Main execution ──

    async def execute(
        self,
        image: Any,
        engine_pool=None,
        yolo_adapter=None,
        vllm_client=None,
    ) -> Dict:
        """Execute the pipeline, return merged results."""
        ctx = PipelineContext(image=image)
        node_map = {n.id: n for n in self.config.pipeline}
        node_timings: Dict[str, float] = {}
        node_status: Dict[str, str] = {}

        for node_id in self._topo_order:
            node = node_map[node_id]
            if not node.enabled:
                node_status[node_id] = "skipped"
                continue

            t0 = time.time()
            try:
                output = await self._run_node(node, ctx, engine_pool, yolo_adapter, vllm_client)
                elapsed = round((time.time() - t0) * 1000, 2)
                node_timings[node_id] = elapsed
                node_status[node_id] = "ok"
                # Add timing to node output
                output["_elapsed_ms"] = elapsed
                ctx.update(node_id, output)
            except Exception as e:
                elapsed = round((time.time() - t0) * 1000, 2)
                node_timings[node_id] = elapsed
                node_status[node_id] = "error"
                err = f"Node {node_id} ({node.type.value}) failed: {e}"
                ctx.errors.append(err)

        result = ctx.to_result()
        result["node_timings"] = node_timings
        result["node_status"] = node_status
        return result

    async def _run_node(self, node: NodeConfig, ctx: PipelineContext,
                        engine_pool, yolo_adapter, vllm_client) -> Dict:
        if node.type == NodeType.INPUT:
            return await self._exec_input(node, ctx)
        elif node.type == NodeType.YOLO:
            return await self._exec_yolo(node, ctx, engine_pool, yolo_adapter)
        elif node.type == NodeType.CONDITION:
            return self._exec_condition(node, ctx)
        elif node.type == NodeType.VLLM:
            return await self._exec_vllm(node, ctx, vllm_client)
        elif node.type == NodeType.OUTPUT:
            return self._exec_output(node, ctx)
        elif node.type == NodeType.CALC:
            return await self._exec_calc(node, ctx)
        else:
            raise ValueError(f"Unknown node type: {node.type}")

    # ── Input node ──

    async def _exec_input(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
        """Handle input node: pass the image through, or fetch from URL if configured."""
        result = {"input_type": node.input_type or "upload", "image_loaded": True}

        if node.input_type == "url" and node.url and ctx.image is None:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(node.url, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            import io as io_mod
                            ctx.image = io_mod.BytesIO(data)
                            result["image_source"] = "url"
                            result["image_loaded"] = True
            except ImportError:
                result["image_source"] = "url_passthrough"
            except Exception as e:
                result["warning"] = f"Failed to fetch image from URL: {e}"
                result["image_loaded"] = False
        elif node.input_type == "upload":
            result["image_source"] = "upload"

        return result

    # ── YOLO node ──

    async def _exec_yolo(self, node: NodeConfig, ctx: PipelineContext,
                         engine_pool, yolo_adapter) -> Dict:
        if engine_pool is None or yolo_adapter is None:
            raise RuntimeError("YOLO node requires engine_pool and yolo_adapter")

        engine = await engine_pool.get(node.model)
        if not engine:
            raise RuntimeError(f"Model not loaded: {node.model}")

        conf = node.params.get("conf", 0.25)
        result = yolo_adapter.infer(
            engine.engine, ctx.image,
            confidence_threshold=conf,
            task_type=engine.metadata.get("task_type", "detect"),
            metadata=engine.metadata,
        )

        dets = result.get("detections", [])
        max_conf = max((d.get("confidence", 0) for d in dets), default=0)
        result["max_conf"] = max_conf

        # Generate annotated image if supervision is available
        try:
            annotated_bytes = yolo_adapter.infer_annotated(
                engine.engine, ctx.image,
                confidence_threshold=conf,
                task_type=engine.metadata.get("task_type", "detect"),
                metadata=engine.metadata,
                existing_detections=dets,
            )
            import base64
            result["annotated_image"] = base64.b64encode(annotated_bytes).decode()
        except Exception as anno_err:
            print(f'[anno_warn] annotated_image generation failed: {anno_err}')

        return result

    # ── Condition node ──

    def _exec_condition(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
        if not node.expression:
            return {"condition_result": True}

        allowed = {
            "max_conf": ctx.max_conf,
            "detection_count": ctx.detection_count,
        }
        try:
            result = bool(eval(node.expression, {"__builtins__": {}}, allowed))
        except Exception:
            result = False
        return {"condition_result": result}

    # ── Calc node ──

    @staticmethod
    def _eval_expression(expression: str, vars: dict) -> float:
        """安全求值数学表达式，仅允许白名单函数和预置变量。"""
        import ast

        SAFE = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'ceil': math.ceil, 'floor': math.floor,
            'pi': math.pi,
        }

        tree = ast.parse(expression.strip(), mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in vars and node.id not in SAFE:
                    raise NameError(f"不支持的变量/函数: {node.id}")
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise TypeError("不支持复杂函数调用")
                if node.func.id not in SAFE:
                    raise NameError(f"不支持的函数: {node.func.id}")

        namespace = {**SAFE, **vars}
        try:
            return float(eval(expression, {"__builtins__": {}}, namespace))
        except Exception as e:
            raise ValueError(f"表达式求值失败: {e}")

    @staticmethod
    def _normalize_points(points) -> list:
        """dict格式 [{'x':n,'y':n},...] → 扁平列表 [x1,y1,x2,y2,...]"""
        if points and isinstance(points[0], dict):
            flat = []
            for p in points:
                flat.extend([float(p.get('x', 0)), float(p.get('y', 0))])
            return flat
        return points

    @staticmethod
    def _calc_area(bbox, points) -> float:
        """计算面积。有 segmentation points 时用鞋带公式，否则用 bbox 面积。"""
        points = PipelineManager._normalize_points(points)
        if points and len(points) >= 6:
            xs = points[0::2]
            ys = points[1::2]
            n = len(xs)
            if n < 3:
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += xs[i] * ys[j]
                area -= xs[j] * ys[i]
            return abs(area) / 2.0
        else:
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            return w * h

    @staticmethod
    def _calc_perimeter(bbox, points) -> float:
        """计算周长。有 segmentation points 时用多边形周长，否则用 bbox 周长。"""
        points = PipelineManager._normalize_points(points)
        if points and len(points) >= 6:
            xs = points[0::2]
            ys = points[1::2]
            n = len(xs)
            if n < 3:
                return 2 * ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]))
            perimeter = 0.0
            for i in range(n):
                j = (i + 1) % n
                perimeter += math.sqrt((xs[j] - xs[i])**2 + (ys[j] - ys[i])**2)
            return perimeter
        else:
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            return 2 * (w + h)

    # ── VLLM node ──

    async def _exec_vllm(self, node: NodeConfig, ctx: PipelineContext,
                         vllm_client) -> Dict:
        if not vllm_client:
            raise RuntimeError("VLLM client not available")

        # Check condition gate if set
        if node.condition:
            gate = ctx.branch_conditions.get(node.condition, False)
            if not gate:
                return {"vllm_result": None, "skipped": True}

        image = ctx.image
        if node.extract_roi and ctx.detections:
            image = vllm_client.crop_roi(ctx.image, ctx.detections)

        result = await vllm_client.analyze(
            image=image,
            prompt=node.prompt or "",
            model=node.model_name or "qwen-vl-chat",
            api_url=node.api_url or "",
            api_key=node.api_key or "",
            temperature=node.params.get("temperature", 0.1),
            max_tokens=node.params.get("max_tokens", 256),
            timeout=node.params.get("timeout", 30),
        )
        return {"vllm_result": result}

    # ── Calc node execution ──

    async def _exec_calc(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
        """执行计算表达式，对每个检测结果求值并追加字段。"""
        # Condition gate check — 与 VLLM 节点行为一致
        if node.condition:
            gate = ctx.branch_conditions.get(node.condition, False)
            if not gate:
                return {"calc_result": None, "skipped": True}

        expression = node.expression or node.params.get('expression', '')
        output_field = node.params.get('output_field', 'computed')

        if not expression:
            return {"detections": ctx.detections or [],
                    "computed_values": []}

        detections = ctx.detections or []
        computed_detections = []
        computed_values = []

        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            points = det.get('points', [])
            # Normalize points format: [{"x":n,"y":n},...] → [x1,y1,x2,y2,...]
            if points and isinstance(points[0], dict):
                flat = []
                for p in points:
                    flat.extend([float(p.get('x', 0)), float(p.get('y', 0))])
                points = flat

            vars = {
                'conf': float(det.get('confidence', 0)),
                'class_id': int(det.get('class_id', 0)),
                'class_name': det.get('class_name', ''),
                'x1': float(bbox[0]), 'y1': float(bbox[1]),
                'x2': float(bbox[2]), 'y2': float(bbox[3]),
                'width': max(0.0, float(bbox[2]) - float(bbox[0])),
                'height': max(0.0, float(bbox[3]) - float(bbox[1])),
            }
            vars['area'] = self._calc_area(bbox, points)
            vars['perimeter'] = self._calc_perimeter(bbox, points)

            try:
                result = self._eval_expression(expression, vars)
                result = round(float(result), 6)
            except Exception as e:
                ctx.warnings.append(
                    f"Calc node {node.id} expression error for detection: {e}")
                result = None

            det[output_field] = result
            computed_detections.append(det)
            if result is not None:
                computed_values.append(result)

        return {
            "detections": computed_detections,
            "computed_values": computed_values,
            "expression": expression,
            "output_field": output_field,
        }

    # ── Output node ──

    def _exec_output(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
        # Merge selected node outputs
        if node.source:
            merged = {}
            for src in node.source:
                if src in ctx.node_outputs:
                    merged[src] = ctx.node_outputs[src]
            return {"merged": merged}
        return ctx.to_result()
