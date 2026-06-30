import base64
import io
import json
import os
import sys
import time
from typing import Optional

# Ensure deploy/ is on the path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine_pool import Engine, EnginePool
from yolo_adapter import YoloAdapter
from pipeline_manager import PipelineManager
from vllm_client import VllmClient
from server_client import ServerClient

app = FastAPI(title="xclabel-deploy", version="1.0.0")

# CORS — allow all origins for dev, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment
CACHE_DIR = os.environ.get("CACHE_DIR", "/app/cache")
MAX_ENGINES = int(os.environ.get("MAX_ENGINES", "10"))
DEFAULT_SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:9924")

# Global state
engine_pool = EnginePool(max_engines=MAX_ENGINES)
yolo_adapter = YoloAdapter()
vllm_client = VllmClient()
server_client = ServerClient(DEFAULT_SERVER_URL)

os.makedirs(f"{CACHE_DIR}/models", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/workflows", exist_ok=True)

# ── Static files & test page ──
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/test", response_class=HTMLResponse)
    async def test_page():
        index_path = os.path.join(static_dir, "test.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>test.html not found</h1>", status_code=404)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return RedirectResponse(url="/test")


# Request/Response models
class InferRequest(BaseModel):
    engine_id: str
    image: Optional[str] = None
    image_url: Optional[str] = None
    confidence_threshold: float = 0.25


class UnloadRequest(BaseModel):
    engine_id: str


class PipelineExecuteRequest(BaseModel):
    workflow_id: str
    image: Optional[str] = None
    image_url: Optional[str] = None
    params: dict = Field(default_factory=dict)


class PredictRequest(BaseModel):
    server_url: Optional[str] = None
    model: str = Field(..., description="Format: project_id/model_version")
    image: Optional[str] = Field(default=None, description="Base64 image data (without data:image/xxx;base64, prefix)")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class LoadWorkflowFileRequest(BaseModel):
    workflow_path: str


class WorkflowExecuteRequest(BaseModel):
    workflow: str = Field(default="custom", description="Workflow name")
    server_url: Optional[str] = Field(default=None, description="Main server URL to download workflow YAML from")
    yaml_content: Optional[str] = Field(default=None, description="Raw YAML content (alternative to server_url)")
    image: Optional[str] = None
    image_url: Optional[str] = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "yolo_available": yolo_adapter.available,
        "vllm_available": vllm_client.available,
        "engines_loaded": len(engine_pool),
    }



@app.post("/infer")
async def infer(req: InferRequest):
    engine = await engine_pool.get(req.engine_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine not found: {req.engine_id}")

    # Prepare image
    try:
        if req.image:
            image_bytes = base64.b64decode(req.image)
            image = io.BytesIO(image_bytes)
        elif req.image_url:
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image = io.BytesIO(resp.content)
        else:
            raise HTTPException(status_code=400, detail="Either image or image_url must be provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Run inference (all models use yolo_adapter now)
    start_time = time.time()
    async with engine.lock:
        try:
            result = yolo_adapter.infer(
                engine.engine, image,
                confidence_threshold=req.confidence_threshold,
                task_type=engine.metadata.get("task_type", "detect"),
                metadata=engine.metadata,
            )

            inference_time_ms = round((time.time() - start_time) * 1000, 2)
            result["engine_id"] = req.engine_id
            result["inference_time_ms"] = inference_time_ms
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/engines")
async def list_engines():
    engines = await engine_pool.list_engines()
    return {"engines": engines, "total": len(engines)}


@app.post("/unload")
async def unload(req: UnloadRequest):
    success = await engine_pool.remove(req.engine_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Engine not found: {req.engine_id}")
    return {"engine_id": req.engine_id, "status": "unloaded"}

@app.post("/unload/all")
async def unload_all():
    count = await engine_pool.clear()
    return {"status": "all_unloaded", "count": count}


@app.post("/v1/predict")
async def predict(req: PredictRequest):
    """One-step prediction: download model, load engine, run inference."""
    # 1. Validate input
    if not req.image and not req.image_url:
        raise HTTPException(status_code=400, detail="Either image or image_url must be provided")

    # 2. Parse model reference
    try:
        project_id, model_version = ServerClient.parse_model_ref(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine_id = req.model  # reuse "project_id/model_version" as engine_id

    # 3. Check engine pool
    existing = await engine_pool.get(engine_id)
    if existing is None or existing.engine is None:
        # Need to download and load
        if not yolo_adapter.available:
            raise HTTPException(status_code=500, detail="YOLO adapter is not available")
        client = server_client
        if req.server_url:
            client = ServerClient(req.server_url)

        try:
            model_dir = client.download_model(req.model, cache_dir=CACHE_DIR)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Model not found on server: {req.model}")
            raise HTTPException(status_code=502, detail=f"Failed to download model from server: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to download model from server: {str(e)}")

        # Find model file
        pt_path = os.path.join(model_dir, "best.pt")
        engine_path = os.path.join(model_dir, "best.engine")
        onnx_path = os.path.join(model_dir, "best.onnx")

        model_file = None
        for candidate in [engine_path, pt_path, onnx_path]:
            if os.path.exists(candidate):
                model_file = candidate
                break

        if not model_file:
            raise HTTPException(
                status_code=502,
                detail=f"No model file found in {model_dir}. Looked for: best.engine, best.pt, best.onnx"
            )

        # Load metadata
        metadata = {}
        info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        if "task" in metadata and "task_type" not in metadata:
            metadata["task_type"] = metadata["task"]

        # Load model
        try:
            model = yolo_adapter.load_model(model_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

        engine = Engine(
            engine_id=engine_id,
            engine_type="yolo_model",
            project_id=project_id,
            engine=model,
            metadata=metadata,
        )
        await engine_pool.add(engine)
        existing = engine

    # 4. Prepare image
    try:
        if req.image:
            image_bytes = base64.b64decode(req.image)
            image = io.BytesIO(image_bytes)
        elif req.image_url:
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image = io.BytesIO(resp.content)
        else:
            raise HTTPException(status_code=400, detail="Either image or image_url must be provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # 5. Run inference
    start_time = time.time()
    async with existing.lock:
        try:
            result = yolo_adapter.infer(
                existing.engine, image,
                confidence_threshold=req.confidence_threshold,
                task_type=existing.metadata.get("task_type", "detect"),
                metadata=existing.metadata,
            )
            inference_time_ms = round((time.time() - start_time) * 1000, 2)
            result["model"] = req.model
            result["inference_time_ms"] = inference_time_ms
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


class PipelineManagerStore:
    """In-memory store for loaded pipeline managers."""
    def __init__(self):
        self._managers: dict = {}

    async def get(self, workflow_id: str) -> PipelineManager | None:
        return self._managers.get(workflow_id)

    async def add(self, workflow_id: str, mgr: PipelineManager):
        self._managers[workflow_id] = mgr

    async def remove(self, workflow_id: str) -> bool:
        return self._managers.pop(workflow_id, None) is not None

    def __len__(self):
        return len(self._managers)

pipeline_store = PipelineManagerStore()


@app.post("/pipeline/execute")
async def pipeline_execute(req: PipelineExecuteRequest):
    """Execute a loaded pipeline workflow."""
    mgr = await pipeline_store.get(req.workflow_id)
    if not mgr:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {req.workflow_id}")

    # Prepare image
    try:
        if req.image:
            image_bytes = base64.b64decode(req.image)
            image = io.BytesIO(image_bytes)
        elif req.image_url:
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image = io.BytesIO(resp.content)
        else:
            raise HTTPException(status_code=400, detail="Either image or image_url must be provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    start_time = time.time()
    try:
        result = await mgr.execute(
            image=image,
            engine_pool=engine_pool,
            yolo_adapter=yolo_adapter,
            vllm_client=vllm_client,
        )
        result["workflow_id"] = req.workflow_id
        result["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.post("/pipeline/load")
async def pipeline_load(req: LoadWorkflowFileRequest):
    """Load a pipeline workflow.yaml file and register the pipeline."""
    if not os.path.exists(req.workflow_path):
        raise HTTPException(status_code=404, detail=f"Workflow file not found: {req.workflow_path}")

    try:
        mgr = PipelineManager(req.workflow_path)
        workflow_id = mgr.config.name
        await pipeline_store.add(workflow_id, mgr)
        return {
            "workflow_id": workflow_id,
            "status": "loaded",
            "nodes": [{"id": n.id, "type": n.type.value} for n in mgr.config.pipeline],
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load workflow: {str(e)}")


class UnloadPipelineRequest(BaseModel):
    workflow_id: str


@app.post("/pipeline/unload")
async def pipeline_unload(req: UnloadPipelineRequest):
    """Unload a pipeline workflow from the pipeline store."""
    success = await pipeline_store.remove(req.workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {req.workflow_id}")
    return {"workflow_id": req.workflow_id, "status": "unloaded"}


@app.get("/pipeline/workflows")
async def list_pipelines():
    """List loaded pipelines."""
    return {"pipelines": list(pipeline_store._managers.keys()), "total": len(pipeline_store)}


@app.post("/v1/workflow/execute")
async def v1_workflow_execute(req: WorkflowExecuteRequest):
    """One-step workflow execution: download YAML, load pipeline, auto-load models, run."""
    if not req.workflow:
        raise HTTPException(status_code=400, detail="workflow name is required")
    if not req.image and not req.image_url:
        raise HTTPException(status_code=400, detail="Either image or image_url must be provided")
    # 1. Get YAML content: from yaml_content param, or download from server
    if req.yaml_content:
        yaml_content = req.yaml_content
    elif req.server_url:
        try:
            yaml_resp = requests.get(
                f"{req.server_url.rstrip('/')}/api/wf/yaml",
                params={"name": req.workflow},
                timeout=30,
            )
            yaml_resp.raise_for_status()
            yaml_content = yaml_resp.text
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Failed to download workflow from server: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either server_url or yaml_content must be provided")
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=CACHE_DIR)
    try:
        tmp.write(yaml_content)
        tmp.close()

        # 2. Load pipeline
        mgr = PipelineManager(tmp.name)
        workflow_id = mgr.config.name

        # 3. Auto-load YOLO models
        for node in mgr.config.pipeline:
            if node.type.value == 'yolo' and node.model:
                existing = await engine_pool.get(node.model)
                if existing and existing.engine is not None:
                    continue
                parts = node.model.split('/', 1)
                if len(parts) != 2:
                    continue
                p_name, p_version = parts
                try:
                    from server_client import ServerClient
                    client = ServerClient(req.server_url)
                    model_dir = client.download_model(p_name, p_version, cache_dir=CACHE_DIR)
                    model_file = None
                    for ext in ('.engine', '.pt', '.onnx'):
                        for f in os.listdir(model_dir):
                            if f.endswith(ext):
                                model_file = os.path.join(model_dir, f)
                                break
                        if model_file:
                            break
                    if not model_file:
                        print(f'[warn] No model file found for {node.model}')
                        continue
                    loaded = yolo_adapter.load_model(model_file)
                    eng = Engine(
                        engine_id=node.model,
                        engine_type="model",
                        project_id=p_name,
                        engine=loaded,
                        metadata={"task_type": node.task or "detect", "model_file": model_file},
                    )
                    await engine_pool.add(eng)
                    print(f'[deploy] Loaded model {node.model}')
                except Exception as load_err:
                    print(f'[deploy] Failed to load model {node.model}: {load_err}')
                    continue

        # 4. Prepare image
        if req.image:
            image = io.BytesIO(base64.b64decode(req.image))
        else:
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image = io.BytesIO(resp.content)

        # 5. Execute pipeline
        start_time = time.time()
        result = await mgr.execute(
            image=image,
            engine_pool=engine_pool,
            yolo_adapter=yolo_adapter,
            vllm_client=vllm_client,
        )
        result["workflow_id"] = workflow_id
        result["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
