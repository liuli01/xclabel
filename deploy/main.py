import asyncio
import base64
import io
import json
import os
import sys
import time
from collections import OrderedDict
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
from pipeline_manager import PipelineConfig, PipelineManager
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
MAX_WORKFLOWS = int(os.environ.get("MAX_WORKFLOWS", "50"))
DEFAULT_SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:9924")

# Global state
engine_pool = EnginePool(max_engines=MAX_ENGINES)
yolo_adapter = YoloAdapter()
vllm_client = VllmClient()
server_client = ServerClient(DEFAULT_SERVER_URL)

os.makedirs(f"{CACHE_DIR}/models", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/workflows", exist_ok=True)

# ── Workflow cache helpers ──

def _workflow_cache_path(name: str) -> str:
    """Get the disk cache path for a workflow YAML."""
    return f"{CACHE_DIR}/workflows/{name}.yaml"

def _load_workflow_from_cache(name: str) -> Optional[str]:
    """Load workflow YAML content from disk cache. Returns None if not found."""
    path = _workflow_cache_path(name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[cache] Failed to read cached workflow '{name}': {e}", flush=True)
    return None

def _save_workflow_to_cache(name: str, yaml_content: str):
    """Save workflow YAML content to disk cache."""
    path = _workflow_cache_path(name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"[cache] Cached workflow '{name}' to {path}", flush=True)
    except Exception as e:
        print(f"[cache] Failed to cache workflow '{name}': {e}", flush=True)

def _fetch_workflow_from_server(name: str, server_url: str) -> str:
    """Download workflow YAML from the server."""
    url = f"{server_url.rstrip('/')}/api/wf/yaml"
    resp = requests.get(url, params={"name": name}, timeout=30)
    resp.raise_for_status()
    return resp.text

def _parse_workflow_yaml(yaml_content: str) -> PipelineConfig:
    """Parse YAML string into PipelineConfig."""
    import yaml as _yaml
    raw = _yaml.safe_load(yaml_content)
    if not raw:
        raise ValueError("Empty or invalid workflow YAML")
    return PipelineConfig(**raw)

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
    workflow_path: Optional[str] = Field(default=None, description="Local file path to workflow YAML")
    name: Optional[str] = Field(default=None, description="Workflow name (used with server_url)")
    server_url: Optional[str] = Field(default=None, description="Server URL to download workflow from")


class WorkflowExecuteRequest(BaseModel):
    workflow: str = Field(default="custom", description="Workflow name")
    server_url: Optional[str] = Field(default=None, description="Main server URL to download workflow YAML from")
    yaml_content: Optional[str] = Field(default=None, description="Raw YAML content (alternative to server_url)")
    image: Optional[str] = None
    image_url: Optional[str] = None
    force_refresh: bool = Field(default=False, description="Force re-download from server, bypassing cache")


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
    """In-memory store for loaded pipeline managers with LRU eviction."""
    def __init__(self, max_workflows: int = 50):
        self.max_workflows = max_workflows
        self._managers: OrderedDict[str, PipelineManager] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, workflow_id: str) -> PipelineManager | None:
        async with self._lock:
            mgr = self._managers.get(workflow_id)
            if mgr is not None:
                self._managers.move_to_end(workflow_id)
            return mgr

    async def add(self, workflow_id: str, mgr: PipelineManager):
        async with self._lock:
            while len(self._managers) >= self.max_workflows:
                eid, _ = self._managers.popitem(last=False)
                print(f"[pipeline_store] LRU evict workflow '{eid}'", flush=True)
            self._managers[workflow_id] = mgr
            print(f"[pipeline_store] Loaded workflow '{workflow_id}', total={len(self._managers)}", flush=True)

    async def remove(self, workflow_id: str) -> bool:
        async with self._lock:
            return self._managers.pop(workflow_id, None) is not None

    async def list_workflows(self) -> list[str]:
        async with self._lock:
            return list(self._managers.keys())

    def __len__(self):
        return len(self._managers)

pipeline_store = PipelineManagerStore(max_workflows=MAX_WORKFLOWS)


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

    start_time = time.perf_counter()
    try:
        result = await mgr.execute(
            image=image,
            engine_pool=engine_pool,
            yolo_adapter=yolo_adapter,
            vllm_client=vllm_client,
        )
        result["workflow_id"] = req.workflow_id
        result["execution_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.post("/pipeline/load")
async def pipeline_load(req: LoadWorkflowFileRequest):
    """Load a pipeline workflow and register it.

    Two modes:
      1. Local file: pass workflow_path (existing behavior)
      2. From server: pass name (+ optional server_url) — deploy fetches & caches
    """
    if req.workflow_path:
        # Mode 1: load from local file path (backward compatible)
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

    elif req.name:
        # Mode 2: fetch by name from disk cache or server
        server = req.server_url or DEFAULT_SERVER_URL
        # Check disk cache first
        yaml_content = _load_workflow_from_cache(req.name)
        if yaml_content is None:
            try:
                yaml_content = _fetch_workflow_from_server(req.name, server)
                _save_workflow_to_cache(req.name, yaml_content)
            except requests.RequestException as e:
                raise HTTPException(status_code=502, detail=f"Failed to download workflow from server: {str(e)}")
        try:
            config = _parse_workflow_yaml(yaml_content)
            mgr = PipelineManager.from_config(config)
            workflow_id = config.name
            await pipeline_store.add(workflow_id, mgr)
            return {
                "workflow_id": workflow_id,
                "status": "loaded",
                "nodes": [{"id": n.id, "type": n.type.value} for n in config.pipeline],
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to load workflow: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Provide either workflow_path or name")


class UnloadPipelineRequest(BaseModel):
    workflow_id: str


@app.post("/pipeline/unload")
async def pipeline_unload(req: UnloadPipelineRequest):
    """Unload a pipeline workflow from the pipeline store."""
    success = await pipeline_store.remove(req.workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {req.workflow_id}")
    return {"workflow_id": req.workflow_id, "status": "unloaded"}


class RefreshPipelineRequest(BaseModel):
    name: str = Field(..., description="Workflow name to refresh")
    server_url: Optional[str] = Field(default=None, description="Server URL to re-download from")


class SaveWorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow name to save")
    yaml_content: str = Field(..., description="Workflow YAML content")


@app.post("/pipeline/refresh")
async def pipeline_refresh(req: RefreshPipelineRequest):
    """Force-reload a workflow from server, updating both disk and memory cache."""
    server = req.server_url or DEFAULT_SERVER_URL
    # Remove from memory cache
    await pipeline_store.remove(req.name)
    # Remove disk cache
    cache_path = _workflow_cache_path(req.name)
    if os.path.exists(cache_path):
        os.remove(cache_path)
    # Re-download from server
    try:
        yaml_content = _fetch_workflow_from_server(req.name, server)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download workflow from server: {str(e)}")
    # Persist to disk
    _save_workflow_to_cache(req.name, yaml_content)
    # Parse and load into memory
    try:
        config = _parse_workflow_yaml(yaml_content)
        mgr = PipelineManager.from_config(config)
        await pipeline_store.add(req.name, mgr)
        return {
            "workflow_id": req.name,
            "status": "refreshed",
            "nodes": [{"id": n.id, "type": n.type.value} for n in config.pipeline],
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse refreshed workflow: {str(e)}")


@app.get("/pipeline/workflows")
async def list_pipelines():
    """List loaded pipelines."""
    workflows = await pipeline_store.list_workflows()
    return {"pipelines": workflows, "total": len(workflows)}


def _scan_cached_models() -> list[dict]:
    """Scan CACHE_DIR/models/ for cached model directories."""
    models_dir = f"{CACHE_DIR}/models"
    if not os.path.isdir(models_dir):
        return []
    result = []
    for entry in sorted(os.listdir(models_dir)):
        entry_path = os.path.join(models_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # Parse project_id_version from directory name
        info = {"directory": entry_path}
        # List model files
        files = sorted(os.listdir(entry_path))
        info["files"] = files
        # Try to read metadata
        meta_path = os.path.join(entry_path, "model_info.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                info["task_type"] = metadata.get("task_type") or metadata.get("task", "?")
                info["classes"] = metadata.get("classes", [])
            except Exception:
                info["task_type"] = "?"
        else:
            info["task_type"] = "?"
        # Determine if it has a valid model file
        has_model = any(f.endswith(('.pt', '.engine', '.onnx')) for f in files)
        info["has_model"] = has_model
        # Extract project / version from dir name (format: project_version)
        if '_' in entry:
            idx = entry.rfind('_')
            info["project_id"] = entry[:idx]
            info["version"] = entry[idx+1:]
        else:
            info["project_id"] = entry
            info["version"] = "?"
        result.append(info)
    return result


def _list_workflow_files() -> list[dict]:
    """Scan CACHE_DIR/workflows/ for cached workflow YAML files."""
    wf_dir = f"{CACHE_DIR}/workflows"
    if not os.path.isdir(wf_dir):
        return []
    result = []
    for fname in sorted(os.listdir(wf_dir)):
        if fname.endswith(('.yaml', '.yml')):
            fpath = os.path.join(wf_dir, fname)
            try:
                fsize = os.path.getsize(fpath)
                entry = {
                    "name": os.path.splitext(fname)[0],
                    "file": fname,
                    "size": fsize,
                    "size_display": f"{fsize} B" if fsize < 1024 else f"{fsize/1024:.1f} KB",
                    "path": fpath,
                }
                # Read YAML content for preview (limit to 50KB to avoid huge responses)
                if fsize < 51200:
                    with open(fpath, "r", encoding="utf-8") as f:
                        entry["yaml_content"] = f.read()
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        entry["yaml_content"] = f.read(50000) + "\n\n... (truncated)"
                result.append(entry)
            except Exception:
                pass
    return result


@app.get("/cache")
async def cache_list():
    """List all cached models and workflows (disk + memory)."""
    # Workflows
    workflow_files = _list_workflow_files()
    loaded_workflows = await pipeline_store.list_workflows()

    # Models on disk
    cached_models = _scan_cached_models()

    # Models in memory (from engine_pool)
    memory_engines = await engine_pool.list_engines()
    memory_models = []
    for eng in memory_engines:
        memory_models.append({
            "engine_id": eng["engine_id"],
            "project_id": eng.get("project_id", "?"),
            "task_type": eng.get("metadata", {}).get("task_type", "?"),
            "inference_count": eng.get("inference_count", 0),
        })

    return {
        "workflows": {
            "disk": workflow_files,
            "memory": loaded_workflows,
            "disk_count": len(workflow_files),
            "memory_count": len(loaded_workflows),
        },
        "models": {
            "disk": cached_models,
            "memory": memory_models,
            "disk_count": len(cached_models),
            "memory_count": len(memory_models),
        },
    }


@app.post("/cache/models/clear")
async def cache_models_clear():
    """Delete all cached models from disk and unload from memory."""
    models_dir = f"{CACHE_DIR}/models"
    removed_dirs = []
    # Remove from memory first
    mem_count = await engine_pool.clear()
    # Remove from disk
    if os.path.isdir(models_dir):
        for entry in sorted(os.listdir(models_dir)):
            entry_path = os.path.join(models_dir, entry)
            if os.path.isdir(entry_path):
                import shutil
                shutil.rmtree(entry_path, ignore_errors=True)
                removed_dirs.append(entry)
    # Ensure directory exists for future use
    os.makedirs(models_dir, exist_ok=True)
    return {
        "status": "ok",
        "removed_from_memory": mem_count,
        "removed_from_disk": removed_dirs,
    }


@app.post("/cache/workflows/save")
async def cache_workflow_save(req: SaveWorkflowRequest):
    """Save edited workflow YAML to disk cache and reload into memory."""
    # Validate YAML before saving
    try:
        config = _parse_workflow_yaml(req.yaml_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")

    # Save to disk cache
    _save_workflow_to_cache(req.name, req.yaml_content)

    # Reload into memory
    mgr = PipelineManager.from_config(config)
    await pipeline_store.add(req.name, mgr)

    return {
        "status": "ok",
        "name": req.name,
        "size": len(req.yaml_content),
    }


@app.post("/v1/workflow/execute")
async def v1_workflow_execute(req: WorkflowExecuteRequest):
    """One-step workflow execution with three-tier caching.

    Cache tiers:
      1. pipeline_store (memory LRU)
      2. CACHE_DIR/workflows/{name}.yaml (disk)
      3. Server fetch (GET /api/wf/yaml)

    Use force_refresh=true or pass yaml_content to bypass cache.
    """
    if not req.workflow:
        raise HTTPException(status_code=400, detail="workflow name is required")
    if not req.image and not req.image_url:
        raise HTTPException(status_code=400, detail="Either image or image_url must be provided")

    # ── Resolve PipelineManager (three-tier cache) ──
    mgr = None
    workflow_id = req.workflow
    server = req.server_url or DEFAULT_SERVER_URL

    if req.yaml_content:
        # One-shot custom YAML — no caching, no store registration
        config = _parse_workflow_yaml(req.yaml_content)
        mgr = PipelineManager.from_config(config)
        workflow_id = config.name
    else:
        # Tier 1: memory cache
        if not req.force_refresh:
            mgr = await pipeline_store.get(workflow_id)
        if mgr is None:
            # Tier 2: disk cache
            yaml_content = None if req.force_refresh else _load_workflow_from_cache(workflow_id)
            if yaml_content is not None:
                try:
                    config = _parse_workflow_yaml(yaml_content)
                    mgr = PipelineManager.from_config(config)
                except Exception as e:
                    print(f"[cache] Cached YAML for '{workflow_id}' is corrupted, re-fetching: {e}", flush=True)
                    yaml_content = None
            # Tier 3: server fetch
            if yaml_content is None:
                try:
                    yaml_content = _fetch_workflow_from_server(workflow_id, server)
                except requests.RequestException as e:
                    raise HTTPException(status_code=502, detail=f"Failed to download workflow from server: {str(e)}")
                config = _parse_workflow_yaml(yaml_content)
                mgr = PipelineManager.from_config(config)
                # Persist to disk cache
                _save_workflow_to_cache(workflow_id, yaml_content)
            # Register in memory cache
            await pipeline_store.add(workflow_id, mgr)

    # ── Auto-load YOLO models ──
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
                client = ServerClient(server)
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
                    print(f'[deploy] No model file found for {node.model}', flush=True)
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
                print(f'[deploy] Loaded model {node.model}', flush=True)
            except Exception as load_err:
                print(f'[deploy] Failed to load model {node.model}: {load_err}', flush=True)
                continue

    # ── Prepare image ──
    try:
        if req.image:
            image = io.BytesIO(base64.b64decode(req.image))
        elif req.image_url:
            print(f'[deploy] Fetching image from URL: {req.image_url}', flush=True)
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image = io.BytesIO(resp.content)
            print(f'[deploy] Image fetched OK ({len(resp.content)} bytes)', flush=True)
        else:
            raise HTTPException(status_code=400, detail="Either image or image_url must be provided")
    except HTTPException:
        raise
    except Exception as e:
        print(f'[deploy] Failed to fetch image: {e}', flush=True)
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # ── Execute pipeline ──
    start_time = time.perf_counter()
    try:
        result = await mgr.execute(
            image=image,
            engine_pool=engine_pool,
            yolo_adapter=yolo_adapter,
            vllm_client=vllm_client,
        )
        result["workflow_id"] = workflow_id
        result["execution_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
