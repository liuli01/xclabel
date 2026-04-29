import base64
import io
import os
import time
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine_pool import Engine, EnginePool
from nndeploy_adapter import NndeployAdapter
from server_client import ServerClient

app = FastAPI(title="xclabel-deploy", version="1.0.0")

# Configuration from environment
CACHE_DIR = os.environ.get("CACHE_DIR", "/app/cache")
MAX_ENGINES = int(os.environ.get("MAX_ENGINES", "10"))
DEFAULT_SERVER_URL = os.environ.get("SERVER_URL", "http://xclabel-server:5000")

# Global state
engine_pool = EnginePool(max_engines=MAX_ENGINES)
nndeploy_adapter = NndeployAdapter()
server_client = ServerClient(DEFAULT_SERVER_URL)

os.makedirs(f"{CACHE_DIR}/models", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/workflows", exist_ok=True)


# Request/Response models
class LoadModelRequest(BaseModel):
    project_id: str
    model_version: str
    server_url: Optional[str] = None


class LoadWorkflowRequest(BaseModel):
    project_id: str
    workflow_id: str
    server_url: Optional[str] = None


class InferRequest(BaseModel):
    engine_id: str
    image: Optional[str] = None
    image_url: Optional[str] = None
    confidence_threshold: float = 0.5


class UnloadRequest(BaseModel):
    engine_id: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "nndeploy_available": nndeploy_adapter.available,
        "engines_loaded": len(engine_pool),
    }


@app.post("/load/model")
async def load_model(req: LoadModelRequest):
    if not nndeploy_adapter.available:
        raise HTTPException(status_code=503, detail="nndeploy is not available")

    engine_id = f"{req.project_id}/{req.model_version}"

    # Check if already loaded
    existing = await engine_pool.get(engine_id)
    if existing:
        return {
            "engine_id": engine_id,
            "type": "model",
            "status": "already_loaded",
            "metadata": existing.metadata,
        }

    # Use custom server URL if provided
    client = server_client
    if req.server_url:
        client = ServerClient(req.server_url)

    try:
        # Download model from server
        model_dir = client.download_model(
            req.project_id, req.model_version, CACHE_DIR
        )

        # Load with nndeploy
        model, metadata = nndeploy_adapter.load_model(model_dir)

        # Add to engine pool
        engine = Engine(
            engine_id=engine_id,
            engine_type="model",
            project_id=req.project_id,
            engine=model,
            metadata=metadata,
        )
        await engine_pool.add(engine)

        return {
            "engine_id": engine_id,
            "type": "model",
            "status": "loaded",
            "metadata": metadata,
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load model: {str(e)}")


@app.post("/load/workflow")
async def load_workflow(req: LoadWorkflowRequest):
    if not nndeploy_adapter.available:
        raise HTTPException(status_code=503, detail="nndeploy is not available")

    engine_id = f"{req.project_id}/{req.workflow_id}"

    # Check if already loaded
    existing = await engine_pool.get(engine_id)
    if existing:
        return {
            "engine_id": engine_id,
            "type": "workflow",
            "status": "already_loaded",
        }

    # Use custom server URL if provided
    client = server_client
    if req.server_url:
        client = ServerClient(req.server_url)

    try:
        # Download workflow from server
        workflow_path = client.download_workflow(
            req.workflow_id, CACHE_DIR
        )

        # Load with nndeploy
        pipeline, workflow_json = nndeploy_adapter.load_workflow(workflow_path)

        # Add to engine pool
        engine = Engine(
            engine_id=engine_id,
            engine_type="workflow",
            project_id=req.project_id,
            engine=pipeline,
            metadata={"workflow": workflow_json},
        )
        await engine_pool.add(engine)

        return {
            "engine_id": engine_id,
            "type": "workflow",
            "status": "loaded",
            "nodes": list(workflow_json.get("nodes", {}).keys()),
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load workflow: {str(e)}")


@app.post("/infer")
async def infer(req: InferRequest):
    engine = await engine_pool.get(req.engine_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine not found: {req.engine_id}")

    # Prepare image
    try:
        if req.image:
            # Base64 decoded image
            image_bytes = base64.b64decode(req.image)
            image = io.BytesIO(image_bytes)
        elif req.image_url:
            # Download from URL
            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image = io.BytesIO(resp.content)
        else:
            raise HTTPException(status_code=400, detail="Either image or image_url must be provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Run inference with lock (serial per engine)
    start_time = time.time()
    async with engine.lock:
        try:
            if engine.engine_type == "model":
                result = nndeploy_adapter.infer_model(
                    engine.engine, image, req.confidence_threshold,
                    metadata=engine.metadata
                )
            else:
                result = nndeploy_adapter.infer_workflow(
                    engine.engine, image,
                    project_id=engine.project_id,
                    server_client=server_client,
                    cache_dir=CACHE_DIR,
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


@app.get("/workflows")
async def list_workflows():
    """List available nndeploy workflows from server."""
    try:
        workflows = server_client.list_workflows()
        return {"workflows": workflows}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to list workflows: {str(e)}")


@app.post("/unload/all")
async def unload_all():
    count = await engine_pool.clear()
    return {"status": "all_unloaded", "count": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
