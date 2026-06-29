# Deploy 一键预测 API 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 xclabel-deploy 中新增 `POST /v1/predict` 一键推理端点，改造 ServerClient 支持 `"project_id/model_version"` 组合参数，简化测试页面。

**Architecture:** ServerClient 新增 `parse_model_ref()` 解析组合标识，内部仍调用 server 现有 API；新增 `/v1/predict` 端点封装下载→加载→推理三步为一步；测试页面新增一键预测 UI 区块。

**Tech Stack:** Python/FastAPI, Pydantic, HTML/JS

**设计文档:** `docs/superpowers/specs/2026-06-29-deploy-predict-api-design.md`

---

### Task 1: ServerClient 改造 — 组合参数支持 + 补全 list_workflows

**Files:**
- Modify: `deploy/server_client.py`

- [ ] **Step 1: 添加 `parse_model_ref()` 静态方法**

在 `download_model` 方法之前添加：

```python
@staticmethod
def parse_model_ref(model_ref: str) -> tuple[str, str]:
    """Parse "project_id/model_version" into (project_id, version)."""
    parts = model_ref.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model ref format: '{model_ref}'. "
            "Expected 'project_id/model_version'"
        )
    return parts[0], parts[1]
```

- [ ] **Step 2: 改造 `download_model()` 支持组合格式**

将方法签名改为接受 `model_ref: str`(组合格式) 或保留旧签名，内部兼容。推荐改造为：

```python
def download_model(self, project_id_or_ref: str, version: Optional[str] = None,
                   cache_dir: str = "/app/cache") -> str:
    """Download model from server.
    
    Supports two calling conventions:
      1. Combined: download_model("proj/v1", cache_dir="/app/cache")
      2. Legacy:   download_model("proj", "v1", cache_dir="/app/cache")
    """
    if version is None:
        project_id, version = self.parse_model_ref(project_id_or_ref)
    else:
        project_id = project_id_or_ref
    # 后续逻辑不变：url拼接、下载、解压
    url = f"{self.server_url}/api/model/download"
    params = {"project": project_id, "version": version}
    response = requests.get(url, params=params, stream=True, timeout=300)
    response.raise_for_status()

    model_dir = f"{cache_dir}/models/{project_id}_{version}"
    os.makedirs(model_dir, exist_ok=True)

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(model_dir)

    return model_dir
```

注意：`version` 参数改为可选，通过 `Optional[str]` 标注。需要从 `typing` 导入 `Optional`（已导入）。

- [ ] **Step 3: 补全 `list_workflows()` 方法**

在 `list_model_versions` 方法之后添加：

```python
def list_workflows(self, project_id: str) -> List[Dict]:
    """List available workflows for a project from the server."""
    url = f"{self.server_url}/api/workflow/list"
    params = {"project": project_id}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("workflows", data if isinstance(data, list) else [])
```

- [ ] **Step 4: 更新 imports（如需）**

确保 `deploy/server_client.py` 顶部已导入 `Optional`。当前文件第5行已有 `from typing import Dict, List, Optional`，无需额外导入。

- [ ] **Step 5: 提交 Task 1**

```bash
git add deploy/server_client.py
git commit -m "feat: ServerClient 支持 project_id/model_version 组合格式

- 新增 parse_model_ref() 静态方法解析组合标识
- download_model() 兼容新旧两种调用方式
- 补全 list_workflows() 方法修复 /workflows 端点 bug

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 新增 `POST /v1/predict` 端点

**Files:**
- Modify: `deploy/main.py`

- [ ] **Step 1: 添加 `PredictRequest` 和 `PredictResponse` 模型**

在 `PipelineExecuteRequest` 类之后（约第95行）添加：

```python
class PredictRequest(BaseModel):
    server_url: Optional[str] = None
    model: str = Field(..., description="Format: project_id/model_version")
    image: Optional[str] = Field(default=None, description="Base64 image data (without data:image/xxx;base64, prefix)")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
```

- [ ] **Step 2: 添加 `POST /v1/predict` 端点**

在 `unload_all` 端点之后（约第316行之后）、`PipelineManagerStore` 类之前添加：

```python
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
        client = server_client
        if req.server_url:
            client = ServerClient(req.server_url)

        try:
            model_dir = client.download_model(req.model, CACHE_DIR)
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
```

- [ ] **Step 3: 修复 `/workflows` 端点传入 project_id**

当前 `deploy/main.py` 第307行调用 `server_client.list_workflows()` 没有传参，但该方法需要 `project_id`。由于 `/workflows` 端点缺少 project_id 上下文，暂时将返回空列表，避免崩溃：

```python
@app.get("/workflows")
async def list_workflows():
    try:
        # TODO: get project_id from query param once needed
        workflows = server_client.list_workflows("")
        return {"workflows": workflows}
    except Exception as e:
        # Return empty list instead of 502 for now
        return {"workflows": []}
```

或者更干净的方案：添加可选 `project_id` 查询参数：

```python
@app.get("/workflows")
async def list_workflows(project_id: str = ""):
    try:
        if not project_id:
            return {"workflows": []}
        workflows = server_client.list_workflows(project_id)
        return {"workflows": workflows}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to list workflows: {str(e)}")
```

采用带 `project_id` 查询参数的方案。

- [ ] **Step 4: 提交 Task 2**

```bash
git add deploy/main.py
git commit -m "feat: 新增 POST /v1/predict 一键预测端点

- 添加 PredictRequest 请求模型
- 一键完成下载→加载→推理全流程
- 引擎自动缓存复用
- 修复 /workflows 端点 project_id 参数

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 测试页面改造 — 新增一键预测 UI

**Files:**
- Modify: `deploy/static/test.html`

- [ ] **Step 1: 在左侧面板顶部新增"一键预测"区块**

在现有左侧面板（`class="panel-left"`）的最前面、加载模型区块之前插入：

```html
<!-- 一键预测 -->
<div class="panel">
    <h2><i class="fas fa-rocket"></i> 一键预测 <span style="color:#999;font-weight:400;font-size:0.75em;">v1/predict</span></h2>
    <div class="load-form">
        <div class="form-row">
            <div class="form-group" style="flex:1;">
                <label>服务端地址</label>
                <input type="text" id="predictServerUrl" value="http://127.0.0.1:9924" placeholder="服务端地址">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group" style="flex:1;">
                <label>模型路径 <span style="color:#999;font-weight:400;">project_id/model_version</span></label>
                <input type="text" id="predictModel" value="sv30_seg/20260618_172731" placeholder="project_id/model_version">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group" style="max-width:120px;">
                <label>置信度</label>
                <input type="number" id="predictConfidence" value="0.25" step="0.05" min="0" max="1">
            </div>
            <div class="form-group" style="flex:1;">
                <label>测试图片</label>
                <div class="upload-zone" id="predictUploadZone" style="min-height:50px;padding:8px;">
                    <span style="color:#666;font-size:0.85em;">点击上传图片</span>
                    <input type="file" id="predictImageInput" accept="image/*">
                </div>
            </div>
        </div>
        <button class="btn btn-success" id="predictBtn"><i class="fas fa-rocket"></i> 一键预测</button>
        <span id="predictStatus" style="font-size:0.82em;color:#999;margin-left:8px;"></span>
    </div>
</div>
```

- [ ] **Step 2: 添加一键预测的 JS 逻辑**

在现有 script 的 `clearResult` 函数之前添加：

```javascript
// ── One-click Predict (v1/predict) ──
let predictImageData = null;

document.getElementById('predictImageInput').addEventListener('change', e => {
    if (e.target.files.length) {
        const file = e.target.files[0];
        if (!file.type.startsWith('image/')) { toast('仅支持图片'); return; }
        const reader = new FileReader();
        reader.onload = evt => {
            predictImageData = evt.target.result;
            document.querySelector('#predictUploadZone span').textContent = file.name;
            toast('图片已加载');
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('predictBtn').addEventListener('click', async () => {
    const model = document.getElementById('predictModel').value.trim();
    if (!model) { toast('请输入模型路径'); return; }
    if (!predictImageData) { toast('请先上传图片'); return; }

    const btn = document.getElementById('predictBtn');
    const status = document.getElementById('predictStatus');
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> 预测中...';
    status.textContent = '下载模型 + 推理中...';

    try {
        const body = {
            model: model,
            image: predictImageData.split(',')[1],
            confidence_threshold: parseFloat(document.getElementById('predictConfidence').value) || 0.25,
        };
        const serverUrl = document.getElementById('predictServerUrl').value.trim();
        if (serverUrl) body.server_url = serverUrl;

        // Show loading in result area
        const loading = document.getElementById('inferLoading');
        loading.classList.add('show');

        const r = await fetch(deployBase + '/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || '预测失败');

        // Display result (reuse existing rendering)
        currentImageData = predictImageData;
        currentResult = data.detections || [];
        currentTaskType = data.task_type || 'detect';
        renderResult(currentImageData, currentResult, currentTaskType);

        const jsonPanel = document.getElementById('resultJsonPanel');
        document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
        jsonPanel.style.display = 'block';

        const info = document.getElementById('resultInfo');
        info.innerHTML = `
            <span>⏱ ${data.inference_time_ms || 0} ms</span>
            <span>|</span>
            <span>📦 ${data.model || model}</span>
            <span>|</span>
            <span>🎯 ${currentResult.length} 个检测</span>
        `;
        status.textContent = '✅ 完成';
        toast('预测完成');

        // Refresh engine list (new engine may have been loaded)
        await refreshEngines();
        checkHealth();
    } catch (e) {
        toast('❌ ' + e.message);
        status.textContent = '❌ 失败';
    } finally {
        btn.disabled = false; btn.innerHTML = '<i class="fas fa-rocket"></i> 一键预测';
        document.getElementById('inferLoading').classList.remove('show');
    }
});
```

- [ ] **Step 3: 提交 Task 3**

```bash
git add deploy/static/test.html
git commit -m "feat: 测试页面新增一键预测 UI

- 左侧面板顶部新增一键预测区块
- 只需填入服务端地址、模型路径、上传图片即可预测
- 结果复用现有标注渲染逻辑

Co-Authored-By: Claude <noreply@anthropic.com>"
```
