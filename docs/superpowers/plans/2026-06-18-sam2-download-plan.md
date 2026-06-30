# SAM2 模型下载功能 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在工程管理页面的设置中新增 SAM2 三个版本权重文件下载功能，从 hf-mirror.com 下载。

**架构:** 后端新增两个 API（检查模型状态 + SSE 流式下载），前端在设置弹框中新增 SAM2 下载卡片 UI，复用现有 YOLO 下载的 SSE 模式。

**Tech Stack:** Python/Flask (后端 SSE), JavaScript/HTML/CSS (前端)

---

### Task 1: 后端 API - SAM 模型检查 `/api/sam/check-models`

**Files:**
- Modify: `app.py`（在现有 `/api/sam/models` 附近新增）

- [ ] **Step 1: 在 `app.py` 的 `/api/sam/models` 之后新增 check-models 端点**

在 `app.py` 约 5007 行处，`/api/sam/models` 之后，`/api/sam/switch-model` 之前，加入：

```python
@app.route('/api/sam/check-models', methods=['GET'])
def sam_check_models():
    """检查 SAM2 各版本权重文件是否存在并返回大小信息"""
    from ai_manager import SAM2Engine
    models_dir = os.environ.get("SAM_MODELS_DIR", os.path.join(BASE_PATH, "models"))
    # 模型大小信息（粗略值）
    model_sizes = {
        "tiny": 155,
        "small": 184,
        "base_plus": 298,
    }
    models = []
    for key, cfg in SAM2Engine.MODEL_CONFIGS.items():
        ckpt_path = os.path.join(models_dir, cfg["checkpoint"])
        size_mb = model_sizes.get(key, 0)
        models.append({
            'id': key,
            'name': f"SAM 2 Hiera {key.capitalize()}",
            'checkpoint': cfg["checkpoint"],
            'available': os.path.exists(ckpt_path),
            'size_mb': size_mb,
        })
    return jsonify({'success': True, 'models': models})
```

### Task 2: 后端 API - SAM 模型下载 `/api/sam/download-models` (SSE)

**Files:**
- Modify: `app.py`（在 check-models 端点之后新增）

- [ ] **Step 1: 在 `app.py` 中新增 download-models SSE 端点**

```python
@app.route('/api/sam/download-models')
def sam_download_models():
    """从 hf-mirror.com 下载 SAM2 权重文件（SSE 流式推送）"""
    import requests
    import time

    models_str = request.args.get('models', '')
    model_ids = [m.strip() for m in models_str.split(',') if m.strip()]

    from ai_manager import SAM2Engine
    models_dir = os.environ.get("SAM_MODELS_DIR", os.path.join(BASE_PATH, "models"))
    os.makedirs(models_dir, exist_ok=True)

    # 下载 URL 映射
    HF_MIRROR_URLS = {
        "tiny": "https://hf-mirror.com/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt",
        "small": "https://hf-mirror.com/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt",
        "base_plus": "https://hf-mirror.com/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt",
    }

    def generate():
        yield f"data: {json.dumps({'status': 'started', 'message': '准备下载 SAM2 模型...', 'progress': 0})}\n\n"
        time.sleep(0.3)

        try:
            # 过滤已存在的模型
            to_download = []
            skipped = []
            for mid in model_ids:
                if mid not in SAM2Engine.MODEL_CONFIGS:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'未知模型: {mid}', 'progress': 0})}\n\n"
                    return
                ckpt = SAM2Engine.MODEL_CONFIGS[mid]["checkpoint"]
                ckpt_path = os.path.join(models_dir, ckpt)
                if os.path.exists(ckpt_path):
                    skipped.append(mid)
                else:
                    to_download.append(mid)

            if skipped:
                yield f"data: {json.dumps({'message': f'已跳过已存在的模型: {", ".join(skipped)}', 'progress': 5})}\n\n"
                time.sleep(0.3)

            if not to_download:
                yield f"data: {json.dumps({'status': 'completed', 'message': '所有选中模型均已存在，无需下载', 'progress': 100})}\n\n"
                return

            total = len(to_download)
            completed_count = 0
            for mid in to_download:
                ckpt = SAM2Engine.MODEL_CONFIGS[mid]["checkpoint"]
                url = HF_MIRROR_URLS.get(mid)
                if not url:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'模型 {mid} 无下载地址', 'progress': 0})}\n\n"
                    return

                save_path = os.path.join(models_dir, ckpt)
                yield f"data: {json.dumps({'status': 'downloading', 'model': ckpt, 'progress': int(completed_count / total * 80) + 10, 'message': f'正在下载 {ckpt}...'})}\n\n"

                try:
                    # 流式下载
                    resp = requests.get(url, stream=True, timeout=300)
                    resp.raise_for_status()

                    total_size = int(resp.headers.get('content-length', 0))
                    downloaded = 0
                    chunk_size = 8192

                    with open(save_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    pct = int(downloaded / total_size * 100)
                                    # 进度范围：10% ~ 90% 之间
                                    overall_pct = int(completed_count / total * 80) + int(pct * 0.8 / total)
                                    downloaded_mb = downloaded / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    yield f"data: {json.dumps({'status': 'downloading', 'model': ckpt, 'progress': min(overall_pct, 99), 'downloaded_mb': round(downloaded_mb, 1), 'total_mb': round(total_mb, 1), 'message': f'下载 {ckpt}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB'})}\n\n"

                    completed_count += 1
                    yield f"data: {json.dumps({'status': 'model_completed', 'model': ckpt, 'progress': int(completed_count / total * 80) + 10, 'message': f'{ckpt} 下载完成'})}\n\n"
                    time.sleep(0.2)

                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'model': ckpt, 'message': f'{ckpt} 下载失败: {str(e)}', 'progress': 0})}\n\n"
                    return

            yield f"data: {json.dumps({'status': 'completed', 'message': '所有 SAM2 模型下载完成', 'progress': 100})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'status': 'error', 'message': f'下载失败: {str(e)}', 'progress': 0, 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')
```

### Task 3: 前端 - projects.html 设置弹框新增 SAM2 下载区域 UI

**Files:**
- Modify: `templates/projects.html`（在设置弹框内，YOLO 区域之后新增）

- [ ] **Step 1: 在 YOLO 训练环境区域之后、设置弹框关闭前新增 SAM2 下载区域 HTML**

在 `projects.html` 约 386 行（YOLO 区域的 `</div>` 之后，设置弹框 `</div>` 关闭之前），插入以下 HTML：

```html
            <!-- SAM2 模型下载 -->
            <div class="sam2-download-section" style="margin-top: 20px; padding-top: 20px; border-top: 2px solid #e9ecef;">
                <h3 style="margin: 0 0 15px 0; font-size: 1.1em; color: #555;">
                    <i class="fas fa-cut"></i> SAM2 分割模型
                </h3>
                <p style="font-size: 0.85em; color: #888; margin-bottom: 15px;">
                    下载地址：<a href="https://hf-mirror.com" target="_blank">hf-mirror.com</a>（中国 HuggingFace 镜像）
                </p>
                <div style="margin-bottom: 10px;">
                    <label style="font-weight: 500; display: block; margin-bottom: 8px;">选择要下载的 SAM2 版本:</label>
                    <div id="samModelsCheckboxContainer" style="display: flex; flex-wrap: wrap; gap: 10px;">
                        <!-- SAM2 模型复选框将通过 JS 动态生成 -->
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 15px; width: fit-content;">
                    <button type="button" class="btn btn-primary icon-btn" id="downloadSamModelsBtn">
                        <i class="fas fa-download"></i> <span class="btn-text">下载选中模型</span>
                    </button>
                    <button type="button" class="btn btn-secondary icon-btn" id="refreshSamModelsBtn">
                        <i class="fas fa-sync-alt"></i> <span class="btn-text">刷新状态</span>
                    </button>
                </div>

                <!-- 已安装模型列表 -->
                <div id="samModelsContainer" style="margin-bottom: 15px;">
                    <h4 style="margin: 0 0 10px 0; font-size: 1em; color: #666;">已安装模型:</h4>
                    <div id="samModelsList" style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; min-height: 40px; font-size: 0.9em;">
                        <!-- 模型列表动态生成 -->
                    </div>
                </div>

                <!-- 下载状态 -->
                <div id="samDownloadStatus" style="margin-top: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px; font-size: 0.9em; display: none;">
                    <i class="fas fa-info-circle"></i> <span id="samStatusText">就绪</span>
                </div>

                <!-- 下载进度条 -->
                <div id="samDownloadProgress" style="margin-top: 8px; display: none;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span style="font-size: 0.85em; color: #666;" id="samProgressLabel">下载进度:</span>
                        <span id="samProgressPercent">0%</span>
                    </div>
                    <div style="width: 100%; height: 6px; background-color: #e9ecef; border-radius: 3px; overflow: hidden;">
                        <div id="samProgressBar" style="width: 0%; height: 100%; background-color: #339af0; transition: width 0.3s ease;"></div>
                    </div>
                </div>
            </div>
```

- [ ] **Step 2: 在 projects.html 的 `<script>` 中新增设置弹框打开时刷新 SAM 模型状态的逻辑**

在 `projects.html` 中现有的 `showSettingsModal` 函数内（约 651 行），在 `loadGlobalConfig()` 之后添加：

```javascript
// 在 showSettingsModal 函数中
function showSettingsModal() {
    loadGlobalConfig();
    document.getElementById('settingsModal').style.display = 'block';
    // 刷新 SAM 模型状态
    if (typeof refreshSamModels === 'function') refreshSamModels();
}
```

- [ ] **Step 3: 在 projects.html 的 `<script>` 中绑定 SAM2 下载和刷新按钮事件**

在 `setupEventListeners` 函数内（约 597 行，YOLO 按钮事件绑定之后），添加：

```javascript
// SAM2 按钮事件
const downloadSamBtn = document.getElementById('downloadSamModelsBtn');
const refreshSamBtn = document.getElementById('refreshSamModelsBtn');
if (downloadSamBtn && typeof downloadSamModels === 'function') {
    downloadSamBtn.addEventListener('click', downloadSamModels);
}
if (refreshSamBtn && typeof refreshSamModels === 'function') {
    refreshSamBtn.addEventListener('click', refreshSamModels);
}
```

### Task 4: 前端逻辑 - script.js 新增 SAM2 模型管理函数

**Files:**
- Modify: `static/script.js`（在 YOLO 模型函数附近新增）

- [ ] **Step 1: 新增 `refreshSamModels()` 函数**

在 `script.js` 中 `refreshModels()` 函数之后（约 2940 行处），新增：

```javascript
// ===== SAM2 模型管理 =====

// 刷新 SAM2 模型状态
function refreshSamModels() {
    const modelsList = document.getElementById('samModelsList');
    if (!modelsList) return;

    modelsList.innerHTML = '<span style="color: #999;"><i class="fas fa-spinner fa-spin"></i> 加载中...</span>';

    fetch('/api/sam/check-models')
        .then(r => r.json())
        .then(data => {
            if (!data.success || !data.models) {
                modelsList.innerHTML = '<span style="color: #999;">加载失败</span>';
                return;
            }
            // 更新模型列表
            let html = '';
            data.models.forEach(m => {
                const statusIcon = m.available ? '✅' : '❌';
                const statusText = m.available ? '已安装' : '未安装';
                html += `<div style="padding: 4px 0;">${statusIcon} ${m.checkpoint} <span style="color: #888; font-size: 0.85em;">(${statusText})</span></div>`;
            });
            modelsList.innerHTML = html || '<span style="color: #999;">暂无模型信息</span>';

            // 更新复选框
            const container = document.getElementById('samModelsCheckboxContainer');
            if (container) {
                container.innerHTML = '';
                data.models.forEach(m => {
                    const label = document.createElement('label');
                    label.style.cssText = 'display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 6px 12px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; font-size: 0.9em;';
                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.name = 'samModels';
                    cb.value = m.id;
                    // 默认勾选未安装的模型
                    if (!m.available) cb.checked = true;
                    const statusText = m.available ? '✅ 已安装' : '⬜ 未下载';
                    label.appendChild(cb);
                    label.appendChild(document.createTextNode(` ${m.name}（约 ${m.size_mb}MB）${statusText}`));
                    container.appendChild(label);
                });
            }
        })
        .catch(err => {
            modelsList.innerHTML = `<span style="color: #999;">加载失败: ${err.message}</span>`;
        });
}
```

- [ ] **Step 2: 新增 `downloadSamModels()` 函数**

在 `refreshSamModels()` 之后新增：

```javascript
// 下载 SAM2 模型
function downloadSamModels() {
    const selectedModels = Array.from(document.querySelectorAll('input[name="samModels"]:checked'))
        .map(cb => cb.value);

    if (selectedModels.length === 0) {
        showToast('请至少选择一个模型');
        return;
    }

    const statusElement = document.getElementById('samDownloadStatus');
    const statusText = document.getElementById('samStatusText');
    const progressElement = document.getElementById('samDownloadProgress');
    const progressBar = document.getElementById('samProgressBar');
    const progressPercent = document.getElementById('samProgressPercent');
    const progressLabel = document.getElementById('samProgressLabel');

    statusElement.style.display = 'block';
    statusText.textContent = '正在下载: ' + selectedModels.join(', ');
    progressElement.style.display = 'block';
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';

    // 禁用按钮
    const downloadBtn = document.getElementById('downloadSamModelsBtn');
    const refreshBtn = document.getElementById('refreshSamModelsBtn');
    downloadBtn.disabled = true;
    refreshBtn.disabled = true;

    // SSE 下载
    const eventSource = new EventSource('/api/sam/download-models?models=' + selectedModels.join(','));

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);

            // 更新进度
            if (data.progress !== undefined) {
                progressBar.style.width = data.progress + '%';
                progressPercent.textContent = data.progress + '%';
            }

            // 更新下载详情
            if (data.downloaded_mb !== undefined && data.total_mb !== undefined) {
                progressLabel.textContent = `正在下载 ${data.model}: ${data.downloaded_mb}MB / ${data.total_mb}MB`;
            } else if (data.message) {
                statusText.textContent = data.message;
                progressLabel.textContent = data.message;
            }

            // 下载完成
            if (data.status === 'completed') {
                eventSource.close();
                statusText.textContent = '✅ ' + (data.message || '所有 SAM2 模型下载完成');
                progressBar.style.width = '100%';
                progressPercent.textContent = '100%';
                // 刷新模型状态
                refreshSamModels();
                // 恢复按钮
                downloadBtn.disabled = false;
                refreshBtn.disabled = false;
                setTimeout(() => {
                    statusElement.style.display = 'none';
                    progressElement.style.display = 'none';
                }, 5000);
            }

            // 下载失败
            if (data.status === 'error') {
                eventSource.close();
                statusText.textContent = '❌ ' + (data.message || '下载失败');
                downloadBtn.disabled = false;
                refreshBtn.disabled = false;
                setTimeout(() => {
                    statusElement.style.display = 'none';
                    progressElement.style.display = 'none';
                }, 8000);
            }
        } catch (error) {
            console.error('解析 SAM 下载进度失败:', error);
        }
    };

    eventSource.onerror = function() {
        eventSource.close();
        statusText.textContent = '❌ 下载连接中断';
        downloadBtn.disabled = false;
        refreshBtn.disabled = false;
    };
}
```

### Task 5: 修改 projects.html 中 showSettingsModal 函数

- [ ] **Step 1: 更新 `showSettingsModal` 函数**

在 `projects.html` 的 `showSettingsModal` 函数中（约 651 行，在 `document.getElementById('settingsModal').style.display = 'block';` 之前），添加 `refreshSamModels()` 调用：

```javascript
function showSettingsModal() {
    loadGlobalConfig();
    document.getElementById('settingsModal').style.display = 'block';
    // 刷新 SAM2 模型状态
    if (typeof refreshSamModels === 'function') refreshSamModels();
}
```

### Task 6: 修改 projects.html 中 setupEventListeners 函数

- [ ] **Step 1: 在 DOMContentLoaded 初始化中添加 SAM2 模型状态加载**

在 `projects.html` 的 `document.addEventListener('DOMContentLoaded', function()` 内（约 408 行），在初始化列表的最后添加：

```javascript
// 初始化 SAM2 模型状态
if (typeof refreshSamModels === 'function') refreshSamModels();
```
