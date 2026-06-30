# Workflow 可视化调试运行 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增强 workflow 编辑器可视化调试能力（结果面板 + 节点状态着色 + 双向联动），清理 nndeploy 残留

**Architecture:** 
- 后端仅增加 `exec_mode` 参数控制执行模式（auto/local/remote），不改动 PipelineManager
- 前端在现有 workflow.html 上增强：属性面板增加结果 tab，LiteGraph 节点结果状态着色，双向联动
- deploy 端保持不变
- 清理：删除 nndeploy 日志，更新 README 和 .claude/settings.json

**Tech Stack:** Flask/Python, FastAPI, LiteGraph.js, JavaScript (vanilla)

---

### Task 1: 清理 nndeploy 残留

**Files:**
- Delete: `logs/nndeploy.log`
- Delete: `logs/nndeploy_error.log`
- Modify: `README.md` (移除 nndeploy 章节)
- Modify: `.claude/settings.json` (移除 nndeploy 相关 allowlist)

- [ ] **Step 1: 删除 nndeploy 日志文件**

```bash
git rm logs/nndeploy.log logs/nndeploy_error.log
```

Expected: Files staged for deletion.

- [ ] **Step 2: 删除 nndeploy 相关 git 跟踪文件（已验证不在仓库中）**

确认 `nndeploy-app/` 和 `deploy/nndeploy_adapter.py` 已不存在于工作区。

- [ ] **Step 3: 更新 README.md — 移除 nndeploy 章节**

修改 `README.md`：

删除以下内容（约从 `#### 3. xclabel-nndeploy` 到 ```` 结束的代码块及其说明段落）：

删除从 `#### 3. xclabel-nndeploy` 开始到下一个二级标题 `#### 4.` 之前的所有内容。

具体定位文本（README.md 第 146-162 行）：
```
#### 3. xclabel-nndeploy
...
\`\`\`
```
替换为空字符串。

同时更新端口表格（README.md 约第 225-228 行），删除 nndeploy-app 那一行：

删除前：
```
| xclabel-server | 9924 | xclabel-server-gpu | 标注/训练 WebUI |
| nndeploy-app | 8002 | xclabel-nndeploy | Workflow 编排 WebUI |
| xclabel-deploy-cpu | 8000 | xclabel-deploy-cpu | CPU 推理 API |
```
删除后：
```
| xclabel-server | 9924 | xclabel-server-gpu | 标注/训练 WebUI |
| xclabel-deploy-cpu | 8000 | xclabel-deploy-cpu | CPU 推理 API |
```

同时更新架构图（约第 252-256 行），删除 nndeploy-app 框。

同时更新目录结构（约第 422-433 行），删除 `nndeploy-app/` 行和相关说明。

同时更新 Workflow 执行说明（约第 269-273 行），去掉 "nndeploy DAG" 等描述。

同时更新 docker-compose 引用（约第 366 行），删除 `nndeploy_adapter.py` 的 docker cp 示例。

- [ ] **Step 4: 更新 .claude/settings.json — 移除 nndeploy 相关规则**

编辑 `.claude/settings.json`，移除包含 `nndeploy` 的 allowlist 条目（共约 8 行 Bash 规则，全部以 `Bash(...nndeploy...)` 形式）。

具体删除内容：找到包含 `nndeploy` 的数组元素，从第 19-28 行附近，删除这些元素。

```json
// 删除前允许的条目如:
"Bash(curl -s \"http://127.0.0.1:9924/api/nndeploy/workflows\")",
"Bash(curl -s \"http://127.0.0.1:9924/api/nndeploy/workflow/download?id=...\")",
// ... 所有 nndeploy 相关的行
```

- [ ] **Step 5: 提交清理改动**

```bash
git add README.md .claude/settings.json
git rm logs/nndeploy.log logs/nndeploy_error.log
git commit -m "chore: 清理 nndeploy 残留文件与引用

移除废弃的 nndeploy 日志、README 章节和 Claude 配置中的历史规则。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 后端增加 exec_mode 参数

**Files:**
- Modify: `app.py:5843-5921` (`/api/wf/execute` 端点)

- [ ] **Step 1: 修改 execute 端点增加 exec_mode 参数**

在 `app.py` 的 `wf_execute()` 函数中，在获取 `workflow_id` 后读取 `exec_mode` 参数：

```python
@app.route('/api/wf/execute', methods=['POST'])
def wf_execute():
    """Execute a workflow locally (no separate deploy service needed)."""
    data = request.json or {}
    name = data.get('workflow_id', '')
    exec_mode = data.get('exec_mode', 'auto')  # 'auto', 'local', 'remote'
    if not name:
        return jsonify({'error': '缺少 workflow_id 参数'}), 400

    # 1. Try external deploy service (unless forced local)
    if exec_mode != 'local':
        deploy_url = os.environ.get('DEPLOY_SERVER_URL', 'http://127.0.0.1:8000')
        try:
            resp = requests.post(
                f'{deploy_url}/pipeline/execute',
                json=data,
                timeout=8,
            )
            resp.raise_for_status()
            result = resp.json()
            result['mode'] = 'remote_deploy'
            return jsonify(result)
        except requests.RequestException:
            if exec_mode == 'remote':
                return jsonify({'error': '远程 deploy 服务不可用，请检查 DEPLOY_SERVER_URL'}), 502
            pass  # fall back to local engine

    # 2. Local execution with integrated ML engine (rest unchanged)
    import yaml as _yaml
    yaml_path = _wf_yaml_path(name)
    json_path = _wf_path(name)
    # ... 其余代码保持不变 ...
```

注意：保持现有逻辑完整不变，只添加 `exec_mode` 参数判断和 `mode` 返回值。

- [ ] **Step 2: 确认现有结果格式已包含 mode 字段**

确认原有代码中 `result['mode'] = 'local_engine'` 已存在（第 5914 行）。✅

- [ ] **Step 3: 提交后端改动**

```bash
git add app.py
git commit -m "feat: /api/wf/execute 增加 exec_mode 参数

支持 auto(默认尝试远程，失败回退本地) / local(强制本地) / remote(仅远程) 三种模式。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: workflow.html — 结果面板 Tab

**Files:**
- Modify: `templates/workflow.html`

- [ ] **Step 1: 添加结果面板的 CSS 样式**

在 `templates/workflow.html` 的 `<style>` 块中（在 `editor-area` 样式之后，约第 161 行），添加结果面板相关样式：

```css
/* ── Props sidebar tabs ── */
.props-tabs{display:flex;border-bottom:1px solid #0f3460;flex-shrink:0}
.props-tab{padding:8px 14px;font-size:.76em;cursor:pointer;color:#888;border-bottom:2px solid transparent;transition:all .15s;display:flex;align-items:center;gap:4px}
.props-tab:hover{color:#e0e0e0}
.props-tab.active{color:#e94560;border-bottom-color:#e94560}

/* ── Result panel ── */
.result-panel{display:none;flex:1;overflow-y:auto;padding:0}
.result-panel.open{display:block}
.result-empty{padding:20px 12px;color:#666;font-size:.8em;text-align:center}
.result-summary{padding:8px 12px;background:#0d1b2a;border-bottom:1px solid #0f3460;font-size:.74em;display:flex;gap:16px;flex-wrap:wrap}
.result-summary .stat{display:flex;align-items:center;gap:4px}
.result-summary .stat .val{color:#e0e0e0;font-weight:600}
.result-summary .stat .lbl{color:#888}
.result-node{padding:6px 12px;border-bottom:1px solid rgba(255,255,255,0.04);cursor:pointer;transition:background .1s;font-size:.76em}
.result-node:hover{background:rgba(233,69,96,0.08)}
.result-node .node-header{display:flex;align-items:center;gap:6px}
.result-node .node-status-icon{width:16px;text-align:center}
.result-node .node-title{color:#e0e0e0;font-weight:500;flex:1}
.result-node .node-time{color:#888;font-size:.9em}
.result-node .node-detail{margin-top:4px;margin-left:22px;padding:4px 8px;background:#0d1b2a;border-radius:3px;display:none;font-size:.95em}
.result-node .node-detail.open{display:block}
.result-node .node-detail img{max-width:100%;max-height:180px;border-radius:3px;margin:4px 0;cursor:pointer}
.result-node .node-detail img:active{max-height:none}
.result-node .node-detail .det-row{padding:1px 0;color:#aaa}
.result-node .node-detail .det-row .cls{color:#4caf50}
.result-node .node-detail .det-row .conf{color:#ffa726}
```

- [ ] **Step 2: 改造属性面板区域，增加 Tab 切换**

替换现有的属性面板 injection 代码（约第 410-415 行）：

将：
```javascript
// ── Inject right sidebar (property panel) ──
var propsPanel = document.createElement('div');
propsPanel.className = 'props-sidebar';
propsPanel.id = 'propsSidebar';
propsPanel.innerHTML = '<h3 id="propsTitle"><i class="fas fa-sliders-h"></i> 节点属性</h3><div id="propsContent"><div class="no-selection">点击节点编辑属性</div></div>';
content.appendChild(propsPanel);
```

替换为：
```javascript
// ── Inject right sidebar (property panel with result panel tab) ──
var propsPanel = document.createElement('div');
propsPanel.className = 'props-sidebar';
propsPanel.id = 'propsSidebar';

// Tab bar
var tabBar = document.createElement('div');
tabBar.className = 'props-tabs';
tabBar.innerHTML =
  '<div class="props-tab active" data-tab="props" onclick="switchPropsTab(\'props\')"><i class="fas fa-sliders-h"></i> 属性</div>' +
  '<div class="props-tab" data-tab="result" onclick="switchPropsTab(\'result\')"><i class="fas fa-chart-bar"></i> 运行结果</div>';
propsPanel.appendChild(tabBar);

// Properties content
var propsContent = document.createElement('div');
propsContent.id = 'propsContent';
propsContent.innerHTML = '<div class="no-selection">点击节点编辑属性</div>';
propsPanel.appendChild(propsContent);

// Result panel (hidden by default)
var resultPanel = document.createElement('div');
resultPanel.className = 'result-panel';
resultPanel.id = 'resultPanel';
resultPanel.innerHTML = '<div class="result-empty">尚未执行工作流</div>';
propsPanel.appendChild(resultPanel);

content.appendChild(propsPanel);
```

- [ ] **Step 3: 添加 Tab 切换函数**

在 `hideProperties` 函数之前（约第 517 行），添加 Tab 切换函数：

```javascript
// ── Result panel / Props tab switching ──
var lastRunResult = null;

function switchPropsTab(tab){
  // Update tab buttons
  document.querySelectorAll('.props-tab').forEach(function(t){
    t.classList.toggle('active', t.dataset.tab === tab);
  });
  // Show/hide panels
  document.getElementById('propsContent').style.display = (tab === 'props') ? 'block' : 'none';
  document.getElementById('resultPanel').classList.toggle('open', tab === 'result');
  // Ensure sidebar is open
  document.getElementById('propsSidebar').classList.add('open');
  // If switching to result tab and we have data, render
  if(tab === 'result' && lastRunResult){
    renderResultPanel(lastRunResult);
  }
}
```

- [ ] **Step 4: 实现 renderResultPanel 函数**

在 `switchPropsTab` 之后添加：

```javascript
function renderResultPanel(data){
  var panel = document.getElementById('resultPanel');
  if(!data) return;
  
  var nodeOutputs = data.node_outputs || {};
  var nodeTimings = data.node_timings || {};
  var nodeStatus = data.node_status || {};
  var errors = data.errors || [];
  var totalMs = data.execution_time_ms || 0;
  var mode = data.mode || 'unknown';
  
  // Build graph info lookup
  var graphInfo = {};
  if(graph && graph._nodes){
    graph._nodes.forEach(function(n){
      graphInfo[n.id] = {title: n.title || n.type, type: n.type.replace('xclabel/', '')};
    });
  }
  
  // Summary
  var modeLabels = {'local_engine':'本地引擎', 'remote_deploy':'远程 Deploy'};
  var statusEmoji = errors.length ? '❌' : '✅';
  var summaryHtml =
    '<div class="result-summary">' +
    '<div class="stat"><span class="val">'+statusEmoji+'</span><span class="lbl">状态</span></div>' +
    '<div class="stat"><span class="val">'+totalMs+'ms</span><span class="lbl">总耗时</span></div>' +
    '<div class="stat"><span class="val">'+(modeLabels[mode]||mode)+'</span><span class="lbl">执行模式</span></div>' +
    '<div class="stat"><span class="val">'+Object.keys(nodeOutputs).length+'</span><span class="lbl">节点数</span></div>';
  if(errors.length){
    summaryHtml += '<div class="stat"><span class="val" style="color:#ff6b6b">'+errors.length+'</span><span class="lbl">错误</span></div>';
  }
  summaryHtml += '</div>';
  
  // Node list
  var nodeHtml = '';
  Object.keys(nodeOutputs).sort().forEach(function(nid){
    var info = graphInfo[nid] || {};
    var ntype = info.type || 'unknown';
    var ntitle = info.title || nid;
    var out = nodeOutputs[nid] || {};
    var timing = nodeTimings[nid] || '';
    var status = nodeStatus[nid] || 'ok';
    var timingStr = timing ? Math.round(timing)+'ms' : '';
    
    // Status icon
    var statusIcon = status === 'ok' ? '✅' : status === 'error' ? '❌' : '⏭';
    var statusColor = status === 'ok' ? '#4caf50' : status === 'error' ? '#ff6b6b' : '#ffa726';
    
    // Node type icon
    var typeIcons = {input:'fa-cloud-upload-alt', yolo:'fa-eye', condition:'fa-code-branch', vllm:'fa-brain', output:'fa-file-export'};
    var typeIcon = typeIcons[ntype] || 'fa-circle';
    
    // Detail content per node type
    var detailHtml = '';
    if(ntype === 'yolo' && out.annotated_image){
      detailHtml += '<img src="data:image/jpeg;base64,'+out.annotated_image+'" onclick="this.style.maxHeight=this.style.maxHeight?\'\':\'none\'" title="点击缩放">';
    }
    if(out.detections && out.detections.length){
      detailHtml += '<div style="color:#888;margin:4px 0">检测到 <b style="color:#e0e0e0">'+out.detections.length+'</b> 个目标</div>';
      out.detections.slice(0, 10).forEach(function(d){
        detailHtml += '<div class="det-row"><span class="cls">'+d.class_name+'</span> <span class="conf">'+(d.confidence?d.confidence.toFixed(3):'')+'</span></div>';
      });
      if(out.detections.length > 10){
        detailHtml += '<div style="color:#888">... 还有 '+(out.detections.length-10)+' 个</div>';
      }
    }
    if(ntype === 'yolo' && out.max_conf !== undefined){
      detailHtml += '<div style="color:#888;margin-top:2px">最高置信度: <b style="color:#ffa726">'+out.max_conf.toFixed(3)+'</b></div>';
    }
    if(ntype === 'condition'){
      detailHtml += '<div style="color:#888">表达式: <code style="color:#ce93d8">'+(out.expression || '')+'</code></div>';
      if(out.evaluated !== undefined){
        detailHtml += '<div style="color:#888">结果: <b style="color:'+(out.evaluated?'#4caf50':'#ff6b6b')+'">'+out.evaluated+'</b></div>';
      }
    }
    if(ntype === 'vllm' && out.vllm_result){
      var txt = out.vllm_result;
      detailHtml += '<div style="color:#e0e0e0;white-space:pre-wrap;word-break:break-all;max-height:120px;overflow:hidden">'+(txt.length>200?txt.slice(0,200)+'...':txt)+'</div>';
      if(txt.length > 200){
        detailHtml += '<div style="color:#6a9fb5;cursor:pointer;margin-top:2px" onclick="this.previousElementSibling.style.maxHeight=\'none\';this.style.display=\'none\'">展开全文 ▾</div>';
      }
    }
    if(ntype === 'output'){
      detailHtml += '<div style="color:#888">合并上游节点输出</div>';
    }
    
    nodeHtml +=
      '<div class="result-node" data-node-id="'+nid+'" onclick="onResultNodeClick(this,\''+nid+'\')">' +
      '<div class="node-header">' +
      '<span class="node-status-icon">'+statusIcon+'</span>' +
      '<i class="fas '+typeIcon+'" style="color:#e94560;width:14px;text-align:center"></i>' +
      '<span class="node-title">'+ntitle+'</span>' +
      '<span class="node-time" style="color:'+statusColor+'">'+timingStr+'</span>' +
      '</div>' +
      '<div class="node-detail" id="detail-'+nid+'">'+detailHtml+'</div>' +
      '</div>';
  });
  
  if(!nodeHtml){
    nodeHtml = '<div class="result-empty">无节点输出</div>';
  }
  
  panel.innerHTML = summaryHtml + nodeHtml;
}

function onResultNodeClick(el, nid){
  // Toggle detail
  var detail = document.getElementById('detail-'+nid);
  if(detail) detail.classList.toggle('open');
  
  // Select node in graph
  if(graph && graph._nodes){
    for(var i=0; i<graph._nodes.length; i++){
      var n = graph._nodes[i];
      if(String(n.id) === String(nid)){
        graphcanvas.selectNode(n);
        break;
      }
    }
  }
}
```

- [ ] **Step 5: 修改 executeWorkflow — 保存结果到 lastRunResult 并自动切换到结果 tab**

在 `executeWorkflow` 函数的 `.then(function(data){` 回调中，在 `resultEl.textContent = ...`（约第 1016 行）之后添加：

```javascript
// Save for result panel
lastRunResult = data;
// Auto-switch to result tab
switchPropsTab('result');
```

同时删除或保留现有的模态框显示逻辑。我们保留模态框（因为有图片预览和日志），但运行完成后自动切换到右侧结果面板。

- [ ] **Step 6: 提交 workflow 结果面板改动**

```bash
git add templates/workflow.html
git commit -m "feat: workflow 编辑器增加运行结果面板

右侧属性面板增加「运行结果」tab，按节点展示执行状态/耗时/详细输出，
支持 YOLO 标注图、检测列表、Condition 分支、VLLM 文本等差异化展示。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: workflow.html — 节点状态着色 + 双向联动

**Files:**
- Modify: `templates/workflow.html`

- [ ] **Step 1: 添加节点状态重置和着色逻辑**

在 `executeWorkflow` 函数开始时（约第 882 行，`addLog` 调用之前），添加节点状态重置：

```javascript
function executeWorkflow(){
  var name = getWfName();
  if(!name){ toast('请输入名称'); return; }
  
  // Reset node colors
  resetNodeColors();
  
  // ... rest of existing code ...
```

在 `resetNodeColors` 函数（在 `executeWorkflow` 之前添加）：

```javascript
function resetNodeColors(){
  if(!graph || !graph._nodes) return;
  graph._nodes.forEach(function(n){
    n.boxcolor = null; // reset to default
    // Restore original title if we modified it
    if(n._originalTitle) n.title = n._originalTitle;
  });
  graph.setDirtyCanvas(true);
}
```

- [ ] **Step 2: 结果返回后着色节点**

在 `executeWorkflow` 的 `.then(function(data){` 回调中，在解析 `nodeStatus` 等变量之后（约第 922 行），添加节点着色逻辑：

```javascript
// Colorize nodes by status
var statusColors = {'ok': '#27ae60', 'error': '#e74c3c', 'skipped': '#f39c12'};
if(graph && graph._nodes){
  graph._nodes.forEach(function(n){
    var s = nodeStatus[n.id];
    if(s && statusColors[s]){
      n.boxcolor = statusColors[s];
    }
    // Append timing to title
    if(s === 'ok' && nodeTimings[n.id] !== undefined){
      if(!n._originalTitle) n._originalTitle = n.title;
      n.title = n._originalTitle + ' ['+Math.round(nodeTimings[n.id])+'ms]';
    } else if(s === 'error'){
      if(!n._originalTitle) n._originalTitle = n.title;
      n.title = n._originalTitle + ' ❌';
    } else if(s === 'skipped'){
      if(!n._originalTitle) n._originalTitle = n.title;
      n.title = n._originalTitle + ' ⏭';
    }
  });
  graph.setDirtyCanvas(true);
}
```

- [ ] **Step 3: 节点选中 → 结果面板滚动联动**

在 `initEditor` 函数的 `graphcanvas.onNodeMousedown` 回调（约第 321 行）中添加滚动逻辑：

```javascript
graphcanvas.onNodeMousedown = function(e, node, slot, event){
  showProperties(node);
  // Scroll result panel to matching node
  scrollToResultNode(node.id);
};
```

添加 `scrollToResultNode` 函数：

```javascript
function scrollToResultNode(nid){
  var panel = document.getElementById('resultPanel');
  if(!panel || !panel.classList.contains('open')) return;
  var row = panel.querySelector('.result-node[data-node-id="'+nid+'"]');
  if(row){
    row.scrollIntoView({behavior:'smooth', block:'nearest'});
    // Brief highlight
    row.style.background = 'rgba(233,69,96,0.2)';
    setTimeout(function(){ row.style.background = ''; }, 1000);
  }
}
```

- [ ] **Step 4: 新建/打开工作流时重置节点颜色**

在 `onNew` 函数（约第 614 行）开头添加 `resetNodeColors();`

在 `loadWorkflow` 函数（约第 697 行）的 `.then(function(data){` 回调开头添加 `resetNodeColors();`

- [ ] **Step 5: 提交节点状态着色改动**

```bash
git add templates/workflow.html
git commit -m "feat: workflow 节点状态着色与双向联动

运行后节点按状态着色(绿/红/黄)，节点标题附加耗时；
结果面板点击行→高亮画布节点，画布选中节点→滚动结果面板。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Run 模式选择（exec_mode 选择器）

**Files:**
- Modify: `templates/workflow.html`

- [ ] **Step 1: 在 Run 模态框增加执行模式选择**

在 `templates/workflow.html` 的运行模态框（约第 183 行），在执行按钮前增加模式下拉框：

```html
<div class="field" style="display:flex;gap:10px;align-items:center;margin-top:8px">
  <label style="font-size:.76em;color:#aaa">执行模式</label>
  <select id="execModeSelect" style="background:#0f3460;color:#e0e0e0;border:1px solid #533483;padding:3px 6px;border-radius:3px;font-size:.78em">
    <option value="auto">自动 (远程→本地)</option>
    <option value="local">仅本地引擎</option>
    <option value="remote">仅远程 Deploy</option>
  </select>
  <span style="font-size:.7em;color:#666;margin-left:4px" id="execModeHint">自动尝试远程，失败回退本地</span>
</div>
```

- [ ] **Step 2: 在执行请求中发送 exec_mode**

修改 `executeWorkflow` 函数中的 payload 构建，在约第 885 行：

```javascript
var payload = {workflow_id: name};
// Add exec_mode from selector
var modeSelect = document.getElementById('execModeSelect');
if(modeSelect) payload.exec_mode = modeSelect.value;
```

- [ ] **Step 3: 切换 exec_mode 时更新提示**

```javascript
// 放在全局位置，与 switchPropsTab 同级
document.addEventListener('change', function(e){
  if(e.target.id === 'execModeSelect'){
    var hints = {auto:'自动尝试远程，失败回退本地', local:'仅在本地引擎执行，不连接远程服务', remote:'仅通过远程 Deploy 执行，失败则报错'};
    var hintEl = document.getElementById('execModeHint');
    if(hintEl) hintEl.textContent = hints[e.target.value] || '';
  }
});
```

- [ ] **Step 4: 提交 exec_mode 选择器改动**

```bash
git add templates/workflow.html
git commit -m "feat: workflow 运行增加执行模式选择

支持自动/本地/远程三种模式，用户可在运行模态框中切换。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 最终检查与验证

- [ ] **Step 1: 检查所有 nndeploy 残留是否清理干净**

```bash
# 确认没有 nndeploy 的活跃文件
grep -r "nndeploy" --include="*.py" --include="*.html" --include="*.js" --include="*.yml" --include="*.yaml" --include="*.md" --include="*.json" app.py templates/ static/ deploy/ docker-compose*.yml .vscode/launch.json README.md

# 预期只有 openspec/ 归档目录中有引用（那是历史文档，无需清理）
```

- [ ] **Step 2: 验证 app.py 语法正确**

```bash
python -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: 验证 deploy 服务无破坏**

```bash
# 检查 deploy/main.py 的关键端点是否完整
grep -n "@app.*post\|@app.*get" deploy/main.py
```

Expected: `/health`, `/infer`, `/engines`, `/unload`, `/workflows`, `/v1/predict`, `/pipeline/execute`, `/pipeline/load`, `/pipeline/unload`, `/pipeline/workflows`, `/test`

- [ ] **Step 4: 集成提交**

如果所有改动都是干净的，创建一个最终的集成提交：

```bash
git add -A
git commit -m "feat: workflow 可视化调试与清理

- 后端: /api/wf/execute 增加 exec_mode 参数 (auto/local/remote)
- 前端: 右侧结果面板，按节点展示运行输出
- 前端: 节点状态着色(绿/红/黄)+耗时显示
- 前端: 双向联动(点击结果行↔画布节点)
- 前端: 执行模式选择器(自动/本地/远程)
- 清理: 删除 nndeploy 日志与 README 残留引用

Co-Authored-By: Claude <noreply@anthropic.com>"
```

如果更偏好逐任务独立提交，前面的 Task 1-5 已经提供了每个步骤的提交命令。

---

## 自检清单

- [x] **Spec 覆盖**: 设计文档中所有需求都有对应任务
  - 结果面板 → Task 3
  - 节点状态着色 → Task 4
  - 双向联动 → Task 4
  - exec_mode 参数 → Task 2
  - 清理 nndeploy → Task 1
  - 执行模式选择器 → Task 5
- [x] **占位符检查**: 无 TBD/TODO/implement later 等占位符
- [x] **类型一致性**: 
  - `lastRunResult` 在 Task 3 定义和赋值，在 Task 4 中引用
  - `resetNodeColors()` 在 Task 4 定义，在 `onNew` 和 `loadWorkflow` 中调用
  - `switchPropsTab()` 在 Task 3 定义，在 `executeWorkflow` 结果返回后调用
  - `exec_mode` 参数在 Task 2 后端实现，在 Task 5 前端选择器中发送
