# 计算节点 Trigger 机制实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 CalcNode 添加 condition/trigger 门控机制，使 Condition 节点的 boolean 输出能连接到 CalcNode 的 trigger 输入，并由 condition 结果决定 CalcNode 是否执行。

**Architecture:** 完全镜像 LLM 节点的现有 trigger 模式，涉及前端（LiteGraph 节点类 + 属性面板）和后端（转换逻辑 + DAG 构建 + 执行门控）共 3 个文件、5 个任务。

**Tech Stack:** Python (Flask), JavaScript (LiteGraph), YAML workflow engine

---

### Task 1: 前端 — CalcNode 添加 trigger 输入和 condition 属性

**Files:**
- Modify: `templates/workflow.html:353-375`
- Modify: `templates/workflow.html:659-662`

- [ ] **Step 1: 修改 CalcNode 类，增加 trigger 输入和 condition 属性**

修改 `templates/workflow.html` 第 353-360 行：

```javascript
// Calc Node
function CalcNode(){
    this.properties = {
        expression: '(width * height) / 100',
        output_field: 'computed',
        condition: ''
    };
    this.addInput('in', 'array');
    this.addInput('trigger', 'boolean');
    this.addOutput('out', 'array');
    this.size = [200, 90];
}
```

- [ ] **Step 2: 修改属性面板，增加 condition 字段**

修改 `templates/workflow.html` 第 659-662 行，在 calc 分支的 output_field 行后插入 condition 字段：

```javascript
else if(type === 'xclabel/calc' || type === 'calc'){
    html += '<div class="field"><label>expression</label><textarea onchange="setProp(\'expression\',this.value)" placeholder="(width * height) / 100" style="font-family:Consolas,monospace;min-height:60px">'+(p.expression||'')+'</textarea></div>';
    html += fieldHtml('text', 'output_field', p.output_field || 'computed', '结果字段名');
    html += fieldHtml('text', 'condition', p.condition || '', '条件节点 ID');
    html += '<div class="help" style="padding:0 12px;margin:-4px 0 6px;line-height:1.6">可用变量: conf, class_id, class_name, x1, y1, x2, y2, width, height, area, perimeter<br>可用函数: abs, round, sqrt, ceil, floor, min, max, pi</div>';
}
```

- [ ] **Step 3: 验证 — 浏览器检查**

重新加载工作流编辑器，确认：
1. 新建 CalcNode 显示 "trigger" 输入端口
2. Condition 节点的 "result" 输出可以连接到 CalcNode 的 "trigger" 输入
3. 属性面板显示 "condition" 字段

---

### Task 2: 后端 — `_derive_node_conditions` 函数扩展

**Files:**
- Modify: `app.py:5722`
- Modify: `app.py:5726-5744`
- Modify: `app.py:5758`

- [ ] **Step 1: 将 `_derive_vllm_conditions` 重命名为 `_derive_node_conditions` 并扩展目标类型**

修改 `app.py` 第 5726-5744 行：

```python
def _derive_node_conditions(graph_data: dict):
    """Patch VLLM/Calc node properties.condition from incoming links.

    The front-end ``condition`` property may be empty when the graph was
    last saved, but the YAML derive step already reads it from link
    topology.  This function brings the JSON property in sync so the
    property panel shows the correct value.
    """
    if not graph_data or 'links' not in graph_data or 'nodes' not in graph_data:
        return
    node_map = {n.get('id'): n for n in graph_data['nodes']}
    TARGET_TYPES = ('vllm', 'calc')
    for link in graph_data['links']:
        if isinstance(link, (list, tuple)):
            link = {'origin_id': link[1], 'target_id': link[3]}
        dst = node_map.get(link.get('target_id'))
        dst_type = (dst.get('type', '') or '').lower() if dst else ''
        if any(t in dst_type for t in TARGET_TYPES):
            src = node_map.get(link.get('origin_id'))
            if src and 'condition' in (src.get('type', '') or '').lower():
                dst.setdefault('properties', {})['condition'] = str(src['id'])
```

- [ ] **Step 2: 更新所有调用点**

修改 `app.py` 第 5722 行（`wf_get` 路由内）：

```python
        _derive_node_conditions(graph_data)
```

修改 `app.py` 第 5758 行（`wf_save` 路由内）：

```python
    _derive_node_conditions(graph_data)
```

---

### Task 3: 后端 — `_litegraph_to_workflow` 增加 calc condition 处理

**Files:**
- Modify: `app.py:5443-5447`
- Modify: `app.py:5488-5489`

- [ ] **Step 1: calc 节点序列化增加 condition 字段**

修改 `app.py` 第 5443-5447 行：

```python
        elif _type == "calc":
            cfg["expression"] = props.get("expression", "")
            cfg["condition"] = props.get("condition", "")
            cfg["params"] = {
                "output_field": props.get("output_field", "computed"),
            }
```

- [ ] **Step 2: condition→calc 连线设置 condition 门控**

修改 `app.py` 第 5488-5489 行，将原先的简单 `_add_source` 替换为与 VLLM 分支逻辑一致的条件判断：

```python
        if _dst_type == "calc":
            src_node_obj = next((n for n in graph_json.get("nodes", []) if n.get("id") == link.get("origin_id")), None)
            _src_type = src_node_obj.get("type", "").lower().replace("node", "").split("/")[-1] if src_node_obj else ""
            # Link from condition → calc sets the condition gate
            if _src_type == "condition":
                cond_node = next((n for n in nodes if n["id"] == str(src_node.get("id"))), None)
                if cond_node:
                    dst_calc = next((n for n in nodes if n["id"] == str(dst_node.get("id"))), None)
                    if dst_calc:
                        dst_calc["condition"] = cond_node["id"]
            # Other incoming links (input/yolo) add as source dependency for DAG ordering
            else:
                _add_source(str(dst_node.get("id")), str(src_node.get("id")), nodes)
```

---

### Task 4: 后端 — DAG 构建增加 calc condition 依赖

**Files:**
- Modify: `deploy/pipeline_manager.py:154-166`

- [ ] **Step 1: `_build_dag` 中 calc 节点添加 condition 依赖**

修改 `deploy/pipeline_manager.py` 第 159-161 行：

```python
        for node in nodes:
            if node.type in (NodeType.VLLM, NodeType.CALC) and node.condition:
                dag[node.id].append(node.condition)
            if node.source:
                for src in node.source:
                    if src in node_map:
                        dag[node.id].append(src)
```

---

### Task 5: 后端 — `_exec_calc` 增加 condition 门控检查

**Files:**
- Modify: `deploy/pipeline_manager.py:447-499`

- [ ] **Step 1: `_exec_calc` 函数开头添加 condition 门控逻辑**

修改 `deploy/pipeline_manager.py` 第 447-454 行，在 expression 读取之前插入门控检查：

```python
    async def _exec_calc(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
        """执行计算表达式，对每个检测结果求值并追加字段。"""
        # Condition gate check — 与 VLLM 节点行为一致
        if node.condition:
            gate = ctx.branch_conditions.get(node.condition, False)
            if not gate:
                return {"calc_result": None, "skipped": True}

        expression = node.expression or node.params.get('expression', '')
        output_field = node.params.get('output_field', 'computed')
        # ... 后续代码保持不变
```

---

### Task 6: 端到端验证

- [ ] **Step 1: 构造测试工作流**

创建一个包含以下节点的工作流：
1. Input 节点
2. YOLO 节点（连接 Input）
3. Condition 节点（连接 YOLO，表达式 `detection_count > 0`）
4. Calc 节点（连接 YOLO，contition 连线来自 3 的 trigger）
5. Output 节点（连接 Calc）

- [ ] **Step 2: 序列化测试 — 检查 YAML 输出**

通过编辑器保存，检查生成的 YAML 中 calc 节点包含 `condition: <cond_node_id>`。

- [ ] **Step 3: 执行测试 — 条件满足**

运行 `detection_count > 0`（图片中有检测结果），确认 CalcNode 正常执行，结果中包含计算的字段。

- [ ] **Step 4: 执行测试 — 条件不满足**

运行 `detection_count > 100`（没有这么多检测），确认 CalcNode 被跳过，返回 `skipped: True`。
