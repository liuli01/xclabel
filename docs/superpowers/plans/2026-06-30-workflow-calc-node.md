# Workflow 计算节点 (CalcNode) 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在工作流系统中新增"计算（Calc）"节点，支持数学表达式求值，将计算结果追加到检测数据中。

**Architecture:** 
- 后端：在 `PipelineManager` 中新增 `CALC` 节点类型和表达式安全求值引擎
- 前端：在 LiteGraph 编辑器中新增 `CalcNode` 节点类，调色板入口，属性面板
- 数据流：YOLO → Calc → Condition/Output，每个检测逐条计算

**Tech Stack:** Python (Flask), JavaScript (LiteGraph.js), YAML

---

### Task 1: 后端 — 节点枚举与执行引擎

**Files:**
- Modify: `deploy/pipeline_manager.py`

- [ ] **Step 1: NodeType 枚举新增 CALC**

在 `deploy/pipeline_manager.py:24-29`，在 `OUTPUT = "output"` 后添加：

```python
class NodeType(str, Enum):
    INPUT = "input"
    YOLO = "yolo"
    CONDITION = "condition"
    VLLM = "vllm"
    OUTPUT = "output"
    CALC = "calc"       # ← 新增
```

- [ ] **Step 2: NodeConfig 添加 calc 相关字段**

此时无需新增字段，calc 节点复用已有的 `expression` 字段和 `params` 字典：
- `expression` — 数学表达式字符串
- `params.output_field` — 计算结果字段名，默认 `"computed"`

- [ ] **Step 3: 实现安全表达式求值方法**

在 `PipelineManager` 类中新增 `_eval_expression()` 和辅助方法，放在 `_exec_condition` 方法之后：

```python
    # ── Calc node ──

    @staticmethod
    def _eval_expression(expression: str, vars: dict) -> float:
        """安全求值数学表达式，仅允许白名单函数和预置变量。"""
        import ast
        import math

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
    def _calc_area(bbox, points) -> float:
        """计算面积。有 segmentation points 时用鞋带公式，否则用 bbox 面积。"""
        if points and len(points) >= 6:
            # Shoelace formula for polygon area
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
```

注意：需要在文件顶部加 `import math`（若不存在）。

- [ ] **Step 4: 实现 `_exec_calc()` 执行方法**

在 `_exec_output()` 方法之前新增：

```python
    async def _exec_calc(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
        """执行计算表达式，对每个检测结果求值并追加字段。"""
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
```

- [ ] **Step 5: `_run_node()` 调度器增加 calc 分支**

在 `deploy/pipeline_manager.py:234-247`，在 `OUTPUT` 分支后添加：

```python
        elif node.type == NodeType.CALC:
            return await self._exec_calc(node, ctx)
```

- [ ] **Step 6: 验证后端改动**

```bash
cd e:/coding/project/xclabel && python -c "
from deploy.pipeline_manager import PipelineManager, NodeType
import math

# 验证 NodeType 枚举
assert NodeType.CALC == 'calc'
print('✓ NodeType.CALC =', NodeType.CALC)

# 验证 _calc_area bbox
area = PipelineManager._calc_area([0, 0, 100, 200], [])
assert area == 20000.0, f'bbox area should be 20000, got {area}'
print('✓ bbox area =', area)

# 验证 _calc_area polygon (4x4 square)
area_poly = PipelineManager._calc_area([0,0,4,4], [0,0,4,0,4,4,0,4])
assert area_poly == 16.0, f'polygon area should be 16, got {area_poly}'
print('✓ polygon area =', area_poly)

# 验证 _eval_expression
result = PipelineManager._eval_expression('(width * height) / 100', {'width': 200, 'height': 300})
assert result == 600.0, f'should be 600, got {result}'
print('✓ expression (width * height) / 100 =', result)

# 验证安全限制
try:
    PipelineManager._eval_expression('__import__("os")', {})
    assert False, 'should have raised'
except NameError:
    print('✓ unsafe variable blocked')

print('\\n✅ All backend checks passed')
"
```
预期输出：全部验证通过。


### Task 2: 后端 — YAML 转换

**Files:**
- Modify: `app.py`

- [ ] **Step 1: `_litegraph_to_workflow()` 增加 calc 类型转换**

在 `app.py:5436`（output 分支后），新增 calc 分支：

```python
        elif _type == "calc":
            cfg["expression"] = props.get("expression", "")
            cfg["params"] = {
                "output_field": props.get("output_field", "computed"),
            }
```

- [ ] **Step 2: 增加 calc 节点的链接处理**

在 `app.py:5472-5477`，在 `_dst_type == "condition"` 分支后增加 calc 的 source 处理：

```python
        if _dst_type == "calc":
            _add_source(str(dst_node.get("id")), str(src_node.get("id")), nodes)
```

- [ ] **Step 3: 验证 YAML 转换**

```bash
cd e:/coding/project/xclabel && python -c "
from app import _litegraph_to_workflow

graph = {
    'title': 'test_calc',
    'nodes': [
        {'id': 1, 'type': 'xclabel/input', 'properties': {'input_type': 'upload'}},
        {'id': 2, 'type': 'xclabel/yolo', 'properties': {'model': 'test', 'task': 'segment', 'conf': 0.25, 'iou': 0.5}},
        {'id': 3, 'type': 'xclabel/calc', 'properties': {'expression': '(width * height) / 100', 'output_field': 'area_scaled'}},
        {'id': 4, 'type': 'xclabel/output', 'properties': {}},
    ],
    'links': [
        [1, 1, 0, 2, 0, 'image'],
        [2, 2, 0, 3, 0, 'array'],
        [3, 3, 0, 4, 0, 'array'],
    ]
}

result = _litegraph_to_workflow(graph)
calc_node = next(n for n in result['pipeline'] if n['type'] == 'calc')
assert calc_node['expression'] == '(width * height) / 100', f'expression mismatch'
assert calc_node['params']['output_field'] == 'area_scaled', f'output_field mismatch'
print('✓ calc node YAML conversion:', calc_node)
print('\\n✅ YAML conversion passed')
"
```


### Task 3: 前端 — CalcNode 节点类与注册

**Files:**
- Modify: `templates/workflow.html`

- [ ] **Step 1: 新增 CalcNode LiteGraph 节点类**

在 `templates/workflow.html` 中 `OutputNode` 定义之后（约 350 行），添加：

```javascript
// Calc Node
function CalcNode(){
this.properties = {
expression: '(width * height) / 100',
output_field: 'computed'
};
this.addInput('in', 'array');
this.addOutput('out', 'array');
this.size = this.computeSize();
}
CalcNode.title = '计算';
CalcNode.desc = '数学表达式计算';
CalcNode.prototype.onNodeSelected = function(){ showProperties(this); };
CalcNode.prototype.onDeselected = function(){ hideProperties(); };
CalcNode.prototype.onDrawBackground = function(ctx){
if(this.properties.expression){
ctx.fillStyle = '#aaa';
ctx.font = '11px monospace';
var text = this.properties.expression.length > 22
? this.properties.expression.substring(0, 22) + '...'
: this.properties.expression;
ctx.fillText(text, 10, this.size[1] - 10);
}
};
```

- [ ] **Step 2: 注册 CalcNode**

在 `templates/workflow.html` 约 376 行（注册语句区域），增加：

```javascript
LiteGraph.registerNodeType('xclabel/calc', CalcNode);
```

以及向后兼容的无前缀注册：

```javascript
LiteGraph.registerNodeType('calc', CalcNode);
```

- [ ] **Step 3: 调色板增加"计算"节点条目**

在 `templates/workflow.html` 约 494-499 行（palette-sidebar 的 innerHTML），在 Output 项前增加：

```html
'<div class="palette-item" data-type="xclabel/calc" draggable="true"><i class="fas fa-calculator"></i>计算<span class="badge">表达式</span></div>' +
```

注意保持与上下文一致的字符串拼接语法。最终 palette 区域变为：

```javascript
palette.innerHTML =
'<h4>节点</h4>' +
'<div class="palette-item" data-type="xclabel/input" draggable="true"><i class="fas fa-cloud-upload-alt"></i>Input<span class="badge">上传/URL/流</span></div>' +
'<div class="palette-item" data-type="xclabel/yolo" draggable="true"><i class="fas fa-eye"></i>YOLO<span class="badge">检测/分割</span></div>' +
'<div class="palette-item" data-type="xclabel/condition" draggable="true"><i class="fas fa-code-branch"></i>条件<span class="badge">分支</span></div>' +
'<div class="palette-item" data-type="xclabel/calc" draggable="true"><i class="fas fa-calculator"></i>计算<span class="badge">表达式</span></div>' +
'<div class="palette-item" data-type="xclabel/vllm" draggable="true"><i class="fas fa-brain"></i>VLLM<span class="badge">分析</span></div>' +
'<div class="palette-item" data-type="xclabel/output" draggable="true"><i class="fas fa-file-export"></i>输出<span class="badge">合并</span></div>';
```


### Task 4: 前端 — 属性面板与 YAML 预览

**Files:**
- Modify: `templates/workflow.html`

- [ ] **Step 1: `showProperties()` 增加 calc 分支**

在 `templates/workflow.html` 约 628 行（output 分支后，content.innerHTML 赋值前），增加：

```javascript
else if(type === 'xclabel/calc' || type === 'calc'){
html += '<div class="field"><label>expression</label><textarea onchange="setProp(\'expression\',this.value)" placeholder="(width * height) / 100" style="font-family:Consolas,monospace;min-height:60px">'+(p.expression||'')+'</textarea></div>';
html += fieldHtml('text', 'output_field', p.output_field || 'computed', '结果字段名');
html += '<div class="help" style="padding:0 12px;margin:-4px 0 6px;line-height:1.6">可用变量: conf, class_id, class_name, x1, y1, x2, y2, width, height, area, perimeter<br>可用函数: abs, round, sqrt, ceil, floor, min, max, pi</div>';
}
```

- [ ] **Step 2: `updateYamlPreview()` 增加 calc 分支**

在 `templates/workflow.html` 约 896 行（vllm 分支后，output 分支前），增加：

```javascript
else if(t === 'calc'){
y += '    expression: "' + (p.expression || '').replace(/"/g, '\\"') + '"\n';
y += '    params:\n      output_field: "' + (p.output_field || 'computed') + '"\n';
}
```

- [ ] **Step 3: `formatNodeType()` 增加 calc 映射**

在 `templates/workflow.html` 约 1202 行，增加映射：

```javascript
var map = {input:'输入', yolo:'YOLO检测', condition:'条件判断', calc:'计算', vllm:'VLM分析', output:'结果输出'};
```

- [ ] **Step 4: 结果面板增加 calc 节点展示**

在 `renderResultPanel()` 函数（约 763 行，condition 分支后），为 calc 类型增加显示：

```javascript
if(ntype === 'calc'){
  detailHtml += '<div style="color:#888">表达式: <code style="color:#ce93d8">'+(out.expression || '')+'</code></div>';
  detailHtml += '<div style="color:#888">输出字段: <b style="color:#e0e0e0">'+(out.output_field || 'computed')+'</b></div>';
  if(out.computed_values && out.computed_values.length){
    detailHtml += '<div style="color:#888;margin:3px 0">计算结果:</div>';
    out.computed_values.slice(0, 15).forEach(function(v, i){
      detailHtml += '<div class="det-row">#'+(i+1)+': <b style="color:#ffa726">'+v+'</b></div>';
    });
    if(out.computed_values.length > 15){
      detailHtml += '<div style="color:#888">... 还有 '+(out.computed_values.length-15)+' 个</div>';
    }
  }
}
```

同时更新 `renderResultPanel()` 内的 `typeIcons` 映射（约 725 行），增加 calc：

```javascript
var typeIcons = {input:'fa-cloud-upload-alt', yolo:'fa-eye', condition:'fa-code-branch', calc:'fa-calculator', vllm:'fa-brain', output:'fa-file-export'};
```


### Task 5: 端到端验证

- [ ] **Step 1: 启动应用**

```bash
cd e:/coding/project/xclabel && python app.py
```

- [ ] **Step 2: 打开浏览器访问工作流编辑器**
打开 `http://localhost:5000/workflow`

验证以下内容：
1. 调色板中出现"计算"节点（带 ∑ 图标）
2. 拖拽到画布，节点显示标题"计算"
3. 选中节点，右侧属性面板显示 expression 文本域和 output_field 输入框
4. 属性面板显示变量/函数提示

- [ ] **Step 3: 构建一个完整工作流并运行**
1. 添加 Input 节点（上传模式）
2. 添加 YOLO 节点（连接 Input 的 image 输出）
3. 添加计算节点（连接 YOLO 的 detections 输出）
4. 设置表达式 `(width * height) / 100`，输出字段 `area_scaled`
5. 添加 Output 节点（连接计算的 out 输出）
6. 保存工作流
7. 上传图片执行

预期结果：
- 执行日志显示计算节点完成
- 结果面板显示计算节点的表达式和计算结果列表
- YOLO 检测结果中每个框都有一个 `area_scaled` 字段
- 结果面板切换查看原始 JSON 确认 `area_scaled` 存在

- [ ] **Step 4: 验证错误处理**
1. 设置非法表达式如 `unknown_var + 1`
2. 执行，预期：计算节点报错但仍然完成流程（其他节点正常）
3. 设置空表达式，预期：计算节点返回空结果，不报错
