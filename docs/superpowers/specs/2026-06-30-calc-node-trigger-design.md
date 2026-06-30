# 计算节点 Trigger 机制设计

## 概述

为 CalcNode（计算节点）添加 condition/trigger 门控机制，使 Condition 节点的 boolean 输出能连接到 CalcNode 的 trigger 输入，CalcNode 根据 condition 结果决定是否执行。

## 背景

LLM 节点（VllmNode）已有完整的 trigger 机制。本设计将该模式完整复用到 CalcNode。

当前状态：

- CalcNode 只有 `addInput('in', 'array')` 一个输入
- Condition 节点输出类型为 `boolean`，无法连接到 CalcNode 的 `array` 输入
- CalcNode 没有任何条件门控，每次 DAG 到达时必然执行

## 设计

### 改动文件清单

| 文件 | 改动类型 |
|------|----------|
| `templates/workflow.html` | 前端 CalcNode 类 + 属性面板 |
| `app.py` | `_derive_vllm_conditions` 扩展 + `_litegraph_to_workflow` 链接处理 |
| `deploy/pipeline_manager.py` | `_build_dag` + `_exec_calc` 条件门控 |

### 1. 前端 CalcNode 类

```javascript
function CalcNode(){
    this.properties = {
        expression: '(width * height) / 100',
        output_field: 'computed',
        condition: ''          // 条件节点 ID
    };
    this.addInput('in', 'array');
    this.addInput('trigger', 'boolean');  // 新增
    this.addOutput('out', 'array');
    this.size = [200, 90];
}
```

属性面板增加 `condition` 字段输入框，与 LLM 节点一致。

### 2. 属性面板

calc 分支增加 condition 字段：

```html
html += fieldHtml('text', 'condition', p.condition || '', '条件节点 ID');
```

### 3. 链接条件自动推导

`_derive_vllm_conditions` 重命名为 `_derive_node_conditions`，目标节点类型扩展到 `vllm` + `calc`。

当 condition 节点连线到 calc 节点的 trigger 输入时，自动填充 `calc.properties.condition = cond_node_id`。

### 4. YAML 转换 — 链接处理

`_litegraph_to_workflow` 中，condition→calc 的连接设置 `dst_calc["condition"]` 而不是 `_add_source`，与 condition→vllm 逻辑一致。

calc 节点序列化时写入 `condition` 字段：

```python
cfg["condition"] = props.get("condition", "")
```

### 5. DAG 构建

`_build_dag` 中 calc 节点有 `condition` 时，将 condition 节点加入依赖列表：

```python
if node.type in (NodeType.VLLM, NodeType.CALC) and node.condition:
    dag[node.id].append(node.condition)
```

### 6. Calc 执行 — 条件门控

`_exec_calc` 执行前检查 condition：

```python
if node.condition:
    gate = ctx.branch_conditions.get(node.condition, False)
    if not gate:
        return {"calc_result": None, "skipped": True}
```

跳过行为与 LLM 节点一致：返回 `skipped=True`，不修改 detections。

## 数据流

```
Condition 节点
  │ boolean output ───→ CalcNode.trigger input
  │                    (自动设置 condition 属性)
  ▼
PipelineContext.branch_conditions[node_id] = True/False
  │
  ▼
_exec_calc 执行前检查:
  ├─ condition=True  → 正常执行计算
  └─ condition=False → 跳过，返回 skipped=True
```

## 兼容性

- 现有无 condition 的 CalcNode 不受影响
- `condition` 为空字符串时跳过门控检查
- YAML 格式兼容：`condition` 字段已在 NodeConfig 中定义为 `Optional[str]`
