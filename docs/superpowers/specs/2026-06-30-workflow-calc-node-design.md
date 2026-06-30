---
title: Workflow 计算节点（CalcNode）设计
date: 2026-06-30
status: draft
---

# Workflow 计算节点（CalcNode）设计

## 1. 动机

当前 xclabel 工作流支持 YOLO 检测 → Condition 条件判断 → VLLM 分析 → Output 输出的流水线，但缺少对检测结果做**数学计算**的能力。用户在标注场景中常需要对检测框/分割多边形的尺寸做运算（缩放面积、计算长宽比、归一化等），以便后续条件判断或输出展示。

目标：新增一个通用**表达式计算节点**，对上游节点的每个检测结果执行数学表达式求值，将计算结果追加为新的检测字段，供下游节点使用。

## 2. 设计原则

- **遵从现有架构**：与已有 5 种节点相同的注册模式、执行调度、YAML 转换流程
- **安全求值**：表达式在受限命名空间中执行，防止任意代码注入
- **通用性强**：预置语义变量覆盖 bbox 和 segmentation 两种模式，数学函数满足常见运算

## 3. 范围

### In Scope

| 模块 | 内容 |
|------|------|
| 后端枚举 & 执行 | `NodeType.CALC = "calc"`、`_exec_calc()` |
| 安全表达式引擎 | 受限 `math` 命名空间 + AST 解析 |
| 前端节点定义 | `CalcNode` LiteGraph 类、输入/输出/属性 |
| 属性面板 | expression 编辑框 + output_field 输入 |
| 节点面板 | 调色板新增"计算"节点图标 |
| YAML 转换 | `_litegraph_to_workflow()` 增加 calc 分支 |

### Out of Scope

- 聚合计算（汇总统计如 sum/avg 等多检测聚合）— 当前仅逐检测计算
- 条件分支逻辑（计算节点的结果不直接驱动分支，由下游 Condition 节点处理）
- 自定义 Python 脚本（仅表达式模式，不引入完整脚本执行）

## 4. 详细设计

### 4.1 节点定义

**名称**：计算（Calc）

**输入**：
- `in`（`array` 类型）— 上游检测结果列表

**输出**：
- `out`（`array` 类型）— 追加了计算字段的检测结果列表

**属性**：

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `expression` | text | `""` | 数学表达式，如 `(width * height) / 100` |
| `output_field` | text | `computed` | 计算结果存储的字段名 |

**节点外观**：
- 标题：`计算`
- 颜色：建议橙色系（区别于 YOLO 蓝、VLLM 紫、Condition 黄、Output 绿）
- 节点体展示 expression 摘要（截取前 30 字符）

### 4.2 预置语义变量

对每个检测结果（逐条），可用变量：

| 变量 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `conf` | float | detection.confidence | 置信度 0~1 |
| `class_id` | int | detection.class_id | 类别编号 |
| `class_name` | str | detection.class_name | 类别名称（可用于字符串比较） |
| `x1`, `y1` | float | detection.bbox[0..1] | 左上角坐标 |
| `x2`, `y2` | float | detection.bbox[2..3] | 右下角坐标 |
| `width` | float | `x2 - x1` | bbox 宽度 |
| `height` | float | `y2 - y1` | bbox 高度 |
| `area` | float | 存在 points 时用鞋带公式，否则 `width * height` | 面积 |
| `perimeter` | float | 存在 points 时用多边形周长，否则 `2*(width+height)` | 周长 |

**鞋带公式计算多边形面积**：

```
area = 0.5 * | Σ(xi * yi+1 - xi+1 * yi) |
```

其中 `(xi, yi)` 为多边形闭合点序列。

### 4.3 数学函数

安全命名空间提供以下函数和常量：

| 名称 | 来源 | 说明 |
|------|------|------|
| `abs(x)` | 内置 | 绝对值 |
| `round(x, n)` | 内置 | 四舍五入到 n 位小数 |
| `min(a, b)` | 内置 | 取小值 |
| `max(a, b)` | 内置 | 取大值 |
| `sqrt(x)` | `math.sqrt` | 平方根 |
| `ceil(x)` | `math.ceil` | 向上取整 |
| `floor(x)` | `math.floor` | 向下取整 |
| `pi` | `math.pi` | 圆周率 3.14159... |

### 4.4 示例表达式

| 场景 | 表达式 | output_field |
|------|--------|-------------|
| 面积缩放至百分比 | `(width * height) / 100` | `area_scaled` |
| 长宽比 | `round(width / height, 2)` | `aspect_ratio` |
| 归一化宽度 | `round(width / 1920, 4)` | `norm_width` |
| 置信度加权面积 | `area * conf` | `weighted_area` |
| 多边形面积 | `area`（自动鞋带公式） | `polygon_area` |
| 周长平方根 | `sqrt(perimeter)` | `sqrt_perimeter` |
| 面积阈值判断 | `max(area - 5000, 0)` | `area_above_threshold` |

### 4.5 YAML 格式

```yaml
- id: '3'
  type: calc
  expression: "(width * height) / 100"
  output_field: area_scaled
```

### 4.6 后端实现

#### 4.6.1 类型枚举

`deploy/pipeline_manager.py`：

```python
class NodeType(str, Enum):
    INPUT = "input"
    YOLO = "yolo"
    CONDITION = "condition"
    VLLM = "vllm"
    OUTPUT = "output"
    CALC = "calc"       # ← 新增
```

#### 4.6.2 安全表达式求值

在 `PipelineManager` 中新增方法：

```python
def _eval_expression(self, expression: str, vars: dict) -> float:
    """安全求值数学表达式"""
    import ast
    import math
    
    # 白名单函数
    SAFE = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sqrt': math.sqrt, 'ceil': math.ceil, 'floor': math.floor,
        'pi': math.pi,
    }
    
    # 验证表达式仅包含安全节点
    tree = ast.parse(expression, mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in vars and node.id not in SAFE:
                raise NameError(f"不允许的变量/函数: {node.id}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypeError("不支持复杂函数调用")
    
    namespace = {**SAFE, **vars}
    return eval(expression, {"__builtins__": {}}, namespace)
```

#### 4.6.3 计算执行

```python
async def _exec_calc(self, node: NodeConfig, ctx: PipelineContext) -> Dict:
    """执行计算表达式"""
    expression = node.params.get('expression', '')
    output_field = node.params.get('output_field', 'computed')
    
    if not expression:
        return {'detections': ctx.detections}
    
    computed_detections = []
    for det in (ctx.detections or []):
        bbox = det.get('bbox', [0, 0, 0, 0])
        points = det.get('points', [])
        
        vars = {
            'conf': det.get('confidence', 0),
            'class_id': det.get('class_id', 0),
            'class_name': det.get('class_name', ''),
            'x1': bbox[0], 'y1': bbox[1],
            'x2': bbox[2], 'y2': bbox[3],
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1],
        }
        vars['area'] = self._calc_area(bbox, points)
        vars['perimeter'] = self._calc_perimeter(bbox, points)
        
        result = self._eval_expression(expression, vars)
        det[output_field] = round(float(result), 6)
        computed_detections.append(det)
    
    return {'detections': computed_detections}
```

#### 4.6.4 调度器注册

```python
async def _run_node(self, node, ctx, ...):
    ...
    elif node.type == NodeType.CALC:
        return await self._exec_calc(node, ctx)
```

### 4.7 前端实现

#### 4.7.1 LiteGraph 节点类

```javascript
function CalcNode() {
  this.addInput('in', 'array');
  this.addOutput('out', 'array');
  
  this.properties = {
    expression: '',
    output_field: 'computed',
  };
  
  this.size = [180, 70];
}

CalcNode.title = '计算';
CalcNode.desc = '数学表达式计算';

CalcNode.prototype.onDrawBackground = function(ctx) {
  // 显示表达式摘要
  if (this.properties.expression) {
    ctx.fillStyle = '#aaa';
    ctx.font = '12px monospace';
    const text = this.properties.expression.length > 24
      ? this.properties.expression.substring(0, 24) + '...'
      : this.properties.expression;
    ctx.fillText(text, 10, 50);
  }
};

LiteGraph.registerNodeType('xclabel/calc', CalcNode);
```

#### 4.7.2 节点面板

在调色板侧边栏增加输入项：

```html
<div class="palette-item" draggable="true" data-node-type="xclabel/calc">
  <div class="palette-icon" style="background:#e67e22;">∑</div>
  <div class="palette-label">计算</div>
</div>
```

#### 4.7.3 属性面板

在 `showProperties()` 中新增 `calc` 分支：

- `expression`：多行文本域（textarea），等宽字体
- `output_field`：单行文本输入

提供变量提示区域（可折叠），展示可用变量名和函数。

#### 4.7.4 YAML 转换

`_litegraph_to_workflow()` 增加 `xclabel/calc` → `calc` 映射。

`updateYamlPreview()` 增加 calc 节点的 YAML 生成。

### 4.8 节点运行结果显示

在结果面板中，calc 节点的展示内容：

- 表达式原文
- 计算字段名
- 计算值列表（检测 1: 12.5, 检测 2: 8.3, ...），点击展开全部

## 5. 实施步骤

### Step 1: 后端 — 节点类型与执行

- `deploy/pipeline_manager.py`：
  - `NodeType` 枚举新增 `CALC`
  - 实现 `_eval_expression()` 安全求值
  - 实现 `_calc_area()` / `_calc_perimeter()` 辅助方法
  - 实现 `_exec_calc()` 方法
  - `_run_node()` 调度器增加 calc 分支
  - `_build_dag()` 检查是否有对 calc 节点的特殊连接处理（通常无需）

### Step 2: 前端 — 节点定义与注册

- `templates/workflow.html`：
  - `CalcNode` 构造函数 + 注册
  - 调色板面板增加"计算"节点条目
  - `showProperties()` 增加 calc 分支
  - `updateYamlPreview()` 增加 calc 支持

### Step 3: 后端 — YAML 转换

- `app.py` 中 `_litegraph_to_workflow()` 增加 `xclabel/calc` → `calc` 映射

### Step 4: 验证

- 本地运行工作流：Input → YOLO → Calc → Output
- 测试表达式：`(width * height) / 100`，验证结果追加到检测数据
- 测试条件分支：YOLO → Calc → Condition（使用计算后的字段做判断）
- 测试 segmentation 多边形面积计算
- 验证错误表达式提示（非法变量、语法错误）

## 6. 安全问题

- 表达式求值使用 AST 预检 + 限制命名空间，`__builtins__` 设为空字典
- 仅白名单函数可调用，禁止 `__import__`、`exec`、`eval`、`open` 等
- 变量名从预置语义变量中提取，不暴露 `ctx` 等内部对象
- 建议后续增加表达式超时保护（`signal.alarm` 或 `threading.Timer`）
