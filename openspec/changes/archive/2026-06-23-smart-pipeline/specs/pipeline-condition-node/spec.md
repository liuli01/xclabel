## ADDED Requirements

### Requirement: Condition 节点基于表达式做分支判断
Condition 节点 SHALL 解析 expression 字段中的 Python 表达式，使用 PipelineContext 中的变量（max_conf、detection_count）作为输入。
表达式求值结果 SHALL 写入 ctx.branch_conditions[node.id]。
系统 SHALL 使用 sandbox 模式执行 eval，仅暴露白名单变量，禁用 __builtins__。

#### Scenario: 条件表达式为真
- **WHEN** expression="max_conf > 0.5" 且 ctx.max_conf=0.7
- **THEN** 条件判断结果为 true

#### Scenario: 条件表达式为假
- **WHEN** expression="max_conf > 0.5" 且 ctx.max_conf=0.3
- **THEN** 条件判断结果为 false

#### Scenario: 表达式语法错误
- **WHEN** expression 包含非法语法
- **THEN** 视为 false，继续执行后续节点
