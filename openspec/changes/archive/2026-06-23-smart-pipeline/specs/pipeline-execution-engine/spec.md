## ADDED Requirements

### Requirement: PipelineManager 加载并解析 workflow.yaml
系统 SHALL 支持从文件路径加载 workflow.yaml，并使用 Pydantic 校验其结构完整性。
系统 SHALL 在加载成功后构建节点 DAG，检测循环依赖。

#### Scenario: 加载有效 workflow.yaml
- **WHEN** 提供有效的 workflow.yaml 路径
- **THEN** 系统成功解析 YAML，返回节点列表和拓扑序

#### Scenario: 检测循环依赖
- **WHEN** workflow.yaml 中的节点链接形成环
- **THEN** 系统抛出明确的循环依赖错误

### Requirement: PipelineManager 按拓扑序执行节点
系统 SHALL 按 DAG 拓扑排序结果依次执行节点，每个节点接收上一个节点的输出作为上下文输入。
节点间数据通过 PipelineContext 传递，包含 image、detections、vllm_result、max_conf 等字段。

#### Scenario: 正常执行线性流水线
- **WHEN** 执行 YOLO→Condition→Output 流水线
- **THEN** 各节点按序执行，最终输出合并结果

### Requirement: PipelineManager 处理节点执行异常
系统 SHALL 在 YOLO/VLLM 节点执行失败时跳过该节点并记录错误，不影响后续节点执行。

#### Scenario: VLLM 节点超时
- **WHEN** VLLM 节点 30 秒无响应
- **THEN** 系统跳过 VLLM 节点，返回 YOLO 结果并在输出中包含 warning 信息
