## ADDED Requirements

### Requirement: YOLO 节点复用 YoloAdapter 执行推理
YOLO 节点 SHALL 从引擎池中按 model 字段获取已加载的 YOLO 模型实例，调用 YoloAdapter.infer() 执行推理。
推理参数（conf、iou）从 workflow.yaml 节点配置中读取，覆盖默认值。

#### Scenario: YOLO 节点推理成功
- **WHEN** YOLO 节点执行，引擎池中存在指定模型
- **THEN** 返回检测结果，包含 class_name、confidence、bbox、points（seg 模型）

#### Scenario: YOLO 模型未加载
- **WHEN** YOLO 节点执行，引擎池中不存在指定模型
- **THEN** 系统尝试自动从 server 加载模型，失败则抛出错误

### Requirement: YOLO 节点输出 max_conf
YOLO 节点 SHALL 计算所有检测结果的最高置信度 max_conf 并写入 PipelineContext，供后续 Condition 节点使用。

#### Scenario: YOLO 输出 max_conf
- **WHEN** YOLO 节点完成推理且有检测结果
- **THEN** ctx.max_conf 被设为所有检测框中 confidence 的最大值
