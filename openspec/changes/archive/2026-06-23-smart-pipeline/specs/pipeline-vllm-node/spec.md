## ADDED Requirements

### Requirement: VLLM 节点通过 OpenAI 兼容 API 调用大模型
VLLM 节点 SHALL 使用 AsyncOpenAI 客户端调用指定 api_url 的大模型。
请求参数（temperature、max_tokens）从 workflow.yaml 节点配置中读取。
请求超时 SHALL 为 30 秒。

#### Scenario: VLLM 节点调用成功
- **WHEN** VLLM 节点执行且 API 服务可用
- **THEN** 返回大模型回复内容并写入 ctx.vllm_result

#### Scenario: VLLM 节点超时
- **WHEN** VLLM 请求超过 30 秒
- **THEN** 跳过该节点，ctx.vllm_result 设为 None，输出包含 warning

### Requirement: ROI 裁剪（可选）
当 workflow.yaml 中 extract_roi 为 true 时，VLLM 节点 SHALL 使用 supervision 库将 YOLO 检测到的目标区域裁剪后发送给大模型。

#### Scenario: 启用 ROI 裁剪
- **WHEN** extract_roi=true 且有检测结果
- **THEN** 仅将裁剪后的目标区域图片发送给 VLLM，而非整图
