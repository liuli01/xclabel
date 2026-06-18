## Purpose

将 VLM 视觉语言模型的物体检测能力与 SAM 2 的精细化分割能力串联，实现「VLM 识别类别 → SAM 精细化分割」的一键批量标注管道。

## ADDED Requirements

### Requirement: VLM→SAM 精细化标注管道

系统 SHALL 提供 VLM 检测 + SAM 2 精细化分割的串联管道，支持批量自动标注。

#### Scenario: 成功运行 VLM→SAM 管道

- **WHEN** 用户在 AI 模态框选择「VLM+SAM 精细化」模式，选中图片，点击「开始执行」
- **THEN** 系统对每张图片依次执行：VLM 推理获取检测框 → SAM 2 用 bbox 做 Box Prompt 生成 Mask → 掩码转多边形 → 保存标注
- **THEN** 前端通过 SocketIO 接收每张图的进度推送（状态：vlm/sam/done）
- **THEN** 完成后弹窗提示处理结果

#### Scenario: VLM 无检测结果

- **WHEN** VLM 对某张图返回空检测列表
- **THEN** 跳过该图，继续处理下一张
- **THEN** 进度中标记该图无检测

#### Scenario: SAM 2 推理失败

- **WHEN** SAM 2 对某个 bbox 生成空 mask
- **THEN** 降级为该 bbox 的原始矩形框标注，标注来源标记为 vlm_sam_fallback

#### Scenario: GPU 锁被占用

- **WHEN** GPU 正被 YOLO 训练或其他任务占用
- **THEN** 返回错误提示「GPU 正忙，请稍后重试」

### Requirement: AI 模态框新增模式

AI 标注模态框 SHALL 提供「VLM+SAM 精细化」模式选项。

#### Scenario: 模式切换

- **WHEN** 用户在 AI 模态框中选择「VLM+SAM 精细化」
- **THEN** 显示 VLM API 配置区域（用于查看/修改提示词）
- **THEN** YOLO 配置区域隐藏

#### Scenario: 执行 VLM+SAM 标注

- **WHEN** 用户点击「开始执行」，且当前为 VLM+SAM 模式
- **THEN** 调用 `POST /api/auto-label/vlm-sam`
- **THEN** 进度区域显示每张图的处理状态
- **THEN** 完成后刷新标注列表和图片列表

### Requirement: SocketIO 进度推送

系统 SHALL 在 VLM→SAM 管道处理过程中实时推送进度。

#### Scenario: 进度事件

- **WHEN** VLM→SAM 管道处理每张图时
- **THEN** 后端推送 `vlm_sam_progress` 事件，包含当前序号、总数、图片名、状态（vlm/sam/done）、检测数
