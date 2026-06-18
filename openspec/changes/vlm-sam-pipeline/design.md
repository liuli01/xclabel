## Context

当前 xclabel 已有 VLM AI 标注功能（通过 AiUtils.py 调用 LMStudio/Ollama/OpenAI 等外部 API）和 SAM 2 交互式分割功能（ai_manager.py），但两者相互独立。用户标注时需要先在 AI 模态框跑 VLM 出矩形框，再手动用 SAM 修成多边形，无法一键完成。

## Goals / Non-Goals

**Goals:**
- 新增一条 VLM→SAM 的自动化管道：VLM 检测出 label + bbox → SAM 2 用 bbox 做 Box Prompt 生成分割 Mask → 自动保存多边形标注
- AI 模态框新增「VLM+SAM 精细化」模式，与现有的「VLM API 标注」「YOLO 自动检测」并列

**Non-Goals:**
- 不修改现有的 VLM AI 标注逻辑
- 不修改 SAM 2 交互式分割工具
- 不新增外部依赖

## Decisions

| 决策 | 选择 | 理由 |
|------|------|------|
| 新路由 vs 改现有路由 | 新路由 `/api/auto-label/vlm-sam` | 现有路由已耦合 VLM+保存逻辑，加 SAM 会变复杂；三种模式对应三个独立路由更清晰 |
| GPU 锁策略 | VLM 推理 + SAM 推理连续占用 GPU | VLM 调用外部 API 实际不占 GPU，但为扩展性统一管理 |
| SAM 降级策略 | SAM 失败时降级为矩形框 | 不影响整体流程，用户后面可手动修正 |
| 进度推送 | SocketIO 事件 `vlm_sam_progress` | 复用现有 SocketIO 机制 |

## Risks / Trade-offs

- [性能] 每张图先跑 VLM（网络延迟）再跑 SAM（本地推理），单张耗时较长 → 前端进度推送保持透明，用户可看到每张图的状态
- [VLM 不稳定] 外部 API 可能超时或返回空结果 → 自动跳过该图继续下一张
- [SAM 精度] 对极小物体或遮挡严重的 bbox，SAM 可能生成空 mask → 降级为矩形框
