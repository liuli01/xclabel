## Context

Smart Pipeline 升级后，所有 YOLO 推理已统一使用 ultralytics YOLO adapter。nndeploy 作为 ONNX 推理的回退路径仅在 `.pt`/`.engine` 文件不存在时才被触发，实际已不再需要。移除后能消除 ONNX 精度偏差、减少维护成本。

## Goals / Non-Goals

**Goals:**
- 删除 `deploy/nndeploy_adapter.py` 文件
- 删除所有 nndeploy 的 import 和引用
- 从 requirements.txt 移除 nndeploy 包
- 简化 `/load/model` 逻辑，只保留 ultralytics 路径

**Non-Goals:**
- 不修改 EnginePool 逻辑
- 不修改 yolo_adapter / pipeline_manager / vllm_client
- 不修改 server 端代码

## Decisions

| 决策 | 选择 | 理由 |
|------|------|------|
| nndeploy_adapter.py 处理 | 直接删除 | 无其他依赖引用该模块 |
| load/model 回退逻辑 | 删除 ONNX 回退，model 加载失败直接报错 | 简化，用户只需提供 .pt/.engine |
| Dockerfile | 不修改 | nndeploy 安装行保留无害，移除可后续清理 |

## Risks / Trade-offs

- 如果已有 workflow 依赖 ONNX 模型路径 → 需切换为 .pt 路径（轻微，因为 ultralytics 同样支持 .onnx 文件加载）
- 无回退路径 → 如果 .pt/.engine 文件缺失，直接报错，不再尝试 ONNX
