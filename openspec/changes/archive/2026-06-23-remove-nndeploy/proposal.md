## Why

xclabel-deploy 已完成 PipelineManager 智能流水线升级，所有 YOLO 模型推理已统一通过 ultralytics `predict()` 执行。nndeploy 作为 ONNX 推理的后备方案，存在以下问题：ONNX 输出解析与 Ultralytics 不一致（置信度、bbox坐标、mask多边形均有偏差）；维护两套推理路径增加复杂度；nndeploy 库更新缓慢。移除 nndeploy 可简化部署栈，消除精度偏差风险。

## What Changes

1. **删除 `deploy/nndeploy_adapter.py`** — nndeploy 模型加载和推理适配器
2. **删除 `deploy/main.py` 中 nndeploy 相关代码** — NndeployAdapter 初始化、health 状态、异常处理分支
3. **`deploy/requirements.txt` 移除 nndeploy 依赖**
4. **`deploy/main.py` 简化 `/load/model` 逻辑** — 只保留 ultralytics 加载路径，删除 ONNX 回退
5. **保持 EnginePool 不变** — 引擎池管理逻辑与推理后端解耦，无需修改

## Capabilities

### New Capabilities
- *(无，本次为清理变更)*

### Modified Capabilities
- *(无，不修改 spec 级别行为)*

## Impact

- **移除文件**: `deploy/nndeploy_adapter.py`
- **修改文件**: `deploy/main.py`（移除 nndeploy 引用和回退逻辑）、`deploy/requirements.txt`（移除 nndeploy）
- **无影响**: `deploy/engine_pool.py`、`deploy/yolo_adapter.py`、`deploy/pipeline_manager.py`、`deploy/vllm_client.py`
