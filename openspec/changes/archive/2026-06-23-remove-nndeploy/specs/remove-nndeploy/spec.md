## REMOVED Requirements

### Requirement: nndeploy ONNX 模型加载
**Reason**: 所有 YOLO 推理已统一使用 ultralytics YOLO adapter，nndeploy ONNX 加载路径不再需要
**Migration**: 使用 ultralytics 加载 .pt/.engine/.onnx 文件替代

### Requirement: nndeploy ONNX 推理执行
**Reason**: nndeploy 推理输出与 Ultralytics 结果不一致（置信度、坐标、mask均有偏差）
**Migration**: 使用 YoloAdapter / yolo_adapter.py 替代

### Requirement: nndeploy workflow 执行
**Reason**: Smart Pipeline 的 PipelineManager 已完全替代 nndeploy workflow 功能
**Migration**: 使用 PipelineManager / pipeline_manager.py 替代
