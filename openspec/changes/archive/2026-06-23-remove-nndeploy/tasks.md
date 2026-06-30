## 1. 移除 nndeploy 代码

- [x] 1.1 删除 `deploy/nndeploy_adapter.py` 文件
- [x] 1.2 `deploy/main.py` 移除 NndeployAdapter 的 import 和初始化
- [x] 1.3 `deploy/main.py` 精简 load_model：删除 ONNX 回退逻辑，只保留 ultralytics
- [x] 1.4 `deploy/main.py` 精简 `/infer` 端点：移除 nndeploy 推理分支
- [x] 1.5 `deploy/main.py` 移除 load/workflow 中对 nndeploy 的依赖（已有新 pipeline 端点）
- [x] 1.6 `deploy/main.py` health 端点移除 nndeploy_available 字段

## 2. 清理依赖

- [x] 2.1 `deploy/requirements.txt` 移除 nndeploy

## 3. 验证

- [x] 3.1 启动 deploy 服务，确认无 nndeploy 报错
- [x] 3.2 加载 YOLO 模型并推理，确认结果正确
- [x] 3.3 执行 pipeline 端到端测试
