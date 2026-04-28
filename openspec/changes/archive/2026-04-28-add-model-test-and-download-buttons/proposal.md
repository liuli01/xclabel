## Why

目前用户在工程管理页面只能进入标注或训练，缺少一个快速验证模型效果的入口。新增模型测试功能可以让用户直接选择已安装的 YOLO 模型、上传单张图片并实时查看推理结果（画框/关键点/分割等），显著提升调试效率。同时新增模型下载按钮，方便用户快速进入模型管理界面获取预训练权重。

## What Changes

- 在 `templates/projects.html` 的工程卡片操作区新增「模型测试」和「模型下载」两个按钮
- 新增模型测试弹窗：支持选择已安装的 YOLO 模型（按版本和任务类型筛选）、上传图片、执行推理
- 新增后端 `/api/model-test/infer` API：加载指定 YOLO 模型，对上传图片执行推理，返回检测结果（bbox/segment/pose/keypoints/obb/classify）
- 新增前端结果渲染：在弹窗内显示原图叠加检测结果的 canvas，支持切换显示/隐藏标注
- 模型下载按钮点击后跳转/打开设置面板中的模型下载区域

## Capabilities

### New Capabilities
- `model-inference-testing`: 模型推理测试能力，包括模型选择、图片上传、推理执行、结果可视化

### Modified Capabilities
- `project-management`: 工程管理页面的工程卡片操作区新增「模型测试」和「模型下载」按钮

## Impact

- **前端**: `templates/projects.html`（新增按钮和弹窗）、`static/script.js`（弹窗交互、图片上传、canvas 渲染）
- **后端**: `app.py`（新增 `/api/model-test/infer` API，调用 ultralytics YOLO 推理）
- **依赖**: 依赖已安装的 ultralytics 环境（YOLOv8/11/26）
