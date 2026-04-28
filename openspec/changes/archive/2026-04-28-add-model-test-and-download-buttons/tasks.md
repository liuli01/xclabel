## 1. 前端 — 工程卡片新增按钮

- [x] 1.1 修改 `templates/projects.html` 工程卡片渲染逻辑：在操作区新增「模型测试」和「模型下载」按钮
- [x] 1.2 为新增按钮绑定点击事件：模型测试跳转至独立页面 `/model-test?project=<name>`、模型下载打开训练模型列表弹窗
- [x] 1.3 调整工程卡片按钮布局样式，确保 6 个按钮排布合理

## 2. 前端 — 模型测试页面 UI

- [x] 2.1 新建独立页面 `templates/model-test.html`，展示训练模型列表及测试工作区
- [x] 2.2 实现模型列表加载：调用 `/api/train/model-info` 获取已训练模型版本，展示任务/YOLO/基础模型/mAP 等指标
- [x] 2.3 点击模型进入测试工作区，自动带入该模型的任务类型和 YOLO 版本
- [x] 2.4 实现默认加载工程 test 数据集图片：页面加载时扫描 `projects/<project>/test/images/`，加载前 4 张作为默认测试样本
- [x] 2.5 实现图片上传功能：支持点击选择和拖拽上传自定义图片，预览上传图片
- [x] 2.6 实现「开始推理」按钮及加载状态展示
- [x] 2.7 实现「显示/隐藏标注」切换开关
- [x] 2.8 实现返回模型列表清理逻辑：清空 Canvas、推理结果、上传状态，重置工作区

## 3. 后端 — 模型推理 API

- [x] 3.1 在 `app.py` 中新增 `/model-test` 页面路由，渲染 `model-test.html`
- [x] 3.2 在 `app.py` 中新增 `/api/model-test/infer` POST API，接收模型路径、图片文件、任务类型
- [x] 3.3 实现推理逻辑：通过对应版本的虚拟环境 Python 调用 ultralytics YOLO 加载模型并推理
- [x] 3.4 实现任务类型适配：detect/segment/pose/obb/classify 分别返回对应格式的结果
- [x] 3.5 添加 GPU 资源冲突检测：推理前检查是否有正在运行的训练或 AI 标注任务
- [x] 3.6 添加环境校验：检查选中版本的虚拟环境是否已安装
- [x] 3.7 实现推理结果结构化返回：包含 bbox/polygon/keypoints/confidence/label 等字段
- [x] 3.8 统一返回格式为 Roboflow 兼容 JSON：`predictions` 数组，每项含 `x`/`y`/`width`/`height`/`confidence`/`class`/`class_id`/`detection_id`
- [x] 3.9 segment 任务额外返回 `points` 多边形点数组
- [x] 3.10 pose 任务额外返回 `keypoints` 数组（含 `name`/`confidence`）
- [x] 3.11 obb 任务返回 `points` 四点角点坐标
- [x] 3.12 classify 任务返回简化 predictions（仅 `class`/`confidence`/`class_id`）

## 4. 前端 — 推理结果渲染

- [x] 4.1 在 `templates/model-test.html` 内联脚本中实现 Canvas 绘制函数：detect 任务画矩形框+标签
- [x] 4.2 实现 segment 任务绘制：半透明多边形掩码+矩形框+标签
- [x] 4.3 实现 pose 任务绘制：矩形框+关键点+骨架连线
- [x] 4.4 实现 obb 任务绘制：4 点旋转框+标签
- [x] 4.5 实现 classify 任务绘制：图片上方显示分类结果文字
- [x] 4.6 实现「显示/隐藏标注」切换时的 Canvas 重绘逻辑

## 5. 前端 — 模型下载弹窗

- [x] 5.1 在 `templates/projects.html` 中新增训练模型列表弹窗 HTML 结构
- [x] 5.2 实现模型下载按钮点击事件：打开弹窗并调用 `/api/train/model-info` 加载该工程的训练版本列表
- [x] 5.3 复用 train.html 中的版本卡片样式和渲染逻辑，展示版本信息及 mAP 指标
- [x] 5.4 实现「下载 .pt」按钮：调用 `/api/train/download-model`
- [x] 5.5 实现「下载 ONNX / 导出 ONNX」按钮：调用 `/api/train/download-model?format=onnx` 或 `/api/train/export-onnx`

## 6. 测试与验证

- [x] 6.1 运行 `ruff check .` 确保后端代码风格合规（本次变更未引入新错误，剩余 5 处为历史遗留问题）
- [ ] 6.2 测试模型测试页面加载训练模型列表并正确展示版本信息
- [ ] 6.3 测试点击模型进入测试工作区并正确带入任务类型
- [ ] 6.4 测试图片上传和预览功能
- [ ] 6.5 测试 detect 任务推理及结果可视化
- [ ] 6.6 测试 segment/pose/obb/classify 任务推理及结果可视化
- [ ] 6.7 测试 GPU 冲突检测和提示
- [ ] 6.8 测试模型下载按钮是否正确打开训练模型列表弹窗并展示版本信息
- [ ] 6.9 测试无模型时的空状态提示
