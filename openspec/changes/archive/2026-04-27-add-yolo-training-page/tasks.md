## 1. 后端 — 训练环境管理

- [x] 1.1 实现 `install_ultralytics_env()`：在 `plugins/ultralytics` 创建 venv 并安装 ultralytics
- [x] 1.2 实现 `check_ultralytics_install()`：检测训练环境是否已安装
- [x] 1.3 实现 `/api/check-ultralytics-install`：返回安装状态及版本信息
- [x] 1.4 实现 `/api/install-ultralytics`：通过 EventSource 推送安装进度（复用 YOLO11 安装模式）

## 2. 后端 — 训练数据准备

- [x] 2.1 实现 `export_yolo_dataset_for_training(project_name)`：将工程图片和标注导出为 YOLO 格式到临时目录
- [x] 2.2 实现训练/验证集划分：按配置比例拆分已标注图片（默认 80/20）
- [x] 2.3 自动生成 `data.yaml` 配置文件（train/val 路径、类别列表）
- [x] 2.4 实现训练数据量校验：已标注图片少于 10 张时返回错误

## 3. 后端 — 训练任务执行

- [x] 3.1 创建 `YOLOTrainingTask` 类：封装训练进程管理（启动、停止、状态查询）
- [x] 3.2 实现训练脚本 `train_yolo.py`：调用 ultralytics API 训练，支持从官方权重或工程已有权重启动
- [x] 3.3 实现 GPU 资源冲突检测：训练启动前检查是否有 AI 标注任务在运行，反之亦然
- [x] 3.4 实现 `/api/train/start`：接收训练参数，校验后启动训练任务
- [x] 3.5 实现 `/api/train/status`：返回当前训练任务状态（idle/running/completed/error）
- [x] 3.6 实现 `/api/train/cancel`：终止正在运行的训练任务
- [x] 3.7 训练完成后自动执行验证（val），提取 mAP/precision/recall
- [x] 3.8 将最佳权重和评估结果保存到 `projects/<project>/models/best.pt`

## 4. 后端 — 训练进度推送

- [x] 4.1 实现训练日志解析器：从 ultralytics 输出中提取 epoch、loss、mAP 等字段
- [x] 4.2 通过 SocketIO `train_progress` 事件向前端实时推送训练进度
- [x] 4.3 实现训练完成/失败的事件推送

## 5. 前端 — 训练页面

- [x] 5.1 创建 `templates/train.html`：训练主页面布局
- [x] 5.2 实现环境安装状态检测：未安装时显示安装引导，已安装时显示配置表单
- [x] 5.3 实现训练参数配置表单：模型类型、epochs、batch、imgsz、device、train_val_ratio
- [x] 5.4 实现"基础模型"选择器：官方预训练权重 / 本工程已有模型（动态检测是否存在）
- [x] 5.5 实现参数校验：正整数校验、模型名称白名单校验
- [x] 5.6 实现 GPU 冲突提示：启动训练前检测 AI 标注任务状态并给出阻塞提示
- [x] 5.5 实现训练进度展示面板：epoch 进度条、loss 曲线占位、mAP 指标
- [x] 5.7 实现"开始训练"和"取消训练"按钮及状态切换
- [x] 5.8 实现训练进度展示面板：epoch 进度条、loss 曲线占位、mAP 指标
- [x] 5.9 实现训练完成后的评估结果展示：mAP50、mAP50-95、precision、recall、训练时长
- [x] 5.10 实现训练失败/取消的状态提示与错误信息显示

## 6. 前端 — 工程管理页面入口

- [x] 6.1 修改 `templates/projects.html`：在每个工程卡片操作区新增"进入训练"按钮
- [x] 6.2 点击"进入训练"跳转至 `/train?project=<name>`

## 7. 测试与验证

- [x] 7.1 运行 `ruff check .` 确保代码风格合规
- [x] 7.2 测试训练环境安装流程
- [x] 7.3 测试训练参数校验逻辑
- [x] 7.4 测试训练任务启动、进度推送、取消流程
- [x] 7.5 测试训练/验证集划分逻辑
- [x] 7.6 测试基于已有权重的 Fine-tune 流程
- [x] 7.7 测试 GPU 资源冲突检测（训练 vs AI 标注互锁）
- [x] 7.8 测试训练完成后自动评估及结果保存
