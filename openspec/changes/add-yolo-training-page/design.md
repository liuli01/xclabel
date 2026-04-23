## Context

xclabel 目前支持图片标注、AI 自动标注和数据集导出。用户完成标注后，需要手动搭建训练环境、转换数据格式、编写训练脚本，才能基于标注数据训练 YOLO 模型。项目已在 `plugins/yolo11` 目录下支持通过独立 venv 安装 YOLO11 环境，但仅用于 AI 自动标注推理，尚未提供训练能力。

## Goals / Non-Goals

**Goals:**
- 在工程管理页面为每个工程提供"进入训练"入口
- 提供独立的训练页面，支持配置训练参数（模型类型、轮数、批次大小、图像尺寸等）
- 基于当前工程的标注数据自动导出 YOLO 格式并启动训练
- 使用 `plugins/ultralytics` 独立 venv 环境运行训练，避免污染主环境
- 实时展示训练进度（epoch、loss、mAP、当前状态）
- 训练完成后自动保存最佳权重到工程目录 `models/` 下

**Non-Goals:**
- 多 GPU 分布式训练
- 模型量化、剪枝、导出 ONNX/TensorRT
- 超参数自动调优（AutoML）
- 支持除 YOLOv8/v11 之外的其他检测框架
- 训练历史版本管理（仅保留最新权重）

## Decisions

### 1. 训练环境：独立 venv 安装 Ultralytics

在 `plugins/ultralytics` 目录下创建 Python venv，通过 pip 安装 `ultralytics`。训练脚本通过 `subprocess` 调用该 venv 的 Python 执行，与主应用环境隔离。

**Rationale**: 与现有 YOLO11 安装逻辑保持一致，避免主环境依赖冲突。

### 2. 训练数据流：工程标注 → YOLO 格式 → 训练

训练前自动将当前工程的图片和标注数据导出为 YOLO 格式（临时目录），训练完成后清理临时数据，仅保留权重文件。

**Rationale**: Ultralytics 训练接口原生支持 YOLO 格式目录结构，无需额外转换逻辑。

### 3. 训练进程管理：独立 Python 进程 + SocketIO 推送

训练任务在独立进程中运行（避免阻塞 Flask 主线程），通过文件或 SocketIO 向前端推送进度。

**Rationale**: YOLO 训练耗时较长且 CPU/GPU 占用高，独立进程可防止 Web 服务无响应。与现有 AI 标注任务的 SocketIO 进度推送机制保持一致。

### 4. 权重保存路径：`projects/<project>/models/`

每个工程的训练结果保存在各自目录下，天然隔离。

**Rationale**: 与工程数据隔离架构保持一致，便于备份和迁移。

## Risks / Trade-offs

- **[Risk]** 训练过程占用大量 GPU/CPU 资源，可能影响 AI 标注推理性能 → **Mitigation**: 同一时间仅允许一个训练任务运行；训练前检查资源占用并提示用户
- **[Risk]** Ultralytics 版本升级可能破坏训练接口 → **Mitigation**: 安装时锁定版本号；在训练脚本中做版本兼容检查
- **[Trade-off]** 训练数据每次都要重新导出为 YOLO 格式 → 训练时间增加取决于数据量，但简化了代码逻辑（无需维护训练专用缓存）

## Migration Plan

1. 用户首次使用训练功能时，系统自动检测 `plugins/ultralytics` 是否存在，若不存在提示安装
2. 安装流程复用现有 YOLO11 安装的 EventSource 进度推送 UI 模式
3. 无需数据迁移，训练功能对现有工程无侵入
