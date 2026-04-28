## Context

xclabel 当前在工程管理页面 (`/projects`) 为每个工程卡片提供「进入标注」、「模型训练」、重命名、删除四个操作。用户在训练完模型后，需要一个快速验证模型效果的入口，而不是跳转到其他工具或手动写脚本。同时，预训练模型的下载入口仅在设置面板中，用户在工程页面无法快速触达。

## Goals / Non-Goals

**Goals:**
- 在工程卡片上提供一键进入模型测试和模型下载的入口
- 模型测试弹窗支持选择已安装的 YOLO 模型、使用数据集 test 图片或上传自定义图片、实时推理并可视化结果
- 推理结果支持所有 5 种 YOLO 任务类型（detect/segment/pose/obb/classify）
- 模型下载按钮打开训练模型列表弹窗，展示已训练版本及下载操作

**Non-Goals:**
- 批量图片推理（仅支持单张图片测试）
- 模型测试结果的保存或导出
- 视频推理
- 修改现有训练/标注流程

## Decisions

**1. 弹窗内嵌 Canvas 渲染推理结果**
- **Rationale**: 单张图片测试不需要完整的标注页面，弹窗内嵌 Canvas 足够轻量且体验连贯。使用原生 Canvas 2D API 绘制 bbox/polygon/keypoints，无需引入额外图表库。
- **Alternative considered**: 跳转到标注页面加载模型并自动标注 — 太重，且会干扰现有标注状态。

**2. 后端推理复用 ultralytics 虚拟环境**
- **Rationale**: 训练环境已安装了 ultralytics 及依赖，无需额外环境。通过 `plugins/<version>/venv/bin/python` 或 `Scripts/python.exe` 调用 YOLO 推理。
- **Alternative considered**: 在主进程直接 `import ultralytics` — 版本冲突风险高，且用户可能未安装。

**3. 模型选择按 YOLO 版本 + 任务类型筛选**
- **Rationale**: 不同版本（YOLOv8/11/26）的模型不能混用，且任务类型决定输出格式。弹窗中先选版本，再选任务类型，最后列出该组合下已安装的 `.pt` 模型文件。
- **Alternative considered**: 列出所有已安装模型不分版本 — 用户容易选错版本导致环境不匹配。

**4. 弹窗默认加载工程数据集 test 目录前 4 张图片**
- **Rationale**: 用户训练完模型后，test 集是最自然的验证数据来源。弹窗打开时自动扫描 `projects/<name>/test/images/` 目录，加载前 4 张图片作为默认测试样本。用户仍可上传自定义图片覆盖。

**5. 弹窗关闭时清理临时状态**
- **Rationale**: 避免内存泄漏和隐私问题。关闭弹窗后清空 Canvas、推理结果 JSON、用户上传的临时文件（存放在 `temp/model-test/<project>/`），下次打开时恢复初始状态。

**6. 模型下载按钮打开训练模型列表弹窗**
- **Rationale**: 复用 train.html 中已有的 `loadModelInfo()` 功能，展示该工程已训练的所有模型版本（含任务、YOLO 版本、基础模型、epochs、训练集/验证集数量、mAP 指标），并提供「下载 .pt」和「下载 ONNX / 导出 ONNX」操作。弹窗内调用 `/api/train/model-info` 和 `/api/train/download-model` 实现，无需额外后端开发。

## Risks / Trade-offs

- **[Risk]** 用户选择的模型版本与已安装环境不匹配 → **Mitigation**: 模型列表只显示当前选中版本对应的已安装环境目录中的 `.pt` 文件；推理前校验虚拟环境是否存在。
- **[Risk]** 大模型首次加载耗时长 → **Mitigation**: 推理 API 返回加载进度信息；首次加载后 Python 进程退出即释放显存。
- **[Risk]** GPU 资源与训练/AI标注冲突 → **Mitigation**: 复用现有的 GPU 互锁机制，推理前检查是否有正在运行的训练或 AI 标注任务。
- **[Trade-off]** 单张图片测试不支持批量 — 批量推理属于生产环境需求，不在本功能范围内。

## Migration Plan

无需数据迁移。本功能为纯新增能力，不影响现有工程、标注和训练数据。

## Inference Result Data Format

后端 `/api/model-test/infer` 返回的 JSON **严格兼容 Roboflow 推理 API 格式**，统一结构如下：

```json
{
  "predictions": [
    {
      "x": 1030,
      "y": 85,
      "width": 26,
      "height": 26,
      "confidence": 0.946,
      "class": "10",
      "class_id": 1,
      "detection_id": "ca9f6eea-0c80-4522-8a86-17b5d77285ff"
    }
  ]
}
```

各任务类型字段差异：

| 任务 | 基础字段 | 额外字段 |
|------|---------|---------|
| **detect** | `x`, `y`, `width`, `height`, `confidence`, `class`, `class_id`, `detection_id` | — |
| **segment** | 同上 | `points`: 归一化多边形点数组 `[{"x":0.1,"y":0.2}, ...]` |
| **pose** | 同上 | `keypoints`: `[{"x":100,"y":200,"confidence":0.95,"name":"nose"}, ...]` |
| **obb** | — | `points`: 4 个角点像素坐标 `[{"x":100,"y":200}, ...]`，`confidence`, `class`, `class_id`, `detection_id` |
| **classify** | — | 顶层 `predictions` 改为 `predictions: [{"class":"cat","confidence":0.99,"class_id":0}]`，无坐标字段 |

> **坐标约定**：`x`/`y` 为 bbox 中心点像素坐标，`width`/`height` 为像素尺寸。`detection_id` 由后端生成 UUID。`class` 为工程 classes.json 中的类别名称。

## Open Questions

- 是否需要支持用户上传自定义模型（非官方预训练权重）进行测试？当前方案只支持 `.pt` 文件列表中的模型。
