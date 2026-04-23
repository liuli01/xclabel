## Context

xclabel 当前支持三种数据集导入方式：图片文件夹上传、视频抽帧、LabelMe 数据集导入。LabelMe 数据集导入要求用户选择已解压的文件夹，不支持 ZIP 压缩包。Roboflow 作为主流数据集平台，其导出格式（YOLOv8）通常为 ZIP 压缩包，内含图片、`data.yaml` 类别定义和 YOLO 格式的 `.txt` 标注文件。用户目前需要手动解压并整理文件后才能导入，流程繁琐。

本项目采用前后端不分离架构（Flask + Jinja2 模板 + Vanilla JS），数据以 JSON 文件（`annotations.json`、`classes.json`）存储，图片存放在 `uploads/` 目录。新增 Roboflow 数据集导入需遵循现有代码组织模式，在单文件 `app.py` 中新增路由，在 `index.html` 和 `script.js` 中扩展 UI。

## Goals / Non-Goals

**Goals:**
- 用户在"添加数据集"弹窗中可选择"Roboflow 数据集"选项卡
- 支持上传单个 ZIP 文件（如 `river_color.v4i.yolov8.zip`）
- 后端自动解压 ZIP，提取图片到 `uploads/`
- 解析 `data.yaml` 中的类别信息并同步到 `classes.json`
- 解析 YOLO `.txt` 标注文件并转换为内部格式保存到 `annotations.json`
- 处理有标注和无标注两种情况（ZIP 中可能不含 `labels/` 目录）
- 修改 YOLO 导出功能，多边形标注导出为 YOLO 分割格式（`class_id x1 y1 x2 y2 ...`），矩形标注仍导出为边界框格式
- 图片格式统一：导入支持 `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.gif`；导出读取也同步扩展

**Non-Goals:**
- 不支持非 YOLO 格式的 Roboflow 导出（如 COCO JSON、VOC XML 等）
- 不处理训练/验证/测试集划分（导入时合并所有子集中的图片和标注）
- 不实现从 Roboflow API 直接下载数据集（仅支持本地上传 ZIP）
- 不修改现有的 LabelMe 或图片上传功能
- 不新增导出格式选项（仅增强现有 YOLO 导出的标注表示方式）

## Decisions

**1. 多边形标注导出为 YOLO 分割格式，矩形标注导出为边界框格式**
- **Rationale**: YOLOv8+ 支持分割任务，格式为 `class_id x1 y1 x2 y2 ...`（归一化多边形点）。xclabel 内部标注有 `type` 字段（`polygon`/`rectangle`/`line`），可根据类型决定导出格式。矩形保持 bbox 导出以兼容检测任务；多边形保持完整形状以兼容分割任务。
- **Alternatives considered**: 统一全部导出为 bbox — 会导致多边形精度丢失；全部导出为多边形 — 矩形导出为四边形虽可行但会增加标签文件大小且对检测任务不必要。

**2. 使用标准库 `zipfile` 解压，PyYAML 解析 `data.yaml`**
- **Rationale**: `zipfile` 为 Python 标准库，无需额外依赖。PyYAML 通常已随 Python 数据科学生态安装（OpenCV 等依赖常间接引入），且 `data.yaml` 是 YOLO 数据集的标准配置方式。
- **Alternatives considered**: 使用 `shutil.unpack_archive` — 功能类似但 `zipfile` 提供更精细的控制（如安全解压路径检查）。

**3. YOLO 标注坐标转换为绝对像素坐标后存储**
- **Rationale**: Roboflow/YOLO 格式使用归一化坐标（0-1 范围），而 xclabel 内部使用绝对像素坐标。转换时机设在导入时，避免运行时重复计算。
- **Alternatives considered**: 存储归一化坐标并在前端渲染时转换 — 会增加渲染复杂度，且与现有 LabelMe 导入逻辑（存储绝对坐标）不一致。

**4. 遍历 ZIP 内标准 YOLO 目录合并导入，不保留 train/val/test 划分**
- **Rationale**: xclabel 当前没有数据集划分的概念，所有图片平铺在 `uploads/` 中。保留划分需要引入新的数据模型，超出当前需求范围。扫描时优先识别标准 YOLO 目录结构（`{train,val,valid,test}/images/`），若未找到再递归扫描整个 ZIP，避免误识别嵌套的非数据集图片。
- **Alternatives considered**: 在 `annotations.json` 中增加 split 字段 — 非目标，且会修改现有数据模型；直接递归扫描整个 ZIP — 可能误识别 ZIP 内嵌套的示例图或文档配图。

**5. 复用现有 `classes.json` 和 `annotations.json` 存储机制，类别颜色分配复用 LabelMe 导入的 `hash(label) % 0x1000000` 逻辑**
- **Rationale**: xclabel 内部 `classes.json` 为纯列表格式 `[{name, color}]`，标注中直接存储类别名称（`ann['class']`）。当 Roboflow `data.yaml` 中的类别名称与现有类别名称匹配时，复用其颜色；不匹配时新增类别并自动分配颜色。

**6. 不修改 ZIP 文件内容，解压到临时目录处理后清理（`shutil.rmtree` 使用 `ignore_errors=True`）**
- **Rationale**: 避免在 `uploads/` 中留下不必要的目录结构（如 `train/labels/` 等）。仅将图片文件复制到 `uploads/`，标注解析后直接存入 JSON。
- **Alternatives considered**: 直接解压整个 ZIP 到 `uploads/` — 会引入多余的目录层级和标注文件，与现有文件组织结构冲突。

## Risks / Trade-offs

- **[Risk] ZIP 文件可能包含恶意路径（路径遍历攻击）** → **Mitigation**: 使用 `zipfile` 时检查每个条目的 `filename`，拒绝包含 `..` 或以 `/` 开头的绝对路径条目。
- **[Risk] 同名图片文件覆盖已有数据** → **Mitigation**: 若 `uploads/` 中已存在同名文件，保留现有文件并跳过（或重命名新文件，此处选择跳过以符合现有上传逻辑）。
- **[Risk] `data.yaml` 格式不统一或位置不确定** → 不同 Roboflow 版本导出格式略有差异；某些打包工具会在 ZIP 根目录套一层文件夹（如 `river_color/data.yaml`）。→ **Mitigation**: 支持常见变体：`names` 字典或列表、`path` 字段存在或不存在。查找 `data.yaml` 时同时搜索 ZIP 根目录和第一层子目录。若解析失败，回退到使用标注文件中出现的类别编号，并提示用户手动编辑类别名称。
- **[Risk] 大 ZIP 文件导致内存或磁盘压力** → **Mitigation**: 使用流式解压（`zipfile` 逐个提取），不在内存中保留整个 ZIP 内容。设置合理的文件大小限制（如 500MB）。
- **[Risk] PyYAML 未显式安装导致导入失败** → **Mitigation**: 在 `pyproject.toml` 中显式添加 `pyyaml` 依赖，并在实施时执行 `uv sync`。
- **[Risk] Roboflow ZIP 内部目录结构差异** → Roboflow 导出可能使用 `valid/` 而非 `val/` 作为验证集目录名。→ **Mitigation**: 扫描时同时识别 `train/`、`val/`、`valid/`、`test/` 四个目录名。
- **[Trade-off] 不保留 train/val/test 划分信息** → 用户若需要划分，需导出时使用 xclabel 的导出功能重新划分。

## Migration Plan

无需数据迁移。本功能为纯新增能力，不影响现有数据。
部署步骤：
1. 更新代码（`app.py`、`index.html`、`script.js`）
2. 确保 `PyYAML` 已安装（检查 `pyproject.toml` / `requirements.txt`）
3. 重启 Flask 服务
4. 测试导入示例 ZIP 文件（`river_color.v4i.yolov8.zip`）

## Open Questions

1. 是否需要支持多文件 ZIP 上传（如用户同时选择多个 ZIP）？当前按单个 ZIP 设计。
2. 当 `data.yaml` 中 `names` 为列表时（如 `['class_a', 'class_b']`），类别 ID 从 0 开始还是从 1 开始？YOLO 标准为从 0 开始，与 xclabel 内部标注存储方式一致。

## 边界情况处理

- **空 ZIP / 无图片文件**：返回明确错误 "ZIP 文件中未找到图片"
- **图片格式范围**：导入支持 `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.gif`（与 LabelMe 导入保持一致，新增 `.webp` 以覆盖 Roboflow 常见格式）
- **图片无对应标注文件**：作为无标注图片导入，不报错
- **标注文件无对应图片**：忽略该标注文件
- **临时目录清理**：无论导入成功或失败，`finally` 块中使用 `shutil.rmtree(temp_dir, ignore_errors=True)` 删除临时解压目录，避免 Windows 文件句柄未释放导致删除失败
- **同名文件冲突**：若 `uploads/` 中已存在同名图片，保留现有文件并跳过（与现有上传逻辑一致）
- **类别颜色分配**：新类别颜色使用与 LabelMe 导入一致的 `hash(label) % 0x1000000` 逻辑
