## ADDED Requirements

### Requirement: Roboflow ZIP 数据集上传接口
系统 SHALL 提供 `/api/upload/roboflow` API 端点，接受用户上传的单个 ZIP 文件，并返回导入结果（成功导入的文件列表、类别信息、标注统计）。

#### Scenario: 成功上传并导入带标注的 Roboflow 数据集
- **WHEN** 用户在"Roboflow 数据集"选项卡中选择一个有效的 `.zip` 文件（如 `river_color.v4i.yolov8.zip`）并点击确认
- **THEN** 系统解压 ZIP 文件，解析 `data.yaml` 中的类别定义，提取所有图片到 `uploads/`，解析所有 `.txt` 标注文件并保存到 `annotations.json`，返回成功消息和导入统计

#### Scenario: 成功上传并导入无标注的 Roboflow 数据集
- **WHEN** 用户上传的 ZIP 文件中仅包含图片和 `data.yaml`，不含 `labels/` 目录
- **THEN** 系统仅导入图片和类别信息，不生成标注数据，返回成功消息并提示"未检测到标注文件"

#### Scenario: 上传非 ZIP 文件
- **WHEN** 用户选择的文件扩展名不是 `.zip`
- **THEN** 系统拒绝上传，返回错误消息"仅支持 .zip 格式的 Roboflow 数据集"

#### Scenario: ZIP 文件中缺少 data.yaml
- **WHEN** 用户上传的 ZIP 文件中不包含 `data.yaml`
- **THEN** 系统尝试从 `labels/` 目录下的 `.txt` 文件中推断类别数量，使用默认类别名称（如 `class_0`, `class_1`），并发出警告提示用户手动编辑类别名称

### Requirement: ZIP 文件安全解压
系统 SHALL 在解压 ZIP 文件时验证每个文件条目的路径，防止路径遍历攻击，仅解压安全的文件到临时目录。

#### Scenario: ZIP 中包含恶意路径
- **WHEN** ZIP 文件中的某个条目文件名包含 `..` 或以 `/` 开头的绝对路径
- **THEN** 系统跳过该条目，继续处理其他安全条目，并在返回结果中包含警告信息

### Requirement: YOLO 标注格式解析与转换
系统 SHALL 能够解析 YOLO 格式的 `.txt` 标注文件，将归一化的多边形或边界框坐标转换为绝对像素坐标，并存储为 xclabel 内部标注格式。

#### Scenario: 解析多边形标注
- **WHEN** YOLO `.txt` 文件中某一行包含类别 ID 和多个坐标对（如 `0 0.5 0.3 0.6 0.4 0.7 0.5`）
- **THEN** 系统将该行解析为对应类别的多边形标注，坐标根据图片尺寸转换为绝对像素值，存储到 `annotations.json`

#### Scenario: 解析边界框标注
- **WHEN** YOLO `.txt` 文件中某一行包含类别 ID 和 4 个坐标值（如 `1 0.5 0.5 0.2 0.3`）
- **THEN** 系统将该行解析为边界框标注（中心 x, 中心 y, 宽, 高），转换为左上角和右下角绝对坐标，存储为矩形标注

### Requirement: YOLO 导出支持多边形标注
系统 SHALL 在导出 YOLO 数据集时，根据标注类型选择输出格式：多边形标注（`type == 'polygon'` 且点数 >= 6）导出为 YOLO 分割格式 `class_id x1 y1 x2 y2 ...`（归一化坐标）；矩形标注（`type == 'rectangle'`）导出为 YOLO 边界框格式 `class_id cx cy w h`；线段标注（`type == 'line'`）跳过不导出。

#### Scenario: 导出多边形标注
- **WHEN** 导出数据集中包含多边形标注（`type == 'polygon'`，点数 >= 6）
- **THEN** 系统输出 `class_id x1 y1 x2 y2 ...` 格式，所有坐标按图片宽高归一化

#### Scenario: 导出矩形标注
- **WHEN** 导出数据集中包含矩形标注（`type == 'rectangle'`）
- **THEN** 系统输出 `class_id cx cy w h` 格式，中心点和宽高按图片宽高归一化

#### Scenario: 导出线段标注
- **WHEN** 导出数据集中包含线段标注（`type == 'line'`）
- **THEN** 系统跳过该标注，不写入标签文件（YOLO 格式不支持线段）

### Requirement: 类别信息同步
系统 SHALL 解析 `data.yaml` 中的 `names` 字段，将类别信息同步到 `classes.json`，并为新类别自动分配颜色。若类别已存在，复用现有类别名称和颜色。

#### Scenario: data.yaml 中 names 为字典格式
- **WHEN** `data.yaml` 中 `names` 字段为字典格式（如 `{0: 'class_a', 1: 'class_b'}`）
- **THEN** 系统正确提取类别名称，同步到 `classes.json`

#### Scenario: data.yaml 中 names 为列表格式
- **WHEN** `data.yaml` 中 `names` 字段为列表格式（如 `['class_a', 'class_b']`）
- **THEN** 系统将列表索引对应值作为类别名称，同步到 `classes.json`
