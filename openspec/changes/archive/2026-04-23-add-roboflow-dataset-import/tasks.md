## 1. 前端 UI 扩展

- [ ] 1.1 在 `templates/index.html` 的"添加数据集"弹窗中新增"Roboflow 数据集"选项卡按钮（紧邻 LabelMe 选项卡之后）
- [ ] 1.2 在 `templates/index.html` 中新增选项卡内容区域：包含 ZIP 文件选择输入框（`accept=".zip"`）、文件信息展示、确认按钮
- [ ] 1.3 在 `static/script.js` 中为"Roboflow 数据集"选项卡添加事件监听器：文件选择变化时显示文件名和大小
- [ ] 1.4 在 `static/script.js` 中实现 ZIP 上传逻辑：构造 FormData（单个 `file` 字段），调用 `/api/upload/roboflow`，处理成功/失败回调并显示 Toast 提示
- [ ] 1.5 上传成功后调用 `loadImages()` 刷新图片列表，并自动切换到第一张图片

## 2. 依赖检查

- [ ] 2.1 检查 `pyproject.toml` 中是否已包含 `pyyaml` 依赖，若无则添加
- [ ] 2.2 运行 `uv sync` 确保依赖同步

## 3. 后端 API 实现

- [ ] 3.1 在 `app.py` 中新增 `/api/upload/roboflow` POST 路由，接收单个 `file` 字段
- [ ] 3.2 实现 ZIP 文件验证逻辑：检查文件扩展名是否为 `.zip`，文件大小是否超过限制（500MB）
- [ ] 3.3 实现安全解压函数：使用 `zipfile` 模块解压到临时目录（`tempfile.mkdtemp`），过滤包含 `..` 或绝对路径的恶意条目
- [ ] 3.4 实现临时目录清理：无论导入成功或失败，`finally` 块中使用 `shutil.rmtree(temp_dir, ignore_errors=True)` 删除临时解压目录
- [ ] 3.5 实现 ZIP 内部目录扫描：优先识别标准 YOLO 目录结构 `{train,val,valid,test}/images/` 和 `{train,val,valid,test}/labels/`；若未找到图片，再递归扫描整个 ZIP
- [ ] 3.6 实现图片提取逻辑：遍历所有子目录中的图片文件（`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`），复制到 `UPLOAD_FOLDER`，跳过已存在的同名文件
- [ ] 3.7 实现 `data.yaml` 解析函数：在 ZIP 解压后的临时目录中查找 `data.yaml`（优先根目录，再搜索第一层子目录），使用 `yaml.safe_load` 读取，支持 `names` 字典和列表两种格式，提取类别名称。注意 xclabel 内部 `classes.json` 为 `[{name, color}]` 格式，无 ID 字段
- [ ] 3.8 实现类别同步逻辑：将 `data.yaml` 中的类别名称与现有 `classes.json` 对比，名称匹配时复用现有颜色，不匹配时新增类别并使用 `hash(label) % 0x1000000` 分配颜色（与 LabelMe 导入逻辑一致）
- [ ] 3.9 实现 YOLO `.txt` 标注解析函数：
  - 根据图片文件名（不含扩展名）在对应 `labels/` 目录中查找 `.txt` 文件
  - 解析每一行的类别 ID 和坐标
  - 5 个数值为边界框（cx, cy, w, h），转换为左上角/右下角绝对像素坐标
  - 多于 5 个数值为多边形（YOLOv8 分割格式），将所有坐标对转换为绝对像素坐标
- [ ] 3.10 实现标注数据保存逻辑：将解析后的标注按图片文件名组织，写入 `annotations.json`，更新 `classes.json`。标注中 `class` 字段存储类别名称而非 ID
- [ ] 3.11 实现缺失 `data.yaml` 的回退逻辑：从 `labels/` 目录下的 `.txt` 文件中推断最大类别编号，生成默认类别名称（`class_0`, `class_1` 等），并在返回结果中附带警告信息
- [ ] 3.12 实现边界情况处理：
  - 空 ZIP / 无图片文件：返回错误 "ZIP 文件中未找到图片"
  - 图片无对应标注文件：作为无标注图片导入
  - 标注文件无对应图片：忽略该标注文件
- [ ] 3.13 返回统一的 JSON 响应（包含导入图片数、标注数、类别列表、警告信息）

## 4. 后端导出增强

- [ ] 4.1 修改 `app.py` 的 `export_dataset()` 函数中标签写入逻辑（[app.py:2225-2299](app.py#L2225-L2299)）
- [ ] 4.2 实现多边形标注导出：当 `ann['type'] == 'polygon'` 且有效点数 >= 6 时，输出 `class_id x1 y1 x2 y2 ...` 格式（所有坐标按图片宽高归一化，保留原始多边形形状）
- [ ] 4.3 实现矩形标注导出：当 `ann['type'] == 'rectangle'` 时，输出 `class_id cx cy w h` 格式（与现有逻辑一致）
- [ ] 4.4 实现线段标注处理：当 `ann['type'] == 'line'` 时，跳过不写入标签文件（YOLO 格式不支持线段）
- [ ] 4.5 扩展导出端图片读取格式：将 `export_dataset()` 中读取 `UPLOAD_FOLDER` 图片的格式从 `.png` `.jpg` `.jpeg` `.bmp` 扩展为 `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.gif`，与导入端保持一致
- [ ] 4.6 确保 `data.yaml` 中 `names` 字段格式与导出方式一致（列表格式）

## 5. 代码质量与测试

- [ ] 5.1 运行 `ruff check .` 确保代码风格合规，修复所有 lint 错误
- [ ] 5.2 使用示例文件 `river_color.v4i.yolov8.zip` 测试完整导入流程：验证图片是否出现在列表中，类别是否正确加载，标注是否正确显示
- [ ] 5.3 测试无标注 ZIP（仅含图片和 `data.yaml`）：验证图片导入成功，无标注数据，提示信息正确
- [ ] 5.4 测试错误场景：上传非 ZIP 文件、ZIP 中无图片、ZIP 中无 `data.yaml`、空 ZIP
- [ ] 5.5 测试同名文件冲突：重复导入同一 ZIP，验证不覆盖已有文件
- [ ] 5.6 测试导出多边形标注：创建多边形标注后导出 YOLO 数据集，验证 `.txt` 文件包含多边形点格式（`class_id x1 y1 x2 y2 ...`）
- [ ] 5.7 测试导出矩形标注：创建矩形标注后导出 YOLO 数据集，验证 `.txt` 文件仍为边界框格式（`class_id cx cy w h`）
- [ ] 5.8 测试导出混合标注：同时包含多边形和矩形标注的图片，导出后各标注使用对应格式
- [ ] 5.9 验证导出的 ZIP 可通过本功能的导入功能完整还原（round-trip 测试）
- [ ] 5.10 验证现有功能未受影响：图片上传、视频上传、LabelMe 导入、原有导出功能正常工作
