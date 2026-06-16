## xclabel
* 作者：北小菜 
* 作者主页：https://www.yuturuishi.com
* gitee开源地址：https://gitee.com/Vanishi/xclabel
* github开源地址：https://github.com/liuli01/xclabel

### 软件介绍
xclabel是一款功能强大的开源图像标注工具，采用Python+Flask开发，跨平台支持Windows/Linux/Mac。

**核心功能：**
- 支持多种标注类型（矩形、多边形等）
- 支持导入图片文件夹、视频文件、LabelMe格式数据集
- 支持RTSP流处理和网络摄像头标注
- AI自动标注功能，支持大模型对图片和视频进行自动标注
- 集成YOLO11模型管理，支持安装、卸载和预训练模型下载
- 支持导出YOLO格式数据集，可自定义训练/验证/测试比例
- 内置文件管理系统，支持文件浏览、上传、下载等操作
- 简洁直观的用户界面，易于使用

**技术特点：**
- 可通过源码运行或直接运行打包后的exe文件
- 支持命令行参数配置
- 标注数据安全存储
- 完善的目录自动创建机制

### 软件截图
<img width="720" alt="1" src="https://gitee.com/Vanishi/images/raw/master/xclabel/v2.1/1.png">
<img width="720" alt="2" src="https://gitee.com/Vanishi/images/raw/master/xclabel/v2.1/2.png">
<img width="720" alt="3" src="https://gitee.com/Vanishi/images/raw/master/xclabel/v2.1/3.png">

### 版本历史

查看完整的版本更新记录，请参考 [CHANGELOG.md](CHANGELOG.md)

### 主要功能
1. **图像标注**：支持矩形、多边形等多种标注类型
2. **数据集管理**：
   - 支持图像、视频、LabelMe数据集导入
   - 视频抽帧时使用视频文件名作为前缀，便于管理
3. **AI自动标注**：
   - 支持多种推理工具（LMStudio、vLLM、ollama、阿里云大模型）
   - 支持图片和视频的AI自动标注
   - 实现AI标注弹框，包含API配置、提示词输入和标签选择
   - 支持显示标注进度，包括已执行数量、总量、总耗时和进度条
4. **API配置管理**：支持保存和加载API配置参数
5. **标注导出**：支持YOLO格式数据集导出，可自定义训练/验证/测试比例
6. **标签管理**：支持添加、编辑、删除标签，自定义标签颜色
7. **YOLO11集成**：
   - 自动安装和卸载YOLO11
   - 支持预训练模型下载和管理
   - 手动拖放模型文件支持
   - CUDA支持自动检测
8. **文件管理系统**：
   - 支持文件系统导航和路径浏览
   - 支持图片预览和放大查看
   - 支持文件选择、全选、批量下载和删除
   - 支持新建文件夹和文件上传
9. **RTSP流处理**：支持直接对网络摄像头流进行标注
10. **快捷键支持**：提高标注效率
11. **实时保存**：标注数据实时保存，避免数据丢失
12. **UI优化**：
    - 在导航栏添加AI按钮
    - 实现左侧侧边栏宽度保存功能，使用localStorage持久化
    - 优化AI标注弹框样式，使元素更紧凑
    - 改进左侧侧边栏图片列表，解决序号和文件名重叠问题

### 使用说明

**源码运行（推荐 uv）：**
```bash
# 安装依赖
uv sync

# 启动服务
uv run python app.py --host 0.0.0.0 --port 5000
```

**或使用 pip：**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 5000
```

### Docker 镜像

打 `v*.*.*` 标签推送到 GitHub 后，CI/CD 自动构建并推送以下镜像至 **ghcr.io** 和 **Docker Hub**：

- ghcr.io：`ghcr.io/liuli01/<image>:latest` / `ghcr.io/liuli01/<image>:v*.*.*`
- Docker Hub：`liuli01/<image>:latest` / `liuli01/<image>:v*.*.*`

---

#### 1. xclabel-server-cpu

| 项目 | 说明 |
|------|------|
| **用途** | 标注/训练主服务（CPU），提供图像标注、YOLO 训练、AI 标注等全部 WebUI 功能 |
| **镜像** | `liuli01/xclabel-server-cpu` / `ghcr.io/liuli01/xclabel-server-cpu` |
| **端口** | 5000（映射建议 9924） |
| **基础镜像** | `python:3.12-slim-bookworm` |
| **构建文件** | `Dockerfile.server.cpu` |

```bash
docker run -d \
  --name xclabel-server \
  -p 9924:5000 \
  -v projects:/app/projects \
  -v uploads:/app/uploads \
  -v annotations:/app/static/annotations \
  -v plugins:/app/plugins \
  -e FLASK_DEBUG=false \
  -e FLASK_HOST=0.0.0.0 \
  ghcr.io/liuli01/xclabel-server-cpu:latest
```

---

#### 2. xclabel-server-gpu

| 项目 | 说明 |
|------|------|
| **用途** | 标注/训练主服务（GPU），支持 CUDA 加速训练和推理 |
| **镜像** | `liuli01/xclabel-server-gpu` / `ghcr.io/liuli01/xclabel-server-gpu` |
| **端口** | 5000（映射建议 9924） |
| **基础镜像** | `nvidia/cuda:12.2.2-runtime-ubuntu22.04` |
| **构建文件** | `Dockerfile.server.gpu` |

```bash
docker run -d \
  --name xclabel-server \
  --gpus all \
  -p 9924:5000 \
  -v projects:/app/projects \
  -v uploads:/app/uploads \
  -v annotations:/app/static/annotations \
  -v plugins:/app/plugins \
  -e FLASK_DEBUG=false \
  ghcr.io/liuli01/xclabel-server-gpu:latest
```

---

#### 3. xclabel-nndeploy

| 项目 | 说明 |
|------|------|
| **用途** | Workflow 编排服务，提供可视化 WebUI 编辑推理 workflow（DAG 编排） |
| **镜像** | `liuli01/xclabel-nndeploy` / `ghcr.io/liuli01/xclabel-nndeploy` |
| **端口** | 8002 |
| **构建文件** | `nndeploy-app/Dockerfile` |

```bash
docker run -d \
  --name nndeploy-app \
  -p 8002:8002 \
  -v nndeploy-resources:/app/resources \
  ghcr.io/liuli01/xclabel-nndeploy:latest \
  nndeploy-app --port 8002 --resources /app/resources
```

WebUI 访问：`http://localhost:8002`

---

#### 4. xclabel-deploy-cpu

| 项目 | 说明 |
|------|------|
| **用途** | 推理部署服务（CPU），从 xclabel-server 拉取模型提供 REST API 推理，支持 ONNX 和 workflow 两种模式 |
| **镜像** | `liuli01/xclabel-deploy-cpu` / `ghcr.io/liuli01/xclabel-deploy-cpu` |
| **端口** | 8000 |
| **基础镜像** | `python:3.12-slim-bookworm` |
| **构建文件** | `Dockerfile.deploy.cpu` |

```bash
docker run -d \
  --name xclabel-deploy-cpu \
  -p 8000:8000 \
  -e SERVER_URL=http://xclabel-server:5000 \
  -e CACHE_DIR=/app/cache \
  -e MAX_ENGINES=10 \
  -v deploy-cache:/app/cache \
  ghcr.io/liuli01/xclabel-deploy-cpu:latest
```

---

#### 5. xclabel-deploy-gpu

| 项目 | 说明 |
|------|------|
| **用途** | 推理部署服务（GPU），支持 CUDA 加速推理 |
| **镜像** | `liuli01/xclabel-deploy-gpu` / `ghcr.io/liuli01/xclabel-deploy-gpu` |
| **端口** | 8001 |
| **基础镜像** | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` |
| **构建文件** | `Dockerfile.deploy.gpu` |

```bash
docker run -d \
  --name xclabel-deploy-gpu \
  --gpus all \
  -p 8001:8000 \
  -e SERVER_URL=http://xclabel-server:5000 \
  -e MAX_ENGINES=5 \
  -v deploy-cache:/app/cache \
  ghcr.io/liuli01/xclabel-deploy-gpu:latest
```

---

### Docker Compose 部署（完整栈）

一键启动所有服务（推荐）：

```bash
docker compose -f docker-compose.server.yml up -d
```

启动后的服务清单：

| 服务 | 端口 | 镜像 | 说明 |
|------|------|------|------|
| xclabel-server | 9924 | xclabel-server-gpu | 标注/训练 WebUI |
| nndeploy-app | 8002 | xclabel-nndeploy | Workflow 编排 WebUI |
| xclabel-deploy-cpu | 8000 | xclabel-deploy-cpu | CPU 推理 API |
| xclabel-deploy-gpu | 8001 | xclabel-deploy-gpu | GPU 推理 API |

访问 `http://localhost:9924` 即可使用标注工具。

---

### xclabel-server 架构说明

```
┌─────────────────────────────────────────────────────────┐
│                    xclabel-server                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ 标注工具  │  │ YOLO训练  │  │ AI自动标注             │  │
│  │ (WebUI)  │  │ (WebUI)  │  │ (LMStudio/vLLM/Ollama) │  │
│  └──────────┘  └────┬─────┘  └───────────────────────┘  │
│                      │                                    │
│                      ▼                                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │           模型发布 → ONNX 导出                    │   │
│  │           resources/models/<project>_<ver>.onnx   │   │
│  └─────────────────────┬────────────────────────────┘   │
└────────────────────────┼────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   nndeploy-app      │
              │   (Workflow 编排)   │
              │   localhost:8002    │
              └────────┬────────────┘
                       │ workflow .json
                       ▼
              ┌─────────────────────┐
              │   xclabel-deploy    │
              │   (推理部署 API)     │
              │   CPU:8000 GPU:8001 │
              └─────────────────────┘
```

### 推理部署 API 使用

xclabel-deploy 提供 REST API，支持模型推理和 workflow 执行两种模式。

**Workflow 执行（nndeploy DAG）：**

1. **查看可用 workflow 列表：**
```bash
curl http://localhost:8000/workflows
```

2. **加载 workflow：**
```bash
curl -X POST http://localhost:8000/load/workflow \
  -H "Content-Type: application/json" \
  -d '{"project_id": "test", "workflow_id": "9c924d24-97fa-423d-b3f3-3e7356c9ea1e"}'
```

3. **执行 workflow 推理：**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "test/9c924d24-97fa-423d-b3f3-3e7356c9ea1e",
    "image_url": "http://example.com/image.jpg"
  }'
```

**模型推理 API：**

1. **加载模型：**
```bash
curl -X POST http://localhost:8000/load/model \
  -H "Content-Type: application/json" \
  -d '{"project_id": "test", "model_version": "20260428_104927"}'
```

2. **执行推理（base64 图片）：**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "test/20260428_104927",
    "image": "<base64-encoded-image>",
    "confidence_threshold": 0.5
  }'
```

3. **执行推理（图片 URL）：**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "test/20260428_104927",
    "image_url": "http://example.com/image.jpg"
  }'
```

4. **查看已加载引擎：**
```bash
curl http://localhost:8000/engines
```

5. **卸载模型：**
```bash
curl -X POST http://localhost:8000/unload \
  -H "Content-Type: application/json" \
  -d '{"engine_id": "test/20260428_104927"}'
```

**环境变量：**
| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SERVER_URL` | `http://xclabel-server:5000` | xclabel-server 地址 |
| `CACHE_DIR` | `/app/cache` | 本地缓存目录 |
| `MAX_ENGINES` | `10` | 最大同时加载引擎数 |

**连接宿主机 Server（不通过 Docker Compose）：**

如果 xclabel-server 直接在宿主机运行（`python app.py`），deploy 容器可通过 `host.docker.internal` 连接：

```bash
curl -X POST http://localhost:8000/load/model \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test",
    "model_version": "20260428_172800",
    "server_url": "http://host.docker.internal:9924"
  }'
```

`server_url` 参数会覆盖环境变量 `SERVER_URL`，适用于临时指向不同的 server 实例。

**更新 Deploy 代码（开发调试）：**

如需修改 deploy 代码后快速测试，可将文件拷贝到运行中的容器，然后重启容器使变更生效：

```bash
# 拷贝修改后的文件到容器
docker cp deploy/nndeploy_adapter.py xclabel-deploy-cpu:/app/nndeploy_adapter.py

# 重启容器加载新代码
docker restart xclabel-deploy-cpu
```

**ONNX 输出格式说明：**

deploy 容器支持两种 ONNX 导出格式：
1. **含 NMS**（推荐）：输出形状 `[batch, num_dets, 6]`，每行 `[x1, y1, x2, y2, confidence, class_id]`
2. **原始 YOLO**：输出形状 `[batch, 4+num_classes, num_anchors]`，需手动解析 bbox 和类别

训练完成后自动导出的 ONNX 默认包含 NMS。如需手动导出，使用 Ultralytics 的 `model.export(format='onnx', nms=True)`。

### 源码运行

```bash
# 1. 克隆仓库
git clone https://github.com/liuli01/xclabel.git
cd xclabel

# 2. 安装依赖
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 3. 启动服务
python app.py --host 0.0.0.0 --port 5000
```

启动后访问 http://localhost:5000 即可开始使用。

**YOLO11 管理：**
- 点击右上角"设置"按钮打开设置弹框
- 在 YOLO11 安装部分，点击"安装 YOLO11"按钮进行安装
- 选择要下载的预训练模型，点击"下载选中模型"
- 可以手动拖放模型文件到指定区域进行安装
- 点击"卸载 YOLO11"按钮彻底卸载 YOLO11

### 项目结构
```
xclabel/
├── app.py                    # 主应用文件 (Flask Server)
├── AiUtils.py                # AI自动标注工具类
├── app.spec                  # PyInstaller打包配置文件
├── CHANGELOG.md              # 版本更新记录
├── CLAUDE.md                 # 项目开发规范
├── LICENSE                   # 授权协议
├── README.md                 # 项目说明文档
├── pyproject.toml            # 项目依赖与元数据 (uv)
├── uv.lock                   # uv 锁定文件
├── Dockerfile.server.cpu     # Server CPU 镜像构建
├── Dockerfile.server.gpu     # Server GPU 镜像构建
├── Dockerfile.deploy.cpu     # Deploy CPU 镜像构建
├── Dockerfile.deploy.gpu     # Deploy GPU 镜像构建
├── docker-compose.server.yml # 完整栈 Docker Compose
├── .gitignore               # Git忽略文件配置
├── nndeploy-app/             # Workflow 编排服务
│   ├── Dockerfile            # nndeploy 镜像构建
│   ├── app/                  # FastAPI WebUI
│   └── ...
├── deploy/                   # 推理部署服务源码
│   ├── main.py               # FastAPI 推理服务入口
│   ├── requirements.txt      # Deploy 依赖列表
│   ├── engine_pool.py        # 推理引擎池管理
│   ├── nndeploy_adapter.py   # nndeploy 推理适配器
│   └── server_client.py      # Server 端 HTTP 客户端
├── static/
│   ├── all.min.css           # Font Awesome图标库
│   ├── script.js             # 脚本文件
│   ├── style.css             # 样式文件
│   ├── annotations/          # 标注数据存储目录
│   ├── fonts/                # 字体文件目录
│   └── images/               # 图片资源目录
├── templates/
│   ├── projects.html         # 主页面模板 (Vue SPA)
│   ├── ai_config.html        # AI配置页面
│   └── file_manager.html     # 文件管理页面
├── config/                   # 运行时配置（不提交Git）
├── uploads/                  # 上传的图片和视频（运行时创建）
├── projects/                 # 工程数据目录（运行时创建）
│   └── <project>/
│       ├── annotations/
│       ├── models/           # 训练模型存储
│       └── workflows/        # Workflow 定义
├── resources/                # 共享资源目录（运行时创建）
│   ├── models/               # 已发布的 ONNX 模型
│   ├── workflow/             # workflow JSON
│   ├── images/
│   ├── videos/
│   ├── audios/
│   ├── db/                   # nndeploy-app SQLite
│   └── template/             # 内置模板
└── plugins/                  # 插件目录（运行时创建）
```

### 标注流程
1. **添加数据集**：点击右上角"添加数据集"按钮，选择要标注的图片、视频或LabelMe数据集
2. **创建标签**：在右侧标签管理中添加需要的标签，设置颜色
3. **开始标注**：
   - **手动标注**：选择左侧图片列表中的图片，使用左侧工具进行标注
   - **AI自动标注**：点击导航栏"AI"按钮，在弹框中配置API参数、输入提示词、选择标签，然后点击"开始执行"
4. **导出数据集**：标注完成后，点击右上角"导出数据集"按钮，选择导出格式和参数

### AI标注使用说明
1. **打开AI标注弹框**：点击导航栏中的"AI"按钮
2. **配置API参数**：检查并确认API配置信息，包括推理工具、模型、API地址等
3. **输入提示词**：在提示词输入框中输入用于标注的提示词
4. **选择标签**：从标签列表中选择一个用于标注的标签
5. **开始AI标注**：确保左侧侧边栏至少选中一个图片文件，然后点击"开始执行"
6. **查看标注进度**：在弹框中查看标注进度，包括已执行数量、总量、总耗时和进度条
7. **完成标注**：标注完成后，系统会自动更新标注数据，可在左侧图片列表中查看已标注的图片

### YOLO11使用流程
1. **安装YOLO11**：在设置弹框中点击"安装YOLO11"按钮
2. **下载预训练模型**：选择要下载的模型，点击"下载选中模型"
3. **手动添加模型**：将模型文件拖放到指定区域
4. **使用模型**：安装完成后，可用于模型推理和训练
5. **卸载YOLO11**：点击"卸载YOLO11"按钮彻底删除

### 快捷键说明
- **Ctrl+S**：保存标注
- **Ctrl+Shift+D**：清除标注

### 技术栈
- **后端**：Flask
- **前端**：HTML, CSS, JavaScript
- **数据库**：JSON文件存储
- **图像处理**：OpenCV, PIL
- **YOLO11集成**：Ultralytics YOLO11

### 授权协议
- 本项目自有代码使用宽松的MIT协议，在保留版权信息的情况下可以自由应用于各自商用、非商业的项目。
- 本项目使用了一些第三方库，使用本项目时请遵循相应第三方库的授权协议。
- 由于使用本项目而产生的商业纠纷或侵权行为一概与本项目及开发者无关，请自行承担法律风险。

