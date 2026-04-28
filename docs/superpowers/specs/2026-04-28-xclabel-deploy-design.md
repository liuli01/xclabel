# xclabel-deploy 推理部署容器设计

## 1. 概述

xclabel-deploy 是 xclabel 的独立推理部署容器，支持从 xclabel-server 动态拉取模型和工作流，对外提供 REST API 推理服务。参考 roboflow-inference-server 的设计，支持同时加载多个模型/工作流。

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     外部客户端                                │
│            (调用 /infer 获取推理结果)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  xclabel-deploy (推理容器)                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  FastAPI 服务 (port 8000)                              │  │
│  │  ├── POST /load/model   → 下载模型 → nndeploy 加载      │  │
│  │  ├── POST /load/workflow → 下载 workflow → 构建 Pipeline│  │
│  │  ├── POST /infer        → 按 engine_id 路由执行推理     │  │
│  │  ├── POST /unload       → 卸载引擎释放内存              │  │
│  │  └── GET  /engines      → 返回已加载引擎列表            │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  推理引擎池 (Inference Engine Pool)                    │  │
│  │  {                                                     │  │
│  │    "test/20260428_104927": {                          │  │
│  │      "type": "model",                                 │  │
│  │      "engine": <nndeploy.Model>,                      │  │
│  │      "metadata": {...}                                │  │
│  │    },                                                  │  │
│  │    "test/detection-pipeline": {                       │  │
│  │      "type": "workflow",                              │  │
│  │      "engine": <nndeploy.Pipeline>,                   │  │
│  │      "workflow_json": {...}                           │  │
│  │    }                                                   │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  本地缓存 (/app/cache/)                                │  │
│  │  ├── models/test_20260428_104927/                     │  │
│  │  │   ├── best.pt                                      │  │
│  │  │   └── model_info.json                              │  │
│  │  └── workflows/test_detection-pipeline.json           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    HTTP GET (下载模型/workflow)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  xclabel-server (服务端)                                    │
│  ├── 模型管理: /api/train/download-model                   │
│  ├── 模型列表: /api/train/model-info                       │
│  ├── Workflow: /api/workflow/{name}/export                 │
│  └── 存储: projects/<project>/models/{version}/            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 容器关系

| 容器 | 职责 | 镜像 |
|------|------|------|
| xclabel-server | 标注、训练、模型管理、workflow 编排 | 现有 Dockerfile |
| xclabel-deploy-cpu | CPU 推理服务 | deploy/Dockerfile.cpu |
| xclabel-deploy-gpu | GPU 推理服务（CUDA + TensorRT） | deploy/Dockerfile.gpu |

## 3. 推理引擎池

引擎池维护所有已加载的模型和工作流，key 格式为 `{project_id}/{version_or_name}`。

### 3.1 引擎类型

**模型引擎（type: model）**
- 来源：xclabel-server 训练后自动导出的 `.onnx`（server 端负责将 `.pt` 转为 `.onnx`）
- 加载方式：nndeploy Python API 直接加载 ONNX 模型文件
- 适用：单模型直接推理
- **注意**：deploy 容器只支持 ONNX 格式，不直接加载 `.pt`

**工作流引擎（type: workflow）**
- 来源：xclabel-server workflow 编排页面导出的 `workflow.json`
- 加载方式：nndeploy 解析 workflow.json 构建 Pipeline
- 适用：多节点处理流程（预处理 → 推理 → 后处理）

### 3.2 生命周期

1. **加载**：首次调用 `/load/*` 时从 server HTTP 下载到本地缓存，nndeploy 初始化后放入引擎池
2. **复用**：同一 engine_id 再次推理时直接使用，无需重复下载/初始化
3. **卸载**：调用 `/unload` 或内存不足 LRU 淘汰时释放

### 3.3 并发安全

由于 nndeploy 的线程安全特性未知，每个引擎在引擎池中配有一个 **asyncio Lock**：
- 同一引擎的多个并发推理请求会串行执行（排队等待锁）
- 不同引擎之间的推理可以并行（独立锁）
- 未来确认 nndeploy 线程安全后，可移除锁以提升吞吐量

## 4. API 设计

### 4.1 加载模型

```bash
POST /load/model
Content-Type: application/json

{
    "project_id": "test",
    "model_version": "20260428_104927",
    "server_url": "http://xclabel-server:5000"  // 可选，默认使用环境变量 SERVER_URL
}

Response 200 OK:
{
    "engine_id": "test/20260428_104927",
    "type": "model",
    "status": "loaded",
    "metadata": {
        "yolo_version": "yolo11",
        "task": "detect",
        "class_count": 5,
        "input_size": 640
    }
}
```

### 4.2 加载工作流

```bash
POST /load/workflow
Content-Type: application/json

{
    "project_id": "test",
    "workflow_name": "detection-pipeline",
    "server_url": "http://xclabel-server:5000"  // 可选，默认使用环境变量 SERVER_URL
}

Response 200 OK:
{
    "engine_id": "test/detection-pipeline",
    "type": "workflow",
    "status": "loaded",
    "nodes": ["preprocess", "inference", "nms", "output"]
}
```

### 4.3 执行推理

```bash
POST /infer
Content-Type: application/json

{
    "engine_id": "test/20260428_104927",
    "image": "<base64-encoded-image>",
    "confidence_threshold": 0.5
}

# 或图片 URL
{
    "engine_id": "test/detection-pipeline",
    "image_url": "http://example.com/image.jpg",
    "confidence_threshold": 0.5
}

Response 200 OK:
{
    "engine_id": "test/20260428_104927",
    "inference_time_ms": 45,
    "detections": [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.92,
            "bbox": [100, 200, 150, 300]
        }
    ]
}
```

### 4.4 引擎管理

```bash
# 列出已加载引擎
GET /engines
Response:
{
    "engines": [
        {
            "engine_id": "test/20260428_104927",
            "type": "model",
            "loaded_at": "2026-04-28T10:00:00Z",
            "last_used_at": "2026-04-28T10:05:00Z"
        }
    ],
    "total": 1
}

# 卸载指定引擎
POST /unload
{
    "engine_id": "test/20260428_104927"
}

# 卸载全部引擎
POST /unload/all
```

### 4.5 健康检查

```bash
GET /health
Response: {"status": "ok", "version": "1.0.0"}
```

## 5. Dockerfile 设计

### 5.1 CPU 镜像

```dockerfile
# ============================================
# xclabel-deploy CPU 推理容器
# ============================================
FROM python:3.12-slim-bookworm

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 nndeploy (PyPI 官方包，支持 CPU 推理)
RUN pip install --no-cache-dir nndeploy

# 复制服务代码
COPY deploy/*.py ./

# 创建缓存目录
RUN mkdir -p /app/cache/models /app/cache/workflows

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 GPU 镜像

```dockerfile
# ============================================
# xclabel-deploy GPU 推理容器
# ============================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装 Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 python3.12-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
COPY deploy/requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# 安装 nndeploy (PyPI 官方包，自动识别 CUDA 环境启用 GPU 后端)
RUN python3.12 -m pip install --no-cache-dir nndeploy

# 复制服务代码
COPY deploy/*.py ./

# 创建缓存目录
RUN mkdir -p /app/cache/models /app/cache/workflows

EXPOSE 8000

CMD ["python3.12", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 6. docker-compose 编排

```yaml
version: "3.8"

services:
  xclabel-server:
    build: .
    container_name: xclabel-server
    ports:
      - "9924:5000"
    volumes:
      - ./uploads:/app/uploads
      - ./static/annotations:/app/static/annotations
      - ./plugins:/app/plugins
      - ./projects:/app/projects
    environment:
      - FLASK_HOST=0.0.0.0
      - FLASK_PORT=5000
      - FLASK_DEBUG=false
    restart: unless-stopped

  xclabel-deploy-cpu:
    build:
      context: .
      dockerfile: deploy/Dockerfile.cpu
    container_name: xclabel-deploy-cpu
    ports:
      - "8000:8000"
    environment:
      - SERVER_URL=http://xclabel-server:5000
      - CACHE_DIR=/app/cache
      - MAX_ENGINES=10
    volumes:
      - deploy-cache-cpu:/app/cache
    depends_on:
      - xclabel-server
    restart: unless-stopped

  xclabel-deploy-gpu:
    build:
      context: .
      dockerfile: deploy/Dockerfile.gpu
    container_name: xclabel-deploy-gpu
    ports:
      - "8001:8000"
    environment:
      - SERVER_URL=http://xclabel-server:5000
      - CACHE_DIR=/app/cache
      - MAX_ENGINES=5
    volumes:
      - deploy-cache-gpu:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - xclabel-server
    restart: unless-stopped

volumes:
  deploy-cache-cpu:
  deploy-cache-gpu:
```

## 7. 与 xclabel-server 的交互

### 7.0 Server 端 ONNX 导出（前置条件）

deploy 容器只接受 ONNX 格式模型，因此 xclabel-server 需在训练完成后自动导出：

1. 训练完成 → Ultralytics `model.export(format='onnx')` → 生成 `best.onnx`
2. 将 `best.onnx` 放入版本目录 `projects/<project>/models/<version>/`
3. deploy 容器下载的是包含 `best.onnx` 的模型包

### 7.1 模型下载流程

```
deploy                         server
  |                              |
  |-- GET /api/model/download ---|
  |   ?project=test              |
  |   &version=20260428_104927   |
  |                              |
  |<-- 200 OK (zip stream) ------|
  |   (projects/test/models/     |
  |    20260428_104927/ 打包)    |
```

### 7.2 工作流下载流程

```
deploy                         server
  |                              |
  |-- GET /api/workflow/export --|
  |   ?project=test              |
  |   &name=detection-pipeline   |
  |                              |
  |<-- 200 OK (json) ------------|
  |   workflow.json +            |
  |   引用的模型包下载链接        |
```

### 7.3 服务端需新增 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/model/download` | GET | 按 project + version 下载模型包 |
| `/api/model/versions` | GET | 列出项目的所有模型版本 |
| `/api/workflow/export` | GET | 导出 workflow.json |
| `/api/workflow/list` | GET | 列出项目的工作流 |

## 8. 目录结构

```
xclabel/
├── Dockerfile                          # server 镜像（已有）
├── docker-compose.yml                  # 编排文件（更新）
├── app.py                              # Flask 服务端（已有）
├── deploy/                             # deploy 容器代码（新增）
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   ├── main.py                         # FastAPI 入口
│   ├── requirements.txt
│   ├── engine_pool.py                  # 引擎池管理
│   ├── nndeploy_adapter.py            # nndeploy 封装
│   └── server_client.py               # 调用 xclabel-server 的客户端
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-28-xclabel-deploy-design.md
```

## 9. 错误处理

| 场景 | HTTP 状态码 | 错误信息 |
|------|------------|---------|
| 引擎未找到 | 404 | `Engine not found: {engine_id}` |
| 模型下载失败 | 502 | `Failed to download model from server` |
| 模型格式不支持 | 400 | `Unsupported model format` |
| nndeploy 加载失败 | 500 | `Failed to load model with nndeploy` |
| 推理输入无效 | 400 | `Invalid image data` |
| 内存不足 | 503 | `Insufficient memory, please unload engines` |

## 10. 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SERVER_URL` | `http://xclabel-server:5000` | xclabel-server 地址 |
| `CACHE_DIR` | `/app/cache` | 本地缓存目录 |
| `MAX_ENGINES` | `10` | 最大同时加载引擎数 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 11. 后续工作

1. xclabel-server 端需实现 `/api/model/download` 和 `/api/workflow/export` 端点
2. 调研 nndeploy Python API 的具体调用方式，完善 `nndeploy_adapter.py`
3. 支持批量推理（多张图片同时传入）
4. 支持视频流推理
5. 支持推理结果回调（webhook）
