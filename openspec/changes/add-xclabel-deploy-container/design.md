## Context

xclabel 当前是 Flask 单文件应用，支持数据标注、工程管理、YOLO 模型训练。训练产出的模型按时间戳版本存储在 `projects/<project>/models/<version>/` 目录下。用户需要手动下载模型后自行部署，缺乏标准化的推理服务能力。

当前状态：
- 模型存储：`projects/<project>/models/20260428_104927/best.pt`
- 已有 Dockerfile 和 docker-compose.yml 用于部署 server 端
- 缺乏独立的推理执行容器
- 缺乏从 server 拉取模型/工作流的 API

通过引入 deploy 容器，xclabel 可以形成完整的 MLOps 闭环：server 负责训练与编排，deploy 负责推理执行。

## Goals / Non-Goals

**Goals:**
- 提供独立的推理部署容器，与 server 分离部署
- 支持从 xclabel-server 动态拉取模型（按 project + version）
- 支持从 xclabel-server 拉取 workflow.json 并执行
- 支持同时加载多个模型/工作流（引擎池）
- 提供 CPU 和 GPU 两种镜像
- 对外提供 REST API 推理服务

**Non-Goals:**
- 不在 deploy 容器中实现模型训练（训练仍在 server 端）
- 不在 deploy 容器中实现 workflow 编排界面（编排仍在 server 端）
- 不实现分布式推理或多机部署
- 不实现模型 A/B 测试或金丝雀发布

## Decisions

### 1. deploy 容器使用 FastAPI 而非 Flask
- **Decision**: deploy 容器使用 FastAPI 框架
- **Rationale**: FastAPI 异步性能更好，适合 I/O 密集型的推理服务；自动 API 文档生成便于调试
- **Alternative**: 复用 Flask — 拒绝，Flask 同步模型不利于并发推理请求

### 2. 引擎池使用内存驻留而非每次加载
- **Decision**: 加载后的模型/工作流驻留内存，通过 engine_id 复用
- **Rationale**: 模型初始化耗时（尤其 GPU），驻留内存可避免重复加载开销
- **Alternative**: 每次推理重新加载 — 拒绝，延迟不可接受

### 3. 本地缓存 + HTTP 拉取
- **Decision**: deploy 容器本地缓存模型文件，首次从 server HTTP 下载
- **Rationale**: 简单可靠，server 和 deploy 可以跨机部署
- **Alternative**: 共享卷 — 拒绝，仅适合单机，限制部署灵活性

### 4. CPU/GPU 镜像分离
- **Decision**: 提供两个独立的 Dockerfile，分别基于 python:3.12-slim 和 nvidia/cuda
- **Rationale**: CPU 镜像更轻量（无 CUDA 依赖），GPU 镜像包含完整 CUDA 运行时
- **Alternative**: 单镜像通过运行时检测 — 拒绝，镜像体积过大，CUDA 依赖复杂

### 5. deploy 容器只支持 ONNX 格式
- **Decision**: deploy 容器只加载 `.onnx` 模型，server 端负责训练后自动导出
- **Rationale**: ONNX 更轻量，无需在 deploy 容器安装 PyTorch；nndeploy 对 ONNX 支持最成熟
- **Alternative**: 同时支持 `.pt` — 拒绝，会大幅增加镜像体积，且需要处理 torch 版本兼容性

### 6. server_url 配置方式
- **Decision**: `SERVER_URL` 通过环境变量配置，API 参数可覆盖
- **Rationale**: 大多数场景一个 deploy 容器只连一个 server，环境变量更简洁；API 参数覆盖支持临时切换
- **Alternative**: 强制每次 API 传 server_url — 拒绝，冗余且容易出错

### 7. engine_id 格式
- **Decision**: `engine_id = "{project_id}/{version_or_name}"`
- **Rationale**: 直观唯一，与 server 端的 project 结构对齐
- **Alternative**: UUID — 拒绝，用户需要可读性强的 ID 来管理引擎

## Risks / Trade-offs

- **[Risk]** nndeploy 线程安全性未知 → **Mitigation**: 每个引擎配 asyncio Lock，串行化同一引擎的推理请求；不同引擎可并行
- **[Risk]** nndeploy Python API 可能不稳定或文档不全 → **Mitigation**: 封装适配层，必要时降级到 CLI 调用
- **[Risk]** 多引擎同时驻留导致 GPU OOM → **Mitigation**: 设置 MAX_ENGINES 限制，LRU 淘汰策略
- **[Risk]** 模型文件较大，首次下载慢 → **Mitigation**: 异步下载 + 进度反馈，本地缓存持久化到卷
- **[Risk]** server 和 deploy 版本不一致导致 workflow.json 不兼容 → **Mitigation**: workflow.json 中包含 nndeploy 版本信息，deploy 加载时校验
- **[Trade-off]** 内存占用 vs 响应速度：驻留内存快但占资源，需根据场景配置 MAX_ENGINES

## Migration Plan

1. **Phase 1**: 实现 deploy 容器基础代码（FastAPI + 引擎池 + 缓存）
2. **Phase 2**: 实现 server 端模型下载 API
3. **Phase 3**: 实现 Dockerfile.cpu 和本地验证
4. **Phase 4**: 实现 Dockerfile.gpu 和 CUDA 推理验证
5. **Phase 5**: 更新 docker-compose.yml 编排

Rollback: 停止 deploy 容器，删除 deploy/ 目录，不影响 server 端功能。

## Open Questions

- nndeploy Python API 的具体调用方式需要调研确认
- 是否需要支持模型热更新（同名版本重新加载）？
- GPU 镜像中 TensorRT 版本的兼容性矩阵？
