## Why

xclabel 目前已有完整的标注和训练能力，但训练好的模型只能下载到本地手动部署，缺乏标准化的推理服务能力。通过引入独立的 deploy 推理容器，可以将 xclabel 升级为完整的 MLOps 平台：server 负责训练与编排，deploy 负责推理执行，形成"训练-编排-部署-推理"的完整闭环。

## What Changes

- 新增 `deploy/` 目录，包含独立的推理部署容器代码（FastAPI + nndeploy）
- 新增 CPU 和 GPU 两种 Dockerfile（`Dockerfile.cpu`、`Dockerfile.gpu`）
- 新增推理引擎池，支持同时加载多个模型和工作流，动态切换
- 更新 `docker-compose.yml`，支持 `xclabel-server` + `xclabel-deploy-cpu` + `xclabel-deploy-gpu` 三服务编排
- xclabel-server 新增模型下载 API 和工作流导出 API，供 deploy 容器调用
- deploy 容器提供 REST API：`/load/model`、`/load/workflow`、`/infer`、`/engines`、`/unload`

## Capabilities

### New Capabilities
- `deploy-inference-service`: 独立推理部署容器，支持从 server 拉取模型/工作流并对外提供 REST API 推理服务
- `model-download-api`: xclabel-server 端新增按 project + version 下载模型的 API
- `workflow-export-api`: xclabel-server 端新增导出 workflow.json 的 API
- `multi-engine-pool`: deploy 容器内部推理引擎池，支持同时驻留多个模型/工作流

### Modified Capabilities
- `yolo-model-training`: 训练完成后自动触发模型导出，生成 deploy 可消费的模型包格式

## Impact

- **后端**: 新增 deploy 目录及 FastAPI 服务代码（约 4-5 个 Python 文件）
- **API**: xclabel-server 新增 `/api/model/download`、`/api/model/versions`、`/api/workflow/export`、`/api/workflow/list`
- **Docker**: 新增 `deploy/Dockerfile.cpu`、`deploy/Dockerfile.gpu`，更新根目录 `docker-compose.yml`
- **依赖**: deploy 容器依赖 nndeploy、fastapi、uvicorn
- **部署**: 支持单机 docker-compose 部署，也支持 server 与 deploy 分离部署（通过 `SERVER_URL` 环境变量）
