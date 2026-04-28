## 1. Deploy 容器基础架构

- [x] 1.1 创建 `deploy/` 目录结构
- [x] 1.2 编写 `deploy/requirements.txt`（fastapi, uvicorn, nndeploy, requests, Pillow, numpy）
- [x] 1.3 编写 `deploy/main.py`（FastAPI 应用入口，含路由注册）
- [x] 1.4 编写 `deploy/engine_pool.py`（推理引擎池管理：加载、卸载、LRU 淘汰）
- [x] 1.5 编写 `deploy/server_client.py`（调用 xclabel-server 的 HTTP 客户端）
- [x] 1.6 编写 `deploy/nndeploy_adapter.py`（nndeploy 模型/工作流加载封装）

## 2. Deploy 容器 API 实现

- [x] 2.1 实现 `POST /load/model` 端点（下载模型 → nndeploy 加载 → 放入引擎池）
- [x] 2.2 实现 `POST /load/workflow` 端点（下载 workflow → 构建 Pipeline → 放入引擎池）
- [x] 2.3 实现 `POST /infer` 端点（支持 base64 图片和 image_url）
- [x] 2.4 实现 `GET /engines` 端点（列出已加载引擎）
- [x] 2.5 实现 `POST /unload` 端点（卸载指定引擎）
- [x] 2.6 实现 `POST /unload/all` 端点（卸载全部引擎）
- [x] 2.7 实现 `GET /health` 端点（健康检查）

## 3. Dockerfile 构建

- [x] 3.1 编写 `deploy/Dockerfile.cpu`（基于 python:3.12-slim）
- [x] 3.2 编写 `deploy/Dockerfile.gpu`（基于 nvidia/cuda:12.1.1-runtime）
- [x] 3.3 本地构建 CPU 镜像并验证 `docker build -f deploy/Dockerfile.cpu .`
- [ ] 3.4 本地构建 GPU 镜像并验证 `docker build -f deploy/Dockerfile.gpu .`

## 4. xclabel-server API 扩展

- [x] 4.1 实现 `GET /api/model/download`（按 project + version 下载模型包）
- [x] 4.2 实现 `GET /api/model/versions`（列出项目的所有模型版本）
- [x] 4.3 实现 `GET /api/workflow/export`（导出 workflow.json）
- [x] 4.4 实现 `GET /api/workflow/list`（列出项目的工作流）

## 5. Docker Compose 编排

- [x] 5.1 更新根目录 `docker-compose.yml`，添加 `xclabel-deploy-cpu` 服务
- [x] 5.2 更新根目录 `docker-compose.yml`，添加 `xclabel-deploy-gpu` 服务
- [x] 5.3 配置服务间网络通信（server → deploy 通过服务名）
- [x] 5.4 配置缓存卷持久化（deploy-cache-cpu, deploy-cache-gpu）

## 6. 训练流程扩展

- [x] 6.1 训练完成后自动导出 ONNX 到版本目录
- [x] 6.2 生成 `deploy_metadata.json` 包含模型元数据
- [x] 6.3 训练完成通知添加部署提示信息

## 7. 集成测试

- [x] 7.1 测试 deploy 容器加载模型并推理（CPU 模式）
- [x] 7.2 测试 deploy 容器加载工作流并推理
- [x] 7.3 测试多引擎同时加载和切换
- [x] 7.4 测试 LRU 淘汰策略
- [x] 7.5 测试本地缓存命中/未命中
- [x] 7.6 测试 GPU 镜像推理（如有 CUDA 环境）

## 8. 文档更新

- [x] 8.1 更新 README.md，添加 deploy 容器使用说明
- [x] 8.2 添加 deploy API 文档（FastAPI 自动生成的 /docs 页面说明）
- [x] 8.3 添加 docker-compose 部署指南
