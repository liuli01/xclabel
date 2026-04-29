## Why

当前 xclabel 的训练模型通过 ONNX 导出后，deploy 容器可以直接加载推理，但缺乏可视化的 workflow 编排能力。用户本地环境已安装 nndeploy 并启动 nndeploy-app 服务，希望将其集成到 xclabel 架构中：server 端同时提供 Flask 标注/训练服务和 nndeploy-app workflow 编排服务，deploy 端可以拉取 server 端生成的 workflow JSON 并执行推理。

## What Changes

- nndeploy-app 作为独立 Docker 容器运行（端口 8002），与 xclabel-server 共享 `resources` 命名卷
- xclabel-server 新增 API：手动发布 ONNX 模型到共享 `resources/models/`、代理 nndeploy-app workflow 列表和下载
- deploy 容器扩展：支持从 server 拉取 nndeploy-app 格式的 workflow 并加载执行
- workflow 创建和编辑完全由 nndeploy-app WebUI 完成，xclabel 不介入

## Capabilities

### New Capabilities

- `nndeploy-workflow-management`: nndeploy-app 服务集成、resources 目录关联、workflow 的创建/编辑/导出/列表管理
- `deploy-workflow-execution`: deploy 容器从 server 拉取 nndeploy workflow 并执行推理

### Modified Capabilities

- （无现有 spec 需要修改，本次为纯新增功能）

## Impact

- `docker-compose.yml`: 新增 nndeploy-app 独立服务、定义共享 resources 命名卷
- `app.py`: 新增 `/api/model/publish` 端点、新增 workflow 代理 API
- 前端模型版本列表：新增"发布"按钮
- `deploy/`: 扩展 server_client.py 和 workflow 加载逻辑
- `Dockerfile`: server 镜像需安装 nndeploy（nndeploy-app 容器复用同一镜像）
