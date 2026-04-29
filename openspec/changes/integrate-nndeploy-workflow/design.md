## Context

当前 xclabel 采用双容器架构：xclabel-server（Flask）负责标注和训练，xclabel-deploy（FastAPI）负责推理部署。训练完成后自动导出 ONNX 模型，deploy 容器通过 HTTP 从 server 下载模型并加载到 nndeploy 引擎池执行推理。

用户本地已安装 nndeploy PyPI 包，并通过 `nndeploy-app --port 8000` 启动了可视化 workflow 编排服务。nndeploy-app 自动生成 `resources` 目录，包含 `models/`、`workflow/`、`images/` 等子目录。用户希望将这个能力集成到 xclabel 中，使 server 端不仅能训练模型，还能通过可视化界面编排推理 workflow；deploy 端可以拉取这些 workflow 并执行。

## Goals / Non-Goals

**Goals:**
- nndeploy-app 作为独立 Docker 容器运行（端口 8002），与 xclabel-server 共享 `resources/` 卷
- nndeploy-app 的 resources 目录与 xclabel `projects/` 目录通过共享卷关联，实现模型和 workflow 的统一存储
- 提供 API 让 deploy 容器发现和下载 server 端的 workflow JSON
- deploy 容器支持加载 nndeploy-app 格式的 workflow 并执行推理
- xclabel 模型版本列表增加"发布到 nndeploy-app"按钮，手动将 ONNX 模型移动到 `resources/models/<project>_<version>.onnx`，供用户在 WebUI 中编辑 workflow 时选用

**Non-Goals:**
- 不替换当前 FastAPI deploy 容器为 nndeploy-app（保持现有架构）
- 不修改 nndeploy-app 前端 UI（使用原生界面）
- 不支持 GPU 推理 workflow 的自动分配（保持现有 CPU/GPU 服务分离）

## Decisions

**1. nndeploy-app 作为独立 Docker 容器运行**
- 理由：生命周期独立，nndeploy-app 重启不影响 Flask；资源隔离；职责更清晰。
- xclabel-server 和 nndeploy-app 通过命名卷共享 `resources/` 目录。
- 替代方案：子进程。拒绝原因：子进程崩溃会导致整个 server 容器重启；nndeploy-app 和 Flask 日志混合。

**2. nndeploy-app 使用全局 resources/ 目录，不绑定 project**
- 理由：nndeploy-app 自身为全局 workflow 编辑器，无 project 概念。workflow 编辑统一由 WebUI 完成。
- 结构：`resources/{models,workflow,images,videos,db,...}`
- xclabel 模型版本列表提供"发布"按钮，手动将 ONNX 模型复制到 `resources/models/<project>_<version>.onnx`，供用户在 nndeploy-app WebUI 中选用。

**3. Flask 代理 nndeploy-app 的 workflow 列表和下载 API**
- 理由：deploy 容器统一通过 Flask 入口发现资源。Flask 转发请求到 nndeploy-app（端口 8002），对外暴露 `/api/nndeploy/workflows` 和 `/api/nndeploy/workflow/download`。
- workflow 编辑由 nndeploy-app WebUI 完成，Flask 不提供上传/保存接口。

**4. Workflow 执行复用现有 deploy 容器的 engine_pool**
- 理由：现有 engine_pool 已支持 LRU 淘汰和并发锁，workflow 作为另一种引擎类型加载即可。

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| nndeploy-app 前端下载耗时（首次启动需下载 dist.zip） | Dockerfile 构建时预下载前端资源，避免运行时延迟 |
| nndeploy-app 和 Flask 端口冲突 | nndeploy-app 使用 8002 端口，Flask 保持 5000 |
| resources 目录膨胀 | 定期清理未使用的模型和 workflow，由用户手动管理 |
| deploy 端解析 nndeploy-app workflow 格式困难 | workflow 为完整 DAG 定义，deploy 端通过 nndeploy DAG API 加载执行 |

## Migration Plan

1. 更新 docker-compose.yml：新增 nndeploy-app 服务、定义共享 resources 卷
2. 更新 app.py：新增 `/api/model/publish` 端点（复制模型到共享 resources/models/）、新增 workflow 代理 API
3. 更新前端模型版本列表：删除按钮前增加"发布"按钮
4. 更新 deploy 端：扩展 server_client.py 支持 workflow 拉取
5. 本地验证：启动完整栈，测试手动发布模型→WebUI 编辑 workflow→下载→执行流程

## Open Questions

- nndeploy-app 前端资源 dist.zip 是否在 Dockerfile 构建时预下载？（还是依赖运行时首次下载）
- deploy 端执行 workflow 的方式：直接通过 nndeploy DAG API 加载完整 DAG，还是解析 workflow JSON 提取模型路径后用现有推理逻辑？（当前决策为 DAG 加载，但需验证可行性）
- 已发布到 nndeploy-app 的模型文件是否需要和 project 下的原始模型保持同步？（如 project 下删除模型后，resources/models/ 是否同步删除）
