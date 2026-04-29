## ADDED Requirements

### Requirement: nndeploy-app 作为独立 Docker 容器运行
nndeploy-app SHALL 作为独立的 Docker 服务运行，与 xclabel-server 容器通过命名卷共享 `resources/` 目录。

#### Scenario: 容器启动后服务可用
- **WHEN** docker-compose 启动 nndeploy-app 容器
- **THEN** nndeploy-app 在 8002 端口响应请求
- **THEN** xclabel-server 容器可通过共享卷访问同一 `resources/` 目录

### Requirement: nndeploy-app 使用全局 resources 目录
nndeploy-app 的 resources 目录 SHALL 通过 Docker 命名卷挂载，包含 models、workflow、images、db 等子目录。workflow 编辑由 nndeploy-app WebUI 独立完成，xclabel 不介入 workflow 的创建和编辑。

#### Scenario: nndeploy-app 启动后识别全局 resources
- **WHEN** nndeploy-app 容器启动
- **THEN** 自动创建并加载 `resources/` 目录下的 workflow、template 和模型

### Requirement: 手动发布模型到 nndeploy-app
xclabel 前端模型版本列表 SHALL 提供"发布"按钮。用户点击后，系统将指定版本的 ONNX 模型复制到共享卷 `resources/models/`。

#### Scenario: 用户手动发布模型
- **WHEN** 用户在模型版本列表点击"发布"按钮
- **THEN** 系统确认弹窗显示目标路径
- **THEN** 系统将 `projects/<project>/models/<version>/best.onnx` 复制到 `resources/models/<project>_<version>.onnx`
- **THEN** 用户在 nndeploy-app WebUI 中可以看到已发布的模型并选用

#### Scenario: 重复发布同一模型
- **WHEN** 用户重复发布已存在的模型
- **THEN** 系统覆盖原文件，保持幂等

### Requirement: Flask 提供 workflow 列表和下载 API
Flask SHALL 提供 REST API 供外部查询和下载 nndeploy-app 管理的 workflow JSON。Flask 代理转发到 nndeploy-app 的 API，自身不提供 workflow 的创建、编辑或上传接口。

#### Scenario: 列出 workflow
- **WHEN** 客户端调用 `GET /api/nndeploy/workflows`
- **THEN** Flask 转发请求到 nndeploy-app 的 `/api/workflows`
- **THEN** 返回 workflow 列表（id、name_、developer_、desc_）

#### Scenario: 下载指定 workflow
- **WHEN** 客户端调用 `GET /api/nndeploy/workflow/download?id=<workflow_id>`
- **THEN** Flask 转发请求到 nndeploy-app 的 `/api/workflow/download/{id}`
- **THEN** 返回对应的 workflow JSON 文件内容
