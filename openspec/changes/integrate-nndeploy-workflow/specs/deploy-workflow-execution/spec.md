## ADDED Requirements

### Requirement: Deploy 容器支持下载 nndeploy workflow
xclabel-deploy 容器 SHALL 通过 HTTP 从 xclabel-server 下载 nndeploy-app 格式的 workflow JSON。

#### Scenario: 下载 workflow 到本地缓存
- **WHEN** deploy 容器调用 `GET /api/nndeploy/workflow/download?project=test&name=detect.json`
- **THEN** workflow JSON 保存到 `/app/cache/workflows/test_detect.json`

### Requirement: Deploy 容器加载并执行 nndeploy workflow
deploy 容器 SHALL 支持加载 nndeploy-app 格式的 workflow JSON 并通过 nndeploy DAG 执行推理。

#### Scenario: 加载 workflow 到引擎池
- **WHEN** 客户端调用 `POST /load/workflow` 指定 nndeploy workflow 名称
- **THEN** deploy 容器下载 workflow JSON
- **THEN** 使用 nndeploy DAG 构建执行图
- **THEN** workflow 放入引擎池管理

#### Scenario: 执行 workflow 推理
- **WHEN** 客户端调用 `POST /infer` 指定 workflow 引擎
- **THEN** deploy 容器通过 nndeploy DAG 执行 workflow
- **THEN** 返回检测结果

### Requirement: Deploy 端列出可用的 nndeploy workflow
deploy 容器 SHALL 提供 API 查询 server 端可用的 nndeploy workflow 列表。

#### Scenario: 查询 workflow 列表
- **WHEN** 客户端调用 deploy 的 `GET /workflows?project=test`
- **THEN** deploy 容器转发请求到 server 的 `/api/nndeploy/workflows`
- **THEN** 返回 workflow 名称列表
