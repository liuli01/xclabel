## ADDED Requirements

### Requirement: 列出项目下的所有 workflow
系统 SHALL 提供 GET /api/workflow/list?project=<name> 端点，列出指定项目下的所有 workflow 文件。

#### Scenario: 列出 workflow
- **WHEN** GET /api/workflow/list?project=sv30_seg
- **THEN** 返回 workflow 名称列表和状态

### Requirement: 获取单个 workflow
系统 SHALL 提供 GET /api/workflow/get?project=<name>&workflow=<name> 端点，返回指定 workflow 的完整配置。

#### Scenario: 获取 workflow
- **WHEN** GET /api/workflow/get?project=sv30_seg&workflow=ccrc_fall_detection
- **THEN** 返回 workflow.yaml 内容

### Requirement: 保存 workflow
系统 SHALL 提供 POST /api/workflow/save 端点，接收 LiteGraph JSON 格式，转换为 workflow.yaml 存储到 projects/<project>/workflows/<name>.yaml。

#### Scenario: 保存新 workflow
- **WHEN** POST /api/workflow/save 提供 project、name 和 graph JSON
- **THEN** 系统将 graph 转换为 workflow.yaml 并存盘

### Requirement: 删除 workflow
系统 SHALL 提供 POST /api/workflow/delete 端点，删除指定 workflow 文件。

#### Scenario: 删除 workflow
- **WHEN** POST /api/workflow/delete 提供 project 和 name
- **THEN** 对应 workflow.yaml 被删除

### Requirement: 部署 workflow 到 deploy 服务
系统 SHALL 提供 POST /api/workflow/deploy 端点，将 workflow.yaml 推送到 deploy 服务。

#### Scenario: 部署 workflow
- **WHEN** POST /api/workflow/deploy 提供 project 和 name
- **THEN** 调用 deploy 服务的 /load/workflow 端点加载该 workflow
