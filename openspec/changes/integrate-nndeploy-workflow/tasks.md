## 1. nndeploy-app 独立容器部署

- [x] 1.1 确认 nndeploy-app 安装方式（基于 xclabel-server 镜像或单独构建）
- [x] 1.2 更新 `docker-compose.yml`：新增 `nndeploy-app` 服务（端口 8002）
- [x] 1.3 更新 `docker-compose.yml`：定义命名卷 `nndeploy-resources`，挂载到 server 和 nndeploy-app 容器
- [x] 1.4 确保 nndeploy-app 容器启动后自动创建 `resources/{models,workflow,images,db}` 目录结构

## 2. 模型手动发布到 nndeploy-app

- [x] 2.1 更新前端模型版本列表：每行删除按钮前增加"发布"按钮
- [x] 2.2 前端点击"发布"后调用确认弹窗，显示目标路径 `resources/models/<project>_<version>.onnx`
- [x] 2.3 后端实现 `POST /api/model/publish`：从 `projects/<project>/models/<version>/best.onnx` 复制到共享卷 `resources/models/<project>_<version>.onnx`
- [x] 2.4 后端实现发布幂等：同一模型重复发布覆盖原文件
- [x] 2.5 后端返回发布结果（成功/已存在/模型文件不存在）
- [x] 2.6 确保 nndeploy-app WebUI 可识别已发布的模型（通过文件系统扫描或刷新机制）

## 3. Flask Workflow API 代理

- [x] 3.1 实现 `GET /api/nndeploy/workflows`：转发到 nndeploy-app `/api/workflows`，返回 workflow 列表
- [x] 3.2 实现 `GET /api/nndeploy/workflow/download`：转发到 nndeploy-app `/api/workflow/download/{id}`，返回 workflow JSON
- [x] 3.3 配置 Flask 到 nndeploy-app 的内部网络地址（docker-compose 服务名或环境变量）

## 4. Deploy 容器 Workflow 执行扩展

- [x] 4.1 更新 `server_client.py`：增加下载 nndeploy workflow 的方法
- [x] 4.2 更新 `nndeploy_adapter.py`：实现 nndeploy DAG workflow 加载和执行
- [x] 4.3 更新 `engine_pool.py`：workflow 引擎生命周期管理
- [x] 4.4 更新 `main.py`：新增 `/workflows` 端点转发 workflow 列表

## 5. 集成测试

- [ ] 5.1 测试 docker-compose 启动 nndeploy-app 独立容器，WebUI 可访问
- [ ] 5.2 测试手动发布模型后，resources/models/ 下出现 `<project>_<version>.onnx`
- [ ] 5.3 测试 nndeploy-app WebUI 中基于已发布模型编辑 workflow
- [ ] 5.4 测试 deploy 容器下载并执行 nndeploy workflow
- [ ] 5.5 测试多 project 的模型发布隔离（文件名冲突避免）

## 6. 文档更新

- [x] 6.1 更新 README.md：nndeploy-app 独立容器集成说明
- [x] 6.2 更新 README.md：模型发布到 nndeploy-app 的操作流程
- [x] 6.3 更新 README.md：workflow 创建和使用流程
- [x] 6.4 更新 docker-compose 部署指南
