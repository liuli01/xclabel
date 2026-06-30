## 1. PipelineManager 核心引擎 (deploy 端)

- [x] 1.1 创建 `deploy/pipeline_manager.py`，实现 workflow.yaml 加载与 Pydantic 校验
- [x] 1.2 实现 DAG 构建与拓扑排序（检测循环依赖）
- [x] 1.3 实现 `PipelineContext` 数据传递类
- [x] 1.4 实现 `PipelineManager.execute()` 主循环：按拓扑序执行节点
- [x] 1.5 实现 YOLO 节点执行器（复用 YoloAdapter，从引擎池获取模型）
- [x] 1.6 实现 Condition 节点执行器（安全 eval 沙箱）
- [x] 1.7 实现 Output 节点执行器（合并各节点结果）

## 2. VLLM 节点 (deploy 端)

- [x] 2.1 创建 `deploy/vllm_client.py`，封装 AsyncOpenAI 调用
- [x] 2.2 实现 ROI 裁剪功能（supervision 库提取检测区域）
- [x] 2.3 实现超时/重试逻辑（30s 超时，超时跳过）

## 3. Deploy API 扩展

- [x] 3.1 `deploy/main.py` 新增 `POST /pipeline/execute` 端点
- [x] 3.2 `deploy/main.py` 新增 `POST /pipeline/load` 端点（加载 workflow.yaml）
- [x] 3.3 `deploy/requirements.txt` 新增 openai / supervision / pyyaml 依赖

## 4. Server 端 Workflow CRUD API

- [x] 4.1 `app.py` 实现 `GET /api/workflow/list` 端点
- [x] 4.2 `app.py` 实现 `GET /api/workflow/get` 端点
- [x] 4.3 `app.py` 实现 `POST /api/workflow/save` 端点（LiteGraph JSON → workflow.yaml）
- [x] 4.4 `app.py` 实现 `POST /api/workflow/delete` 端点
- [x] 4.5 `app.py` 实现 `POST /api/workflow/deploy` 端点（调用 deploy 服务）
- [x] 4.6 `app.py` 实现 `POST /api/workflow/undeploy` 端点

## 5. 前端工作流编辑器

- [x] 5.1 创建 `templates/workflow.html` 页面框架
- [x] 5.2 引入 LiteGraph.js 并初始化画布
- [x] 5.3 注册 YOLO Node 类型（模型选择、conf/iou 参数面板）
- [x] 5.4 注册 Condition Node 类型（表达式输入面板）
- [x] 5.5 注册 VLLM Node 类型（API URL、模型名、prompt 编辑面板）
- [x] 5.6 注册 Output Node 类型
- [x] 5.7 实现保存按钮：序列化画布 → POST /api/workflow/save
- [x] 5.8 实现部署按钮：先保存 → POST /api/workflow/deploy
- [x] 5.9 导航栏添加"流程编排"入口，跳转到 workflow.html
- [x] 5.10 已创建 workflow 列表展示和加载功能

## 6. 集成测试

- [x] 6.1 编写 PipelineManager 单元测试
- [x] 6.2 编写 VLLM 客户端单元测试
- [x] 6.3 端到端测试：创建 workflow → 保存 → 部署 → 执行 → 查看结果
- [x] 6.4 测试 YOLO→Condition→Output 正常路径
- [x] 6.5 测试 YOLO→Condition→VLLM→Output 智能核实路径
- [x] 6.6 测试 VLLM 超时容错
