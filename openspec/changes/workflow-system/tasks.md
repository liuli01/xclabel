## 1. 完善工作流编辑器

- [x] 1.1 `templates/workflow.html` 修复 LiteGraph.js 节点注册（YOLO/Condition/VLLM/Output）
- [x] 1.2 实现保存按钮：画布序列化 → POST /api/workflow/save
- [x] 1.3 实现部署按钮：先保存 → POST /api/workflow/deploy
- [x] 1.4 已创建 workflow 列表展示和加载功能
- [x] 1.5 修复 projects.html 中"工作流"按钮跳转（getCurrentProjectName 未定义问题）

## 2. 工作流多模型支持

- [x] 2.1 `pipeline_manager.py` 支持 pipeline 中引用多个不同模型
- [x] 2.2 `main.py` 的 `/pipeline/execute` 处理多模型加载依赖

## 3. Deploy 标注图生成

- [x] 3.1 `deploy/yolo_adapter.py` 新增 `infer_annotated()` 方法（supervision 渲染标注图）
- [x] 3.2 `deploy/main.py` 新增 `/infer/annotated` 端点

## 4. 端到端测试与完善

- [x] 4.1 验证 `_litegraph_to_workflow` 转换函数正确性
- [x] 4.2 新增：workflow 执行页面（编辑器加"运行"按钮，选择图片执行 pipeline）
- [x] 4.3 新增：workflow 取消部署（卸载）按钮
- [ ] 4.4 验证 workflow 保存 → 部署 → 执行完整链路
- [ ] 4.5 验证 `/infer/annotated` 标注图端点
