## 1. 后端：新增 API 路由

- [x] 1.1 在 `app.py` 中添加 `POST /api/auto-label/vlm-sam` 路由，实现 VLM 检测 → SAM 精化的串联管道
- [x] 1.2 集成 GPU 锁：`acquire_gpu("vlm_sam")` / `release_gpu("vlm_sam")`
- [x] 1.3 添加 SAM 降级策略（SAM 失败时保存 VLM 矩形框）
- [x] 1.4 SocketIO 进度推送 `vlm_sam_progress` 事件

## 2. 前端：AI 模态框新增模式

- [x] 2.1 在 `templates/index.html` 的 AI 模态框添加「VLM+SAM 精细化」模式选项
- [x] 2.2 在 `templates/index.html` 内联脚本中添加 `startVlmSamLabel()` 函数

## 3. 测试

- [x] 3.1 手动验证 VLM → SAM 管道能完整走通
