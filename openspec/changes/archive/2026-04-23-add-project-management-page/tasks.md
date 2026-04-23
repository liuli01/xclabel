## 1. 后端基础架构

- [x] 1.1 在 `app.py` 中新增 `PROJECTS_FOLDER` 常量，路径为 `projects/`
- [x] 1.2 实现 `get_current_project()` 辅助函数，从 session 读取当前工程名，默认为 "default"
- [x] 1.3 实现 `get_project_path(project_name)` 辅助函数，返回工程完整目录路径
- [x] 1.4 实现 `ensure_default_project()` 函数：首次启动时检查 `projects/` 是否存在，若不存在则创建 `projects/default/` 并迁移现有 `uploads/` 数据
- [x] 1.5 修改所有硬编码的 `UPLOAD_FOLDER` 引用，改为基于当前工程路径动态获取
- [x] 1.6 修改 `ANNOTATIONS_FOLDER`、`ANNOTATIONS_FILE`、`CLASSES_FILE` 为基于当前工程路径

## 2. 后端 API — 工程管理

- [x] 2.1 实现 `GET /api/projects`：列出所有工程，返回名称、图片数量、最后修改时间
- [x] 2.2 实现 `POST /api/projects`：创建新工程，参数 `{name}`，验证名称唯一性和合法性
- [x] 2.3 实现 `PUT /api/projects/<name>`：重命名工程，参数 `{new_name}`，验证新名称唯一性
- [x] 2.4 实现 `DELETE /api/projects/<name>`：删除工程及所有数据，增加确认机制
- [x] 2.5 实现 `POST /api/projects/switch`：切换当前工程，参数 `{name}`，写入 session
- [x] 2.6 实现 `GET /api/projects/current`：获取当前工程信息

## 3. 后端 API — 现有接口适配

- [x] 3.1 修改 `/api/images` 及相关路由，使用当前工程路径读取图片
- [x] 3.2 修改 `/api/annotations/<image_name>` 及相关路由，使用当前工程路径读写标注
- [x] 3.3 修改 `/api/upload/roboflow` 及相关导入路由，导入到当前工程目录
- [x] 3.4 修改 `/api/export` 及相关导出路由，从当前工程目录导出
- [x] 3.5 修改 AI 标注相关路由，使用当前工程路径

## 4. 前端 — 工程管理页面

- [x] 4.1 创建 `templates/projects.html`：工程管理入口页面，包含工程列表卡片布局
- [x] 4.2 实现工程列表渲染：展示工程名称、图片数量、最后修改时间、操作按钮
- [x] 4.3 实现创建工程弹窗：输入工程名称，验证并提交
- [x] 4.4 实现重命名功能：点击重命名按钮弹出输入框
- [x] 4.5 实现删除功能：点击删除按钮弹出确认对话框
- [x] 4.6 实现"进入标注"按钮：点击后切换工程并跳转到 `/label`

## 5. 前端 — 标注页面适配

- [x] 5.1 修改 `index.html` 路由改为 `/label`，新增返回工程管理页面链接
- [x] 5.2 在标注页面顶部显示当前工程名称
- [x] 5.3 修改 `script.js` 初始化逻辑，加载当前工程的图片和标注

## 6. 前端 — 工程状态管理

- [x] 6.1 实现 `localStorage` 缓存当前工程名，页面刷新后自动恢复
- [x] 6.2 实现工程切换时的状态清理（清空图片缓存、标注缓存、撤销栈等）

## 7. 测试与验证

- [x] 7.1 运行 `ruff check .` 确保代码风格合规（无新增错误）
- [x] 7.2 测试创建工程：验证目录创建、初始文件生成
- [x] 7.3 测试切换工程：验证图片和标注数据隔离
- [x] 7.4 测试导入数据到指定工程：验证图片和标注正确保存
- [x] 7.5 测试导出数据：验证从当前工程正确导出
- [x] 7.6 测试删除工程：验证目录和数据完全清除
- [x] 7.7 测试默认工程迁移：验证现有数据无感迁移
