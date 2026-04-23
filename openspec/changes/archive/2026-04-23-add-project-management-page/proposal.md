## Why

当前 xclabel 只有一个全局数据集和标注空间，无法同时管理多个独立的标注任务。用户需要一种方式来隔离不同项目的数据、类别和标注，避免混淆和覆盖。

## What Changes

- **新增工程管理页面**：作为应用入口页面，展示所有标注工程的列表
- **工程 CRUD 操作**：创建新工程、重命名工程、删除工程、切换当前工程
- **工程数据隔离**：每个工程拥有独立的图片目录、标注文件（annotations.json）和类别配置（classes.json）
- **工程级导入/导出**：数据集导入和导出均在当前工程范围内操作
- **标注页面适配**：进入标注页面时自动加载当前工程的图片和标注数据

## Capabilities

### New Capabilities
- `project-management`: 工程的创建、列表、切换、重命名和删除
- `project-data-isolation`: 工程级别的文件目录隔离和配置管理

### Modified Capabilities
- （无现有 spec 需要修改）

## Impact

- 前端：新增 `projects.html` 工程管理页面，修改 `index.html` 以支持工程上下文
- 后端：新增 `/api/projects` 系列路由，修改现有 API 以接受 `project_id` 参数
- 数据存储：从单一 `uploads/` 目录改为 `projects/<project_name>/` 结构
- 初始化流程：首次访问时若不存在工程，自动创建默认工程
