## Context

xclabel 目前将所有图片存放在 `uploads/` 目录，标注数据存放在 `uploads/annotations/annotations.json`，类别配置存放在 `uploads/annotations/classes.json`。这种单一空间设计无法支持多个独立标注任务的数据隔离。

## Goals / Non-Goals

**Goals:**
- 支持创建、列出、切换、重命名、删除标注工程
- 每个工程的图片、标注、类别完全隔离
- 工程管理页面作为应用入口，用户先选工程再进入标注
- 向后兼容：现有数据自动迁移到默认工程

**Non-Goals:**
- 工程级别的权限控制（多用户）
- 工程之间的数据复制或合并
- 工程模板功能
- 工程历史版本管理

## Decisions

### 1. 目录结构：`projects/<project_name>/`

每个工程一个独立目录，内部结构与当前 `uploads/` 一致：
```
projects/
  project-a/
    images/
    annotations/
      annotations.json
      classes.json
  project-b/
    ...
```

**Rationale**: 简单、文件系统天然隔离、易于备份和迁移。不需要数据库。

**Alternative considered**: 数据库记录工程元数据 + 统一目录用前缀区分。Rejected：增加复杂度，无额外收益。

### 2. 当前工程状态存储：服务端 session + 前端 localStorage

后端通过 `session['current_project']` 跟踪当前工程，前端通过 `localStorage` 缓存当前工程名称用于页面刷新后恢复。

**Rationale**: 无状态 HTTP 请求需要知道上下文。Session 适合服务端，localStorage 适合前端刷新恢复。

### 3. 默认工程自动创建

若 `projects/` 目录为空，自动创建名为 "default" 的工程，并将现有 `uploads/` 数据迁移进去。

**Rationale**: 保证向后兼容，现有用户无感知升级。

### 4. 工程名作为目录名，限制字符

工程名只允许字母、数字、中文、下划线和连字符。用于目录创建时进行安全清理。

**Rationale**: 防止路径遍历和特殊字符导致的文件系统问题。

## Risks / Trade-offs

- **[Risk]** 工程重命名需要重命名目录，大量文件时可能耗时 → **Mitigation**: 异步处理或限制重命名频率
- **[Risk]** 删除工程会永久删除数据 → **Mitigation**: 增加确认弹窗，考虑未来增加回收站
- **[Trade-off]** 工程数量受文件系统限制 → 预期个人/小团队使用，工程数量不会很大（<100）

## Migration Plan

1. 首次启动时检查 `projects/` 是否存在
2. 若不存在，创建 `projects/default/` 并将 `uploads/` 内容移入
3. 后续所有文件操作指向 `projects/<current_project>/`
4. `uploads/` 目录在完成迁移后可删除（保留一个版本作为备份）
