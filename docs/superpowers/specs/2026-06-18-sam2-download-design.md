# SAM2 模型下载功能设计

## 概述

在工程管理页面（`projects.html`）的设置模态框中，新增 SAM2 分割模型下载功能。允许用户从中国 HF 镜像站（`hf-mirror.com`）下载三个版本的 SAM2 权重文件，并实时查看下载进度。

## 背景

- 项目使用 SAM2 进行智能分割标注（VLM+SAM 管线）
- 三个模型版本在 `ai_manager.py` 中定义：`tiny`、`small`、`base_plus`
- 权重文件存放于 `models/` 目录
- 当前仅 `tiny` 和 `small` 的权重文件存在，`base_plus` 缺失
- 下载源为中国 HF 镜像站 `hf-mirror.com`，解决国内网络访问 HuggingFace 慢的问题

## 后端 API 设计

### 1. 检查模型状态 `GET /api/sam/check-models`

返回三个 SAM2 版本的本地存在状态和大小信息。

```json
{
  "success": true,
  "models": [
    {"id": "tiny", "name": "SAM 2 Hiera Tiny", "checkpoint": "sam2_hiera_tiny.pt", "available": true, "size_mb": 155},
    {"id": "small", "name": "SAM 2 Hiera Small", "checkpoint": "sam2_hiera_small.pt", "available": true, "size_mb": 184},
    {"id": "base_plus", "name": "SAM 2 Hiera Base+", "checkpoint": "sam2_hiera_base_plus.pt", "available": false, "size_mb": 298}
  ]
}
```

### 2. 下载模型 `GET /api/sam/download-models?models=tiny,base_plus`

SSE（Server-Sent Events）流式端点，与现有 YOLO 模型下载模式一致。

**请求参数：**
- `models`：逗号分隔的模型标识列表（`tiny`、`small`、`base_plus`）

**SSE 事件流：**

```json
// 开始下载某个模型
{"status": "downloading", "model": "sam2_hiera_base_plus.pt", "progress": 10, "downloaded_mb": 30, "total_mb": 298}

// 单个模型下载完成
{"status": "model_completed", "model": "sam2_hiera_base_plus.pt", "progress": 60}

// 全部完成
{"status": "completed", "models": ["sam2_hiera_tiny.pt", "sam2_hiera_base_plus.pt"]}

// 错误
{"status": "error", "model": "sam2_hiera_base_plus.pt", "error": "下载失败: ...", "progress": 0}
```

**下载源 URL 映射：**

| 标识 | HF 镜像 URL |
|---|---|
| `tiny` | `https://hf-mirror.com/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt` |
| `small` | `https://hf-mirror.com/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt` |
| `base_plus` | `https://hf-mirror.com/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt` |

## 前端 UI 设计

### 位置

在 `projects.html` 的设置模态框（`#settingsModal`）中，YOLO 训练环境区域下方，新增一个带分隔线的 SAM2 下载卡片。

### HTML 结构

```
┌─────────────────────────────────────────────────┐
│  [🧠] SAM2 分割模型                              │
│  下载地址：hf-mirror.com（中国镜像）               │
│                                                  │
│  ☐ SAM 2 Hiera Tiny（快速）      约 155MB  [已下载]│
│  ☐ SAM 2 Hiera Small（推荐）      约 184MB  [已下载]│
│  ☑ SAM 2 Hiera Base+（高精度）    约 298MB  [未下载]│
│                                                  │
│  [📥 下载选中模型]  [🔄 刷新状态]                  │
│                                                  │
│  ── 已安装模型 ──                                │
│  ✅ sam2_hiera_tiny.pt                           │
│  ✅ sam2_hiera_small.pt                          │
│  ❌ sam2_hiera_base_plus.pt                      │
│                                                  │
│  ┌─ 下载进度 ─────────────────────────────────┐  │
│  │ 正在下载: sam2_hiera_base_plus.pt            │  │
│  │ 135MB / 298MB  ████████░░░░░░  45%          │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 交互逻辑

1. **页面加载 / 打开设置**：调用 `/api/sam/check-models`，更新模型可用状态
2. **勾选模型**：复选框根据可用状态显示对应文字（已下载/未下载）
3. **点击下载**：
   - 禁用下载按钮和刷新按钮
   - 创建 EventSource 连接到 SSE 端点
   - 实时更新进度条和状态文本
   - 下载完成后自动刷新模型状态
   - 恢复按钮可用状态
4. **模型已存在时**：勾选并点击下载会跳过（不重复下载），在状态信息中提示已跳过

## 修改文件清单

| 文件 | 改动 |
|---|---|
| `app.py` | 新增 `/api/sam/check-models` 和 `/api/sam/download-models` 两个端点 |
| `projects.html` | 在设置模态框中新增 SAM2 下载区域 HTML，绑定事件 |
| `static/script.js` | 新增 `refreshSamModels()` 和 `downloadSamModels()` 函数 |

## 不修改的文件

- `ai_manager.py` — 模型加载逻辑不受影响
- `templates/index.html` — 标注页面无需改动
- `templates/ai_config.html` — AI 配置页面的 SAM 模型选择保持不变
- `templates/train.html` — 训练页面无需改动

## 注意事项

- 下载使用 `requests` 库的流式下载，避免大文件占用过多内存
- 对于已存在的权重文件，跳过下载并在返回信息中提示
- 进度信息以 MB 为单位显示，更直观
- 错误处理：网络错误、磁盘空间不足、文件校验等
