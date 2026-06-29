---
title: Deploy 端一键预测 API 设计
date: 2026-06-29
status: draft
---

# Deploy 端一键预测 API 设计

## 1. 动机

当前 xclabel-deploy 的推理流程需要两步：先 `POST /load/model` 加载模型，再 `POST /infer` 进行推理。第三方用户需要了解引擎 ID、分步调用，使用门槛高。

参考 Roboflow Inference API 的设计理念，将 deploy→server 的连接参数简化为两个核心参数，并新增一步式推理端点，降低第三方集成成本。

## 2. 核心变更

### 2.1 连接参数简化

将原先的三个参数（`server_url`、`project_id`、`model_version`）简化为两个：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `server_url` | 服务端地址 | `http://127.0.0.1:9924/` |
| `model` | 项目 + 版本组合标识，格式 `project_id/model_version` | 必填 |

`model` 参数在 `ServerClient` 内部通过 `parse_model_ref()` 静态方法解析为 `(project_id, version)`。

### 2.2 新增 POST /v1/predict 端点

提供一步式推理，第三方用户只需传入三个核心参数即可获得推理结果。

**请求体 (PredictRequest)：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `server_url` | string | 否 | `http://127.0.0.1:9924/` | Server 端地址 |
| `model` | string | 是 | — | 格式 `project_id/model_version` |
| `image` | string | 否* | — | Base64 编码的图片数据，不含 `data:image/xxx;base64,` 前缀部分 |
| `image_url` | string | 否* | — | 图片 URL |
| `confidence_threshold` | number | 否 | 0.25 | 置信度阈值 |

*`image` 和 `image_url` 必须二选一。

**内部流程：**

1. 从 `model` 解析出 `project_id` 和 `model_version`
2. 构造 `engine_id = model`（复用 `"project_id/model_version"` 作为引擎标识）
3. 查询引擎池是否已加载该引擎
   - 已加载 → 直接复用
   - 未加载 → 使用 `ServerClient`（可指定 `server_url`）下载模型并加载到引擎池
4. 运行推理
5. 返回推理结果 JSON

**响应体：**

```json
{
  "model": "sv30_seg/20260618_172731",
  "inference_time_ms": 45.2,
  "detections": [
    {
      "class_id": 1,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 500],
      "points": []
    }
  ],
  "image_width": 1920,
  "image_height": 1080
}
```

**错误响应：**

```json
{
  "detail": "错误描述信息"
}
```

错误码：
- `400` — 参数错误（缺 image、model 格式不对等）
- `502` — 从 server 下载模型失败
- `500` — 推理执行失败

**注意：** `server_url` 末尾的斜杠会被自动规范化，`http://127.0.0.1:9924/` 与 `http://127.0.0.1:9924` 等价。

**缓存策略：** 加载过的模型会保留在引擎池中，后续相同 `model` 的请求直接复用。引擎池的 LRU 淘汰机制由现有 `EnginePool` 管理。

### 2.3 ServerClient 改造

**新增 `parse_model_ref()` 静态方法：**

```python
@staticmethod
def parse_model_ref(model_ref: str) -> tuple[str, str]:
    parts = model_ref.split("/", 1)
    if len(parts) != 2:
        raise ValueError("model_ref 格式须为 project_id/model_version")
    return parts[0], parts[1]
```

**修改 `download_model()` 方法签名：**

支持传 `model_ref` 组合字符串，内部自动解析。同时保留对旧格式的兼容。

**补全 `list_workflows()` 方法：**

当前 `deploy/main.py` 的 `/workflows` 端点调用 `server_client.list_workflows()`，但该方法未实现，属于 bug。一并补充，调用 server 端的 `GET /api/workflow/list?project=xxx`。

### 2.4 测试页面改造

在现有页面基础上，新增"一键预测"功能区块。

**布局方案：**

左侧面板分为上下两区：
- **上区：一键预测** — 只包含简化后的参数输入
  - 服务端地址输入框（默认 `http://127.0.0.1:9924/`）
  - 模型路径输入框（格式 `project_id/model_version`）
  - 置信度滑块
  - 图片上传区
  - "🚀 一键预测"按钮
- **下区：已加载引擎列表** — 保留现有引擎选择、双击卸载等高级功能

右侧面板：结果展示（标注图 + JSON），保持不变。

**交互流程：**
1. 用户填写服务端地址和模型路径
2. 上传一张图片
3. 点击"一键预测"
4. 页面自动调用 `POST /v1/predict`
5. 结果展示在右侧

## 3. 向后兼容

| 现有功能 | 兼容性 |
|---------|--------|
| `POST /load/model` | 完全保留，不受影响 |
| `POST /infer` | 完全保留 |
| `GET /engines` | 完全保留，`/v1/predict` 加载的引擎也会出现在列表中 |
| `POST /unload` | 完全保留 |
| `POST /pipeline/*` | 完全保留 |
| 现有测试页面功能 | 保留在页面下方，作为高级模式 |

## 4. 文件变更清单

| 文件 | 变更类型 | 变更内容 |
|------|---------|---------|
| `deploy/server_client.py` | 修改 | 新增 `parse_model_ref()`、`list_workflows()`；`download_model` 支持组合格式 |
| `deploy/main.py` | 修改 | 新增 `PredictRequest` schema + `POST /v1/predict` 端点 |
| `deploy/static/test.html` | 修改 | 新增"一键预测"UI 区块，简化参数输入 |

## 5. 测试场景

1. **正常流程：** 传入合法参数 → 返回推理结果 JSON + 图片标注
2. **模型已缓存：** 相同 `model` 第二次请求 → 直接复用，更快响应
3. **server_url 自定义：** 传入不同 server_url → 从指定 server 下载模型
4. **参数验证：** `model` 格式错误 → 返回 400，提示正确格式
5. **无图片：** `image` 和 `image_url` 都为空 → 返回 400
6. **server 不可达：** server_url 无效 → 返回 502
7. **向后兼容：** 现有 `/load/model` + `/infer` 端点照常工作
