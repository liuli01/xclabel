# deploy/demo — API 调用示例

三个示例脚本演示如何通过 HTTP API 调用 Deploy 服务。

## 前提

1. **Deploy 服务已启动**（默认 http://127.0.0.1:8000）
2. **主服务已启动**（默认 http://127.0.0.1:9924），示例 1、2 需要从此下载模型/工作流
3. 依赖：`requests`（可选，示例使用 `urllib` 无额外依赖）

## 示例说明

| 脚本 | API | 说明 |
|------|-----|------|
| `predict_model.py` | `POST /v1/predict` | 单模型推理：下载模型 → 加载 → 推理 |
| `run_server_workflow.py` | `POST /v1/workflow/execute` + `server_url` | 从主服务下载工作流 YAML 后执行 |
| `run_custom_workflow.py` | `POST /v1/workflow/execute` + `yaml_content` | 直接传入 YAML 内容执行 |

## 运行

```bash
# 1. 单模型推理
python demo/predict_model.py

# 2. 从服务器加载工作流
python demo/run_server_workflow.py

# 3. 自定义 YAML（使用内嵌示例）
python demo/run_custom_workflow.py

# 3b. 自定义 YAML（从文件读取）
python demo/run_custom_workflow.py /path/to/my-pipeline.yaml
```

## API 速查

### 模型推理

```json
POST /v1/predict
{
  "model": "project_id/model_version",
  "server_url": "http://127.0.0.1:9924",
  "image_url": "https://example.com/test.jpg",
  "confidence_threshold": 0.25
}
```

### 工作流执行（服务端加载）

```json
POST /v1/workflow/execute
{
  "workflow": "demo-seg-pipeline",
  "server_url": "http://127.0.0.1:9924",
  "image_url": "https://example.com/test.jpg"
}
```

### 工作流执行（自定义 YAML）

```json
POST /v1/workflow/execute
{
  "workflow": "my-pipeline",
  "yaml_content": "version: '1.0'\nname: my-pipeline\npipeline:\n  ...",
  "image_url": "https://example.com/test.jpg"
}
```

也可在 test.html 页面上直接上传 `.yaml` 文件或粘贴 YAML 内容执行。
