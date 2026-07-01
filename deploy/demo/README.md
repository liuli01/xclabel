# xclabel-deploy Workflow 缓存架构与示例

## 三级缓存架构

```
调用 v1/workflow/execute
        │
        ▼
┌─────────────────────┐
│  1. pipeline_store   │  ← 内存 LRU (MAX_WORKFLOWS=50)
│     (OrderedDict)    │     命中直接执行，最快
└────────┬────────────┘
         │ 未命中
         ▼
┌─────────────────────┐
│  2. 磁盘缓存          │  ← CACHE_DIR/workflows/{name}.yaml
│     (YAML 文件)       │     重启后依然有效
└────────┬────────────┘
         │ 未命中
         ▼
┌─────────────────────┐
│  3. Server 拉取       │  ← GET /api/wf/yaml?name=xxx
│     (HTTP)           │     拉取后自动缓存到 磁盘 + 内存
└─────────────────────┘
```

## 重启行为

| 缓存层 | 重启后 | 恢复方式 |
|--------|--------|----------|
| pipeline_store (内存) | ❌ 清空 | 首次请求自动从磁盘恢复 |
| 磁盘 CACHE_DIR/workflows/ | ✅ 保留 | Docker volume 持久化 |
| EnginePool (内存) | ❌ 清空 | 模型需重新加载到内存 |
| 磁盘 CACHE_DIR/models/ | ✅ 保留 | 文件存在但当前代码未做磁盘预检 |

## API 端点

| 端点 | 用途 |
|------|------|
| `POST /v1/workflow/execute` | 一站式执行（三级缓存查找） |
| `POST /pipeline/load` | 预加载 workflow（支持 name 拉取模式） |
| `POST /pipeline/refresh` | 强制刷新缓存 |
| `POST /pipeline/execute` | 执行已预加载的 workflow |
| `GET /pipeline/workflows` | 列出已加载的 workflow |

## 运行示例

```bash
# 1. 首次运行（需要 server 在线）
python deploy/demo/run_workflow.py --image demo.jpg --mode first

# 2. 离线运行（使用缓存，可断开 server）
python deploy/demo/run_workflow.py --image demo.jpg --mode offline

# 3. 强制刷新
python deploy/demo/run_workflow.py --image demo.jpg --mode refresh

# 4. 全部演示
python deploy/demo/run_workflow.py --image demo.jpg --mode all
```
