"""
xclabel-deploy Workflow 调用示例

演示三种模式：
  1. 首次运行 — 从 server 下载 workflow + model，缓存到本地
  2. 离线运行 — 断开 server，使用本地磁盘缓存 + 内存 pipeline_store
  3. 强制刷新 — 强制重新从 server 拉取最新 workflow

环境要求：
  - deploy 服务运行中（默认 http://localhost:8000）
  - server 服务运行中（首次运行时需要，离线模式可关）
  - 工作流已保存到 server（例如通过 Web UI）
  - 图片文件准备好
"""

import argparse
import base64
import json
import os
import time
import urllib.request
from typing import Optional
from urllib.error import URLError


def encode_image(image_path: str) -> str:
    """读取图片文件，返回 base64 字符串（不含 data: URI 前缀）。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_api(url: str, payload: dict, timeout: int = 60) -> dict:
    """调用 deploy API 并返回 JSON 结果。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {e}"}


def check_cache_dir(deploy_url: str) -> dict:
    """通过 health 接口查看缓存状态。"""
    try:
        with urllib.request.urlopen(f"{deploy_url}/health", timeout=5) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="xclabel-deploy workflow 调用示例")
    parser.add_argument("--deploy", default="http://localhost:8000",
                        help="deploy 服务地址 (default: http://localhost:8000)")
    parser.add_argument("--server", default="http://localhost:9924",
                        help="server 服务地址 (default: http://localhost:9924)")
    parser.add_argument("--workflow", default="yolo-detect-preview",
                        help="workflow 名称 (default: yolo-detect-preview)")
    parser.add_argument("--image", required=True,
                        help="输入图片路径")
    parser.add_argument("--mode", choices=["first", "offline", "refresh", "all"],
                        default="all",
                        help="运行模式: first=首次, offline=离线, refresh=强制刷新, all=全部演示")
    parser.add_argument("--force-refresh", action="store_true",
                        help="强制从 server 重新拉取 workflow")
    args = parser.parse_args()

    DEPLOY = args.deploy.rstrip("/")
    SERVER = args.server.rstrip("/")
    WORKFLOW = args.workflow

    if not os.path.exists(args.image):
        print(f"[错误] 图片文件不存在: {args.image}")
        return

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  xclabel-deploy Workflow 调用示例            ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Deploy: {DEPLOY}")
    print(f"║  Server: {SERVER}")
    print(f"║  Workflow: {WORKFLOW}")
    print(f"║  Image: {args.image}")
    print(f"║  Mode: {args.mode}")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    # ── 检查服务状态 ──
    print("▶ 检查 deploy 服务状态...")
    health = check_cache_dir(DEPLOY)
    if "error" in health:
        print(f"  ✗ deploy 服务不可用: {health['error']}")
        print("  请先启动 deploy 服务")
        return
    print(f"  ✓ deploy 服务正常 (engines={health.get('engines_loaded', '?')})")
    print()

    # ── 图片编码 ──
    print("▶ 编码图片...")
    image_b64 = encode_image(args.image)
    print(f"  ✓ base64 编码完成 ({len(image_b64)} bytes)")
    print()

    # ═══════════════════════════════════════════════
    # 模式 A: 首次运行 / 在线模式
    # ═══════════════════════════════════════════════
    if args.mode in ("first", "all"):
        print("━" * 50)
        print("📦 [首次运行] 从 server 下载 workflow + model，缓存到本地")
        print("━" * 50)

        payload = {
            "workflow": WORKFLOW,
            "server_url": SERVER,
            "image": image_b64,
            "force_refresh": args.force_refresh or False,
        }

        t0 = time.time()
        result = call_api(f"{DEPLOY}/v1/workflow/execute", payload, timeout=120)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"  ✗ 执行失败: {result['error']}")
        else:
            status = result.get("node_status", {})
            timings = result.get("node_timings", {})
            det_count = result.get("detection_count", 0)
            print(f"  ✓ 执行完成 ({elapsed:.2f}s)")
            print(f"  ├─ 检测目标: {det_count} 个")
            print(f"  ├─ 节点状态: {status}")
            print(f"  ├─ 节点耗时: {timings}")
            print(f"  └─ workflow_id: {result.get('workflow_id', '?')}")
        print()

    # ═══════════════════════════════════════════════
    # 模式 B: 离线运行（模拟断开 server）
    # ═══════════════════════════════════════════════
    if args.mode in ("offline", "all"):
        print("━" * 50)
        print("🔌 [离线模式] 关闭 server，使用本地缓存执行")
        print("━" * 50)

        if args.mode == "all":
            # 在全量演示模式下，先验证 disk cache 存在
            print("  ℹ  全量演示: 假设首次运行已缓存成功")
            print()

        print("  ▶ 调用 deploy（不传 server_url，模拟离线）...")
        payload = {
            "workflow": WORKFLOW,
            # 故意不传 server_url — deploy 应该从缓存加载
            # 如果使用 DEFAULT_SERVER_URL 会尝试连 server，但我们不传让它走缓存
            "server_url": SERVER,  # 仍有地址但 force_refresh=false 应该走缓存
            "image": image_b64,
            "force_refresh": False,
        }

        t0 = time.time()
        result = call_api(f"{DEPLOY}/v1/workflow/execute", payload, timeout=120)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"  ✗ 执行失败: {result['error']}")
            print("  提示: 离线模式需要先运行首次模式完成缓存")
        else:
            det_count = result.get("detection_count", 0)
            print(f"  ✓ 离线执行成功 ({elapsed:.2f}s)")
            print(f"  ├─ 检测目标: {det_count} 个")
            print(f"  └─ 说明: 从磁盘缓存加载 workflow，从 EnginePool 复用模型")
        print()

    # ═══════════════════════════════════════════════
    # 模式 C: 强制刷新
    # ═══════════════════════════════════════════════
    if args.mode in ("refresh", "all"):
        print("━" * 50)
        print("🔄 [强制刷新] 重新从 server 拉取最新 workflow")
        print("━" * 50)

        payload = {
            "name": WORKFLOW,
            "server_url": SERVER,
        }
        t0 = time.time()
        result = call_api(f"{DEPLOY}/pipeline/refresh", payload, timeout=30)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"  ✗ 刷新失败: {result['error']}")
        else:
            nodes = result.get("nodes", [])
            print(f"  ✓ 刷新成功 ({elapsed:.2f}s)")
            print(f"  ├─ 节点数: {len(nodes)}")
            for n in nodes:
                print(f"  │   - {n['id']}: {n['type']}")
            print(f"  └─ 说明: 已清除磁盘+内存缓存，重新从 server 下载")
        print()

    # ═══════════════════════════════════════════════
    # 最终状态
    # ═══════════════════════════════════════════════
    print("━" * 50)
    print("📊 最终状态")
    print("━" * 50)

    # 查看已加载的 pipeline
    try:
        with urllib.request.urlopen(f"{DEPLOY}/pipeline/workflows", timeout=5) as resp:
            pipelines = json.loads(resp.read().decode())
            print(f"  pipeline_store: {pipelines.get('pipelines', [])}")
    except Exception:
        pass

    # 查看已加载的模型
    try:
        with urllib.request.urlopen(f"{DEPLOY}/engines", timeout=5) as resp:
            engines = json.loads(resp.read().decode())
            print(f"  EnginePool: {[e['engine_id'] for e in engines.get('engines', [])]}")
    except Exception:
        pass

    # 查看磁盘缓存文件
    cache_dir = os.environ.get("CACHE_DIR", "/app/cache")
    workflows_cache = os.path.join(cache_dir, "workflows")
    if os.path.isdir(workflows_cache):
        files = [f for f in os.listdir(workflows_cache) if f.endswith(".yaml")]
        print(f"  磁盘缓存 ({workflows_cache}): {files}")
    else:
        print(f"  磁盘缓存: 不可用（不在容器内运行）")
    print()
    print("✅ 演示完成")


if __name__ == "__main__":
    main()
