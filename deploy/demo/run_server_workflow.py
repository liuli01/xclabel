"""
示例 2：调用服务器工作流 (v1/workflow/execute + server_url)

从主服务下载工作流 YAML → 加载 pipeline → 自动下载模型 → 执行。
等价于 test.html 的工作流执行 + 服务端地址模式。

用法：
    python demo/run_server_workflow.py
"""

import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DEPLOY_URL = "http://127.0.0.1:8005"
MAIN_URL = "http://127.0.0.1:9924"
TEST_IMAGE = "https://free.boltp.com/2026/06/29/6a42116caa1e2.webp"


def run_server_workflow(name: str, image_url: str):
    """通过工作流名称从主服务下载并执行。

    Args:
        name: 工作流名称（必须在主服务上已保存）
        image_url: 图片 URL
    """
    body = json.dumps({
        "workflow": name,
        "server_url": MAIN_URL,
        "image_url": image_url,
    }).encode()

    req = Request(
        f"{DEPLOY_URL}/v1/workflow/execute",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

    print(f"工作流: {data.get('workflow_id', name)}")
    print(f"耗时: {data.get('execution_time_ms', 0):.0f} ms")
    print(f"检测: {data.get('detection_count', 0)} 个目标")

    # 打印各节点状态
    node_status = data.get("node_status", {})
    node_outputs = data.get("node_outputs", {})
    for nid, status in sorted(node_status.items()):
        out = node_outputs.get(nid, {})
        if out.get("condition_result") is not None:
            print(f"  节点 {nid} [Condition]: {out['condition_result']}")
        elif out.get("vllm_result"):
            r = out["vllm_result"]
            print(f"  节点 {nid} [VLLM]: {r[:80]}...")
        elif out.get("skipped"):
            print(f"  节点 {nid} [VLLM]: 跳过（条件不满足）")
        elif out.get("detections"):
            print(f"  节点 {nid} [YOLO]: {len(out['detections'])} 个目标")
        else:
            print(f"  节点 {nid}: {status}")

    if data.get("errors"):
        print(f"错误: {data['errors']}")

    return data


if __name__ == "__main__":
    run_server_workflow("demo-seg-pipeline", TEST_IMAGE)
