"""
示例 3：调用自定义工作流 (v1/workflow/execute + yaml_content)

直接传入 YAML 内容，不依赖主服务。
适合测试本地编写的 pipeline 配置。

用法：
    python demo/run_custom_workflow.py                          # 使用默认的 demo YAML
    python demo/run_custom_workflow.py /path/to/my-pipeline.yaml
"""

import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DEPLOY_URL = "http://127.0.0.1:8005"
TEST_IMAGE = "https://free.boltp.com/2026/06/29/6a42116caa1e2.webp"

# 默认 YAML 文件路径（同目录下的 demo-seg-pipeline.yaml）
DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "demo-seg-pipeline.yaml")


def run_custom_workflow(yaml_content: str, image_url: str, name: str = "custom"):
    """直接传入 YAML 内容执行工作流。

    Args:
        yaml_content: 完整的 workflow YAML 字符串
        image_url: 图片 URL
        name: 工作流名称（自定义）
    """
    body = json.dumps({
        "workflow": name,
        "yaml_content": yaml_content,
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
    print(f"结果: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}...")
    return data


if __name__ == "__main__":
    # 从命令行参数或默认路径读取 YAML 文件
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    run_custom_workflow(yaml_content, TEST_IMAGE)
