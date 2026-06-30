"""
示例 1：调用模型 (v1/predict)

从主服务下载模型、加载到引擎池、运行推理，一步完成。
等价于 test.html 的「一键预测」。

用法：
    python demo/predict_model.py
"""

import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DEPLOY_URL = "http://127.0.0.1:8005"
MAIN_URL = "http://127.0.0.1:9924"
TEST_IMAGE = "https://free.boltp.com/2026/06/29/6a42116caa1e2.webp"


def predict_model(model_ref: str, image_url: str, conf: float = 0.25):
    """调用 v1/predict 执行单模型推理。

    Args:
        model_ref: 模型路径，格式 project_id/model_version
        image_url: 图片 URL
        conf: 置信度阈值
    """
    body = json.dumps({
        "model": model_ref,
        "server_url": MAIN_URL,
        "image_url": image_url,
        "confidence_threshold": conf,
    }).encode()

    req = Request(
        f"{DEPLOY_URL}/v1/predict",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

    print(f"模型: {data.get('model', model_ref)}")
    print(f"耗时: {data.get('inference_time_ms', 0):.0f} ms")
    print(f"检测: {len(data.get('detections', []))} 个目标")
    for d in data.get("detections", []):
        print(f"  - {d['class_name']} ({d['confidence']:.3f})")
    return data


if __name__ == "__main__":
    predict_model("sv30_seg/20260618_172731", TEST_IMAGE)
