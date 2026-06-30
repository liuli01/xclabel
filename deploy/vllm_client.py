"""
VLLM 客户端 — 通过 OpenAI 兼容 API 调用大模型。
支持 ROI 裁剪、超时/重试。
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VllmClient:
    """Async VLLM client with ROI cropping and timeout support."""

    def __init__(self):
        self._openai = None
        self._available = False
        self._init_client()

    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI
            self._available = True
        except ImportError:
            logger.warning("openai package not installed")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ── ROI cropping ──

    @staticmethod
    def crop_roi(image: Any, detections: List[Dict]) -> Image.Image:
        """Crop the largest detection region from the image."""
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.open(image).convert("RGB") if hasattr(image, 'read') else Image.fromarray(image)

        # Find the largest detection by bbox area
        best_bbox = None
        best_area = 0
        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > best_area:
                best_area = area
                best_bbox = bbox

        if best_bbox:
            x1, y1, x2, y2 = map(int, best_bbox)
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            if x2 > x1 and y2 > y1:
                return img.crop((x1, y1, x2, y2))

        return img  # fallback: return full image

    @staticmethod
    def _image_to_b64(image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    # ── VLLM inference ──

    async def analyze(
        self,
        image: Any,
        prompt: str,
        model: str = "qwen-vl-chat",
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 256,
        timeout: int = 30,
    ) -> Optional[str]:
        """Send image + prompt to VLLM and return text response."""
        if not self._available:
            logger.warning("openai not available, skipping VLLM call")
            return None

        # Convert image to PIL for b64
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = Image.open(image).convert("RGB") if hasattr(image, 'read') else Image.fromarray(image)

        client = self._openai(
            base_url=api_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": self._image_to_b64(pil_img)},
                            },
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Extract content from nested response (OpenAI format or DeepSeek wrapper)
            choices = None
            if response and response.choices:
                choices = response.choices
            elif hasattr(response, 'data') and isinstance(response.data, dict):
                choices = response.data.get('choices')
            if choices and len(choices) > 0:
                return choices[0].message.content if hasattr(choices[0], 'message') else choices[0].get('message', {}).get('content', str(choices[0]))
            # Log raw response for debugging
            import pprint
            details = pprint.pformat(response.model_dump() if hasattr(response, 'model_dump') else str(response))
            logger.warning(f"VLLM response missing choices. Raw: {details[:2000]}")
            return f"[VLLM response error: no choices in response]"
        except Exception as e:
            msg = f"VLLM call failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
