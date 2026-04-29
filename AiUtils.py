import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List

import cv2
import requests

# 尝试导入OpenAI库，用于调用阿里云大模型
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 默认日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AIAutoLabeler:
    """AI自动标注工具类，封装了与大模型API交互和视频处理的核心功能"""

    def __init__(self, model_api_url: str, api_key: str = None, prompt: str = None, timeout: int = 30, inference_tool: str = "LMStudio", model: str = "qwen/qwen3-vl-8b"):
        """初始化自动标注器
        
        Args:
            model_api_url: 大模型API地址
            api_key: API密钥（如果需要）
            prompt: 自定义提示词
            timeout: HTTP请求超时时间（秒）
            inference_tool: 推理工具，支持LMStudio、vLLM、ollama
            model: 模型名称
        """
        self.model_api_url = model_api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.timeout = timeout
        self.inference_tool = inference_tool
        self.model = model
        # 默认提示词
        self.default_prompt = "检测图中物体，返回JSON：{\"detections\":[{\"label\":\"类别\",\"confidence\":0.9,\"bbox\":[x1,y1,x2,y2]}]}"
        # 使用用户自定义提示词或默认提示词
        self.prompt = prompt if prompt else self.default_prompt

        # 定义颜色映射（不同类别使用不同颜色）
        self.colors = {
            "person": (0, 255, 0),
            "car": (255, 0, 0),
            "bicycle": (0, 0, 255),
            "dog": (255, 255, 0),
            "cat": (255, 0, 255),
            "人": (0, 255, 0),
            "车": (255, 0, 0),
            "自行车": (0, 0, 255),
            "狗": (255, 255, 0),
            "猫": (255, 0, 255),
            "default": (0, 255, 255)
        }

    def analyze_image_alibaba(self, image_path: str) -> Dict[str, Any]:
        """调用阿里云大模型API分析图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            大模型返回的分析结果
        """
        import base64


        if OpenAI is None:
            raise Exception("OpenAI库未安装，请使用pip install openai安装")

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 保存原始图片尺寸
        original_h, original_w = img.shape[:2]

        # 阿里云大模型可能需要较小的图像尺寸
        # 按照参考代码中的缩放比例
        resize_h = int(original_h / 3)
        resize_w = int(original_w / 3)
        image = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

        # 转换为JPEG格式
        encoded_image_byte = cv2.imencode(".jpg", image)[1].tobytes()
        image_base64 = base64.b64encode(encoded_image_byte).decode("utf-8")

        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 构建请求消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # 发送请求
        try:
            t1 = time.time()
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            t2 = time.time()
            t_len = t2 - t1
            logging.info(f"阿里云大模型请求耗时: {t_len:.2f}秒")

            content = completion.choices[0].message.content
            logging.info(f"阿里云大模型原始响应: {content}")

            # 尝试解析阿里云返回的特殊格式
            # 阿里云返回格式可能是：```json{"detections":[...]``` 或数组格式 [ {...} ]
            # 尝试多种解析方式
            result_json = {"detections": []}

            # 尝试去除可能的Markdown格式
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # 初始化detections变量，避免引用前未赋值的错误
            detections = []

            # 首先尝试直接解析JSON，这是最可靠的方法
            try:
                # 直接解析JSON
                parsed_json = json.loads(content)

                # 检查解析结果类型
                if isinstance(parsed_json, dict):
                    # 如果是字典，直接获取detections字段
                    detections = parsed_json.get("detections", [])
                    if isinstance(detections, dict):
                        detections = [detections]
                elif isinstance(parsed_json, list):
                    # 如果是数组，直接作为检测结果
                    detections = parsed_json
                else:
                    # 其他类型，默认为空列表
                    detections = []

                # 处理检测结果
                scale = 3.0  # 因为之前缩小了1/3，所以需要放大3倍
                for detection in detections:
                    if isinstance(detection, dict):
                        label = detection.get("label", "unknown")
                        confidence = detection.get("confidence", 0.0)
                        bbox = detection.get("bbox", [])

                        # 确保bbox是有效的
                        if bbox:
                            # 处理不同格式的bbox
                            if isinstance(bbox, list):
                                # 如果bbox是列表，确保只取前4个值
                                bbox_values = list(map(float, bbox[:4]))
                            else:
                                # 如果bbox是字符串或其他类型，尝试转换
                                bbox_str = str(bbox)
                                # 清理bbox值，只保留数字和逗号
                                clean_bbox = re.sub(r'[^0-9, ]', '', bbox_str)
                                # 移除多余空格
                                clean_bbox = re.sub(r'\s+', ' ', clean_bbox).strip()
                                # 确保格式为"x1,y1,x2,y2"
                                clean_bbox = re.sub(r'\s*,\s*', ',', clean_bbox)
                                # 分割并转换为浮点数
                                bbox_values = list(map(float, clean_bbox.split(',')))
                                # 只取前4个值
                                bbox_values = bbox_values[:4]

                            if len(bbox_values) == 4:
                                # 转换坐标到原始尺寸
                                x1, y1, x2, y2 = bbox_values
                                x1 = int(x1 * scale)
                                y1 = int(y1 * scale)
                                x2 = int(x2 * scale)
                                y2 = int(y2 * scale)

                                # 添加到检测结果
                                result_json["detections"].append({
                                    "label": label,
                                    "confidence": confidence,
                                    "bbox": [x1, y1, x2, y2]
                                })

                # 如果直接解析JSON成功并获取到了检测结果，直接返回
                if result_json["detections"]:
                    return result_json
            except json.JSONDecodeError:
                # 如果直接解析JSON失败，再尝试正则表达式提取
                logging.info(f"直接解析JSON失败，尝试使用正则表达式提取: {content}")

                # 使用正则表达式提取检测信息
                import re

                # 匹配模式：{"label":"自行车","confidence":0.9,"bbox":[[672,18,745,83]}
                detection_pattern = r'\{[^}]*"label"\s*:\s*"([^"]+)"[^}]*"confidence"\s*:\s*([0-9.]+)[^}]*"bbox"\s*:\s*\[*([0-9, ]+)\]*[^}]*\}'
                matches = re.findall(detection_pattern, content, re.DOTALL)

                if matches:
                    scale = 3.0  # 因为之前缩小了1/3，所以需要放大3倍
                    for match in matches:
                        label = match[0]
                        confidence = float(match[1])
                        bbox_str = match[2]

                        # 清理bbox值，只保留数字和逗号
                        clean_bbox = re.sub(r'[^0-9, ]', '', bbox_str)
                        # 移除多余空格
                        clean_bbox = re.sub(r'\s+', ' ', clean_bbox).strip()
                        # 确保格式为"x1,y1,x2,y2"
                        clean_bbox = re.sub(r'\s*,\s*', ',', clean_bbox)

                        # 分割并转换为浮点数
                        try:
                            bbox_values = list(map(float, clean_bbox.split(',')))
                            # 只取前4个值
                            bbox_values = bbox_values[:4]

                            if len(bbox_values) == 4:
                                # 转换坐标到原始尺寸
                                x1, y1, x2, y2 = bbox_values
                                x1 = int(x1 * scale)
                                y1 = int(y1 * scale)
                                x2 = int(x2 * scale)
                                y2 = int(y2 * scale)

                                # 添加到检测结果
                                result_json["detections"].append({
                                    "label": label,
                                    "confidence": confidence,
                                    "bbox": [x1, y1, x2, y2]
                                })
                        except ValueError:
                            # 如果转换失败，跳过此检测
                            logging.warning(f"无法解析bbox值: {clean_bbox}")
                            continue
                else:
                    # 尝试另一种方法：手动解析字符串
                    try:
                        # 提取label
                        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', content)
                        # 提取confidence
                        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
                        # 提取bbox值
                        bbox_match = re.search(r'"bbox"\s*:\s*\[*([0-9, ]+)\]*', content)

                        if label_match and confidence_match and bbox_match:
                            label = label_match.group(1)
                            confidence = float(confidence_match.group(1))
                            bbox_str = bbox_match.group(1)

                            # 清理bbox值
                            clean_bbox = re.sub(r'[^0-9, ]', '', bbox_str)
                            clean_bbox = re.sub(r'\s+', ' ', clean_bbox).strip()
                            clean_bbox = re.sub(r'\s*,\s*', ',', clean_bbox)

                            # 分割并转换为浮点数
                            bbox_values = list(map(float, clean_bbox.split(',')))
                            bbox_values = bbox_values[:4]

                            if len(bbox_values) == 4:
                                scale = 3.0
                                x1, y1, x2, y2 = bbox_values
                                x1 = int(x1 * scale)
                                y1 = int(y1 * scale)
                                x2 = int(x2 * scale)
                                y2 = int(y2 * scale)

                                result_json["detections"].append({
                                    "label": label,
                                    "confidence": confidence,
                                    "bbox": [x1, y1, x2, y2]
                                })
                    except Exception:
                        # 最后尝试直接从字符串中提取数字
                        try:
                            # 提取所有数字
                            all_numbers = re.findall(r'\d+\.?\d*', content)
                            if len(all_numbers) >= 5:  # label + confidence + 4 bbox values
                                # 假设格式是：label, confidence, x1, y1, x2, y2
                                label = "unknown"  # 默认标签
                                confidence = float(all_numbers[0])
                                x1 = float(all_numbers[1])
                                y1 = float(all_numbers[2])
                                x2 = float(all_numbers[3])
                                y2 = float(all_numbers[4])

                                scale = 3.0
                                x1 = int(x1 * scale)
                                y1 = int(y1 * scale)
                                x2 = int(x2 * scale)
                                y2 = int(y2 * scale)

                                result_json["detections"].append({
                                    "label": label,
                                    "confidence": confidence,
                                    "bbox": [x1, y1, x2, y2]
                                })
                        except Exception:
                            error_msg = f"无法解析阿里云模型返回的JSON: {content}"
                            logging.error(error_msg)
                            # 不再抛出异常，而是返回空结果，这样不会导致整个标注失败
                            result_json = {"detections": []}

            return result_json
        except Exception as e:
            error_msg = f"阿里云大模型分析图像失败: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def analyze_image_hyperlpr(self, image_path: str) -> Dict[str, Any]:
        """调用HyperLPR API分析图像进行车牌识别
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            车牌识别结果
        """
        import os

        # 构建请求数据
        files = {
            "file": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")
        }

        # 确保API地址以正确的端点结尾
        api_endpoint = self.model_api_url
        if not api_endpoint.endswith("/api/v1/rec"):
            if api_endpoint.endswith("/"):
                api_endpoint = f"{api_endpoint}api/v1/rec"
            else:
                api_endpoint = f"{api_endpoint}/api/v1/rec"

        # 发送请求
        try:
            response = self.session.post(api_endpoint, files=files, timeout=self.timeout)

            # 记录请求详情以便调试
            logging.info(f"发送HyperLPR API请求到: {api_endpoint}")

            # 关闭文件
            files["file"][1].close()

            # 检查响应状态码
            if not response.ok:
                # 记录响应详情
                logging.error(f"HyperLPR API请求失败，状态码: {response.status_code}")
                logging.error(f"响应内容: {response.text}")
                raise Exception(f"HyperLPR API请求失败，状态码: {response.status_code}")

            result = response.json()
            logging.info(f"HyperLPR API响应: {json.dumps(result, ensure_ascii=False)}")

            # 解析车牌识别结果
            detections = []
            if result.get("code") == 5000 and result.get("result"):
                plate_list = result["result"].get("plate_list", [])
                for plate in plate_list:
                    detections.append({
                        "label": plate.get("code", "未知车牌"),
                        "confidence": plate.get("conf", 0.0),
                        "bbox": plate.get("box", [0, 0, 0, 0]),
                        "plate_type": plate.get("plate_type", "蓝牌")
                    })

            return {"detections": detections}
        except Exception as e:
            # 确保文件被关闭
            if "file" in locals() and hasattr(files["file"][1], "close"):
                files["file"][1].close()

            error_msg = f"HyperLPR分析图像失败: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """调用大模型API分析图像

        Args:
            image_path: 图像文件路径

        Returns:
            大模型返回的分析结果
        """
        # 根据推理工具类型调用不同的分析方法
        if self.inference_tool == "阿里云大模型":
            result = self.analyze_image_alibaba(image_path)
            # 确保返回的是字典格式
            if isinstance(result, dict):
                return result
            else:
                logging.error(f"阿里云大模型返回了非字典格式结果: {result}")
                return {"detections": []}
        elif self.inference_tool == "HyperLPR":
            result = self.analyze_image_hyperlpr(image_path)
            # 确保返回的是字典格式
            if isinstance(result, dict):
                return result
            else:
                logging.error(f"HyperLPR返回了非字典格式结果: {result}")
                return {"detections": []}

        # 原有的分析逻辑保留
        import base64


        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 确保API地址以正确的端点结尾
        api_endpoint = self.model_api_url
        if api_endpoint.endswith("/chat/completions"):
            pass
        elif api_endpoint.endswith("/v1"):
            api_endpoint = f"{api_endpoint}/chat/completions"
        else:
            from urllib.parse import urlparse
            path = urlparse(api_endpoint).path.rstrip('/')
            if not path:
                api_endpoint = f"{api_endpoint.rstrip('/')}/v1/chat/completions"
            else:
                api_endpoint = f"{api_endpoint.rstrip('/')}/chat/completions"

        # 读取图像并压缩，减少base64编码后的大小
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 保存原始图片尺寸（不缩放，直接使用原始尺寸）
        original_h, original_w = img.shape[:2]
        scaled_w, scaled_h = original_w, original_h  # 不缩放，直接使用原始尺寸
        scale = 1.0  # 缩放比例为1，不进行缩放
        upscale = 1.0  # 放大比例为1，不进行放大

        # 不压缩图像，直接使用原始尺寸
        # 这样大模型返回的坐标就是基于原始尺寸的，不需要进行坐标转换

        # 转换为JPEG格式，降低质量
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # 构建API请求体
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.prompt
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "text"
            }
        }

        # 发送请求
        try:
            response = self.session.post(api_endpoint, headers=headers, json=payload, timeout=self.timeout)

            # 记录请求详情以便调试
            logging.info(f"发送API请求到: {api_endpoint}")
            logging.info(f"请求头: {headers}")
            logging.info(f"请求体: {json.dumps(payload, ensure_ascii=False)}")

            # 检查响应状态码
            if not response.ok:
                # 记录响应详情
                logging.error(f"API请求失败，状态码: {response.status_code}")
                logging.error(f"响应内容: {response.text}")
                raise Exception(f"API请求失败，状态码: {response.status_code}，响应: {response.text[:200]}...")

            result = response.json()
            logging.info(f"API响应: {json.dumps(result, ensure_ascii=False)}")

            # 解析API返回的结果
            # 兼容标准OpenAI格式（choices在顶层）和包装格式（choices在data内）
            choices = None
            if "choices" in result and len(result["choices"]) > 0:
                choices = result["choices"]
            elif "data" in result and "choices" in result["data"] and len(result["data"]["choices"]) > 0:
                choices = result["data"]["choices"]

            if choices:
                content = choices[0]["message"]["content"]
                # 尝试解析JSON内容
                try:
                    # 去除Markdown格式标记
                    if content.startswith('```json'):
                        content = content[7:]  # 移除开头的```json
                    if content.endswith('```'):
                        content = content[:-3]  # 移除结尾的```
                    content = content.strip()  # 去除首尾空白

                    # 解析JSON结果
                    result_json = json.loads(content)

                    # 确保返回的是包含detections键的字典
                    if isinstance(result_json, dict):
                        if "detections" not in result_json:
                            result_json["detections"] = []
                        elif not isinstance(result_json["detections"], list):
                            result_json["detections"] = [result_json["detections"]]
                    else:
                        result_json = {"detections": []}

                    # 将检测到的坐标从缩放后的尺寸转换回原始图片尺寸
                    for detection in result_json["detections"]:
                        if "bbox" in detection:
                            bbox = detection["bbox"]
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = map(float, bbox)

                                # 如果图片被缩小了，则检测框需要等比放大
                                # upscale = 1.0 / scale
                                x1 = int(x1 * upscale)
                                y1 = int(y1 * upscale)
                                x2 = int(x2 * upscale)
                                y2 = int(y2 * upscale)

                                detection["bbox"] = [x1, y1, x2, y2]

                    return result_json
                except json.JSONDecodeError:
                    error_msg = f"无法解析模型返回的JSON: {content}"
                    logging.error(error_msg)
                    raise Exception(error_msg)

            return {"detections": []}
        except requests.exceptions.ConnectionError as e:
            error_msg = f"无法连接到API服务器: {str(e)}. 请检查API地址是否正确，服务器是否正在运行。"
            logging.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.Timeout as e:
            error_msg = f"API请求超时: {str(e)}. 请检查网络连接或增加超时时间。"
            logging.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"分析图像失败: {str(e)}"
            logging.error(f"分析图像 {image_path} 失败: {e}")
            logging.error(f"使用的API端点: {api_endpoint}")
            raise Exception(error_msg)

    def render_detections(self, image_path: str, detections: List[Dict[str, Any]]) -> str:
        """将检测结果渲染到图像上
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            
        Returns:
            渲染后的图像路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 渲染检测框和标签
        for detection in detections:
            # 解析检测结果
            if isinstance(detection, dict):
                label = detection.get("label", "unknown")
                confidence = detection.get("confidence", 0.0)
                bbox = detection.get("bbox", [0, 0, 0, 0])
            else:
                continue

            # 转换为整数坐标
            x1, y1, x2, y2 = map(int, bbox)

            # 获取颜色
            color = self.colors.get(label, self.colors["default"])

            # 绘制检测框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签和置信度（支持中文）
            label_text = f"{label}: {confidence:.2f}"

            # 尝试使用PIL库渲染中文
            try:
                import numpy as np
                from PIL import Image, ImageDraw, ImageFont

                # 转换为PIL图像
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                # 加载默认中文字体或指定字体文件
                try:
                    # 尝试使用系统默认中文字体
                    font = ImageFont.truetype("simhei.ttf", 16)
                except IOError:
                    # 如果没有找到，使用PIL默认字体
                    font = ImageFont.load_default()

                # 绘制文本
                text_x = x1
                text_y = y1 - 20 if y1 > 20 else y1 + 20
                draw.text((text_x, text_y), label_text, font=font, fill=tuple(color[::-1]))

                # 转换回OpenCV图像
                image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 如果PIL渲染失败，使用OpenCV默认渲染（可能会有乱码）
                logging.warning(f"中文渲染失败，使用默认渲染: {e}")
                cv2.putText(image, label_text, (x1, y1 - 10 if y1 > 10 else y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存渲染后的图像
        import os
        base_name, ext = os.path.splitext(image_path)
        rendered_path = f"{base_name}_labeled{ext}"
        cv2.imwrite(rendered_path, image)
        return rendered_path

    def process_video(self, video_path: str, output_dir: str, frame_interval: int = 1, save_rendered: bool = True):
        """处理视频完整流程，支持本地视频和RTSP流
        
        Args:
            video_path: 视频文件路径或RTSP流地址
            output_dir: 输出目录
            frame_interval: 抽帧间隔
            save_rendered: 是否保存渲染后的帧
        """
        # 记录开始时间
        start_time = datetime.now()
        logging.info(f"🚀 开始处理视频流: {video_path}")
        logging.info(f"📁 输出目录: {output_dir}")
        logging.info(f"⏱️  抽帧间隔: {frame_interval}")
        logging.info(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 创建输出目录
        raw_frames_dir = os.path.join(output_dir, "raw_frames")
        labeled_frames_dir = os.path.join(output_dir, "labeled_frames")
        os.makedirs(raw_frames_dir, exist_ok=True)
        if save_rendered:
            os.makedirs(labeled_frames_dir, exist_ok=True)

        logging.info(f"📁 原始帧目录: {raw_frames_dir}")
        if save_rendered:
            logging.info(f"📁 渲染帧目录: {labeled_frames_dir}")

        # 初始化变量
        cap = None
        frame_count = 0
        processed_count = 0
        is_rtsp = video_path.lower().startswith("rtsp://")
        max_reconnect_attempts = 50  # 最大重连次数，0表示无限重试
        reconnect_delay = 5  # 重连延迟（秒）
        reconnect_count = 0
        last_status_time = datetime.now()  # 上次输出状态的时间
        status_interval = 60  # 状态输出间隔（秒）

        try:
            while True:
                try:
                    # 检查是否需要打开或重新打开视频流
                    if cap is None or not cap.isOpened():
                        if reconnect_count > 0:
                            logging.info(f"🔄 尝试重新连接 RTSP 流... (尝试 {reconnect_count}/{max_reconnect_attempts if max_reconnect_attempts > 0 else '无限'})")
                        else:
                            logging.info(f"📡 打开视频流: {video_path}")

                        # 打开视频或RTSP流
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            raise ValueError(f"无法打开视频流: {video_path}")

                        if reconnect_count > 0:
                            logging.info("✅ RTSP 流重新连接成功")
                            reconnect_count = 0  # 重置重连计数

                    # 读取一帧
                    cap.grab()  # 只抓取帧，不解码，提高响应速度
                    ret, frame = cap.retrieve()  # 解码帧

                    if not ret:
                        if is_rtsp:
                            # RTSP流中断，尝试重连
                            logging.info(f"⚠️  RTSP 流中断，{reconnect_delay}秒后尝试重连...")

                            # 关闭当前视频流
                            if cap is not None:
                                cap.release()
                                cap = None

                            # 增加重连计数
                            reconnect_count += 1

                            # 检查是否达到最大重连次数
                            if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                                logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                                break

                            # 等待重连延迟
                            time.sleep(reconnect_delay)
                            continue  # 跳过当前循环，尝试重新连接
                        else:
                            # 本地视频文件结束
                            logging.info("✅ 视频流读取完成")
                            break

                    # 按照指定间隔处理帧
                    if frame_count % frame_interval == 0:
                        logging.info(f"🔄 处理帧 #{frame_count}")

                        # 定义统一的文件名
                        frame_filename = f"frame_{frame_count:06d}.jpg"

                        # 保存临时帧用于处理
                        temp_frame_path = f"temp_{frame_filename}"
                        cv2.imwrite(temp_frame_path, frame)

                        try:
                            # 分析图像（同步处理，阻塞等待结果）
                            result = self.analyze_image(temp_frame_path)

                            # 解析检测结果
                            detections = result.get("detections", [])
                            if isinstance(detections, dict):
                                detections = [detections]

                            # 仅当检测到至少一个目标时，才保存图片
                            if detections and len(detections) > 0:
                                logging.info(f"✅ 检测到 {len(detections)} 个目标")

                                # 保存原始未渲染帧
                                raw_frame_path = os.path.join(raw_frames_dir, frame_filename)
                                cv2.imwrite(raw_frame_path, frame)
                                logging.info(f"✅ 已保存原始帧: {raw_frame_path}")

                                # 保存渲染后的帧
                                if save_rendered:
                                    # 渲染检测结果
                                    rendered_path = self.render_detections(temp_frame_path, detections)

                                    # 移动渲染后的帧到最终目录，保持与原始帧相同的文件名
                                    final_path = os.path.join(labeled_frames_dir, frame_filename)
                                    os.rename(rendered_path, final_path)
                                    logging.info(f"✅ 已保存标注帧: {final_path}")

                                processed_count += 1
                            else:
                                logging.info("ℹ️  未检测到目标，跳过保存")
                        except KeyboardInterrupt:
                            logging.info("\n⚠️  用户中断处理")
                            # 删除未处理完的临时文件
                            if os.path.exists(temp_frame_path):
                                os.remove(temp_frame_path)
                            raise  # 重新抛出异常，让外层处理

                        # 删除临时文件
                        if os.path.exists(temp_frame_path):
                            os.remove(temp_frame_path)

                    frame_count += 1

                    # 定期输出状态信息
                    current_time = datetime.now()
                    if (current_time - last_status_time).total_seconds() >= status_interval:
                        # 计算运行时长
                        elapsed = current_time - start_time
                        # 计算处理速度
                        fps = processed_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0

                        logging.info(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"运行时长: {str(elapsed).split('.')[0]} | "
                              f"总帧数: {frame_count} | "
                              f"已处理: {processed_count}帧 | "
                              f"处理速度: {fps:.2f}帧/秒")
                        last_status_time = current_time

                    # 短暂休眠，提高中断响应速度
                    time.sleep(0.001)

                except KeyboardInterrupt:
                    logging.info("\n⚠️  用户中断处理")
                    raise  # 重新抛出异常，让外层处理
                except Exception as e:
                    if is_rtsp:
                        # RTSP流出现异常，尝试重连
                        logging.warning(f"⚠️  RTSP 流异常: {e}，{reconnect_delay}秒后尝试重连...")

                        # 关闭当前视频流
                        if cap is not None:
                            cap.release()
                            cap = None

                        # 增加重连计数
                        reconnect_count += 1

                        # 检查是否达到最大重连次数
                        if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                            logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                            raise

                        # 等待重连延迟
                        time.sleep(reconnect_delay)
                        continue  # 跳过当前循环，尝试重新连接
                    else:
                        # 本地视频文件异常，直接抛出
                        raise

        except KeyboardInterrupt:
            logging.info("\n🛑 正在停止处理...")
        except Exception as e:
            logging.error(f"❌ 处理异常: {e}")
        finally:
            # 确保视频流被释放
            if cap is not None and cap.isOpened():
                cap.release()
                logging.info("✅ 视频流已释放")

        # 计算结束时间和总运行时长
        end_time = datetime.now()
        total_elapsed = end_time - start_time

        logging.info("\n" + "=" * 60)
        logging.info("📊 完整处理统计:")
        logging.info(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"📅 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"⏱️  总运行时长: {str(total_elapsed).split('.')[0]}")
        logging.info(f"📈 总帧数: {frame_count}")
        logging.info(f"✅ 已处理: {processed_count}帧")
        logging.info(f"📊 处理比例: {processed_count / frame_count * 100:.1f}%" if frame_count > 0 else "📊 处理比例: 0%")
        logging.info(f"⚡ 平均速度: {processed_count / total_elapsed.total_seconds():.2f}帧/秒" if total_elapsed.total_seconds() > 0 else "⚡ 平均速度: 0帧/秒")
        logging.info(f"📁 输出目录: {output_dir}")
        logging.info("=" * 60)
        logging.info("✅ 处理已停止")
