import io
import json
import logging
import math
import os
import requests
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, session
from flask_cors import CORS
from flask_socketio import SocketIO
from PIL import Image

from AiUtils import AIAutoLabeler
from ai_manager import (
    get_sam2_engine,
    release_sam2_engine,
    YOLOAutoLabeler,
    init_sam2_engine,
)

# ── ML Engine (integrated, no separate deploy service needed) ──
from deploy.yolo_adapter import YoloAdapter
from deploy.vllm_client import VllmClient
from deploy.pipeline_manager import PipelineManager
from deploy.engine_pool import Engine, EnginePool

# Lazy-init ML engine pool (asyncio.Lock needs an event loop)
_ml_engine_pool = None
_ml_yolo_adapter = None
_ml_vllm_client = None
ML_CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache"))

def _get_ml_engine_pool():
    global _ml_engine_pool
    if _ml_engine_pool is None:
        from deploy.engine_pool import EnginePool
        _ml_engine_pool = EnginePool(max_engines=10)
    return _ml_engine_pool

def _get_ml_yolo_adapter():
    global _ml_yolo_adapter
    if _ml_yolo_adapter is None:
        _ml_yolo_adapter = YoloAdapter()
    return _ml_yolo_adapter

def _get_ml_vllm_client():
    global _ml_vllm_client
    if _ml_vllm_client is None:
        _ml_vllm_client = VllmClient()
    return _ml_vllm_client

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# 应用版本号
APP_VERSION = "v2.7"

# 配置SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 任务管理系统
tasks = {}

# 连接ID和任务ID的映射字典，用于跟踪客户端断开连接时需要停止的任务
# 格式: {sid: task_id}
connection_task_map = {}

# 任务状态枚举
TASK_STATUS = {
    'IDLE': 'idle',
    'RUNNING': 'running',
    'PAUSED': 'paused',
    'COMPLETED': 'completed',
    'STOPPED': 'stopped',
    'ERROR': 'error'
}

class VideoAnnotationTask:
    """视频标注任务类"""
    def __init__(self, task_id, video_path, frame_interval, output_dir, api_config):
        self.task_id = task_id
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.output_dir = output_dir
        self.api_config = api_config
        self.status = TASK_STATUS['IDLE']
        self.frame_count = 0
        self.processed_count = 0
        self.total_detections = 0
        self.error = None
        self.thread = None
        self.stop_event = threading.Event()
        self.start_time = None

    def start(self):
        """开始任务"""
        import datetime
        self.status = TASK_STATUS['RUNNING']
        self.start_time = datetime.datetime.now().isoformat()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        return self.task_id

    def stop(self):
        """停止任务"""
        self.stop_event.set()
        self.status = TASK_STATUS['STOPPED']
        self.send_progress()
        # 不立即join线程，让线程自己完成清理工作

    def run(self):
        """运行任务"""
        try:
            import base64
            import os
            import time


            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            raw_dir = os.path.join(self.output_dir, 'raw_frames')
            labeled_dir = os.path.join(self.output_dir, 'labeled_frames')
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(labeled_dir, exist_ok=True)

            # 获取API配置
            api_url = self.api_config.get('apiUrl', 'http://127.0.0.1:1234/v1')
            api_key = self.api_config.get('apiKey', '')
            timeout = int(self.api_config.get('timeout', 300))
            prompt = self.api_config.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
            model = self.api_config.get('model', 'qwen/qwen3-vl-8b')
            inference_tool = self.api_config.get('inferenceTool', 'LMStudio')

            # 初始化AIAutoLabeler
            labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)

            # 打开视频流
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error = f'Failed to open video: {self.video_path}'
                self.status = TASK_STATUS['ERROR']
                return

            # 处理视频帧
            while not self.stop_event.is_set():
                # 检查停止信号
                if self.stop_event.is_set():
                    break

                ret, frame = cap.read()
                if not ret:
                    # 对于RTSP流，尝试重新连接
                    if self.video_path.startswith('rtsp://'):
                        # 关闭当前连接
                        cap.release()
                        # 短暂休眠后重新打开
                        time.sleep(1)
                        cap = cv2.VideoCapture(self.video_path)
                        if not cap.isOpened():
                            self.error = f'Failed to reopen RTSP stream: {self.video_path}'
                            self.status = TASK_STATUS['ERROR']
                            self.send_progress()
                            break
                        # 发送进度更新，告知正在重连
                        self.send_progress()
                        # 继续循环，不中断任务
                        continue
                    else:
                        # 对于普通视频文件，退出循环
                        break

                self.frame_count += 1

                # 发送进度更新，即使不处理当前帧，也要更新帧计数
                if self.frame_count % 10 == 0:  # 每10帧发送一次进度更新
                    self.send_progress()

                # 按照指定间隔处理帧
                if self.frame_count % self.frame_interval == 0:
                    # 检查停止信号
                    if self.stop_event.is_set():
                        break

                    # 保存原始帧
                    frame_filename = f"frame_{self.frame_count:06d}.jpg"
                    raw_frame_path = os.path.join(raw_dir, frame_filename)
                    cv2.imwrite(raw_frame_path, frame)

                    # 检查停止信号
                    if self.stop_event.is_set():
                        break

                    # 检查停止信号
                    if self.stop_event.is_set():
                        break

                    # 调用API进行标注
                    try:
                        result = labeler.analyze_image(raw_frame_path)
                        detections = result.get("detections", [])
                        if isinstance(detections, dict):
                            detections = [detections]
                    except Exception as e:
                        # API请求失败，继续处理下一帧
                        logging.error(f"API request failed: {str(e)}")
                        # 发送进度更新，告知API请求失败
                        self.send_progress()
                        continue

                    # 检查停止信号
                    if self.stop_event.is_set():
                        break

                    # 检查停止信号
                    if self.stop_event.is_set():
                        break

                    # 渲染检测结果
                    rendered_path = labeler.render_detections(raw_frame_path, detections)

                    # 保存渲染后的帧
                    labeled_frame_path = os.path.join(labeled_dir, frame_filename)
                    # 如果目标文件已存在，先删除
                    if os.path.exists(labeled_frame_path):
                        os.remove(labeled_frame_path)
                    os.rename(rendered_path, labeled_frame_path)

                    # 读取渲染后的帧用于后续处理
                    labeled_frame = cv2.imread(labeled_frame_path)

                    self.processed_count += 1
                    self.total_detections += len(detections)

                    # 生成当前帧和渲染后图片的Base64数据（用于实时显示）
                    # 压缩当前帧用于显示
                    _, raw_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    current_frame_base64 = base64.b64encode(raw_buffer).decode("utf-8")

                    # 压缩渲染后的帧用于显示
                    _, labeled_buffer = cv2.imencode('.jpg', labeled_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    labeled_frame_base64 = base64.b64encode(labeled_buffer).decode("utf-8")

                    # 发送进度更新，包含当前帧和渲染后的图片
                    self.send_progress(current_frame_base64, labeled_frame_base64)

                    # 短暂休眠，提高响应速度
                    time.sleep(0.001)

            # 确保发送最终的进度更新
            # 如果状态还没有被设置为STOPPED或ERROR，设置为COMPLETED
            if self.status != TASK_STATUS['ERROR'] and self.status != TASK_STATUS['STOPPED']:
                self.status = TASK_STATUS['COMPLETED']
            # 发送最终的进度更新
            self.send_progress()

        except Exception as e:
            self.status = TASK_STATUS['ERROR']
            self.error = str(e)
            self.send_progress()
        finally:
            # 释放资源
            cap.release()

    def send_progress(self, current_frame=None, labeled_frame=None):
        """发送进度更新"""
        import datetime
        progress = {
            'task_id': self.task_id,
            'status': self.status,
            'frame_count': self.frame_count,
            'processed_count': self.processed_count,
            'total_detections': self.total_detections,
            'error': self.error,
            'output_dir': self.output_dir,
            'start_time': self.start_time,
            'current_time': datetime.datetime.now().isoformat()
        }

        # 如果提供了当前帧和渲染后的图片，添加到进度更新中
        if current_frame:
            progress['current_frame'] = current_frame
        if labeled_frame:
            progress['labeled_frame'] = labeled_frame

        socketio.emit('progress_update', progress)

        # 任务完成、停止或出错后，从任务列表中移除任务
        if self.status in [TASK_STATUS['COMPLETED'], TASK_STATUS['STOPPED'], TASK_STATUS['ERROR']]:
            # 使用线程安全的方式移除任务
            if self.task_id in tasks:
                del tasks[self.task_id]

    def get_status(self):
        """获取任务状态"""
        return {
            'task_id': self.task_id,
            'status': self.status,
            'frame_count': self.frame_count,
            'processed_count': self.processed_count,
            'total_detections': self.total_detections,
            'error': self.error,
            'output_dir': self.output_dir
        }

# 配置
import os
import re

# 使用当前工作目录作为基础目录
BASE_PATH = os.getcwd()
STATIC_FOLDER = os.path.join(BASE_PATH, 'static')
PROJECTS_FOLDER = os.path.join(BASE_PATH, 'projects')
LEGACY_UPLOAD_FOLDER = os.path.join(BASE_PATH, 'uploads')

app.config['STATIC_FOLDER'] = STATIC_FOLDER

# ===== SAM 2 引擎初始化（应用启动时预加载）=====
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "small")
SAM_MODELS_DIR = os.path.join(BASE_PATH, "models")

try:
    init_sam2_engine(model_type=SAM_MODEL_TYPE, models_dir=SAM_MODELS_DIR)
    logging.info(f"SAM 2 engine initialized (model={SAM_MODEL_TYPE})")
except Exception as e:
    logging.warning(f"SAM 2 engine initialization failed: {e}. Smart labeling will be unavailable.")

# 工程相关辅助函数

def sanitize_project_name(name):
    """校验工程名称：仅允许字母、数字、下划线。"""
    name = name.strip()
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        return ''
    return name

def get_current_project():
    """获取当前工程名，默认为 'default'。"""
    return session.get('current_project', 'default')

def set_current_project(name):
    """设置当前工程名。"""
    session['current_project'] = name

def get_project_path(project_name):
    """获取指定工程的完整目录路径。"""
    return os.path.join(PROJECTS_FOLDER, project_name)

def get_upload_folder():
    """获取当前工程的上传目录路径。"""
    return get_project_path(get_current_project())

def get_annotations_folder():
    """获取当前工程的标注目录路径。"""
    return os.path.join(get_upload_folder(), 'annotations')

def get_annotations_file():
    """获取当前工程的标注文件路径。"""
    return os.path.join(get_annotations_folder(), 'annotations.json')

def get_classes_file():
    """获取当前工程的类别文件路径。"""
    return os.path.join(get_annotations_folder(), 'classes.json')

def ensure_default_project():
    """确保默认工程存在。首次启动时迁移旧数据。"""
    os.makedirs(PROJECTS_FOLDER, exist_ok=True)
    default_project_path = get_project_path('default')

    if not os.path.exists(default_project_path):
        os.makedirs(default_project_path, exist_ok=True)
        os.makedirs(os.path.join(default_project_path, 'annotations'), exist_ok=True)

    # 确保默认工程的标注和类别文件存在
    ann_file = os.path.join(default_project_path, 'annotations', 'annotations.json')
    cls_file = os.path.join(default_project_path, 'annotations', 'classes.json')
    if not os.path.exists(ann_file):
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    if not os.path.exists(cls_file):
        default_classes = []
        with open(cls_file, 'w', encoding='utf-8') as f:
            json.dump(default_classes, f)
    else:
        # 如果类别文件存在但解析失败，重置为空列表
        try:
            with open(cls_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                with open(cls_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
        except (json.JSONDecodeError, TypeError):
            with open(cls_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

def init_project(project_name, task_type='detect'):
    """初始化新工程的目录和默认文件。"""
    project_path = get_project_path(project_name)
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, 'annotations'), exist_ok=True)
    ann_file = os.path.join(project_path, 'annotations', 'annotations.json')
    cls_file = os.path.join(project_path, 'annotations', 'classes.json')
    if not os.path.exists(ann_file):
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    if not os.path.exists(cls_file):
        with open(cls_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    # 保存工程类型信息
    info_file = os.path.join(project_path, 'project_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump({'task_type': task_type, 'created_at': time.time()}, f)


def get_project_info(project_name):
    """获取工程信息（含类型）。"""
    info_file = os.path.join(get_project_path(project_name), 'project_info.json')
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'task_type': 'detect'}

# 不再自动创建默认工程


@app.route('/')
def index():
    return redirect('/projects')

@app.route('/ai-config')
def ai_config():
    return render_template('ai_config.html', version=APP_VERSION)

@app.route('/file-manager')
def file_manager():
    """文件管理页面"""
    return render_template('file_manager.html', version=APP_VERSION)

@app.route('/api/files')
def get_files():
    """获取指定路径下的文件列表"""
    import mimetypes
    import os
    from datetime import datetime

    # 获取请求参数
    path = request.args.get('path', 'uploads')

    # 安全检查，防止路径遍历攻击
    if '..' in path or path.startswith('/'):
        return jsonify({
            'success': False,
            'error': 'Invalid path'
        }), 400

    # 构建完整路径
    # 确保uploads目录存在
    if not os.path.exists('uploads'):
        os.makedirs('uploads', exist_ok=True)

    # 优先使用当前工作目录下的uploads目录
    base_path = os.getcwd()
    full_path = os.path.join(base_path, path)

    # 检查路径是否存在
    if not os.path.exists(full_path):
        return jsonify({
            'success': False,
            'error': 'Path not found'
        }), 404

    # 检查是否为目录
    if not os.path.isdir(full_path):
        return jsonify({
            'success': False,
            'error': 'Path is not a directory'
        }), 400

    # 获取目录下的所有项目
    items = os.listdir(full_path)
    files = []

    for item in items:
        item_path = os.path.join(full_path, item)
        item_info = {
                'name': item,
                'path': os.path.join(path, item).replace('\\', '/'),
                'relativePath': os.path.relpath(item_path, os.path.join(base_path, 'uploads')).replace('\\', '/') if path.startswith('uploads') else None
            }

        if os.path.isdir(item_path):
            # 文件夹
            item_info['type'] = 'folder'
            item_info['size'] = 0
            # 统计子项目数量
            try:
                item_info['children'] = len(os.listdir(item_path))
            except:
                item_info['children'] = 0
        else:
            # 文件
            # 获取文件类型
            mime_type, _ = mimetypes.guess_type(item_path)
            if mime_type and mime_type.startswith('image/'):
                item_info['type'] = 'image'
                # 获取图片尺寸
                try:
                    from PIL import Image
                    with Image.open(item_path) as img:
                        width, height = img.size
                        item_info['width'] = width
                        item_info['height'] = height
                except:
                    item_info['width'] = 0
                    item_info['height'] = 0
            else:
                item_info['type'] = 'file'

            # 获取文件大小
            item_info['size'] = os.path.getsize(item_path)
            # 格式化文件大小
            def format_size(size):
                """格式化文件大小"""
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size < 1024.0:
                        return f"{size:.1f} {unit}"
                    size /= 1024.0
                return f"{size:.1f} TB"
            item_info['size'] = format_size(item_info['size'])

        # 获取修改时间
        mtime = os.path.getmtime(item_path)
        item_info['mtime'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        files.append(item_info)

    # 按类型排序，文件夹在前，文件在后，然后按名称排序
    files.sort(key=lambda x: (x['type'] != 'folder', x['name'].lower()))

    return jsonify({
        'success': True,
        'files': files
    })

@app.route('/api/classes')
def get_classes():
    """获取所有类别"""
    classes = []
    if os.path.exists(get_classes_file()):
        with open(get_classes_file(), 'r', encoding='utf-8') as f:
            classes = json.load(f)
    return jsonify(classes)


@app.route('/api/classes', methods=['POST'])
def save_classes():
    """保存所有类别"""
    data = request.json

    # 确保get_annotations_folder()目录存在
    os.makedirs(get_annotations_folder(), exist_ok=True)

    with open(get_classes_file(), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return jsonify({'message': 'Classes saved successfully'})


@app.route('/api/images')
def get_images():
    """获取所有上传的图片"""
    images = []

    # 读取标注信息，用于获取每张图片的标注数量
    annotations = {}
    if os.path.exists(get_annotations_file()):
        try:
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except json.JSONDecodeError:
            # 如果JSON文件无效或为空，使用空字典
            annotations = {}
        except Exception as e:
            # 处理其他可能的错误
            print(f"Error reading annotations file: {e}")
            annotations = {}

    # 获取所有图片文件，并按照创建时间排序（最新的在最后）
    upload_folder = get_upload_folder()
    image_files = []

    for filename in os.listdir(upload_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(upload_folder, filename)
            # 获取文件创建时间
            try:
                create_time = os.path.getctime(image_path)
                image_files.append((create_time, filename))
            except Exception as e:
                print(f"Error getting file creation time for {filename}: {e}")
                # 如果获取创建时间失败，使用当前时间作为默认值
                image_files.append((time.time(), filename))

    # 按照创建时间排序，最早的在前面，最新的在后面
    image_files.sort(key=lambda x: x[0])

    # 构建图片列表
    for create_time, filename in image_files:
        # 获取图片尺寸信息
        try:
            image_path = os.path.join(upload_folder, filename)
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width, height = 0, 0

        # 获取标注数量
        annotation_count = len(annotations.get(filename, []))

        images.append({
            'name': filename,
            'width': width,
            'height': height,
            'annotation_count': annotation_count
        })
    return jsonify({'images': images})


@app.route('/api/images/delete', methods=['POST'])
def delete_images():
    """删除指定的图片"""
    data = request.json or {}
    image_names = data.get('images', [])

    deleted_count = 0
    errors = []

    for image_name in image_names:
        try:
            # 删除图片文件
            image_path = os.path.join(get_upload_folder(), image_name)
            if os.path.exists(image_path):
                os.remove(image_path)
                deleted_count += 1

                # 同时删除对应的标注信息
                annotations = {}
                if os.path.exists(get_annotations_file()):
                    with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                        annotations = json.load(f)

                if image_name in annotations:
                    del annotations[image_name]
                    # 确保get_annotations_folder()目录存在
                    os.makedirs(get_annotations_folder(), exist_ok=True)
                    with open(get_annotations_file(), 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, indent=2)
            else:
                errors.append(f"图片 '{image_name}' 不存在")
        except Exception as e:
            errors.append(f"删除图片 '{image_name}' 失败: {str(e)}")

    if errors:
        return jsonify({
            'success': False,
            'deleted_count': deleted_count,
            'error': '; '.join(errors)
        }), 400

    return jsonify({
        'success': True,
        'deleted_count': deleted_count
    })


@app.route('/api/files/delete', methods=['POST'])
def delete_files():
    """删除指定的文件"""
    data = request.json or {}
    file_paths = data.get('files', [])

    deleted_count = 0
    errors = []

    for file_path in file_paths:
        try:
            # 安全检查，防止路径遍历攻击
            if '..' in file_path or file_path.startswith('/'):
                errors.append(f"无效的文件路径: '{file_path}'")
                continue

            # 构建完整路径
            full_path = os.path.join(app.root_path, file_path)

            # 检查文件是否存在
            if not os.path.exists(full_path):
                errors.append(f"文件 '{file_path}' 不存在")
                continue

            # 检查是否为文件
            if not os.path.isfile(full_path):
                errors.append(f" '{file_path}' 不是文件")
                continue

            # 删除文件
            os.remove(full_path)
            deleted_count += 1

            # 如果是图片文件，同时删除对应的标注信息
            if os.path.splitext(file_path)[1].lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                image_name = os.path.basename(file_path)
                annotations = {}
                if os.path.exists(get_annotations_file()):
                    with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                        annotations = json.load(f)

                if image_name in annotations:
                    del annotations[image_name]
                    # 确保get_annotations_folder()目录存在
                    os.makedirs(get_annotations_folder(), exist_ok=True)
                    with open(get_annotations_file(), 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, indent=2)
        except Exception as e:
            errors.append(f"删除文件 '{file_path}' 失败: {str(e)}")

    if errors:
        return jsonify({
            'success': False,
            'deleted_count': deleted_count,
            'error': '; '.join(errors)
        }), 400

    return jsonify({
        'success': True,
        'deleted_count': deleted_count
    })


@app.route('/api/files/create-folder', methods=['POST'])
def create_folder():
    """创建新文件夹"""
    data = request.json or {}
    path = data.get('path', '')
    folder_name = data.get('folderName', '')

    # 参数验证
    if not path or not folder_name:
        return jsonify({
            'success': False,
            'error': '缺少必要参数'
        }), 400

    # 安全检查，防止路径遍历攻击
    if '..' in path or path.startswith('/') or '..' in folder_name or folder_name.startswith('/'):
        return jsonify({
            'success': False,
            'error': '无效的路径或文件夹名称'
        }), 400

    try:
        # 构建完整的文件夹路径
        full_path = os.path.join(app.root_path, path, folder_name)

        # 检查文件夹是否已存在
        if os.path.exists(full_path):
            return jsonify({
                'success': False,
                'error': '文件夹已存在'
            }), 400

        # 创建文件夹
        os.makedirs(full_path, exist_ok=True)

        return jsonify({
            'success': True,
            'message': '文件夹创建成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'创建文件夹失败: {str(e)}'
        }), 500


@app.route('/api/files/upload', methods=['POST'])
def upload_files():
    """上传文件"""
    try:
        # 获取路径参数
        path = request.form.get('path', 'uploads')

        # 安全检查，防止路径遍历攻击
        if '..' in path or path.startswith('/'):
            return jsonify({
                'success': False,
                'error': '无效的路径'
            }), 400

        # 获取上传的文件
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({
                'success': False,
                'error': '没有选择要上传的文件'
            }), 400

        # 构建上传目录路径
        upload_dir = os.path.join(app.root_path, path)

        # 确保上传目录存在
        os.makedirs(upload_dir, exist_ok=True)

        uploaded_count = 0
        errors = []

        # 保存上传的文件
        for file in files:
            if file.filename:
                # 安全检查，防止路径遍历攻击
                if '..' in file.filename or file.filename.startswith('/'):
                    errors.append(f"无效的文件名: '{file.filename}'")
                    continue

                # 构建完整的文件路径
                file_path = os.path.join(upload_dir, file.filename)

                # 检查文件是否已存在
                if os.path.exists(file_path):
                    errors.append(f"文件 '{file.filename}' 已存在")
                    continue

                # 保存文件
                file.save(file_path)
                uploaded_count += 1

        if errors:
            return jsonify({
                'success': False,
                'uploaded_count': uploaded_count,
                'error': '; '.join(errors)
            }), 400

        return jsonify({
            'success': True,
            'uploaded_count': uploaded_count,
            'message': '文件上传成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'上传文件失败: {str(e)}'
        }), 500


@app.route('/api/upload-video', methods=['POST'])
def upload_video_for_label():
    """上传视频文件用于标注"""
    try:
        # 检查是否有文件上传
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有视频文件上传'
            }), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择视频文件'
            }), 400

        # 安全检查，防止路径遍历攻击
        if '..' in file.filename or file.filename.startswith('/'):
            return jsonify({
                'success': False,
                'error': '无效的文件名'
            }), 400

        # 构建上传目录路径
        upload_dir = os.path.join(app.root_path, 'uploads', 'auto', 'video')

        # 确保上传目录存在
        os.makedirs(upload_dir, exist_ok=True)

        # 构建完整的文件路径
        file_path = os.path.join(upload_dir, file.filename)

        # 检查文件是否已存在，如果存在则删除
        if os.path.exists(file_path):
            os.remove(file_path)

        # 保存文件
        file.save(file_path)

        # 返回相对路径，格式为: uploads/auto/video/filename
        relative_path = os.path.join('uploads', 'auto', 'video', file.filename)

        return jsonify({
            'success': True,
            'filePath': relative_path,
            'message': '视频文件上传成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'上传视频文件失败: {str(e)}'
        }), 500


@app.route('/api/files/download', methods=['POST'])
def download_files():
    """批量下载文件，将选中的文件压缩成tar文件后下载"""
    try:
        import tarfile
        import tempfile

        # 获取请求参数
        data = request.json or {}
        file_paths = data.get('files', [])

        if not file_paths:
            return jsonify({
                'success': False,
                'error': '没有选择要下载的文件'
            }), 400

        # 创建临时目录和tar文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建tar文件
            tar_file_path = os.path.join(temp_dir, 'files.tar')

            with tarfile.open(tar_file_path, 'w') as tar:
                # 添加每个文件到tar文件
                for file_path in file_paths:
                    # 安全检查，防止路径遍历攻击
                    if '..' in file_path or file_path.startswith('/'):
                        continue

                    # 构建完整的文件路径
                    full_path = os.path.join(app.root_path, file_path)

                    # 检查文件是否存在且是文件
                    if os.path.exists(full_path) and os.path.isfile(full_path):
                        # 获取相对路径（相对于app.root_path）
                        rel_path = os.path.relpath(full_path, app.root_path)
                        # 获取文件名
                        file_name = os.path.basename(full_path)
                        # 添加文件到tar，使用文件名作为内部名称
                        tar.add(full_path, arcname=file_name)

            # 读取tar文件内容
            with open(tar_file_path, 'rb') as f:
                tar_content = f.read()

        # 设置响应头，返回tar文件
        from flask import make_response
        response = make_response(tar_content)
        response.headers['Content-Type'] = 'application/x-tar'
        response.headers['Content-Disposition'] = 'attachment; filename=files.tar'
        response.headers['Content-Length'] = len(tar_content)

        return response
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'下载文件失败: {str(e)}'
        }), 500


@app.route('/api/image/<filename>')
def get_image(filename):
    """获取指定图片"""
    return send_from_directory(get_upload_folder(), filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """提供uploads目录下的文件访问，支持子目录"""
    import os

    # uploads 路由应始终映射到根目录下的 uploads 文件夹
    uploads_base = LEGACY_UPLOAD_FOLDER

    # 构建完整的文件路径
    full_path = os.path.join(uploads_base, filename)

    # 安全检查，防止路径遍历攻击
    if '..' in filename or filename.startswith('/'):
        print(f"不安全的文件路径: {filename}")
        return jsonify({
            'success': False,
            'error': 'Invalid file path'
        }), 400

    # 检查文件是否存在
    if not os.path.exists(full_path):
        return jsonify({
            'success': False,
            'error': 'File not found',
            'requested_path': filename,
            'full_path': full_path,
            'upload_folder': uploads_base
        }), 404

    return send_from_directory(uploads_base, filename)


@app.route('/api/upload', methods=['POST'])
def upload_folder():
    """上传整个文件夹"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    uploaded_files = []

    for file in files:
        if file.filename != '':
            filepath = os.path.join(get_upload_folder(), file.filename or '')
            file.save(filepath)
            uploaded_files.append(file.filename or '')

    return jsonify({'message': 'Files uploaded successfully', 'files': uploaded_files})


@app.route('/api/upload-labelme', methods=['POST'])
def upload_labelme_dataset():
    """上传LabelMe格式数据集"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        uploaded_files = []
        processed_annotations = 0

        # 读取现有的类别和标注信息
        classes = []
        if os.path.exists(get_classes_file()):
            with open(get_classes_file(), 'r', encoding='utf-8') as f:
                classes = json.load(f)

        annotations = {}
        if os.path.exists(get_annotations_file()):
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)

        # 获取已有类别名称集合，便于快速查找
        existing_class_names = {cls['name'] for cls in classes}

        # 处理上传的文件
        image_files = {}
        json_files = {}

        for file in files:
            if file.filename != '':
                filename = file.filename or ''
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_files[filename] = file
                elif filename.lower().endswith('.json'):
                    json_files[filename] = file

        # 处理图像文件
        for image_filename, image_file in image_files.items():
            # 保存图像文件
            image_path = os.path.join(get_upload_folder(), image_filename)
            image_file.save(image_path)
            uploaded_files.append(image_filename)

            # 查找对应的JSON文件
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            if json_filename in json_files:
                # 读取并解析JSON文件
                json_file = json_files[json_filename]
                json_content = json.loads(json_file.read().decode('utf-8'))

                # 解析LabelMe标注格式
                image_annotations = []
                if 'shapes' in json_content:
                    for shape in json_content['shapes']:
                        label = shape.get('label', '')
                        points = shape.get('points', [])

                        # 如果标签不存在于现有类别中，添加它
                        if label and label not in existing_class_names:
                            # 为新类别分配一个默认颜色
                            new_color = '#{:06x}'.format(hash(label) % 0x1000000)
                            classes.append({'name': label, 'color': new_color})
                            existing_class_names.add(label)

                        # 将points转换为我们的内部格式
                        if points and label:
                            # 查找标签的颜色
                            color = '#000000'  # 默认颜色
                            for cls in classes:
                                if cls['name'] == label:
                                    color = cls['color']
                                    break

                            # 确定形状类型
                            shape_type = shape.get('shape_type', 'polygon')

                            # 转换为我们的内部格式
                            internal_points = points
                            internal_type = shape_type

                            # 处理矩形：LabelMe矩形只有2个点，我们需要转换为4个点的矩形
                            if shape_type == 'rectangle' and len(points) == 2:
                                x1, y1 = points[0]
                                x2, y2 = points[1]
                                internal_points = [
                                    [x1, y1],
                                    [x2, y1],
                                    [x2, y2],
                                    [x1, y2]
                                ]
                                internal_type = 'rectangle'
                            elif shape_type == 'circle' and len(points) == 2:
                                # 处理圆形，转换为多边形（简化处理）
                                cx, cy = points[0]
                                radius = ((points[1][0] - cx) ** 2 + (points[1][1] - cy) ** 2) ** 0.5
                                # 转换为16边形近似圆形
                                internal_points = []
                                for i in range(16):
                                    angle = (i / 16) * 2 * 3.14159
                                    x = cx + radius * math.cos(angle)
                                    y = cy + radius * math.sin(angle)
                                    internal_points.append([x, y])
                                internal_type = 'polygon'
                            elif shape_type == 'line' and len(points) >= 2:
                                internal_type = 'line'
                            else:
                                internal_type = 'polygon'

                            # 创建标注对象
                            annotation = {
                                'class': label,
                                'color': color,
                                'points': internal_points,
                                'type': internal_type
                            }
                            image_annotations.append(annotation)

                # 保存此图像的标注
                annotations[image_filename] = image_annotations
                processed_annotations += 1

        # 保存更新后的类别和标注信息
        # 确保get_annotations_folder()目录存在
        os.makedirs(get_annotations_folder(), exist_ok=True)
        with open(get_classes_file(), 'w', encoding='utf-8') as f:
            json.dump(classes, f, indent=2)

        with open(get_annotations_file(), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)

        return jsonify({
            'message': 'LabelMe dataset uploaded successfully',
            'files': uploaded_files,
            'annotations_processed': processed_annotations
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process LabelMe dataset: {str(e)}'}), 500


@app.route('/api/upload/roboflow', methods=['POST'])
def upload_roboflow_dataset():
    """上传Roboflow格式ZIP数据集"""
    temp_dir = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.zip'):
            return jsonify({'error': '仅支持 .zip 格式的 Roboflow 数据集'}), 400

        # 文件大小限制 500MB
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > 500 * 1024 * 1024:
            return jsonify({'error': '文件大小超过 500MB 限制'}), 400

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()

        # 保存 ZIP 文件到临时目录
        zip_path = os.path.join(temp_dir, 'dataset.zip')
        file.save(zip_path)

        # 解压 ZIP 文件
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # 安全检查：跳过包含 .. 或绝对路径的条目
                if '..' in member or member.startswith('/'):
                    continue
                zip_ref.extract(member, extract_dir)

        # 读取现有的类别和标注信息
        classes = []
        if os.path.exists(get_classes_file()):
            with open(get_classes_file(), 'r', encoding='utf-8') as f:
                classes = json.load(f)

        annotations = {}
        if os.path.exists(get_annotations_file()):
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)

        existing_class_names = {cls['name'] for cls in classes}

        # 查找 data.yaml
        data_yaml_path = None
        # 优先查找根目录
        root_yaml = os.path.join(extract_dir, 'data.yaml')
        if os.path.exists(root_yaml):
            data_yaml_path = root_yaml
        else:
            # 搜索第一层子目录
            for item in os.listdir(extract_dir):
                item_path = os.path.join(extract_dir, item)
                if os.path.isdir(item_path):
                    sub_yaml = os.path.join(item_path, 'data.yaml')
                    if os.path.exists(sub_yaml):
                        data_yaml_path = sub_yaml
                        extract_dir = item_path
                        break

        # 解析类别信息
        class_names = {}  # id -> name
        warnings = []

        if data_yaml_path and os.path.exists(data_yaml_path):
            import yaml
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)

            if data_yaml and 'names' in data_yaml:
                names = data_yaml['names']
                if isinstance(names, dict):
                    class_names = {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, list):
                    class_names = {i: str(name) for i, name in enumerate(names)}

        # 定义图片扩展名
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')

        # 收集所有图片和标注文件
        # 优先扫描标准 YOLO 目录结构
        standard_splits = ['train', 'val', 'valid', 'test']
        image_files = {}  # basename -> full path
        label_files = {}  # basename -> full path

        for split in standard_splits:
            split_dir = os.path.join(extract_dir, split)
            if not os.path.exists(split_dir):
                continue

            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')

            if os.path.exists(images_dir):
                for filename in os.listdir(images_dir):
                    if filename.lower().endswith(image_extensions):
                        basename = os.path.splitext(filename)[0]
                        image_files[basename] = os.path.join(images_dir, filename)

            if os.path.exists(labels_dir):
                for filename in os.listdir(labels_dir):
                    if filename.lower().endswith('.txt'):
                        basename = os.path.splitext(filename)[0]
                        label_files[basename] = os.path.join(labels_dir, filename)

        # 如果标准目录没找到图片，递归扫描整个解压目录
        if not image_files:
            for root, dirs, files in os.walk(extract_dir):
                for filename in files:
                    if filename.lower().endswith(image_extensions):
                        basename = os.path.splitext(filename)[0]
                        if basename not in image_files:
                            image_files[basename] = os.path.join(root, filename)
                    elif filename.lower().endswith('.txt'):
                        basename = os.path.splitext(filename)[0]
                        if basename not in label_files:
                            label_files[basename] = os.path.join(root, filename)

        if not image_files:
            return jsonify({'error': 'ZIP 文件中未找到图片'}), 400

        # 如果没有 data.yaml，从标注文件推断类别
        if not class_names and label_files:
            max_class_id = -1
            for label_path in label_files.values():
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                try:
                                    class_id = int(parts[0])
                                    max_class_id = max(max_class_id, class_id)
                                except ValueError:
                                    pass
            if max_class_id >= 0:
                class_names = {i: f'class_{i}' for i in range(max_class_id + 1)}
                warnings.append('未找到 data.yaml，使用默认类别名称，请手动编辑类别名称')

        # 同步类别信息
        for class_id, name in sorted(class_names.items()):
            if name not in existing_class_names:
                new_color = '#{:06x}'.format(hash(name) % 0x1000000)
                classes.append({'name': name, 'color': new_color})
                existing_class_names.add(name)

        # 处理图片和标注
        uploaded_files = []
        annotations_imported = 0

        for basename, image_path in image_files.items():
            filename = os.path.basename(image_path)
            dest_path = os.path.join(get_upload_folder(), filename)

            # 跳过已存在的同名文件
            if os.path.exists(dest_path):
                continue

            # 复制图片到上传目录
            shutil.copy2(image_path, dest_path)
            uploaded_files.append(filename)

            # 获取图片尺寸
            try:
                img = Image.open(dest_path)
                width, height = img.size
            except Exception:
                width, height = 0, 0

            # 查找对应的标注文件
            if basename in label_files and width > 0 and height > 0:
                label_path = label_files[basename]
                image_annotations = []

                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            continue

                        class_name = class_names.get(class_id, f'class_{class_id}')
                        color = '#000000'
                        for cls in classes:
                            if cls['name'] == class_name:
                                color = cls['color']
                                break

                        coords = [float(p) for p in parts[1:]]

                        if len(coords) == 4:
                            # 边界框格式: cx cy w h
                            cx, cy, bw, bh = coords
                            x_min = (cx - bw / 2) * width
                            y_min = (cy - bh / 2) * height
                            x_max = (cx + bw / 2) * width
                            y_max = (cy + bh / 2) * height
                            points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                            ann_type = 'rectangle'
                        elif len(coords) >= 6 and len(coords) % 2 == 0:
                            # 多边形格式: x1 y1 x2 y2 ...
                            points = []
                            for i in range(0, len(coords), 2):
                                x = coords[i] * width
                                y = coords[i + 1] * height
                                points.append([x, y])
                            ann_type = 'polygon'
                        else:
                            continue

                        annotation = {
                            'class': class_name,
                            'color': color,
                            'points': points,
                            'type': ann_type
                        }
                        image_annotations.append(annotation)
                        annotations_imported += 1

                if image_annotations:
                    annotations[filename] = image_annotations

        # 保存更新后的类别和标注信息
        os.makedirs(get_annotations_folder(), exist_ok=True)
        with open(get_classes_file(), 'w', encoding='utf-8') as f:
            json.dump(classes, f, indent=2, ensure_ascii=False)

        with open(get_annotations_file(), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        result = {
            'message': 'Roboflow dataset imported successfully',
            'images_imported': len(uploaded_files),
            'annotations_imported': annotations_imported,
            'classes': [cls['name'] for cls in classes],
        }
        if warnings:
            result['warnings'] = warnings

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to process Roboflow dataset: {str(e)}'}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/rotate-image', methods=['POST'])
def rotate_image():
    """旋转图片及标注"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        image_name = data.get('image_name')
        angle = data.get('angle', 90)  # 90, -90, 180

        if not image_name:
            return jsonify({'error': 'Image name is required'}), 400

        if angle not in (90, -90, 180):
            return jsonify({'error': 'Angle must be 90, -90, or 180'}), 400

        image_path = os.path.join(get_upload_folder(), image_name)
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_name}'}), 404

        # 读取图片并旋转
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size

            def transform_cw(x, y):
                return (orig_height - 1 - y, x)

            def transform_ccw(x, y):
                return (y, orig_width - 1 - x)

            def transform_180(x, y):
                return (orig_width - 1 - x, orig_height - 1 - y)

            if angle == 90:
                img_rotated = img.transpose(Image.Transpose.ROTATE_270)
                transform = transform_cw
            elif angle == -90:
                img_rotated = img.transpose(Image.Transpose.ROTATE_90)
                transform = transform_ccw
            else:  # 180
                img_rotated = img.transpose(Image.Transpose.ROTATE_180)
                transform = transform_180

            # 覆盖保存旋转后的图片
            img_rotated.save(image_path)

        # 更新标注坐标
        annotations = {}
        if os.path.exists(get_annotations_file()):
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)

        if image_name in annotations and annotations[image_name]:
            rotated_annotations = []
            for ann in annotations[image_name]:
                points = ann.get('points', [])
                if not points:
                    rotated_annotations.append(ann)
                    continue

                rotated_points = []
                for point in points:
                    if isinstance(point, dict) and 'x' in point and 'y' in point:
                        nx, ny = transform(point['x'], point['y'])
                        rotated_points.append({'x': nx, 'y': ny})
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        nx, ny = transform(point[0], point[1])
                        rotated_points.append([nx, ny])
                    else:
                        rotated_points.append(point)

                rotated_ann = dict(ann)
                rotated_ann['points'] = rotated_points
                rotated_annotations.append(rotated_ann)

            annotations[image_name] = rotated_annotations

            with open(get_annotations_file(), 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)

        return jsonify({
            'success': True,
            'message': f'图片已旋转 {angle}°',
            'image_name': image_name
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to rotate image: {str(e)}'}), 500


@app.route('/api/ai-label', methods=['POST'])
def ai_label():
    """AI标注功能"""
    try:
        import datetime
        import json
        import logging
        import os

        # 获取请求数据
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        images = data.get('images', [])
        selected_label = data.get('label')
        api_config = data.get('apiConfig', {})

        if not images:
            return jsonify({'success': False, 'error': 'No images provided'}), 400

        if not selected_label:
            return jsonify({'success': False, 'error': 'No label provided'}), 400

        # 获取API配置
        api_url = api_config.get('apiUrl', 'http://127.0.0.1:1234/v1')
        api_key = api_config.get('apiKey', '')
        timeout = int(api_config.get('timeout', 300))
        prompt = api_config.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
        model = api_config.get('model', 'qwen/qwen3-vl-8b')
        inference_tool = api_config.get('inferenceTool', 'LMStudio')

        # 初始化AIAutoLabeler
        labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)

        # 读取现有的标注信息
        annotations = {}
        if os.path.exists(get_annotations_file()):
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)

        processed_count = 0
        labeled_count = 0
        total_images = len(images)
        start_time = datetime.datetime.now()

        # 处理每张图片
        for image_name in images:
            # 构建图片路径
            image_path = os.path.join(get_upload_folder(), image_name)
            if not os.path.exists(image_path):
                logging.error(f"Image not found: {image_path}")
                continue

            processed_count += 1

            # 发送实时进度更新
            current_time = datetime.datetime.now()
            elapsed_seconds = int((current_time - start_time).total_seconds())
            progress_data = {
                'task_type': 'ai_label',
                'status': 'running',
                'processed': processed_count,
                'total': total_images,
                'elapsed_time': elapsed_seconds,
                'labeled': labeled_count,
                'message': f'正在处理第 {processed_count}/{total_images} 张图片'
            }
            socketio.emit('ai_label_progress', progress_data)

            # 调用API进行标注
            try:
                result = labeler.analyze_image(image_path)
                detections = result.get("detections", [])
                if isinstance(detections, dict):
                    detections = [detections]

                # 如果检测到目标，更新标注状态
                if detections:
                    # 为每张图片创建标注
                    image_annotations = []
                    for detection in detections:
                        # 确保detection是字典
                        if isinstance(detection, dict):
                            label = selected_label  # 使用选中的标签
                            confidence = detection.get("confidence", 0.0)
                            obb_points = detection.get("points", None)

                            if obb_points and len(obb_points) >= 3:
                                # OBB 格式：直接使用 4 角点
                                points = [[float(p[0]), float(p[1])] for p in obb_points[:4]]
                                annotation = {
                                    "id": str(uuid.uuid4()),
                                    "class": label,
                                    "type": "obb",
                                    "points": points,
                                    "confidence": confidence
                                }
                            else:
                                # 矩形框格式（现有逻辑）
                                bbox = detection.get("bbox", [0, 0, 0, 0])
                                bbox = list(map(float, bbox)) if isinstance(bbox, (list, tuple)) else [0, 0, 0, 0]
                                if len(bbox) < 4:
                                    bbox = bbox + [0] * (4 - len(bbox))
                                x1, y1, x2, y2 = bbox[:4]
                                annotation = {
                                    "id": str(uuid.uuid4()),
                                    "class": label,
                                    "type": "rectangle",
                                    "points": [
                                        [x1, y1],
                                        [x2, y1],
                                        [x2, y2],
                                        [x1, y2]
                                    ],
                                    "confidence": confidence
                                }
                            image_annotations.append(annotation)

                    # 更新标注信息
                    annotations[image_name] = image_annotations
                    labeled_count += 1
            except Exception as e:
                logging.error(f"Failed to process image {image_name}: {str(e)}")
                continue

        # 保存更新后的标注信息
        # 确保get_annotations_folder()目录存在
        os.makedirs(get_annotations_folder(), exist_ok=True)
        with open(get_annotations_file(), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        # 发送最终进度更新
        current_time = datetime.datetime.now()
        elapsed_seconds = int((current_time - start_time).total_seconds())
        final_progress = {
            'task_type': 'ai_label',
            'status': 'completed',
            'processed': processed_count,
            'total': total_images,
            'elapsed_time': elapsed_seconds,
            'labeled': labeled_count,
            'message': f'标注完成，成功处理 {processed_count} 张图片，其中 {labeled_count} 张标注成功'
        }
        socketio.emit('ai_label_progress', final_progress)

        return jsonify({
            'success': True,
            'processed': processed_count,
            'labeled': labeled_count,
            'message': f'成功处理 {processed_count} 张图片，其中 {labeled_count} 张标注成功'
        })

    except Exception as e:
        import traceback
        logging.error(f"AI label failed: {str(e)}")

        # 发送错误进度更新
        progress_data = {
            'task_type': 'ai_label',
            'status': 'error',
            'error': str(e),
            'message': f'标注失败: {str(e)}'
        }
        socketio.emit('ai_label_progress', progress_data)

        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """上传视频文件并抽帧"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    frame_interval = int(request.form.get('frame_interval', 30))  # 默认每隔30帧保存一帧

    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    try:
        # 保存视频文件到临时位置
        temp_video_path = os.path.join(get_upload_folder(), 'temp_' + (video_file.filename or 'video'))
        video_file.save(temp_video_path)

        # 抽帧处理，传递原始文件名
        extracted_frames = extract_frames(temp_video_path, frame_interval, video_file.filename)

        # 删除临时视频文件
        os.remove(temp_video_path)

        return jsonify({
            'message': 'Video frames extracted successfully',
            'frames': extracted_frames,
            'count': len(extracted_frames)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process video: {str(e)}'}), 500


def extract_frames(video_path, frame_interval, original_filename=None):
    """从视频中抽帧并保存为图片"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    extracted_frames = []

    # 生成文件名前缀
    if original_filename:
        # 使用原始视频文件名作为前缀
        video_name = os.path.splitext(os.path.basename(original_filename))[0]
    else:
        # 使用视频路径中的文件名作为前缀
        video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔frame_interval帧保存一帧
        if frame_count % frame_interval == 0:
            # 生成文件名
            frame_filename = f"{video_name}_frame_{saved_frame_count:06d}.jpg"
            frame_path = os.path.join(get_upload_folder(), frame_filename)

            # 保存帧为图片
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_filename)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    return extracted_frames


@app.route('/api/annotations/<image_name>')
def get_annotations(image_name):
    """获取特定图片的标注"""
    annotations = {}
    if os.path.exists(get_annotations_file()):
        try:
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except json.JSONDecodeError:
            # 如果JSON文件无效或为空，使用空字典
            annotations = {}
        except Exception as e:
            # 处理其他可能的错误
            print(f"Error reading annotations file: {e}")
            annotations = {}

    image_annotations = annotations.get(image_name, [])
    return jsonify(image_annotations)


@app.route('/api/annotations/<image_name>', methods=['POST'])
def save_annotations(image_name):
    """保存特定图片的标注"""
    data = request.json

    annotations = {}
    if os.path.exists(get_annotations_file()):
        try:
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except json.JSONDecodeError:
            # 如果JSON文件无效或为空，使用空字典
            annotations = {}
        except Exception as e:
            # 处理其他可能的错误
            print(f"Error reading annotations file: {e}")
            annotations = {}

    annotations[image_name] = data

    # 确保get_annotations_folder()目录存在
    os.makedirs(get_annotations_folder(), exist_ok=True)
    with open(get_annotations_file(), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    return jsonify({'message': 'Annotations saved successfully'})


@app.route('/api/ai-annotate', methods=['POST'])
def ai_annotate():
    """执行AI自动标注 - 已停用"""
    return jsonify({
        'error': 'AI自动标注功能已停用',
        'details': '管理员已停用此功能'
    }), 400


# 自动标注相关API
@app.route('/api/save-api-config', methods=['POST'])
def save_api_config():
    """保存API配置"""
    try:
        # 获取配置数据
        config_data = request.json
        if not config_data:
            return jsonify({'success': False, 'error': 'No config data provided'}), 400

        # 确保uploads/config目录存在
        os.makedirs(os.path.join(get_upload_folder(), 'config'), exist_ok=True)

        # 保存配置到文件
        config_path = os.path.join(get_upload_folder(), 'config', 'ai_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'API配置保存成功'})
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/load-api-config', methods=['GET'])
def load_api_config():
    """加载API配置"""
    try:
        # 读取配置文件
        config_path = os.path.join(get_upload_folder(), 'config', 'ai_config.json')
        logging.info(f"[load-api-config] 当前项目: {get_current_project()}, 配置文件: {config_path}")
        if not os.path.exists(config_path):
            # 返回默认配置
            default_config = {
                "inferenceTool": "LMStudio",
                "model": "qwen/qwen3-vl-8b",
                "apiUrl": "http://127.0.0.1:1234/v1",
                "apiKey": "",
                "timeout": 300,
                "prompt": "检测图中物体，返回JSON：{\"detections\":[{\"label\":\"类别\",\"confidence\":0.9,\"bbox\":[x1,y1,x2,y2]}]}"
            }
            return jsonify({'success': True, 'config': default_config})

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        return jsonify({'success': True, 'config': config_data})
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/auto-label/test', methods=['POST'])
def api_test():
    """测试大模型API连接"""
    try:
        # 获取表单数据
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        image_file = request.files['image']
        api_url = request.form.get('api_url', 'http://127.0.0.1:1234/v1')
        api_key = request.form.get('api_key', '')
        timeout = int(request.form.get('timeout', 300))
        prompt = request.form.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
        inference_tool = request.form.get('inferenceTool', 'LMStudio')
        model = request.form.get('model', 'qwen/qwen3-vl-8b')
        use_sam = request.form.get('use_sam', 'false') == 'true'

        # 保存临时图片文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_file.save(temp_file_path)

        try:
            # 初始化AIAutoLabeler
            labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)

            # 调用analyze_image方法测试API
            result = labeler.analyze_image(temp_file_path)

            # SAM 2 精化（可选）
            sam_refinements = []
            detections = result.get("detections", [])
            if use_sam and detections:
                sam_engine = get_sam2_engine()
                if sam_engine and sam_engine.is_loaded():
                    for det in detections:
                        bbox = det.get("bbox", [])
                        if len(bbox) == 4:
                            try:
                                sam_result = sam_engine.predict(temp_file_path, [
                                    {"type": "box", "x1": bbox[0], "y1": bbox[1],
                                     "x2": bbox[2], "y2": bbox[3]}
                                ])
                                sam_refinements.append({
                                    "label": det.get("label", "unknown"),
                                    "bbox": bbox,
                                    "mask_polygons": sam_result.get("mask_polygons", []),
                                    "area": sam_result.get("area", 0),
                                })
                            except Exception as sam_err:
                                sam_refinements.append({
                                    "label": det.get("label", "unknown"),
                                    "bbox": bbox,
                                    "error": str(sam_err),
                                })

            # 渲染检测结果到图片上
            rendered_data = None
            sam_rendered_data = None
            if detections and len(detections) > 0:
                try:
                    rendered_path = labeler.render_detections(temp_file_path, detections)
                    import base64
                    with open(rendered_path, "rb") as f:
                        rendered_data = base64.b64encode(f.read()).decode("utf-8")
                    if os.path.exists(rendered_path):
                        os.remove(rendered_path)

                    # 如果有 SAM 精化结果，单独渲染 Mask 多边形预览图
                    if use_sam and sam_refinements:
                        sam_img = cv2.imread(temp_file_path)
                        for ref in sam_refinements:
                            for poly in ref.get("mask_polygons", []):
                                if len(poly) >= 3:
                                    pts = np.array([poly], dtype=np.int32)
                                    cv2.polylines(sam_img, pts, True, (0, 255, 0), 2)
                                    overlay = sam_img.copy()
                                    cv2.fillPoly(overlay, pts, (0, 255, 0, 80))
                                    cv2.addWeighted(overlay, 0.2, sam_img, 0.8, 0, sam_img)
                        sam_rendered_path = temp_file_path + "_sam.jpg"
                        cv2.imwrite(sam_rendered_path, sam_img)
                        with open(sam_rendered_path, "rb") as f:
                            sam_rendered_data = base64.b64encode(f.read()).decode("utf-8")
                        if os.path.exists(sam_rendered_path):
                            os.remove(sam_rendered_path)
                except Exception as render_err:
                    logging.warning(f"渲染检测结果失败: {render_err}")

            return jsonify({
                'success': True,
                'result': result,
                'rendered_image': rendered_data,
                'sam_rendered_image': sam_rendered_data,
                'sam_refinements': sam_refinements if use_sam else None,
            })
        finally:
            # 确保临时文件被删除
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# 修复后的auto_label_image函数
@app.route('/api/auto-label/image', methods=['POST'])
def auto_label_image():
    """图片自动标注"""
    try:
        import logging
        import os

        # 获取表单数据
        files = request.files.getlist('images')
        output_dir = request.form.get('output_dir', 'output')
        api_url = request.form.get('api_url', 'http://127.0.0.1:1234/v1')
        api_key = request.form.get('api_key', '')
        timeout = int(request.form.get('timeout', 300))
        prompt = request.form.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
        inference_tool = request.form.get('inferenceTool', 'LMStudio')

        if not files:
            return jsonify({'success': False, 'error': 'No image files provided'}), 400

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        raw_dir = os.path.join(output_dir, 'raw_frames')
        labeled_dir = os.path.join(output_dir, 'labeled_frames')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(labeled_dir, exist_ok=True)

        processed_count = 0
        total_detections = 0

        # 初始化图片列表，用于存储每张图片的处理结果和Base64数据
        images = []

        # 获取模型配置
        model = request.form.get('model', 'qwen/qwen3-vl-8b')

        # 初始化AIAutoLabeler
        labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)

        # 处理每张图片
        for file in files:
            if file.filename == '':
                continue

            # 保存原始图片
            filename = os.path.basename(file.filename)
            raw_path = os.path.join(raw_dir, filename)
            file.save(raw_path)

            # 调用API进行标注
            try:
                result = labeler.analyze_image(raw_path)
                detections = result.get("detections", [])
                if isinstance(detections, dict):
                    detections = [detections]
            except Exception as e:
                error_msg = f"处理图片失败: {str(e)}"
                logging.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'processed': processed_count,
                    'detections': total_detections,
                    'output_dir': output_dir
                }), 500

            # 渲染检测结果
            rendered_path = labeler.render_detections(raw_path, detections)

            # 移动渲染后的图片到输出目录
            labeled_path = os.path.join(labeled_dir, filename)
            # 如果目标文件已存在，先删除
            if os.path.exists(labeled_path):
                os.remove(labeled_path)
            os.rename(rendered_path, labeled_path)

            # 生成原始图片的Base64数据
            import base64
            with open(raw_path, "rb") as f:
                raw_image_data = f.read()
            raw_image_base64 = base64.b64encode(raw_image_data).decode("utf-8")
            raw_image_base64 = f"data:image/jpeg;base64,{raw_image_base64}"

            # 生成渲染后图片的Base64数据
            with open(labeled_path, "rb") as f:
                labeled_image_data = f.read()
            labeled_image_base64 = base64.b64encode(labeled_image_data).decode("utf-8")
            labeled_image_base64 = f"data:image/jpeg;base64,{labeled_image_base64}"

            # 将图片信息添加到列表
            images.append({
                'filename': filename,
                'original_image': raw_image_base64,
                'labeled_image': labeled_image_base64,
                'detections': len(detections)
            })

            processed_count += 1
            total_detections += len(detections)

        return jsonify({
            'success': True,
            'processed': processed_count,
            'detections': total_detections,
            'output_dir': output_dir,
            'images': images
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# 修复后的auto_label_video函数
@app.route('/api/auto-label/video', methods=['POST'])
def auto_label_video():
    """视频自动标注"""
    try:
        import logging
        import os

        # 获取请求数据
        data = request.json
        video_path = data.get('video_path')
        frame_interval = int(data.get('frame_interval', 10))
        output_dir = data.get('output_dir', 'output')
        api_config = data.get('api_config', {})

        if not video_path:
            return jsonify({'success': False, 'error': 'No video path provided'}), 400

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        raw_dir = os.path.join(output_dir, 'raw_frames')
        labeled_dir = os.path.join(output_dir, 'labeled_frames')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(labeled_dir, exist_ok=True)

        # 获取API配置
        api_url = api_config.get('apiUrl', 'http://127.0.0.1:1234/v1')
        api_key = api_config.get('apiKey', '')
        timeout = int(api_config.get('timeout', 300))
        prompt = api_config.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
        model = api_config.get('model', 'qwen/qwen3-vl-8b')
        inference_tool = api_config.get('inferenceTool', 'LMStudio')

        # 初始化AIAutoLabeler
        labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)

        # 打开视频流
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': f'Failed to open video: {video_path}'}), 400

        frame_count = 0
        processed_count = 0
        total_detections = 0

        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按照指定间隔处理帧
            if frame_count % frame_interval == 0:
                # 保存原始帧
                frame_filename = f"frame_{frame_count:06d}.jpg"
                raw_frame_path = os.path.join(raw_dir, frame_filename)
                cv2.imwrite(raw_frame_path, frame)

                # 调用API进行标注
                try:
                    result = labeler.analyze_image(raw_frame_path)
                    detections = result.get("detections", [])
                    if isinstance(detections, dict):
                        detections = [detections]
                except Exception as e:
                    error_msg = f"处理视频帧失败: {str(e)}"
                    logging.error(error_msg)
                    return jsonify({
                        'success': False,
                        'error': error_msg,
                        'processed': processed_count,
                        'detections': total_detections,
                        'output_dir': output_dir
                    }), 500

                # 渲染检测结果
                rendered_path = labeler.render_detections(raw_frame_path, detections)

                # 移动渲染后的图片到输出目录
                labeled_path = os.path.join(labeled_dir, frame_filename)
                # 如果目标文件已存在，先删除
                if os.path.exists(labeled_path):
                    os.remove(labeled_path)
                os.rename(rendered_path, labeled_path)

                processed_count += 1
                total_detections += len(detections)

            frame_count += 1

        # 释放资源
        cap.release()

        return jsonify({
            'success': True,
            'processed': processed_count,
            'detections': total_detections,
            'output_dir': output_dir
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/check-yolo11-install')
def check_yolo11_install():
    """检查YOLO11安装状态"""
    import os
    # 检查YOLO11安装路径是否存在
    yolo11_path = os.path.join(app.root_path, 'plugins', 'yolo11')
    is_installed = os.path.exists(yolo11_path) and os.path.isdir(yolo11_path)

    # 初始化安装信息
    install_info = {
        'is_installed': is_installed,
        'install_time': '',
        'has_cuda': False,
        'hardware': 'CPU'
    }

    # 如果已安装，读取详细的安装信息
    if is_installed:
        install_info_path = os.path.join(yolo11_path, 'install_info.json')
        if os.path.exists(install_info_path):
            try:
                with open(install_info_path, 'r', encoding='utf-8') as f:
                    saved_info = json.load(f)
                    # 更新安装信息
                    install_info.update(saved_info)
            except Exception as e:
                print(f"读取安装信息失败: {e}")

    return jsonify(install_info)


@app.route('/api/install-yolo11')
def install_yolo11():
    """安装YOLO11"""
    import datetime
    import os
    import subprocess
    import time
    import venv

    from flask import Response

    # 获取安装路径
    install_path = request.args.get('install_path', 'plugins/yolo11')
    # 确保安装路径是相对于项目根目录的
    if not os.path.isabs(install_path):
        install_path = os.path.join(app.root_path, install_path)

    def generate():
        # 发送初始状态
        yield f"data: {json.dumps({'status': 'started', 'message': '开始安装YOLO11...', 'progress': 0})}\n\n"
        time.sleep(0.5)

        try:
            # 1. 创建安装目录
            yield f"data: {json.dumps({'message': '创建安装目录...', 'progress': 10})}\n\n"
            os.makedirs(install_path, exist_ok=True)
            time.sleep(0.5)

            # 2. 创建Python虚拟环境
            yield f"data: {json.dumps({'message': '创建Python虚拟环境...', 'progress': 20})}\n\n"

            # 创建虚拟环境
            venv_path = os.path.join(install_path, 'venv')
            venv.create(venv_path, with_pip=True)
            time.sleep(0.5)

            # 3. 安装YOLO11的依赖
            yield f"data: {json.dumps({'message': '安装YOLO11依赖...', 'progress': 40})}\n\n"

            # 获取虚拟环境中的pip路径
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(venv_path, 'Scripts', 'pip.exe')
                python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
            else:  # Linux/macOS
                pip_path = os.path.join(venv_path, 'bin', 'pip')
                python_path = os.path.join(venv_path, 'bin', 'python')

            # 升级pip
            result = subprocess.run(
                [python_path, '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=install_path
            )

            if result.returncode != 0:
                yield f"data: {json.dumps({'status': 'error', 'message': f'升级pip失败: {result.stderr}', 'progress': 40})}\n\n"
                return

            # 安装ultralytics
            result = subprocess.run(
                [pip_path, 'install', 'ultralytics'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=install_path
            )

            if result.returncode != 0:
                yield f"data: {json.dumps({'status': 'error', 'message': f'安装ultralytics失败: {result.stderr}', 'progress': 50})}\n\n"
                return

            time.sleep(0.5)

            # 4. 检查硬件支持
            yield f"data: {json.dumps({'message': '检查硬件支持...', 'progress': 70})}\n\n"

            # 检查是否支持CUDA
            has_cuda = False
            try:
                result = subprocess.run(
                    [python_path, '-c', 'import torch; print(torch.cuda.is_available())'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=install_path
                )
                has_cuda = result.stdout.strip().lower() == 'true'
            except Exception as e:
                print(f"检查CUDA支持失败: {e}")

            time.sleep(0.5)

            # 5. 创建models目录
            yield f"data: {json.dumps({'message': '创建models目录...', 'progress': 80})}\n\n"
            models_dir = os.path.join(install_path, 'models')
            os.makedirs(models_dir, exist_ok=True)
            time.sleep(0.5)

            # 6. 记录安装信息
            yield f"data: {json.dumps({'message': '记录安装信息...', 'progress': 90})}\n\n"

            install_info = {
                'is_installed': True,
                'install_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'install_path': install_path,
                'has_cuda': has_cuda,
                'hardware': 'CUDA' if has_cuda else 'CPU'
            }

            # 保存安装信息到文件
            install_info_path = os.path.join(install_path, 'install_info.json')
            with open(install_info_path, 'w') as f:
                json.dump(install_info, f, indent=2, ensure_ascii=False)

            time.sleep(0.5)

            # 7. 安装完成
            yield f"data: {json.dumps({'message': 'YOLO11安装完成！', 'progress': 100, 'status': 'completed', 'has_cuda': has_cuda})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'status': 'error', 'message': f'安装失败: {str(e)}', 'progress': 0, 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/uninstall-yolo11')
def uninstall_yolo11():
    """卸载YOLO11"""
    import os
    import shutil
    import time

    from flask import Response

    # 获取安装路径
    install_path = request.args.get('install_path', 'plugins/yolo11')
    # 确保安装路径是相对于项目根目录的
    if not os.path.isabs(install_path):
        install_path = os.path.join(app.root_path, install_path)

    def generate():
        # 发送初始状态
        yield f"data: {json.dumps({'status': 'started', 'message': '开始卸载YOLO11...', 'progress': 0})}\n\n"
        time.sleep(0.5)

        try:
            # 检查YOLO11是否安装
            if not os.path.exists(install_path) or not os.path.isdir(install_path):
                yield f"data: {json.dumps({'status': 'error', 'message': 'YOLO11未安装', 'progress': 0})}\n\n"
                return

            # 1. 删除安装目录
            yield f"data: {json.dumps({'message': '删除YOLO11安装目录...', 'progress': 50})}\n\n"

            # 强制删除整个YOLO11目录，包括venv文件夹
            # 先尝试使用shutil.rmtree删除
            shutil.rmtree(install_path, ignore_errors=False)

            # 验证是否删除成功
            if os.path.exists(install_path):
                # 如果shutil.rmtree失败，尝试使用os.system强制删除（针对Windows系统）
                if os.name == 'nt':  # Windows系统
                    os.system(f'rmdir /s /q "{install_path}"')
                else:  # Linux/macOS系统
                    os.system(f'rm -rf "{install_path}"')

                # 再次验证
                if os.path.exists(install_path):
                    raise Exception(f'无法删除目录: {install_path}')

            time.sleep(0.5)

            # 2. 卸载完成
            yield f"data: {json.dumps({'message': 'YOLO11卸载完成！', 'progress': 100, 'status': 'completed'})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'status': 'error', 'message': f'卸载失败: {str(e)}', 'progress': 0, 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/download-models')
def download_models():
    """下载YOLO11预训练模型"""
    import os
    import subprocess
    import time

    from flask import Response

    # 获取模型列表和安装路径
    models_str = request.args.get('models', '')
    models = models_str.split(',') if models_str else []
    install_path = request.args.get('install_path', 'plugins/yolo11')

    # 确保安装路径是相对于项目根目录的
    if not os.path.isabs(install_path):
        install_path = os.path.join(app.root_path, install_path)

    def generate():
        # 发送初始状态
        yield f"data: {json.dumps({'status': 'started', 'message': '开始下载模型...', 'progress': 0})}\n\n"
        time.sleep(0.5)

        try:
            # 检查YOLO11是否安装
            if not os.path.exists(install_path) or not os.path.isdir(install_path):
                yield f"data: {json.dumps({'status': 'error', 'message': 'YOLO11未安装', 'progress': 0})}\n\n"
                return

            # 获取虚拟环境中的python路径
            if os.name == 'nt':  # Windows
                python_path = os.path.join(install_path, 'venv', 'Scripts', 'python.exe')
            else:  # Linux/macOS
                python_path = os.path.join(install_path, 'venv', 'bin', 'python')

            # 检查python路径是否存在
            if not os.path.exists(python_path):
                yield f"data: {json.dumps({'status': 'error', 'message': '虚拟环境未找到', 'progress': 0})}\n\n"
                return

            # 预检 ultralytics 是否已安装
            check_result = subprocess.run(
                [python_path, '-c', 'import ultralytics; print(ultralytics.__version__)'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=install_path
            )
            if check_result.returncode != 0:
                err = (check_result.stderr or check_result.stdout)[:300]
                yield f"data: {json.dumps({'status': 'error', 'message': f'训练环境异常，无法导入 ultralytics: {err}', 'progress': 0})}\n\n"
                return

            # 创建models目录
            models_dir = os.path.join(install_path, 'models')
            os.makedirs(models_dir, exist_ok=True)

            # 过滤已安装的模型
            models_to_download = []
            skipped_models = []
            for model in models:
                pt_path = os.path.join(models_dir, model + '.pt')
                if os.path.exists(pt_path):
                    skipped_models.append(model)
                else:
                    models_to_download.append(model)

            if skipped_models:
                skipped_str = ', '.join(skipped_models)
                yield f"data: {json.dumps({'message': '已跳过已安装模型: ' + skipped_str, 'progress': 5})}\n\n"
                time.sleep(0.3)

            if not models_to_download:
                yield f"data: {json.dumps({'message': '所有选中模型均已安装，无需下载', 'progress': 100, 'status': 'completed'})}\n\n"
                return

            # 下载每个模型
            total_models = len(models_to_download)
            for i, model in enumerate(models_to_download):
                yield f"data: {json.dumps({'message': f'正在下载模型: {model}...', 'progress': int((i / total_models) * 50) + 10})}\n\n"

                # 使用ultralytics的CLI下载模型
                result = subprocess.run(
                    [python_path, '-c', f'from ultralytics import YOLO; YOLO("{model}.pt")'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=models_dir
                )

                if result.returncode != 0:
                    err_detail = (result.stderr or result.stdout)[:500]
                    yield f"data: {json.dumps({'status': 'error', 'message': f'下载模型 {model} 失败: {err_detail}', 'progress': 0})}\n\n"
                    return

                time.sleep(0.5)

            # 下载完成
            msg = '模型下载完成！'
            if skipped_models:
                skipped_str = ', '.join(skipped_models)
                msg += ' (已跳过: ' + skipped_str + ')'
            yield f"data: {json.dumps({'message': msg, 'progress': 100, 'status': 'completed'})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'status': 'error', 'message': f'下载失败: {str(e)}', 'progress': 0, 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/list-models')
def list_models():
    """获取已安装的YOLO11模型列表"""
    import os

    # 获取安装路径
    install_path = request.args.get('install_path', 'plugins/yolo11')
    # 确保安装路径是相对于项目根目录的
    if not os.path.isabs(install_path):
        install_path = os.path.join(app.root_path, install_path)

    # 初始化模型列表
    models = []

    # 检查YOLO11是否安装
    if os.path.exists(install_path) and os.path.isdir(install_path):
        # 检查models目录是否存在
        models_dir = os.path.join(install_path, 'models')
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            # 列出models目录下的所有.pt文件
            for file in os.listdir(models_dir):
                if file.endswith('.pt'):
                    models.append(file)

    return jsonify({'models': models})


@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    """上传YOLO11模型文件"""
    import os

    # 获取安装路径
    install_path = request.headers.get('X-Install-Path', 'plugins/yolo11')
    # 确保安装路径是相对于项目根目录的
    if not os.path.isabs(install_path):
        install_path = os.path.join(app.root_path, install_path)

    # 检查YOLO11是否安装
    if not os.path.exists(install_path) or not os.path.isdir(install_path):
        return jsonify({'success': False, 'error': 'YOLO11未安装'})

    # 检查是否有文件上传
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': '未找到上传的文件'})

    # 创建models目录
    models_dir = os.path.join(install_path, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 保存上传的文件
    uploaded_files = []
    files = request.files.getlist('files[]')
    for file in files:
        if file.filename != '' and file.filename.endswith('.pt'):
            # 保存文件到models目录
            file_path = os.path.join(models_dir, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)

    return jsonify({'success': True, 'uploaded_files': uploaded_files})


@app.route('/api/delete-model', methods=['POST'])
def delete_model():
    """删除YOLO11模型文件"""
    import os

    # 获取安装路径
    install_path = request.headers.get('X-Install-Path', 'plugins/yolo11')
    # 确保安装路径是相对于项目根目录的
    if not os.path.isabs(install_path):
        install_path = os.path.join(app.root_path, install_path)

    # 获取模型名称
    data = request.json or {}
    model_name = data.get('model_name', '')

    # 检查YOLO11是否安装
    if not os.path.exists(install_path) or not os.path.isdir(install_path):
        return jsonify({'success': False, 'error': 'YOLO11未安装'})

    # 检查模型名称是否为空
    if not model_name:
        return jsonify({'success': False, 'error': '模型名称不能为空'})

    # 构建模型文件路径
    models_dir = os.path.join(install_path, 'models')
    model_path = os.path.join(models_dir, model_name)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': '模型文件不存在'})

    try:
        # 删除模型文件
        os.remove(model_path)
        return jsonify({'success': True, 'message': f'模型 {model_name} 删除成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'删除模型失败: {str(e)}'})


@app.route('/api/export', methods=['POST'])
def export_dataset():
    """导出数据集"""
    try:
        import datetime

        data = request.json or {}
        # 确保比例值是有效的数字，处理前端可能发送的null或undefined
        train_ratio = float(data.get('train_ratio', 0.7)) if data.get('train_ratio') is not None else 0.7
        val_ratio = float(data.get('val_ratio', 0.2)) if data.get('val_ratio') is not None else 0.2
        test_ratio = float(data.get('test_ratio', 0.1)) if data.get('test_ratio') is not None else 0.1
        selected_classes = data.get('selected_classes', [])
        sample_selection = data.get('sample_selection', 'all')  # 获取样本选择参数，默认为'all'
        export_data_type = data.get('export_data_type', 'yolo')  # 获取导出数据类型参数，默认为'yolo'
        export_prefix = data.get('export_prefix', '')  # 获取导出文件前缀，默认为空字符串

        # 检查导出数据类型是否受支持
        if export_data_type not in ['yolo']:
            return jsonify({'error': '不支持的导出数据类型'}), 400

        # 前端已经检查了比例总和必须等于1，所以这里不需要再归一化
        # 直接使用前端传递的比例值

        # 获取全局类别列表
        classes = []
        if os.path.exists(get_classes_file()):
            with open(get_classes_file(), 'r', encoding='utf-8') as f:
                classes = json.load(f)

        # 创建临时目录用于生成数据集
        import tempfile
        import zipfile
        temp_dir = tempfile.mkdtemp()

        # 生成带时间戳的基础名称，格式：datasets_年月日时分秒
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        base_name = f"datasets_{timestamp}"

        # 不管有没有前缀，zip文件名和内部文件夹名称都使用datasets_年月日时分秒格式
        yolo_base = os.path.join(temp_dir, base_name)

        # 创建符合YOLOv11格式的目录结构
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(yolo_base, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(yolo_base, split, 'labels'), exist_ok=True)

        # 获取所有图片
        images = []
        for filename in os.listdir(get_upload_folder()):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')):
                images.append(filename)

        # 根据样本选择参数过滤图片
        annotations = {}
        if os.path.exists(get_annotations_file()):
            with open(get_annotations_file(), 'r', encoding='utf-8') as f:
                annotations = json.load(f)

        # 根据用户选择过滤图片
        if sample_selection == 'annotated':
            # 只选择有标注的图片
            images = [img for img in images if img in annotations and annotations[img]]
        elif sample_selection == 'unannotated':
            # 只选择没有标注的图片
            images = [img for img in images if img not in annotations or not annotations[img]]
        # 如果是'all'则不进行过滤，使用所有图片

        # 分割数据集
        np.random.shuffle(images)

        total_images = len(images)

        # 彻底重写数据集分割逻辑，确保严格按照比例分割
        # 0比例的数据集绝对为空，多余的数据直接扔掉
        train_images = []
        val_images = []
        test_images = []

        # 只处理比例大于0的数据集
        if train_ratio > 0:
            # 计算训练集数量
            train_count = int(total_images * train_ratio)
            # 只分配计算出的数量的图片
            train_images = images[:train_count]

        # 验证集只在train_ratio > 0时才处理，否则从0开始
        val_start = len(train_images) if train_ratio > 0 else 0
        if val_ratio > 0:
            # 计算验证集数量
            val_count = int(total_images * val_ratio)
            # 只分配计算出的数量的图片
            val_images = images[val_start:val_start + val_count]

        # 测试集只在train_ratio > 0或val_ratio > 0时才处理，否则从0开始
        test_start = (len(train_images) + len(val_images)) if (train_ratio > 0 or val_ratio > 0) else 0
        if test_ratio > 0:
            # 计算测试集数量
            test_count = int(total_images * test_ratio)
            # 只分配计算出的数量的图片
            test_images = images[test_start:test_start + test_count]

        # 确保0比例的数据集绝对为空
        if train_ratio == 0:
            train_images = []
        if val_ratio == 0:
            val_images = []
        if test_ratio == 0:
            test_images = []

        # 处理每个分割的数据集
        splits = [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]

        # 创建数据集配置文件 (YOLOv11格式)
        data_yaml = f"""path: .
train: train/images
val: val/images
test: test/images

nc: {len(selected_classes)}
names: {selected_classes}
"""

        with open(os.path.join(yolo_base, 'data.yaml'), 'w') as f:
            f.write(data_yaml)

        # 复制图片和生成标签文件
        for split_name, split_images in splits:
            for image_name in split_images:
                # 复制图片，添加前缀
                src_img_path = os.path.join(get_upload_folder(), image_name)
                if export_prefix:
                    dst_img_name = f"{export_prefix}_{image_name}"
                else:
                    dst_img_name = image_name
                dst_img_path = os.path.join(yolo_base, split_name, 'images', dst_img_name)

                # 使用PIL读取图片尺寸
                try:
                    img = Image.open(src_img_path)
                    width, height = img.size
                except Exception as e:
                    print(f"无法读取图片 {src_img_path}: {str(e)}")
                    continue

                # 复制图片文件
                from shutil import copyfile
                copyfile(src_img_path, dst_img_path)

                # 生成YOLO格式的标签文件，添加前缀
                base_name = os.path.splitext(image_name)[0]
                if export_prefix:
                    label_name = f"{export_prefix}_{base_name}.txt"
                else:
                    label_name = f"{base_name}.txt"
                label_path = os.path.join(yolo_base, split_name, 'labels', label_name)

                image_annotations = annotations.get(image_name, [])

                # 对于未标注的图片，创建空的标签文件；对于标注的图片，写入标注信息
                with open(label_path, 'w') as f:
                    # 只有当是标注图片并且选择了相关类别时才写入标注信息
                    if image_annotations and sample_selection != 'unannotated':
                        for ann in image_annotations:
                            # 只导出选中的类别
                            if ann['class'] in selected_classes:
                                # 使用全局类别列表中的索引而不是选中类别列表中的索引
                                class_id = None
                                # 从全局类别列表中查找类别ID
                                for i, cls in enumerate(classes):
                                    if cls['name'] == ann['class']:
                                        class_id = i
                                        break

                                # 如果在全局类别中找到了该类别，则写入标签文件
                                if class_id is not None:
                                    ann_type = ann.get('type', 'polygon')
                                    points = ann.get('points', [])

                                    # 跳过线段标注（YOLO 格式不支持）
                                    if ann_type == 'line':
                                        continue

                                    # 处理不同格式的points数据
                                    valid_points = []
                                    if isinstance(points, list) and len(points) > 0:
                                        if isinstance(points[0], dict):
                                            # 对象数组格式 [{x: ..., y: ...}, ...]
                                            for point in points:
                                                if ('x' in point and 'y' in point
                                                        and point['x'] is not None
                                                        and point['y'] is not None):
                                                    valid_points.append([point['x'], point['y']])
                                        else:
                                            # 坐标对数组格式 [[x, y], ...]
                                            for point in points:
                                                if (isinstance(point, (list, tuple))
                                                        and len(point) >= 2
                                                        and point[0] is not None
                                                        and point[1] is not None):
                                                    valid_points.append([point[0], point[1]])

                                    if len(valid_points) >= 3 and ann_type == 'polygon':
                                        # 多边形标注：导出为 YOLO 分割格式 class_id x1 y1 x2 y2 ...
                                        coords_str = ' '.join(
                                            f"{(x / width):.6f} {(y / height):.6f}"
                                            for x, y in valid_points
                                        )
                                        f.write(f"{class_id} {coords_str}\n")
                                    elif len(valid_points) > 0:
                                        # 矩形或其他标注：导出为 YOLO 边界框格式 class_id cx cy w h
                                        points_arr = np.array(valid_points)
                                        x_min = np.min(points_arr[:, 0])
                                        y_min = np.min(points_arr[:, 1])
                                        x_max = np.max(points_arr[:, 0])
                                        y_max = np.max(points_arr[:, 1])

                                        if (x_min is not None and y_min is not None
                                                and x_max is not None and y_max is not None):
                                            center_x = ((x_min + x_max) / 2) / width
                                            center_y = ((y_min + y_max) / 2) / height
                                            bbox_width = (x_max - x_min) / width
                                            bbox_height = (y_max - y_min) / height
                                            f.write(
                                                f"{class_id} {center_x:.6f} "
                                                f"{center_y:.6f} {bbox_width:.6f} "
                                                f"{bbox_height:.6f}\n"
                                            )
                                    elif 'x' in ann and 'y' in ann and 'width' in ann and 'height' in ann:
                                        # 处理矩形格式的标注数据
                                        x = ann['x']
                                        y = ann['y']
                                        w = ann['width']
                                        h = ann['height']

                                        if x is not None and y is not None and w is not None and h is not None:
                                            x_min = x
                                            y_min = y
                                            x_max = x + w
                                            y_max = y + h

                                            center_x = ((x_min + x_max) / 2) / width
                                            center_y = ((y_min + y_max) / 2) / height
                                            bbox_width = (x_max - x_min) / width
                                            bbox_height = (y_max - y_min) / height

                                            f.write(
                                                f"{class_id} {center_x:.6f} "
                                                f"{center_y:.6f} {bbox_width:.6f} "
                                                f"{bbox_height:.6f}\n"
                                            )
                                    else:
                                        # points数据格式无效，跳过该标注
                                        print(f"Invalid points data for annotation: {ann}")
                    # 对于未标注的图片，文件将保持为空（只需创建文件）

        # 创建zip文件，使用带时间戳的名称
        zip_filename = f"{base_name}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(yolo_base):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 使用yolo_base作为基准路径，这样zip文件中的目录结构就是直接的train/images/xxx.jpg
                    arc_name = os.path.relpath(file_path, yolo_base)
                    zipf.write(file_path, arc_name)

        # 返回zip文件
        return send_from_directory(temp_dir, zip_filename, as_attachment=True, download_name=zip_filename)

    except Exception as e:
        import traceback
        print(f"Export error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 异步视频标注相关API
@app.route('/api/auto-label/video/start', methods=['POST'])
def start_video_annotation():
    """启动视频标注任务"""
    try:
        import os

        # 获取请求数据
        data = request.json
        video_path = data.get('video_path')
        frame_interval = int(data.get('frame_interval', 10))
        output_dir = data.get('output_dir', 'output')
        api_config = data.get('api_config', {})

        if not video_path:
            return jsonify({'success': False, 'error': 'No video path provided'}), 400

        # 创建唯一任务ID
        task_id = str(uuid.uuid4())

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建视频标注任务
        task = VideoAnnotationTask(task_id, video_path, frame_interval, output_dir, api_config)

        # 保存任务到任务列表
        tasks[task_id] = task

        # 启动任务
        task.start()

        # 从请求上下文获取当前连接ID
        # 在API请求中，request对象来自flask，不直接包含socketio sid
        # 因此在API请求中我们无法直接获取socketio sid
        # 这里使用特殊的方式获取，通过flask的request对象的环境变量
        sid = None
        if hasattr(request, 'environ') and 'flask_socketio.sid' in request.environ:
            sid = request.environ['flask_socketio.sid']

        if sid:
            # 存储连接ID和任务ID的映射关系
            connection_task_map[sid] = task_id
            print(f"关联连接ID {sid} 到任务ID {task_id}")

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Video annotation task started successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/auto-label/video/stop', methods=['POST'])
def stop_video_annotation():
    """停止视频标注任务"""
    try:
        # 获取请求数据
        data = request.json
        task_id = data.get('task_id')

        if not task_id:
            return jsonify({'success': False, 'error': 'No task ID provided'}), 400

        # 查找任务
        if task_id not in tasks:
            return jsonify({'success': False, 'error': 'Task not found'}), 404

        # 停止任务
        task = tasks[task_id]
        task.stop()

        # 不要立即从任务列表中移除任务，让任务线程自己完成清理工作
        # 任务线程会在完成后发送最终的进度更新

        return jsonify({
            'success': True,
            'message': 'Video annotation task stopped successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/auto-label/video/status/<task_id>', methods=['GET'])
def get_video_annotation_status(task_id):
    """获取视频标注任务状态"""
    try:
        # 查找任务
        if task_id not in tasks:
            return jsonify({'success': False, 'error': 'Task not found'}), 404

        # 获取任务状态
        task = tasks[task_id]
        status = task.get_status()

        return jsonify({
            'success': True,
            'status': status
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# SocketIO事件处理
@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect(sid):
    """处理客户端断开连接"""
    print(f'Client disconnected: {sid}')

    # 检查该连接是否有关联的任务
    if sid in connection_task_map:
        task_id = connection_task_map[sid]
        print(f'检测到断开连接的客户端有关联任务: {task_id}')

        # 检查任务是否存在且正在运行
        if task_id in tasks:
            task = tasks[task_id]
            if task.status == TASK_STATUS['RUNNING']:
                # 停止任务
                print(f'自动停止任务: {task_id}')
                task.stop()

        # 从映射字典中移除该连接
        del connection_task_map[sid]
        print(f'移除连接和任务的关联: {sid} -> {task_id}')

# 工程管理 API
@app.route('/api/projects', methods=['GET'])
def list_projects():
    """列出所有工程。"""
    os.makedirs(PROJECTS_FOLDER, exist_ok=True)
    projects = []
    for name in os.listdir(PROJECTS_FOLDER):
        project_path = os.path.join(PROJECTS_FOLDER, name)
        if os.path.isdir(project_path):
            image_count = len([
                f for f in os.listdir(project_path)
                if os.path.isfile(os.path.join(project_path, f))
                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))
            ])
            ann_file = os.path.join(project_path, 'annotations', 'annotations.json')
            last_modified = os.path.getmtime(ann_file) if os.path.exists(ann_file) else 0
            info = get_project_info(name)
            projects.append({
                'name': name,
                'image_count': image_count,
                'last_modified': last_modified,
                'task_type': info.get('task_type', 'detect'),
            })
    projects.sort(key=lambda x: x['last_modified'], reverse=True)
    return jsonify({'projects': projects})


@app.route('/api/projects', methods=['POST'])
def create_project():
    """创建新工程。"""
    data = request.json or {}
    name = data.get('name', '').strip()
    name = sanitize_project_name(name)
    task_type = data.get('task_type', 'detect')

    if not name:
        return jsonify({'error': '工程名称不能为空'}), 400

    if task_type not in YOLO_TASK_TYPES:
        return jsonify({'error': f'不支持的任务类型: {task_type}'}), 400

    project_path = get_project_path(name)
    if os.path.exists(project_path):
        return jsonify({'error': '工程名称已存在'}), 400

    init_project(name, task_type)
    return jsonify({'success': True, 'name': name})


@app.route('/api/project-info')
def project_info():
    """获取指定工程的信息。"""
    project_name = request.args.get('project') or get_current_project()
    if not project_name:
        return jsonify({'name': '', 'task_type': 'detect'})
    project_path = get_project_path(project_name)
    if not os.path.exists(project_path):
        return jsonify({'error': '工程不存在'}), 404
    info = get_project_info(project_name)
    return jsonify({'name': project_name, **info})


@app.route('/api/projects/<name>', methods=['PUT'])
def rename_project(name):
    """重命名工程。"""
    data = request.json or {}
    new_name = data.get('new_name', '').strip()
    new_name = sanitize_project_name(new_name)

    if not new_name:
        return jsonify({'error': '新名称不能为空'}), 400

    old_path = get_project_path(name)
    new_path = get_project_path(new_name)

    if not os.path.exists(old_path):
        return jsonify({'error': '工程不存在'}), 404

    if os.path.exists(new_path):
        return jsonify({'error': '新名称已存在'}), 400

    os.rename(old_path, new_path)

    # 如果当前工程是被重命名的，更新 session
    if get_current_project() == name:
        set_current_project(new_name)

    return jsonify({'success': True, 'name': new_name})


# 全局配置存储路径
GLOBAL_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'global_config.json')

def load_global_config():
    """加载全局配置。"""
    if os.path.exists(GLOBAL_CONFIG_FILE):
        try:
            with open(GLOBAL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_global_config(config):
    """保存全局配置。"""
    os.makedirs(os.path.dirname(GLOBAL_CONFIG_FILE), exist_ok=True)
    with open(GLOBAL_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


@app.route('/api/global-config', methods=['GET'])
def get_global_config():
    """获取全局配置。"""
    config = load_global_config()
    return jsonify({'success': True, 'config': config})


@app.route('/api/global-config', methods=['PUT'])
def update_global_config():
    """更新全局配置。"""
    data = request.json or {}
    config = load_global_config()
    if 'deploy_server_url' in data:
        config['deploy_server_url'] = data['deploy_server_url'].strip()
    if 'xclabel_server_url' in data:
        config['xclabel_server_url'] = data['xclabel_server_url'].strip()
    save_global_config(config)
    return jsonify({'success': True})


@app.route('/api/projects/<name>', methods=['DELETE'])
def delete_project(name):
    """删除工程。"""
    project_path = get_project_path(name)
    if not os.path.exists(project_path):
        return jsonify({'error': '工程不存在'}), 404

    shutil.rmtree(project_path, ignore_errors=True)

    # 如果删除的是当前工程，清空当前工程（不再自动重建 default）
    if get_current_project() == name:
        session.pop('current_project', None)

    return jsonify({'success': True})


@app.route('/api/projects/switch', methods=['POST'])
def switch_project():
    """切换当前工程。"""
    data = request.json or {}
    name = data.get('name', '').strip()

    project_path = get_project_path(name)
    if not os.path.exists(project_path):
        return jsonify({'error': '工程不存在'}), 404

    set_current_project(name)
    return jsonify({'success': True, 'name': name})


@app.route('/api/projects/current', methods=['GET'])
def get_current_project_info():
    """获取当前工程信息。"""
    name = get_current_project()
    project_path = get_project_path(name)
    if not os.path.exists(project_path):
        return jsonify({'name': '', 'image_count': 0})

    image_count = len([
        f for f in os.listdir(project_path)
        if os.path.isfile(os.path.join(project_path, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))
    ])
    return jsonify({
        'name': name,
        'image_count': image_count
    })


@app.route('/projects')
def projects_page():
    """工程管理页面。"""
    return render_template('projects.html', version=APP_VERSION)


@app.route('/label')
def label_page():
    """标注页面。"""
    return render_template('index.html', version=APP_VERSION)


# ========================== YOLO 模型训练 ==========================

YOLO_TASK_TYPES = {
    'detect': {'name': '目标检测', 'suffix': ''},
    'segment': {'name': '实例分割', 'suffix': '-seg'},
    'pose': {'name': '姿态估计', 'suffix': '-pose'},
    'obb': {'name': '旋转框检测', 'suffix': '-obb'},
    'classify': {'name': '图像分类', 'suffix': '-cls'},
}

YOLO_VERSION_CONFIG = {
    'yolo8': {
        'name': 'YOLOv8',
        'models': {
            'detect': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
            'segment': ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt'],
            'pose': ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt'],
            'obb': ['yolov8n-obb.pt', 'yolov8s-obb.pt', 'yolov8m-obb.pt', 'yolov8l-obb.pt', 'yolov8x-obb.pt'],
            'classify': ['yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt', 'yolov8l-cls.pt', 'yolov8x-cls.pt'],
        },
        'ultralytics_pkg': 'ultralytics==8.0.196',
        'install_path': os.path.join(app.root_path, 'plugins', 'yolo8'),
    },
    'yolo11': {
        'name': 'YOLO11',
        'models': {
            'detect': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
            'segment': ['yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt', 'yolo11l-seg.pt', 'yolo11x-seg.pt'],
            'pose': ['yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt', 'yolo11l-pose.pt', 'yolo11x-pose.pt'],
            'obb': ['yolo11n-obb.pt', 'yolo11s-obb.pt', 'yolo11m-obb.pt', 'yolo11l-obb.pt', 'yolo11x-obb.pt'],
            'classify': ['yolo11n-cls.pt', 'yolo11s-cls.pt', 'yolo11m-cls.pt', 'yolo11l-cls.pt', 'yolo11x-cls.pt'],
        },
        'ultralytics_pkg': 'ultralytics==8.4.41',
        'install_path': os.path.join(app.root_path, 'plugins', 'yolo11'),
    },
    'yolo26': {
        'name': 'YOLO26',
        'models': {
            'detect': ['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt'],
            'segment': ['yolo26n-seg.pt', 'yolo26s-seg.pt', 'yolo26m-seg.pt', 'yolo26l-seg.pt', 'yolo26x-seg.pt'],
            'pose': ['yolo26n-pose.pt', 'yolo26s-pose.pt', 'yolo26m-pose.pt', 'yolo26l-pose.pt', 'yolo26x-pose.pt'],
            'obb': ['yolo26n-obb.pt', 'yolo26s-obb.pt', 'yolo26m-obb.pt', 'yolo26l-obb.pt', 'yolo26x-obb.pt'],
            'classify': ['yolo26n-cls.pt', 'yolo26s-cls.pt', 'yolo26m-cls.pt', 'yolo26l-cls.pt', 'yolo26x-cls.pt'],
        },
        'ultralytics_pkg': 'ultralytics>=8.4.0',
        'install_path': os.path.join(app.root_path, 'plugins', 'yolo26'),
    },
}


def get_yolo_install_path(yolo_version):
    """获取指定 YOLO 版本的安装目录。"""
    cfg = YOLO_VERSION_CONFIG.get(yolo_version)
    if cfg:
        return cfg['install_path']
    return os.path.join(app.root_path, 'plugins', yolo_version)


def get_ultralytics_python_path(yolo_version):
    """获取指定 YOLO 版本 venv 的 Python 路径。"""
    install_path = get_yolo_install_path(yolo_version)
    if os.name == 'nt':
        return os.path.join(install_path, 'venv', 'Scripts', 'python.exe')
    return os.path.join(install_path, 'venv', 'bin', 'python')


def get_ultralytics_pip_path(yolo_version):
    """获取指定 YOLO 版本 venv 的 pip 路径。"""
    install_path = get_yolo_install_path(yolo_version)
    if os.name == 'nt':
        return os.path.join(install_path, 'venv', 'Scripts', 'pip.exe')
    return os.path.join(install_path, 'venv', 'bin', 'pip')


def check_ultralytics_install(yolo_version):
    """检测指定 YOLO 版本的训练环境是否已安装，并返回 GPU 信息。"""
    python_path = get_ultralytics_python_path(yolo_version)
    install_path = get_yolo_install_path(yolo_version)
    if not os.path.exists(python_path):
        return {'is_installed': False, 'gpus': []}
    try:
        result = subprocess.run(
            [python_path, '-c', 'import ultralytics; print(ultralytics.__version__)'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=install_path
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            # 检测 GPU 信息
            gpus = []
            try:
                gpu_script = (
                    "import torch, json; "
                    "avail = torch.cuda.is_available(); "
                    "count = torch.cuda.device_count() if avail else 0; "
                    "devices = [{'id': i, 'name': torch.cuda.get_device_name(i)} for i in range(count)]; "
                    "print(json.dumps(devices))"
                )
                gpu_result = subprocess.run(
                    [python_path, '-c', gpu_script],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=install_path
                )
                if gpu_result.returncode == 0:
                    gpus = json.loads(gpu_result.stdout.strip())
            except Exception:
                pass
            return {'is_installed': True, 'version': version, 'gpus': gpus}
    except Exception:
        pass
    return {'is_installed': False, 'gpus': []}


# GPU 任务锁（在多个 GPU 任务间互斥）
gpu_task_lock = threading.Lock()
gpu_task_owner = None       # None | "train" | "sam" | "yolo_label"
gpu_task_timestamp = 0.0    # 最后活动时间
current_gpu_task = None     # 'train' | 'ai_label' | None (向后兼容)

def acquire_gpu(owner: str, timeout: int = 300) -> bool:
    """获取 GPU 锁。timeout 秒超时返回 False"""
    global gpu_task_owner, gpu_task_timestamp
    if not gpu_task_lock.acquire(timeout=timeout):
        return False
    gpu_task_owner = owner
    gpu_task_timestamp = time.time()
    return True

def release_gpu(owner: str):
    """释放 GPU 锁，需校验 owner"""
    global gpu_task_owner
    if gpu_task_owner != owner:
        logging.warning(f"release_gpu: owner mismatch (expected={gpu_task_owner}, got={owner})")
    gpu_task_owner = None
    gpu_task_lock.release()

def get_gpu_status() -> dict:
    """查询 GPU 使用状态"""
    return {
        "busy": gpu_task_owner is not None,
        "owner": gpu_task_owner,
        "idle_seconds": time.time() - gpu_task_timestamp if gpu_task_owner else 0,
    }

class YOLOTrainingTask:
    """YOLO 训练任务封装。"""
    def __init__(self, project_name, params):
        self.project_name = project_name
        self.params = params
        self.process = None
        self.status = TASK_STATUS['IDLE']
        self.error = None
        self.progress = {}
        self.start_time = None
        self.end_time = None
        self.result = {}

    def start(self):
        """启动训练任务。"""
        global current_gpu_task
        with gpu_task_lock:
            if current_gpu_task is not None:
                return False, f'当前有 {current_gpu_task} 任务正在运行，请先等待完成或取消'
            current_gpu_task = 'train'

        self.status = TASK_STATUS['RUNNING']
        self.start_time = time.time()
        self.progress = {'total_epochs': int(self.params.get('epochs', 100))}

        # 导出训练数据
        export_dir = self._export_dataset()
        if export_dir is None:
            self.status = TASK_STATUS['ERROR']
            self.error = '训练数据导出失败'
            with gpu_task_lock:
                current_gpu_task = None
            return False, self.error

        # 确定 YOLO 版本
        yolo_version = self.params.get('yolo_version', 'yolo11')

        # 确定基础模型
        base_model = self.params.get('model', 'yolo11n.pt')
        if self.params.get('use_project_model'):
            project_model = os.path.join(get_project_path(self.project_name), 'models', 'best.pt')
            if os.path.exists(project_model):
                base_model = project_model
        else:
            # 相对路径的官方预训练模型，从版本对应目录查找
            if not os.path.isabs(base_model):
                version_model_dir = os.path.join(app.root_path, 'plugins', yolo_version, 'models')
                version_model_path = os.path.join(version_model_dir, base_model)
                if os.path.exists(version_model_path):
                    base_model = version_model_path
                else:
                    # 本地不存在则下载到版本目录，避免文件散落到根目录
                    os.makedirs(version_model_dir, exist_ok=True)
                    logging.info(f'本地模型 {base_model} 不存在，尝试下载到 {version_model_dir}')
                    try:
                        python_path = get_ultralytics_python_path(yolo_version)
                        # 利用 Ultralytics 自身的下载逻辑，下载后移动到版本目录
                        download_script = (
                            'import os, shutil; '
                            'from ultralytics import YOLO; '
                            f'model = YOLO("{base_model}"); '
                            f'dest = r"{version_model_path}"; '
                            'os.makedirs(os.path.dirname(dest), exist_ok=True); '
                            f'src = "{base_model}"; '
                            'if os.path.exists(src) and os.path.normpath(src) != os.path.normpath(dest): '
                            '    shutil.move(src, dest); '
                            '    print(f"Moved to {dest}"); '
                            'else: '
                            '    print(f"Already at {dest}")'
                        )
                        result = subprocess.run(
                            [python_path, '-c', download_script],
                            capture_output=True, text=True, timeout=300,
                        )
                        if os.path.exists(version_model_path):
                            base_model = version_model_path
                            logging.info(f'模型已下载到: {version_model_path}')
                        else:
                            logging.warning(f'下载失败，将使用在线模式: {result.stderr[:500]}')
                    except Exception as e:
                        logging.warning(f'自动下载失败，将使用在线模式: {e}')

        # 生成版本号（时间戳）
        from datetime import datetime
        self.version = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = self.version

        # 将数据集统计信息写入临时文件供训练脚本读取
        dataset_stats_path = os.path.join(export_dir, 'dataset_stats.json')
        with open(dataset_stats_path, 'w', encoding='utf-8') as f:
            json.dump(getattr(self, 'dataset_stats', {}), f, ensure_ascii=False)

        # 构建训练命令
        task = self.params.get('task', 'detect')
        cmd = [
            get_ultralytics_python_path(yolo_version),
            os.path.join(app.root_path, 'plugins', 'train_yolo.py'),
            '--data', os.path.join(export_dir, 'data.yaml'),
            '--model', base_model,
            '--epochs', str(self.params.get('epochs', 100)),
            '--batch', str(self.params.get('batch', 8)),
            '--imgsz', str(self.params.get('imgsz', 640)),
            '--device', str(self.params.get('device', '0')),
            '--project', get_project_path(self.project_name),
            '--export-dir', export_dir,
            '--yolo-version', yolo_version,
            '--version', version,
            '--dataset-stats', dataset_stats_path,
            '--task', task,
        ]

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', cwd=app.root_path
            )
            # 启动日志读取线程
            threading.Thread(target=self._read_logs, daemon=True).start()
            return True, '训练任务已启动'
        except Exception as e:
            self.status = TASK_STATUS['ERROR']
            self.error = str(e)
            with gpu_task_lock:
                current_gpu_task = None
            return False, str(e)

    def _export_dataset(self):
        """导出当前工程的标注数据为 YOLO 格式（支持多种任务类型）。"""
        try:
            project_path = get_project_path(self.project_name)
            export_dir = tempfile.mkdtemp(prefix='yolo_train_')
            task = self.params.get('task', 'detect')

            # 使用 project_name 构建标注文件路径，而不是依赖 session 中的当前工程
            annotations_file = os.path.join(project_path, 'annotations', 'annotations.json')
            classes_file = os.path.join(project_path, 'annotations', 'classes.json')

            # 读取标注和类别
            annotations = {}
            if os.path.exists(annotations_file):
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)

            classes = []
            if os.path.exists(classes_file):
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = json.load(f)

            class_names = [c['name'] for c in classes]
            class_map = {name: i for i, name in enumerate(class_names)}

            # 收集已标注图片（按任务类型过滤）
            annotated_images = []
            task_type_map = {
                'detect': {'rectangle'},
                'segment': {'polygon', 'rectangle'},
                'obb': {'obb'},
                'pose': {'pose'},
                'classify': {'classify'},
            }
            allowed_types = task_type_map.get(task, {'rectangle'})
            for img_name, anns in annotations.items():
                if not anns:
                    continue
                img_path = os.path.join(project_path, img_name)
                if not os.path.exists(img_path):
                    continue
                # 检查是否有至少一个符合任务类型的标注
                has_valid = any(ann.get('type') in allowed_types for ann in anns)
                if has_valid:
                    annotated_images.append(img_name)

            if len(annotated_images) < 10:
                self.error = f'已标注图片不足，需要至少10张，当前只有{len(annotated_images)}张'
                return None

            # 划分训练/验证集
            train_ratio = self.params.get('train_val_ratio', 0.8)
            np.random.shuffle(annotated_images)
            split_idx = int(len(annotated_images) * train_ratio)
            train_images = annotated_images[:split_idx]
            val_images = annotated_images[split_idx:]

            if task == 'classify':
                # 分类任务：按类别文件夹组织
                for split_name, img_list in [('train', train_images), ('val', val_images)]:
                    for img_name in img_list:
                        anns = annotations.get(img_name, [])
                        # 取第一个分类标注的类别
                        cls_name = None
                        for ann in anns:
                            if ann.get('type') == 'classify':
                                cls_name = ann.get('class', '')
                                break
                        if not cls_name or cls_name not in class_map:
                            continue
                        src_img = os.path.join(project_path, img_name)
                        cls_dir = os.path.join(export_dir, split_name, cls_name)
                        os.makedirs(cls_dir, exist_ok=True)
                        dst_img = os.path.join(cls_dir, img_name)
                        shutil.copy2(src_img, dst_img)
            else:
                # 检测/分割/旋转框/姿态：使用 images + labels 目录结构
                for split in ['train', 'val']:
                    os.makedirs(os.path.join(export_dir, split, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(export_dir, split, 'labels'), exist_ok=True)

                for split_name, img_list in [('train', train_images), ('val', val_images)]:
                    for img_name in img_list:
                        src_img = os.path.join(project_path, img_name)
                        dst_img = os.path.join(export_dir, split_name, 'images', img_name)
                        shutil.copy2(src_img, dst_img)

                        anns = annotations.get(img_name, [])
                        if not anns:
                            continue
                        try:
                            img = Image.open(src_img)
                            img_w, img_h = img.size
                        except Exception:
                            continue

                        label_lines = []
                        for ann in anns:
                            ann_type = ann.get('type', '')
                            cls_name = ann.get('class', '')
                            if cls_name not in class_map:
                                continue
                            pts = ann.get('points', [])
                            if not pts:
                                continue
                            cls_id = class_map[cls_name]

                            if task == 'detect' and ann_type == 'rectangle' and len(pts) >= 4:
                                xs = [p[0] for p in pts]
                                ys = [p[1] for p in pts]
                                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                x_center = (x1 + x2) / 2 / img_w
                                y_center = (y1 + y2) / 2 / img_h
                                w = (x2 - x1) / img_w
                                h = (y2 - y1) / img_h
                                label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                            elif task == 'segment':
                                if ann_type == 'polygon' and len(pts) >= 3:
                                    norm_pts = ' '.join(f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in pts)
                                    label_lines.append(f"{cls_id} {norm_pts}")
                                elif ann_type == 'rectangle' and len(pts) >= 4:
                                    xs = [p[0] for p in pts]
                                    ys = [p[1] for p in pts]
                                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                    x_center = (x1 + x2) / 2 / img_w
                                    y_center = (y1 + y2) / 2 / img_h
                                    w = (x2 - x1) / img_w
                                    h = (y2 - y1) / img_h
                                    label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                            elif task == 'obb' and ann_type == 'obb' and len(pts) >= 4:
                                norm_pts = ' '.join(f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in pts[:4])
                                label_lines.append(f"{cls_id} {norm_pts}")

                            elif task == 'pose' and ann_type == 'pose':
                                # pose: class_id cx cy w h x1 y1 v1 x2 y2 v2 ...
                                bbox = ann.get('bbox', [])
                                keypoints = ann.get('keypoints', [])
                                if bbox and len(bbox) >= 4 and keypoints:
                                    xs = [p[0] for p in bbox]
                                    ys = [p[1] for p in bbox]
                                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                    cx = (x1 + x2) / 2 / img_w
                                    cy = (y1 + y2) / 2 / img_h
                                    bw = (x2 - x1) / img_w
                                    bh = (y2 - y1) / img_h
                                    kpts = []
                                    for kp in keypoints:
                                        if len(kp) >= 2:
                                            kx = kp[0] / img_w
                                            ky = kp[1] / img_h
                                            kv = kp[2] if len(kp) > 2 else 2
                                            kpts.append(f"{kx:.6f} {ky:.6f} {int(kv)}")
                                    if kpts:
                                        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {' '.join(kpts)}")

                        label_name = os.path.splitext(img_name)[0] + '.txt'
                        label_path = os.path.join(export_dir, split_name, 'labels', label_name)
                        with open(label_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(label_lines))

            # 生成 data.yaml
            data_yaml = {
                'path': export_dir,
                'train': 'train' if task == 'classify' else 'train/images',
                'val': 'val' if task == 'classify' else 'val/images',
                'names': {i: name for i, name in enumerate(class_names)},
                'nc': len(class_names),
            }
            # 姿态估计任务添加 kpt_shape（从类别配置推断或使用默认值）
            if task == 'pose':
                # 尝试从第一个 pose 标注推断关键点数量
                kpt_count = 0
                for img_name, anns in annotations.items():
                    for ann in anns:
                        if ann.get('type') == 'pose':
                            kpts = ann.get('keypoints', [])
                            if kpts:
                                kpt_count = len(kpts)
                                break
                    if kpt_count > 0:
                        break
                if kpt_count == 0:
                    kpt_count = 17  # 默认 COCO 17 关键点
                data_yaml['kpt_shape'] = [kpt_count, 3]

            with open(os.path.join(export_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

            # 保存数据集统计信息供训练脚本使用
            self.dataset_stats = {
                'total_images': len([f for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))]),
                'annotated_images': len(annotated_images),
                'train_count': len(train_images),
                'val_count': len(val_images),
                'test_count': 0,
                'class_count': len(classes),
                'task': task,
            }

            return export_dir
        except Exception as e:
            import traceback
            err_msg = f'导出训练数据失败: {e}'
            print(err_msg)
            print(traceback.format_exc())
            self.error = err_msg
            return None

    def _read_logs(self):
        """读取训练日志并解析进度。"""
        if self.process is None:
            return
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                self._parse_log_line(line)
                socketio.emit('train_progress', {
                    'status': self.status,
                    'progress': self.progress,
                    'message': line[:200],
                })
        except Exception as e:
            print(f'读取训练日志失败: {e}')
        finally:
            self._on_finished()

    def _parse_log_line(self, line):
        """解析 ultralytics 输出日志。"""
        # 尝试匹配 epoch 进度: 1/100 或 epoch 1/100
        import re
        # 匹配 epoch 进度，要求后面紧跟 GPU 显存（如 0G、4.21G），以排除
        # 预训练权重加载（606/708 items）和验证/批次进度（1/1、0/7）的误匹配
        epoch_match = re.search(r'(?:epoch\s+)?(\d+)/(\d+)\s+(\d+\.\d+G|\d+G)', line, re.IGNORECASE)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            self.progress['epoch'] = current
            self.progress['total_epochs'] = total
            self.progress['percentage'] = int(current / total * 100) if total > 0 else 0

        # 尝试匹配 box_loss
        box_loss = re.search(r'box_loss[:\s]+([\d.]+)', line)
        if box_loss:
            self.progress['box_loss'] = float(box_loss.group(1))

        # 尝试匹配 cls_loss
        cls_loss = re.search(r'cls_loss[:\s]+([\d.]+)', line)
        if cls_loss:
            self.progress['cls_loss'] = float(cls_loss.group(1))

        # 尝试匹配 mAP50
        map50 = re.search(r'mAP50[:\s]+([\d.]+)', line)
        if map50:
            self.progress['mAP50'] = float(map50.group(1))

        # 尝试匹配 mAP50-95
        map50_95 = re.search(r'mAP50-95[:\s]+([\d.]+)', line)
        if map50_95:
            self.progress['mAP50-95'] = float(map50_95.group(1))

    def _on_finished(self):
        """训练结束回调。"""
        global current_gpu_task
        self.end_time = time.time()
        if self.process:
            self.process.wait()
            rc = self.process.returncode
            if rc == 0 and self.status != TASK_STATUS['STOPPED']:
                self.status = TASK_STATUS['COMPLETED']
                # 尝试读取评估结果
                self._load_val_results()
                # 自动导出 ONNX
                self._auto_export_onnx()
            elif self.status != TASK_STATUS['STOPPED']:
                self.status = TASK_STATUS['ERROR']
                self.error = f'训练进程返回非零退出码: {rc}'

        # 将最终状态写入版本 model_info.json
        self._save_version_status()

        with gpu_task_lock:
            current_gpu_task = None

        socketio.emit('train_progress', {
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'elapsed': round(self.end_time - self.start_time, 1) if self.start_time else 0,
            'deploy_hint': self._get_deploy_hint() if self.status == TASK_STATUS['COMPLETED'] else None,
        })

    def _save_version_status(self):
        """将训练最终状态写入版本 model_info.json，并生成 deploy_metadata.json。"""
        try:
            version = getattr(self, 'version', None)
            if not version or not self.project_name:
                return
            models_dir = os.path.join(get_project_path(self.project_name), 'models')
            version_dir = os.path.join(models_dir, version)
            info_path = os.path.join(version_dir, 'model_info.json')
            info = {}
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                except Exception:
                    pass
            info['status'] = self.status
            if self.error:
                info['error'] = self.error
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            # 生成 deploy_metadata.json（供 deploy 容器使用）
            deploy_meta = {
                'project': self.project_name,
                'version': version,
                'yolo_version': self.params.get('yolo_version', 'yolo11'),
                'task': self.params.get('task', 'detect'),
                'input_size': self.params.get('imgsz', 640),
                'class_count': len(info.get('classes', [])) if info else 0,
                'classes': info.get('classes', []) if info else [],
                'onnx_file': 'best.onnx' if os.path.exists(os.path.join(version_dir, 'best.onnx')) else None,
                'pt_file': 'best.pt' if os.path.exists(os.path.join(version_dir, 'best.pt')) else None,
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            deploy_meta_path = os.path.join(version_dir, 'deploy_metadata.json')
            with open(deploy_meta_path, 'w', encoding='utf-8') as f:
                json.dump(deploy_meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f'保存版本状态失败: {e}')

    def _load_val_results(self):
        """加载验证结果。"""
        try:
            result_path = os.path.join(get_project_path(self.project_name), 'models', 'val_results.json')
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    self.result = json.load(f)
        except Exception:
            pass

    def _auto_export_onnx(self):
        """训练完成后自动导出 ONNX 格式。"""
        try:
            version = getattr(self, 'version', None)
            if not version or not self.project_name:
                return
            models_dir = os.path.join(get_project_path(self.project_name), 'models')
            version_dir = os.path.join(models_dir, version)
            model_path = os.path.join(version_dir, 'best.pt')
            output_path = os.path.join(version_dir, 'best.onnx')

            if not os.path.exists(model_path) or os.path.exists(output_path):
                return

            yolo_version = self.params.get('yolo_version', 'yolo11')
            python_path = get_ultralytics_python_path(yolo_version)
            cmd = [
                python_path,
                os.path.join(app.root_path, 'plugins', 'export_yolo.py'),
                '--model', model_path,
                '--output', output_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        except Exception as e:
            print(f'自动导出 ONNX 失败: {e}')

    def _get_deploy_hint(self):
        """生成部署提示信息。"""
        try:
            version = getattr(self, 'version', None)
            if not version or not self.project_name:
                return None
            models_dir = os.path.join(get_project_path(self.project_name), 'models')
            version_dir = os.path.join(models_dir, version)
            onnx_path = os.path.join(version_dir, 'best.onnx')
            has_onnx = os.path.exists(onnx_path)
            return {
                'version': version,
                'project': self.project_name,
                'onnx_ready': has_onnx,
                'deploy_api': f'POST /load/model with project_id="{self.project_name}", model_version="{version}"',
            }
        except Exception:
            return None

    def stop(self):
        """停止训练任务。"""
        global current_gpu_task
        self.status = TASK_STATUS['STOPPED']
        if self.process:
            try:
                self.process.terminate()
                time.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception:
                pass
        with gpu_task_lock:
            current_gpu_task = None


# 全局训练任务实例
yolo_training_task = None


@app.route('/api/check-ultralytics-install')
def check_ultralytics_install_api():
    """检查指定 YOLO 版本的训练环境安装状态。"""
    yolo_version = request.args.get('version', 'yolo11')
    if yolo_version not in YOLO_VERSION_CONFIG:
        return jsonify({'error': f'不支持的 YOLO 版本: {yolo_version}'}), 400
    result = check_ultralytics_install(yolo_version)
    result['version_config'] = YOLO_VERSION_CONFIG[yolo_version]
    return jsonify(result)


@app.route('/api/install-ultralytics')
def install_ultralytics():
    """安装指定 YOLO 版本的训练环境。"""
    import venv

    from flask import Response

    yolo_version = request.args.get('version', 'yolo11')
    if yolo_version not in YOLO_VERSION_CONFIG:
        def error_gen():
            yield f"data: {json.dumps({'status': 'error', 'message': f'不支持的 YOLO 版本: {yolo_version}'})}\n\n"
        return Response(error_gen(), mimetype='text/event-stream')

    cfg = YOLO_VERSION_CONFIG[yolo_version]
    install_path = cfg['install_path']
    ultralytics_pkg = cfg['ultralytics_pkg']

    def generate():
        msg = '开始安装 ' + cfg['name'] + ' 训练环境...'
        yield 'data: ' + json.dumps({'status': 'started', 'message': msg, 'progress': 0}) + '\n\n'
        time.sleep(0.5)
        try:
            yield 'data: ' + json.dumps({'message': '创建安装目录...', 'progress': 10}) + '\n\n'
            os.makedirs(install_path, exist_ok=True)
            time.sleep(0.5)

            yield 'data: ' + json.dumps({'message': '创建 Python 虚拟环境...', 'progress': 20}) + '\n\n'
            venv_path = os.path.join(install_path, 'venv')
            venv.create(venv_path, with_pip=True)
            time.sleep(0.5)

            python_path = get_ultralytics_python_path(yolo_version)

            yield 'data: ' + json.dumps({'message': '升级 pip...', 'progress': 30}) + '\n\n'
            subprocess.run([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'], capture_output=True, cwd=install_path)

            yield 'data: ' + json.dumps({'message': '配置 pip 镜像源...', 'progress': 35}) + '\n\n'
            subprocess.run([python_path, '-m', 'pip', 'config', 'set', 'global.index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'], capture_output=True, cwd=install_path)

            yield 'data: ' + json.dumps({'message': '安装 ' + ultralytics_pkg + '...', 'progress': 50}) + '\n\n'
            result = subprocess.run([python_path, '-m', 'pip', 'install', '--no-cache-dir', ultralytics_pkg], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=install_path)
            if result.returncode != 0:
                err_msg = '安装 ' + ultralytics_pkg + ' 失败: ' + (result.stderr[:800] if result.stderr else result.stdout[:800])
                yield 'data: ' + json.dumps({'status': 'error', 'message': err_msg, 'progress': 50}) + '\n\n'
                return

            yield 'data: ' + json.dumps({'message': '检查 CUDA 支持...', 'progress': 80}) + '\n\n'
            has_cuda = False
            try:
                r = subprocess.run([python_path, '-c', 'import torch; print(torch.cuda.is_available())'], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=install_path)
                has_cuda = r.stdout.strip().lower() == 'true'
            except Exception:
                pass

            info = {'is_installed': True, 'has_cuda': has_cuda, 'hardware': 'CUDA' if has_cuda else 'CPU'}
            with open(os.path.join(install_path, 'install_info.json'), 'w') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            yield 'data: ' + json.dumps({'message': '安装完成！', 'progress': 100, 'status': 'completed', 'has_cuda': has_cuda}) + '\n\n'
        except Exception as e:
            yield 'data: ' + json.dumps({'status': 'error', 'message': '安装失败: ' + str(e), 'progress': 0}) + '\n\n'

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/train/start', methods=['POST'])
def train_start():
    """启动训练任务。"""
    global yolo_training_task

    data = request.json or {}
    project_name = data.get('project', get_current_project())

    # 检查 GPU 资源冲突
    with gpu_task_lock:
        if current_gpu_task == 'ai_label':
            return jsonify({'error': 'AI 标注任务正在运行，请先等待完成或取消后再启动训练'}), 409
        if current_gpu_task == 'train' and yolo_training_task and yolo_training_task.status == TASK_STATUS['RUNNING']:
            return jsonify({'error': '训练任务已经在运行中'}), 409

    # 参数校验
    yolo_version = data.get('yolo_version', 'yolo11')
    if yolo_version not in YOLO_VERSION_CONFIG:
        return jsonify({'error': f'不支持的 YOLO 版本: {yolo_version}'}), 400

    # 检查训练环境
    install_info = check_ultralytics_install(yolo_version)
    if not install_info.get('is_installed'):
        return jsonify({'error': f'{YOLO_VERSION_CONFIG[yolo_version]["name"]} 训练环境未安装，请先安装'}), 400

    # 检查工程是否存在
    project_path = get_project_path(project_name)
    if not os.path.exists(project_path):
        return jsonify({'error': '工程不存在'}), 404

    # 参数校验
    task = data.get('task', 'detect')
    if task not in YOLO_TASK_TYPES:
        return jsonify({'error': f'不支持的任务类型: {task}'}), 400

    # 检查模型是否在任务支持的模型列表中
    cfg_models = YOLO_VERSION_CONFIG[yolo_version].get('models', {})
    task_models = cfg_models.get(task, [])
    model = data.get('model', task_models[0] if task_models else 'yolo11n.pt')
    if model not in task_models:
        return jsonify({'error': f'模型 {model} 不支持任务类型 {task}'}), 400

    params = {
        'yolo_version': yolo_version,
        'task': task,
        'model': model,
        'epochs': int(data.get('epochs', 100)),
        'batch': int(data.get('batch', 8)),
        'imgsz': int(data.get('imgsz', 640)),
        'device': data.get('device', '0'),
        'train_val_ratio': float(data.get('train_val_ratio', 0.8)),
        'use_project_model': data.get('use_project_model', False),
    }

    if params['epochs'] < 1 or params['epochs'] > 1000:
        return jsonify({'error': 'epochs 必须在 1-1000 之间'}), 400
    if params['batch'] < 1 or params['batch'] > 256:
        return jsonify({'error': 'batch 必须在 1-256 之间'}), 400
    if params['imgsz'] < 32 or params['imgsz'] > 2048:
        return jsonify({'error': 'imgsz 必须在 32-2048 之间'}), 400
    if not (0.5 <= params['train_val_ratio'] <= 0.95):
        return jsonify({'error': '训练/验证比例必须在 50%-95% 之间'}), 400

    # 创建新任务
    yolo_training_task = YOLOTrainingTask(project_name, params)
    success, msg = yolo_training_task.start()
    if not success:
        return jsonify({'error': msg}), 400
    return jsonify({'success': True, 'message': msg})


@app.route('/api/train/status')
def train_status():
    """获取训练任务状态。"""
    if yolo_training_task is None:
        return jsonify({'status': TASK_STATUS['IDLE']})
    return jsonify({
        'status': yolo_training_task.status,
        'progress': yolo_training_task.progress,
        'project': yolo_training_task.project_name,
        'version': getattr(yolo_training_task, 'version', None),
        'result': yolo_training_task.result,
        'error': yolo_training_task.error,
        'elapsed': round(time.time() - yolo_training_task.start_time, 1) if yolo_training_task.start_time else 0,
    })


@app.route('/api/train/cancel', methods=['POST'])
def train_cancel():
    """取消训练任务。"""
    global yolo_training_task
    if yolo_training_task is None or yolo_training_task.status != TASK_STATUS['RUNNING']:
        return jsonify({'error': '没有正在运行的训练任务'}), 400
    yolo_training_task.stop()
    return jsonify({'success': True})


@app.route('/api/train/delete-version', methods=['POST'])
def train_delete_version():
    """删除指定版本的训练模型。"""
    data = request.json or {}
    project_name = data.get('project')
    version = data.get('version')
    if not project_name or not version:
        return jsonify({'error': '缺少 project 或 version 参数'}), 400
    project_path = get_project_path(project_name)
    models_dir = os.path.join(project_path, 'models')
    version_dir = os.path.join(models_dir, version)
    if not os.path.exists(version_dir):
        return jsonify({'error': '版本不存在'}), 404
    # 安全检查：确保路径在项目目录内
    real_version_dir = os.path.realpath(version_dir)
    real_models_dir = os.path.realpath(models_dir)
    if not real_version_dir.startswith(real_models_dir):
        return jsonify({'error': '非法路径'}), 403
    # 禁止删除正在训练中的版本
    if yolo_training_task is not None and yolo_training_task.status == TASK_STATUS['RUNNING']:
        if yolo_training_task.project_name == project_name and getattr(yolo_training_task, 'version', None) == version:
            return jsonify({'error': '该版本正在训练中，无法删除'}), 409
    try:
        shutil.rmtree(version_dir)
        return jsonify({'success': True, 'message': f'版本 {version} 已删除'})
    except Exception as e:
        return jsonify({'error': f'删除失败: {str(e)}'}), 500


@app.route('/api/train/dataset-info')
def train_dataset_info():
    """获取当前工程数据集统计信息。"""
    project_name = request.args.get('project')
    if not project_name:
        return jsonify({'error': '缺少 project 参数'}), 400
    project_path = get_project_path(project_name)
    if not os.path.exists(project_path):
        return jsonify({'error': '工程不存在'}), 404

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')

    # 统计工程目录下所有图片
    total_images = 0
    for f in os.listdir(project_path):
        if os.path.isfile(os.path.join(project_path, f)) and f.lower().endswith(image_extensions):
            total_images += 1

    # 读取标注和类别
    annotations_file = os.path.join(project_path, 'annotations', 'annotations.json')
    classes_file = os.path.join(project_path, 'annotations', 'classes.json')

    annotations = {}
    if os.path.exists(annotations_file):
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except Exception:
            pass

    classes = []
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = json.load(f)
        except Exception:
            pass

    # 统计已标注图片（按任务类型过滤）
    task = request.args.get('task', 'detect')
    task_type_map = {
        'detect': {'rectangle'},
        'segment': {'polygon', 'rectangle'},
        'obb': {'obb'},
        'pose': {'pose'},
        'classify': {'classify'},
    }
    allowed_types = task_type_map.get(task, {'rectangle'})
    annotated_images = 0
    for img_name, anns in annotations.items():
        if not anns:
            continue
        if not os.path.exists(os.path.join(project_path, img_name)):
            continue
        has_valid = any(ann.get('type') in allowed_types for ann in anns)
        if has_valid:
            annotated_images += 1

    # 计算划分数量
    ratio = float(request.args.get('ratio', 0.8))
    train_count = int(annotated_images * ratio)
    val_count = annotated_images - train_count

    return jsonify({
        'total_images': total_images,
        'annotated_images': annotated_images,
        'train_count': train_count,
        'val_count': val_count,
        'test_count': 0,
        'class_count': len(classes),
    })


@app.route('/api/train/model-info')
def train_model_info():
    """获取当前工程已训练模型信息（含版本列表）。"""
    project_name = request.args.get('project')
    if not project_name:
        return jsonify({'error': '缺少 project 参数'}), 400
    project_path = get_project_path(project_name)
    models_dir = os.path.join(project_path, 'models')

    # 最新模型（向后兼容）
    best_path = os.path.join(models_dir, 'best.pt')
    info_path = os.path.join(models_dir, 'model_info.json')
    result_path = os.path.join(models_dir, 'val_results.json')

    latest = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                latest['info'] = json.load(f)
        except Exception:
            pass
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                latest['val_results'] = json.load(f)
        except Exception:
            pass

    # 扫描所有版本
    versions = []
    if os.path.exists(models_dir):
        for name in sorted(os.listdir(models_dir), reverse=True):
            vdir = os.path.join(models_dir, name)
            if not os.path.isdir(vdir):
                continue
            v_info_path = os.path.join(vdir, 'model_info.json')
            v_result_path = os.path.join(vdir, 'val_results.json')
            v_best = os.path.join(vdir, 'best.pt')
            v_info = {}
            v_val = {}
            if os.path.exists(v_info_path):
                try:
                    with open(v_info_path, 'r', encoding='utf-8') as f:
                        v_info = json.load(f)
                except Exception:
                    pass
            if os.path.exists(v_result_path):
                try:
                    with open(v_result_path, 'r', encoding='utf-8') as f:
                        v_val = json.load(f)
                except Exception:
                    pass
            # 判断是否正在训练中
            is_training = False
            train_progress = {}
            if yolo_training_task is not None and yolo_training_task.status == TASK_STATUS['RUNNING']:
                if yolo_training_task.project_name == project_name and getattr(yolo_training_task, 'version', None) == name:
                    is_training = True
                    train_progress = yolo_training_task.progress

            versions.append({
                'version': name,
                'exists': os.path.exists(v_best),
                'onnx_exists': os.path.exists(os.path.join(vdir, 'best.onnx')),
                'model_path': v_best if os.path.exists(v_best) else None,
                'is_training': is_training,
                'train_progress': train_progress,
                'status': v_info.get('status') if v_info else None,
                'info': v_info,
                'val_results': v_val,
            })

    return jsonify({
        'exists': os.path.exists(best_path),
        'path': best_path if os.path.exists(best_path) else None,
        'latest': latest,
        'versions': versions,
    })


@app.route('/api/train/download-model')
def train_download_model():
    """下载当前工程已训练模型。支持按版本和格式下载。"""
    project_name = request.args.get('project')
    version = request.args.get('version')
    fmt = request.args.get('format', 'pt')
    if not project_name:
        return jsonify({'error': '缺少 project 参数'}), 400
    project_path = get_project_path(project_name)
    models_dir = os.path.join(project_path, 'models')

    ext = 'onnx' if fmt == 'onnx' else 'pt'
    if version:
        model_path = os.path.join(models_dir, version, f'best.{ext}')
        download_name = f'{project_name}_{version}_best.{ext}'
    else:
        model_path = os.path.join(models_dir, f'best.{ext}')
        download_name = f'{project_name}_best.{ext}'

    if not os.path.exists(model_path):
        return jsonify({'error': '模型文件不存在'}), 404
    return send_file(model_path, as_attachment=True, download_name=download_name)


@app.route('/api/train/export-onnx', methods=['POST'])
def train_export_onnx():
    """导出指定版本模型为 ONNX 格式。"""
    data = request.json or {}
    project_name = data.get('project')
    version = data.get('version')
    if not project_name:
        return jsonify({'error': '缺少 project 参数'}), 400
    project_path = get_project_path(project_name)
    models_dir = os.path.join(project_path, 'models')

    if version:
        model_path = os.path.join(models_dir, version, 'best.pt')
        output_path = os.path.join(models_dir, version, 'best.onnx')
        info_path = os.path.join(models_dir, version, 'model_info.json')
    else:
        model_path = os.path.join(models_dir, 'best.pt')
        output_path = os.path.join(models_dir, 'best.onnx')
        info_path = os.path.join(models_dir, 'model_info.json')

    if not os.path.exists(model_path):
        return jsonify({'error': '模型文件不存在'}), 404

    # 读取 model_info.json 获取 yolo_version
    yolo_version = 'yolo11'
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                yolo_version = info.get('yolo_version', 'yolo11')
        except Exception:
            pass

    # 检查 ONNX 是否已导出
    if os.path.exists(output_path):
        return jsonify({'success': True, 'message': 'ONNX 已存在', 'path': output_path})

    # 执行导出
    python_path = get_ultralytics_python_path(yolo_version)
    cmd = [
        python_path,
        os.path.join(app.root_path, 'plugins', 'export_yolo.py'),
        '--model', model_path,
        '--output', output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0 and os.path.exists(output_path):
            return jsonify({'success': True, 'message': 'ONNX 导出成功', 'path': output_path})
        else:
            err = result.stderr[-800:] if result.stderr else result.stdout[-800:]
            return jsonify({'error': f'导出失败: {err}'}), 500
    except Exception as e:
        return jsonify({'error': f'导出异常: {str(e)}'}), 500


@app.route('/api/model/download')
def model_download():
    """Download a model version as a zip package (for deploy container)."""
    project_name = request.args.get('project')
    version = request.args.get('version')
    if not project_name or not version:
        return jsonify({'error': 'Missing project or version parameter'}), 400

    project_path = get_project_path(project_name)
    version_dir = os.path.join(project_path, 'models', version)
    if not os.path.isdir(version_dir):
        return jsonify({'error': 'Model version not found'}), 404

    # Create zip in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(version_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, version_dir)
                zf.write(file_path, arcname)
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{project_name}_{version}.zip'
    )


@app.route('/api/model/versions')
def model_versions():
    """List all model versions for a project (for deploy container)."""
    project_name = request.args.get('project')
    if not project_name:
        return jsonify({'error': 'Missing project parameter'}), 400

    project_path = get_project_path(project_name)
    models_dir = os.path.join(project_path, 'models')
    versions = []

    if os.path.exists(models_dir):
        for name in sorted(os.listdir(models_dir), reverse=True):
            vdir = os.path.join(models_dir, name)
            if not os.path.isdir(vdir):
                continue
            info_path = os.path.join(vdir, 'model_info.json')
            info = {}
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                except Exception:
                    pass
            versions.append({
                'version': name,
                'onnx_exists': os.path.exists(os.path.join(vdir, 'best.onnx')),
                'info': info,
            })

    return jsonify({'versions': versions})


# (nndeploy publish/unpublish endpoints removed — models are served directly via /api/model/download)


@app.route('/api/admin/rebuild-metadata')
def rebuild_metadata():
    """一次性修复：遍历所有模型版本，从 args.yaml -> data.yaml 读取 names 并补全 classes。"""
    import yaml
    projects_dir = os.path.join(app.root_path, 'projects')
    fixed = []
    errors = []
    skipped = []
    if not os.path.exists(projects_dir):
        return jsonify({'fixed': [], 'errors': ['projects 目录不存在']})
    for project_name in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_name)
        if not os.path.isdir(project_path):
            continue
        models_dir = os.path.join(project_path, 'models')
        if not os.path.exists(models_dir):
            continue
        for name in os.listdir(models_dir):
            vdir = os.path.join(models_dir, name)
            if not os.path.isdir(vdir):
                continue
            args_yaml_path = os.path.join(vdir, 'args.yaml')
            if not os.path.exists(args_yaml_path):
                skipped.append(f'{project_name}/{name}: 无 args.yaml')
                continue
            try:
                with open(args_yaml_path, 'r', encoding='utf-8') as f:
                    args_yaml = yaml.safe_load(f)
                data_yaml_path = args_yaml.get('data')
                classes = []
                if data_yaml_path and os.path.exists(data_yaml_path):
                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                        data_yaml = yaml.safe_load(f)
                    names = data_yaml.get('names', {})
                    if isinstance(names, dict):
                        classes = [names[i] for i in sorted(names.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)]
                    elif isinstance(names, list):
                        classes = names
                # 更新 model_info.json
                info_path = os.path.join(vdir, 'model_info.json')
                info = {}
                if os.path.exists(info_path):
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                info['classes'] = classes
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
                # 重新生成 deploy_metadata.json
                deploy_meta = {
                    'project': project_name,
                    'version': name,
                    'yolo_version': info.get('yolo_version', 'yolo11'),
                    'task': info.get('task', 'detect'),
                    'input_size': info.get('imgsz', 640),
                    'class_count': len(classes),
                    'classes': classes,
                    'onnx_file': 'best.onnx' if os.path.exists(os.path.join(vdir, 'best.onnx')) else None,
                    'pt_file': 'best.pt' if os.path.exists(os.path.join(vdir, 'best.pt')) else None,
                    'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                deploy_meta_path = os.path.join(vdir, 'deploy_metadata.json')
                with open(deploy_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(deploy_meta, f, indent=2, ensure_ascii=False)
                fixed.append(f'{project_name}/{name}: {len(classes)} 个类别')
            except Exception as e:
                errors.append(f'{project_name}/{name}: {str(e)}')
    return jsonify({'fixed': fixed, 'errors': errors, 'skipped': skipped})


@app.route('/api/workflow/export')
def workflow_export():
    """Export workflow.json for a project (for deploy container)."""
    project_name = request.args.get('project')
    name = request.args.get('name')
    if not project_name or not name:
        return jsonify({'error': 'Missing project or name parameter'}), 400

    project_path = get_project_path(project_name)
    workflows_dir = os.path.join(project_path, 'workflows')
    workflow_path = os.path.join(workflows_dir, f'{name}.json')

    if not os.path.exists(workflow_path):
        return jsonify({'error': 'Workflow not found'}), 404

    return send_file(workflow_path, mimetype='application/json')


@app.route('/api/workflow/list')
def workflow_list():
    """List all workflows for a project (for deploy container)."""
    project_name = request.args.get('project')
    if not project_name:
        return jsonify({'error': 'Missing project parameter'}), 400

    project_path = get_project_path(project_name)
    workflows_dir = os.path.join(project_path, 'workflows')
    workflows = []

    if os.path.exists(workflows_dir):
        for filename in os.listdir(workflows_dir):
            if filename.endswith('.json'):
                workflow_name = filename[:-5]
                workflow_path = os.path.join(workflows_dir, filename)
                try:
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = {}
                workflows.append({
                    'name': workflow_name,
                    'description': data.get('description', ''),
                    'nodes': list(data.get('nodes', {}).keys()),
                })

    return jsonify({'workflows': workflows})


# (nndeploy workflow proxy endpoints removed — use /api/workflow/ endpoints instead)


@app.route('/train')
def train_page():
    """YOLO 模型训练页面。"""
    return render_template('train.html', version=APP_VERSION)


@app.route('/model-test')
def model_test_page():
    """模型测试页面。"""
    return render_template('model-test.html', version=APP_VERSION)


@app.route('/workflow')
def workflow_page():
    """工作流编排页面。"""
    return render_template('workflow.html', version=APP_VERSION)


@app.route('/api/project-test-images')
def project_test_images():
    """获取工程 test/images/ 目录下的图片列表。"""
    project_name = request.args.get('project', '')
    if not project_name:
        return jsonify({'error': '缺少 project 参数'}), 400
    project_path = get_project_path(project_name)
    test_images_dir = os.path.join(project_path, 'test', 'images')
    images = []
    if os.path.exists(test_images_dir) and os.path.isdir(test_images_dir):
        for fname in sorted(os.listdir(test_images_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                images.append({'name': fname, 'url': f'/api/project-image?project={project_name}&path=test/images/{fname}'})
    else:
        # 回退到工程根目录
        for fname in sorted(os.listdir(project_path)):
            if os.path.isfile(os.path.join(project_path, fname)) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                images.append({'name': fname, 'url': f'/api/project-image?project={project_name}&path={fname}'})
    return jsonify({'images': images})


@app.route('/api/project-image')
def project_image():
    """从工程目录中提供图片文件。"""
    project_name = request.args.get('project', '')
    path = request.args.get('path', '')
    if not project_name or not path:
        return jsonify({'error': '缺少参数'}), 400
    if '..' in path or path.startswith('/'):
        return jsonify({'error': '非法路径'}), 403
    project_path = get_project_path(project_name)
    file_path = os.path.join(project_path, path)
    real_project_path = os.path.realpath(project_path)
    real_file_path = os.path.realpath(file_path)
    if not real_file_path.startswith(real_project_path):
        return jsonify({'error': '路径越界'}), 403
    if not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    return send_from_directory(directory, filename)


@app.route('/api/model-test/infer', methods=['POST'])
def model_test_infer():
    """执行 YOLO 模型推理测试。"""
    import tempfile

    # 检查 GPU 资源冲突
    with gpu_task_lock:
        if current_gpu_task is not None:
            return jsonify({'error': f'当前有 {current_gpu_task} 任务正在运行，请等待完成'}), 409

    model_path = request.form.get('model_path', '')
    model_file = request.form.get('model', '')
    yolo_version = request.form.get('version', 'yolo11')
    task = request.form.get('task', 'detect')
    project_name = request.form.get('project', '')
    image_url = request.form.get('image_url', '')

    # 环境校验
    python_path = get_ultralytics_python_path(yolo_version)
    install_path = get_yolo_install_path(yolo_version)
    if not os.path.exists(python_path):
        return jsonify({'error': 'YOLO 环境未安装，请先安装训练环境'}), 400

    # 模型路径（优先使用前端传入的完整路径，否则按旧逻辑拼接）
    if not model_path:
        models_dir = os.path.join(install_path, 'models')
        model_path = os.path.join(models_dir, model_file)

    # 将 /projects/<name>/... 虚拟路径解析为真实文件系统路径
    if model_path:
        normalized = model_path.replace('\\', '/')
        if normalized.startswith('/projects/'):
            rel = normalized.lstrip('/')
            model_path = os.path.join(BASE_PATH, rel)

    if not os.path.exists(model_path):
        return jsonify({'error': f'模型文件不存在: {model_file or model_path}'}), 400

    # 获取图片
    image_path = None
    if 'image' in request.files:
        image_file = request.files['image']
        temp_dir = tempfile.mkdtemp(prefix='model_test_')
        image_path = os.path.join(temp_dir, image_file.filename or 'test.jpg')
        image_file.save(image_path)
    elif image_url:
        # 从项目路径解析图片
        if image_url.startswith('/api/project-image?'):
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(image_url)
            qs = parse_qs(parsed.query)
            img_project = qs.get('project', [''])[0]
            img_path = qs.get('path', [''])[0]
            if img_project and img_path:
                project_path = get_project_path(img_project)
                image_path = os.path.join(project_path, img_path.replace('/', os.sep))
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': '图片文件不存在'}), 400
    else:
        return jsonify({'error': '请提供图片'}), 400

    # 获取类别名称映射
    class_names = {}
    if project_name:
        classes_path = os.path.join(get_project_path(project_name), 'classes.json')
        if os.path.exists(classes_path):
            try:
                with open(classes_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        class_names[item.get('id', 0)] = item.get('name', str(item.get('id', 0)))
            except Exception:
                pass

    # 构建推理脚本
    script = f'''
import json, sys, os
from ultralytics import YOLO

model = YOLO({repr(model_path)})
results = model({repr(image_path)}, task="{task}", verbose=False)
result = results[0]

predictions = []
img_h, img_w = result.orig_shape

if "{task}" == "classify":
    probs = result.probs
    if probs is not None:
        top_idx = int(probs.top1)
        predictions.append({{
            "class": result.names.get(top_idx, str(top_idx)),
            "confidence": round(float(probs.top1conf), 4),
            "class_id": top_idx
        }})
elif "{task}" == "segment":
    # 分割推理：以 masks 为主输出，bbox 作为辅助信息
    masks = result.masks if hasattr(result, 'masks') else None
    boxes = result.boxes
    if masks is not None:
        for i in range(len(masks)):
            seg = masks.xy[i].cpu().numpy().tolist() if hasattr(masks.xy, 'cpu') else masks.xy[i].tolist()
            # 分割多边形顶点坐标（绝对像素）
            points = [{{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}} for p in seg]
            pred = {{
                "points": points,
                "detection_id": str(os.urandom(16).hex())
            }}
            # 从对应的 bbox 中获取置信度、类别等信息
            if boxes is not None and i < len(boxes):
                pred["confidence"] = round(float(boxes.conf[i]), 4)
                pred["class"] = result.names.get(int(boxes.cls[i]), str(int(boxes.cls[i])))
                pred["class_id"] = int(boxes.cls[i])
                bbox = boxes.xywhn[i].cpu().numpy().tolist() if hasattr(boxes.xywhn, 'cpu') else boxes.xywhn[i].tolist()
                pred["x"] = round(float(bbox[0] * img_w), 2)
                pred["y"] = round(float(bbox[1] * img_h), 2)
                pred["width"] = round(float(bbox[2] * img_w), 2)
                pred["height"] = round(float(bbox[3] * img_h), 2)
            else:
                pred["confidence"] = 0.0
                pred["class"] = ""
                pred["class_id"] = 0
            predictions.append(pred)
    elif boxes is not None:
        # 降级：masks 不存在但有 boxes，按检测输出
        for i in range(len(boxes)):
            bbox = boxes.xywhn[i].cpu().numpy().tolist() if hasattr(boxes.xywhn, 'cpu') else boxes.xywhn[i].tolist()
            cx, cy, w, h = bbox[0] * img_w, bbox[1] * img_h, bbox[2] * img_w, bbox[3] * img_h
            predictions.append({{
                "x": round(float(cx), 2),
                "y": round(float(cy), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
                "confidence": round(float(boxes.conf[i]), 4),
                "class": result.names.get(int(boxes.cls[i]), str(int(boxes.cls[i]))),
                "class_id": int(boxes.cls[i]),
                "detection_id": str(os.urandom(16).hex())
            }})
else:
    boxes = result.boxes
    masks = result.masks if hasattr(result, 'masks') else None
    keypoints = result.keypoints if hasattr(result, 'keypoints') else None
    obbs = result.obb if hasattr(result, 'obb') else None

    if obbs is not None and len(obbs) > 0:
        for i in range(len(obbs)):
            pts = obbs.xyxyxyxy[i].cpu().numpy().tolist() if hasattr(obbs.xyxyxyxy, 'cpu') else obbs.xyxyxyxy[i].tolist()
            predictions.append({{
                "points": [{{"x": float(pts[j][0]), "y": float(pts[j][1])}} for j in range(4)],
                "confidence": round(float(obbs.conf[i]), 4),
                "class": result.names.get(int(obbs.cls[i]), str(int(obbs.cls[i]))),
                "class_id": int(obbs.cls[i]),
                "detection_id": str(os.urandom(16).hex())
            }})
    elif boxes is not None:
        for i in range(len(boxes)):
            bbox = boxes.xywhn[i].cpu().numpy().tolist() if hasattr(boxes.xywhn, 'cpu') else boxes.xywhn[i].tolist()
            cx, cy, w, h = bbox[0] * img_w, bbox[1] * img_h, bbox[2] * img_w, bbox[3] * img_h
            pred = {{
                "x": round(float(cx), 2),
                "y": round(float(cy), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
                "confidence": round(float(boxes.conf[i]), 4),
                "class": result.names.get(int(boxes.cls[i]), str(int(boxes.cls[i]))),
                "class_id": int(boxes.cls[i]),
                "detection_id": str(os.urandom(16).hex())
            }}
            if masks is not None and i < len(masks):
                seg = masks.xy[i].cpu().numpy().tolist() if hasattr(masks.xy, 'cpu') else masks.xy[i].tolist()
                pred["points"] = [{{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}} for p in seg]
            if keypoints is not None and i < len(keypoints):
                kpts = keypoints.data[i].cpu().numpy().tolist() if hasattr(keypoints.data, 'cpu') else keypoints.data[i].tolist()
                kp_names = result.names if hasattr(result, 'names') else {{}}
                pred["keypoints"] = [{{
                    "x": round(float(kp[0]), 2),
                    "y": round(float(kp[1]), 2),
                    "confidence": round(float(kp[2]), 4),
                    "name": str(idx)
                }} for idx, kp in enumerate(kpts)]
            predictions.append(pred)

# 替换 class 为 classes.json 中的名称
class_names = {json.dumps(class_names)}
for p in predictions:
    cid = p.get("class_id", 0)
    if str(cid) in class_names:
        p["class"] = class_names[str(cid)]
    elif cid in class_names:
        p["class"] = class_names[cid]

print(json.dumps({{"predictions": predictions}}, ensure_ascii=False))
'''
    try:
        result = subprocess.run(
            [python_path, '-c', script],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=300
        )
        if result.returncode != 0:
            err = result.stderr[-800:] if result.stderr else result.stdout[-800:]
            return jsonify({'error': f'推理失败: {err}'}), 500
        output = result.stdout.strip()
        # 提取最后一行 JSON
        lines = [ln for ln in output.split('\n') if ln.strip()]
        json_line = lines[-1] if lines else '{}'
        data = json.loads(json_line)
        return jsonify(data)
    except subprocess.TimeoutExpired:
        return jsonify({'error': '推理超时'}), 500
    except Exception as e:
        return jsonify({'error': f'推理异常: {str(e)}'}), 500
    finally:
        # 清理临时文件
        if image_path and 'temp' in image_path.lower():
            try:
                os.remove(image_path)
                os.rmdir(os.path.dirname(image_path))
            except Exception:
                pass


# ========================== SAM 2 交互式分割 API ==========================


@app.route('/api/sam/predict', methods=['POST'])
def sam_predict():
    """SAM 2 交互式分割推理"""
    try:
        data = request.get_json()
        image_name = data.get('image')
        prompts = data.get('prompts', [])

        if not image_name:
            return jsonify({'error': 'Missing image name'}), 400
        if not prompts:
            return jsonify({'error': 'Missing prompts'}), 400

        # 获取图片路径
        project_name = get_current_project()
        image_path = os.path.join(get_upload_folder(), image_name)
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_name}'}), 404

        # 获取 GPU 锁
        if not acquire_gpu("sam", timeout=10):
            status = get_gpu_status()
            return jsonify({
                'error': f'GPU is busy with {status["owner"]} task. Please try later.'
            }), 503

        # 读取 AI 配置中的 mask_mode
        mask_mode = "best"
        try:
            config_path = os.path.join(get_upload_folder(), 'config', 'ai_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    ai_cfg = json.load(f)
                mask_mode = ai_cfg.get('samMaskMode', 'best')
        except Exception:
            pass

        try:
            engine = get_sam2_engine()
            if not engine or not engine.is_loaded():
                return jsonify({'error': 'SAM 2 engine not loaded'}), 503

            result = engine.predict(image_path, prompts, mask_mode=mask_mode)
            return jsonify({
                'success': True,
                'mask_polygons': result['mask_polygons'],
                'scores': result['scores'],
                'area': result['area'],
            })
        finally:
            release_gpu("sam")

    except Exception as e:
        logging.error(f"SAM predict error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sam/reset', methods=['POST'])
def sam_reset():
    """清除 SAM 当前图片缓存"""
    try:
        engine = get_sam2_engine()
        if engine:
            engine.reset()
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"SAM reset error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sam/status', methods=['GET'])
def sam_status():
    """SAM 模型状态"""
    engine = get_sam2_engine()
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    return jsonify({
        'loaded': engine is not None and engine.is_loaded(),
        'model': engine.model_type if engine else None,
        'device': engine.device if engine else None,
        'gpu_available': gpu_available,
        'gpu': get_gpu_status(),
    })


@app.route('/api/sam/models', methods=['GET'])
def sam_models():
    """获取支持的 SAM 模型列表和当前模型"""
    from ai_manager import SAM2Engine
    engine = get_sam2_engine()
    models = []
    for key, cfg in SAM2Engine.MODEL_CONFIGS.items():
        ckpt_path = os.path.join(os.environ.get("SAM_MODELS_DIR", os.path.join(BASE_PATH, "models")), cfg["checkpoint"])
        models.append({
            'id': key,
            'name': f"SAM 2 Hiera {key.capitalize()}",
            'checkpoint': cfg["checkpoint"],
            'available': os.path.exists(ckpt_path),
        })
    return jsonify({
        'models': models,
        'current': engine.model_type if engine else None,
    })


@app.route('/api/sam/check-models', methods=['GET'])
def sam_check_models():
    """检查 SAM2 各版本权重文件是否存在并返回大小信息"""
    from ai_manager import SAM2Engine
    models_dir = os.environ.get("SAM_MODELS_DIR", os.path.join(BASE_PATH, "models"))
    model_sizes = {
        "tiny": 155,
        "small": 184,
        "base_plus": 298,
    }
    models = []
    for key, cfg in SAM2Engine.MODEL_CONFIGS.items():
        ckpt_path = os.path.join(models_dir, cfg["checkpoint"])
        size_mb = model_sizes.get(key, 0)
        models.append({
            'id': key,
            'name': f"SAM 2 Hiera {key.capitalize()}",
            'checkpoint': cfg["checkpoint"],
            'available': os.path.exists(ckpt_path),
            'size_mb': size_mb,
        })
    return jsonify({'success': True, 'models': models})


@app.route('/api/sam/download-models')
def sam_download_models():
    """从 hf-mirror.com 下载 SAM2 权重文件（SSE 流式推送）"""
    import requests
    import time

    models_str = request.args.get('models', '')
    model_ids = [m.strip() for m in models_str.split(',') if m.strip()]

    from ai_manager import SAM2Engine
    models_dir = os.environ.get("SAM_MODELS_DIR", os.path.join(BASE_PATH, "models"))
    os.makedirs(models_dir, exist_ok=True)

    HF_MIRROR_URLS = {
        "tiny": "https://hf-mirror.com/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt",
        "small": "https://hf-mirror.com/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt",
        "base_plus": "https://hf-mirror.com/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt",
    }

    def generate():
        yield f"data: {json.dumps({'status': 'started', 'message': '准备下载 SAM2 模型...', 'progress': 0})}\n\n"
        time.sleep(0.3)

        try:
            to_download = []
            skipped = []
            for mid in model_ids:
                if mid not in SAM2Engine.MODEL_CONFIGS:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'未知模型: {mid}', 'progress': 0})}\n\n"
                    return
                ckpt = SAM2Engine.MODEL_CONFIGS[mid]["checkpoint"]
                ckpt_path = os.path.join(models_dir, ckpt)
                if os.path.exists(ckpt_path):
                    skipped.append(mid)
                else:
                    to_download.append(mid)

            if skipped:
                skipped_str = ', '.join(skipped)
                yield f"data: {json.dumps({'message': f'已跳过已存在的模型: {skipped_str}', 'progress': 5})}\n\n"
                time.sleep(0.3)

            if not to_download:
                yield f"data: {json.dumps({'status': 'completed', 'message': '所有选中模型均已存在，无需下载', 'progress': 100})}\n\n"
                return

            total = len(to_download)
            completed_count = 0
            for mid in to_download:
                ckpt = SAM2Engine.MODEL_CONFIGS[mid]["checkpoint"]
                url = HF_MIRROR_URLS.get(mid)
                if not url:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'模型 {mid} 无下载地址', 'progress': 0})}\n\n"
                    return

                save_path = os.path.join(models_dir, ckpt)
                yield f"data: {json.dumps({'status': 'downloading', 'model': ckpt, 'progress': int(completed_count / total * 80) + 10, 'message': f'正在下载 {ckpt}...'})}\n\n"

                try:
                    resp = requests.get(url, stream=True, timeout=300)
                    resp.raise_for_status()

                    total_size = int(resp.headers.get('content-length', 0))
                    downloaded = 0
                    chunk_size = 8192

                    with open(save_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    pct = int(downloaded / total_size * 100)
                                    overall_pct = int(completed_count / total * 80) + int(pct * 0.8 / total)
                                    downloaded_mb = downloaded / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    yield f"data: {json.dumps({'status': 'downloading', 'model': ckpt, 'progress': min(overall_pct, 99), 'downloaded_mb': round(downloaded_mb, 1), 'total_mb': round(total_mb, 1), 'message': f'下载 {ckpt}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB'})}\n\n"

                    completed_count += 1
                    yield f"data: {json.dumps({'status': 'model_completed', 'model': ckpt, 'progress': int(completed_count / total * 80) + 10, 'message': f'{ckpt} 下载完成'})}\n\n"
                    time.sleep(0.2)

                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'model': ckpt, 'message': f'{ckpt} 下载失败: {str(e)}', 'progress': 0})}\n\n"
                    return

            yield f"data: {json.dumps({'status': 'completed', 'message': '所有 SAM2 模型下载完成', 'progress': 100})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'status': 'error', 'message': f'下载失败: {str(e)}', 'progress': 0, 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/sam/switch-model', methods=['POST'])
def sam_switch_model():
    """切换 SAM 模型类型"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', '')
        if not model_type:
            return jsonify({'error': 'Missing model_type'}), 400

        from ai_manager import SAM2Engine, init_sam2_engine, release_sam2_engine
        if model_type not in SAM2Engine.MODEL_CONFIGS:
            return jsonify({'error': f'Unsupported model: {model_type}'}), 400

        # 释放旧引擎，初始化新引擎
        release_sam2_engine()
        models_dir = os.environ.get("SAM_MODELS_DIR", os.path.join(BASE_PATH, "models"))
        init_sam2_engine(model_type=model_type, models_dir=models_dir)

        engine = get_sam2_engine()
        return jsonify({
            'success': True,
            'model': engine.model_type if engine else None,
            'loaded': engine.is_loaded() if engine else False,
        })
    except Exception as e:
        logging.error(f"Switch SAM model error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========================== YOLO 批量自动标注 API ==========================


@app.route('/api/auto-label/yolo', methods=['POST'])
def auto_label_yolo():
    """YOLO 批量自动检测"""
    import time as time_module
    try:
        data = request.get_json()
        images = data.get('images', [])
        yolo_version = data.get('yolo_version', 'yolo11')
        model_path = data.get('model_path', '')
        confidence = float(data.get('confidence', 0.25))

        if not images:
            return jsonify({'error': 'No images provided'}), 400
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': f'Model not found: {model_path}'}), 400

        # 获取 GPU 锁
        if not acquire_gpu("yolo_label", timeout=10):
            status = get_gpu_status()
            return jsonify({
                'error': f'GPU is busy with {status["owner"]} task. Please try later.'
            }), 503

        try:
            labeler = YOLOAutoLabeler(yolo_version=yolo_version)
            project_name = get_current_project()
            upload_folder = get_upload_folder()

            total = len(images)
            all_annotations = {}
            labeled = 0

            for idx, image_name in enumerate(images):
                image_path = os.path.join(upload_folder, image_name)
                if not os.path.exists(image_path):
                    continue

                dets = labeler.label_image(image_path, model_path, confidence)

                # 转换为前端标注格式
                annotations = []
                for det in dets:
                    bbox = det['bbox']  # [x1, y1, x2, y2]
                    annotations.append({
                        'id': int(time_module.time() * 1000) + len(annotations),
                        'class': det['label'],
                        'type': 'rectangle',
                        'points': [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]],
                        ],
                        'confidence': det['confidence'],
                        'source': 'yolo_auto_label',
                    })

                if annotations:
                    all_annotations[image_name] = annotations
                    labeled += 1

                # SocketIO 进度推送（socketio 已在 app.py 中定义）
                from flask_socketio import emit
                socketio.emit('yolo_label_progress', {
                    'current': idx + 1,
                    'total': total,
                    'found': len(annotations),
                    'image': image_name,
                }, namespace='/')

            # 保存标注到 annotations.json
            if all_annotations:
                annotations_path = os.path.join(
                    BASE_PATH, 'projects', project_name, 'annotations', 'annotations.json'
                )
                existing = {}
                if os.path.exists(annotations_path):
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)

                for img_name, anns in all_annotations.items():
                    if img_name in existing:
                        existing[img_name].extend(anns)
                    else:
                        existing[img_name] = anns

                with open(annotations_path, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

            return jsonify({
                'success': True,
                'total': total,
                'labeled': labeled,
                'annotations': all_annotations,
            })
        finally:
            release_gpu("yolo_label")

    except Exception as e:
        logging.error(f"YOLO auto-label error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/auto-label/vlm-sam', methods=['POST'])
def auto_label_vlm_sam():
    """VLM 检测 → SAM 2 精细化分割的串联管道"""
    import time as time_module
    try:
        data = request.get_json()
        images = data.get('images', [])
        prompt_override = data.get('prompt', '')
        label_filter = data.get('label_filter')

        if not images:
            return jsonify({'error': 'No images provided'}), 400

        # 加载 VLM AI 配置
        project_name = get_current_project()
        config_path = os.path.join(get_upload_folder(), 'config', 'ai_config.json')
        if not os.path.exists(config_path):
            return jsonify({'error': 'AI config not found. Please configure AI settings first.'}), 400

        with open(config_path, 'r', encoding='utf-8') as f:
            ai_config = json.load(f)

        api_url = ai_config.get('apiUrl', 'http://127.0.0.1:1234/v1')
        api_key = ai_config.get('apiKey', '')
        timeout = int(ai_config.get('timeout', 300))
        prompt = prompt_override or ai_config.get('prompt', '')
        model = ai_config.get('model', 'qwen/qwen3-vl-8b')
        inference_tool = ai_config.get('inferenceTool', 'LMStudio')

        if not prompt:
            return jsonify({'error': 'No prompt configured. Please set AI prompt in settings.'}), 400

        # 获取 GPU 锁
        if not acquire_gpu("vlm_sam", timeout=30):
            status = get_gpu_status()
            return jsonify({
                'error': f'GPU is busy with {status["owner"]} task. Please try later.'
            }), 503

        try:
            labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)
            upload_folder = get_upload_folder()
            sam_engine = get_sam2_engine()

            annotations_path = os.path.join(
                BASE_PATH, 'projects', project_name, 'annotations', 'annotations.json'
            )

            total = len(images)
            all_annotations = {}
            labeled = 0

            for idx, image_name in enumerate(images):
                image_path = os.path.join(upload_folder, image_name)
                if not os.path.exists(image_path):
                    continue

                # 阶段 1: VLM 检测
                socketio.emit('vlm_sam_progress', {
                    'current': idx + 1, 'total': total,
                    'image': image_name, 'status': 'vlm', 'found': 0,
                })

                try:
                    vlm_result = labeler.analyze_image(image_path)
                except Exception as e:
                    logging.error(f"VLM inference failed for {image_name}: {e}")
                    vlm_result = {}

                detections = vlm_result.get('detections', [])
                if label_filter:
                    detections = [d for d in detections if d.get('label') == label_filter]

                # 阶段 2: SAM 2 精细化（对每个检测框）
                annotations = []
                for det in detections:
                    bbox = det.get('bbox', [])
                    if len(bbox) != 4:
                        continue

                    label = det.get('label', 'unknown')
                    confidence = det.get('confidence', 0.0)

                    # 尝试 SAM 2 精化
                    sam_success = False
                    if sam_engine and sam_engine.is_loaded():
                        try:
                            sam_result = sam_engine.predict(image_path, [
                                {'type': 'box', 'x1': bbox[0], 'y1': bbox[1],
                                 'x2': bbox[2], 'y2': bbox[3]}
                            ], mask_mode=ai_config.get('samMaskMode', 'best'))
                            polygons = sam_result.get('mask_polygons', [])
                            if polygons and len(polygons) > 0 and len(polygons[0]) >= 3:
                                annotations.append({
                                    'id': int(time_module.time() * 1000) + len(annotations),
                                    'class': label,
                                    'type': 'polygon',
                                    'points': polygons[0],
                                    'confidence': confidence,
                                    'source': 'vlm_sam',
                                })
                                sam_success = True
                        except Exception as e:
                            logging.warning(f"SAM refinement failed for {image_name}/{label}: {e}")

                    # 降级：保存 VLM 矩形框
                    if not sam_success:
                        annotations.append({
                            'id': int(time_module.time() * 1000) + len(annotations),
                            'class': label,
                            'type': 'rectangle',
                            'points': [
                                [bbox[0], bbox[1]],
                                [bbox[2], bbox[1]],
                                [bbox[2], bbox[3]],
                                [bbox[0], bbox[3]],
                            ],
                            'confidence': confidence,
                            'source': 'vlm_sam_fallback',
                        })

                if annotations:
                    all_annotations[image_name] = annotations
                    labeled += 1

                # SocketIO 进度推送
                socketio.emit('vlm_sam_progress', {
                    'current': idx + 1, 'total': total,
                    'image': image_name, 'status': 'done',
                    'found': len(annotations),
                })

            # 批量保存标注
            if all_annotations:
                existing = {}
                if os.path.exists(annotations_path):
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)

                for img_name, anns in all_annotations.items():
                    if img_name in existing:
                        existing[img_name].extend(anns)
                    else:
                        existing[img_name] = anns

                with open(annotations_path, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

            return jsonify({
                'success': True,
                'total': total,
                'labeled': labeled,
                'annotations': all_annotations,
            })
        finally:
            release_gpu("vlm_sam")

    except Exception as e:
        logging.error(f"VLM-SAM auto-label error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========================== Smart Pipeline Workflow API ==========================


def _add_source(target_id: str, source_id: str, nodes: list):
    """Helper: add a source dependency for a DAG node."""
    node = next((n for n in nodes if n["id"] == target_id), None)
    if node and source_id and source_id != target_id:
        node.setdefault("source", []).append(source_id)


def _litegraph_to_workflow(graph_json: dict) -> dict:
    """Convert LiteGraph.js serialized graph to workflow.yaml-compatible dict."""
    nodes = []
    for node in graph_json.get("nodes", []):
        raw_type = node.get("type", "yolo").lower().replace("node", "")
        # Normalize: strip category prefix (e.g. "xclabel/yolo" → "yolo")
        _type = raw_type.split("/")[-1] if "/" in raw_type else raw_type
        cfg = {
            "id": str(node.get("id", node.get("title", "unknown"))),
            "type": _type,
        }
        props = node.get("properties", {})
        # LiteGraph widget values are more current than properties when deserialized,
        # because configure() restores widgets_values without calling widget callbacks.
        wvals = node.get("widgets_values")
        if _type == "input":
            cfg["input_type"] = props.get("input_type", "upload")
            if props.get("url"):
                cfg["url"] = props.get("url")
        elif _type == "yolo":
            cfg["model"] = props.get("model", "")
            cfg["task"] = (wvals[0] if wvals and len(wvals) > 0 and wvals[0]
                           else props.get("task", "detect"))
            cfg["params"] = {
                "conf": float(wvals[1] if wvals and len(wvals) > 1 else props.get("conf", 0.25)),
                "iou": float(wvals[2] if wvals and len(wvals) > 2 else props.get("iou", 0.5)),
            }
        elif _type == "condition":
            cfg["expression"] = props.get("expression", "max_conf > 0.5")
        elif _type == "vllm":
            cfg["api_url"] = props.get("api_url", "")
            cfg["api_key"] = props.get("api_key", "")
            cfg["model_name"] = props.get("model_name", "")
            cfg["prompt"] = props.get("prompt", "")
            cfg["extract_roi"] = props.get("extract_roi", False)
            cfg["params"] = {
                "temperature": float(props.get("temperature", 0.1)),
                "max_tokens": int(props.get("max_tokens", 256)),
            }
            cfg["condition"] = props.get("condition", "")
        elif _type == "output":
            cfg["source"] = [props.get("source", "")] if props.get("source") else []
        nodes.append(cfg)

    # Build links (source->target relationships)
    # Handle both array format [id, origin_id, slot, target_id, slot, type] and object format
    links = graph_json.get("links", [])
    for link in links:
        if isinstance(link, (list, tuple)):
            link = {
                "id": link[0], "origin_id": link[1],
                "origin_slot": link[2], "target_id": link[3],
                "target_slot": link[4],
                "type": link[5] if len(link) > 5 else "",
            }
        src_node = next((n for n in graph_json.get("nodes", [])
                         if n.get("id") == link.get("origin_id")), None)
        dst_node = next((n for n in graph_json.get("nodes", [])
                         if n.get("id") == link.get("target_id")), None)
        if dst_node:
            _dst_type = dst_node.get("type", "").lower().replace("node", "").split("/")[-1]
        else:
            _dst_type = ""
        if _dst_type == "vllm":
            src_node_obj = next((n for n in graph_json.get("nodes", []) if n.get("id") == link.get("origin_id")), None)
            _src_type = src_node_obj.get("type", "").lower().replace("node", "").split("/")[-1] if src_node_obj else ""
            # Link from condition → vllm sets the condition gate
            if _src_type == "condition":
                cond_node = next((n for n in nodes if n["id"] == str(src_node.get("id"))), None)
                if cond_node:
                    dst_vllm = next((n for n in nodes if n["id"] == str(dst_node.get("id"))), None)
                    if dst_vllm:
                        dst_vllm["condition"] = cond_node["id"]
            # Other incoming links (input/yolo) add as source dependency for DAG ordering
            else:
                _add_source(str(dst_node.get("id")), str(src_node.get("id")), nodes)
        if _dst_type == "yolo":
            _add_source(str(dst_node.get("id")), str(src_node.get("id")), nodes)
        if _dst_type == "condition":
            _add_source(str(dst_node.get("id")), str(src_node.get("id")), nodes)
        if _dst_type == "output":
            _add_source(str(dst_node.get("id")), str(src_node.get("id")), nodes)

    title = graph_json.get("title", "untitled")
    return {
        "version": "1.0",
        "name": title.replace(" ", "_").lower(),
        "pipeline": nodes,
    }


@app.route('/api/workflow/list')
def project_workflow_list():
    """List all workflow files for a project."""
    project_name = request.args.get('project')
    if not project_name:
        return jsonify({'error': '缺少 project 参数'}), 400
    workflows_dir = os.path.join(get_project_path(project_name), 'workflows')
    if not os.path.isdir(workflows_dir):
        return jsonify({'workflows': []})
    workflows = []
    for f in sorted(os.listdir(workflows_dir)):
        if f.endswith('.yaml') or f.endswith('.yml'):
            workflows.append({
                'name': f.replace('.yaml', '').replace('.yml', ''),
                'file': f,
            })
    return jsonify({'workflows': workflows})


@app.route('/api/workflow/get')
def workflow_get():
    """Get a single workflow file content. Prefers JSON (for graph restore) over YAML."""
    project_name = request.args.get('project')
    name = request.args.get('workflow')
    if not project_name or not name:
        return jsonify({'error': '缺少 project 或 workflow 参数'}), 400
    workflows_dir = os.path.join(get_project_path(project_name), 'workflows')

    # Try JSON first — contains the full LiteGraph serialization for graph.configure()
    json_path = os.path.join(workflows_dir, f'{name}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return jsonify({'name': name, 'content': json.load(f), 'format': 'json'})

    # Fall back to YAML (read-only, cannot restore graph canvas)
    for ext in ('.yaml', '.yml'):
        path = os.path.join(workflows_dir, f'{name}{ext}')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return jsonify({'name': name, 'content': f.read(), 'format': 'yaml'})

    return jsonify({'error': 'Workflow 文件不存在'}), 404


@app.route('/api/workflow/save', methods=['POST'])
def workflow_save():
    """Save a workflow from LiteGraph JSON → workflow.yaml."""
    data = request.json or {}
    project_name = data.get('project')
    graph_json = data.get('graph')
    name = data.get('name', 'untitled')

    if not project_name or not graph_json:
        return jsonify({'error': '缺少 project 或 graph 参数'}), 400

    workflow_dir = os.path.join(get_project_path(project_name), 'workflows')
    os.makedirs(workflow_dir, exist_ok=True)

    # Convert LiteGraph JSON to workflow YAML
    workflow_dict = _litegraph_to_workflow(graph_json)
    if name != 'untitled':
        workflow_dict['name'] = name

    # Write YAML (for deploy engine)
    import yaml as yaml_lib
    base_name = workflow_dict['name']
    yaml_path = os.path.join(workflow_dir, f"{base_name}.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml_lib.dump(workflow_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Write raw LiteGraph JSON (for re-editing the graph)
    json_path = os.path.join(workflow_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)

    return jsonify({'success': True, 'name': base_name, 'path': yaml_path})


@app.route('/api/workflow/delete', methods=['POST'])
def workflow_delete():
    """Delete a workflow file."""
    data = request.json or {}
    project_name = data.get('project')
    name = data.get('name')
    if not project_name or not name:
        return jsonify({'error': '缺少 project 或 name 参数'}), 400

    workflows_dir = os.path.join(get_project_path(project_name), 'workflows')
    deleted = []
    for ext in ('.yaml', '.yml'):
        path = os.path.join(workflows_dir, f'{name}{ext}')
        if os.path.exists(path):
            os.remove(path)
            deleted.append(f'{name}{ext}')
    json_path = os.path.join(workflows_dir, f'{name}.json')
    if os.path.exists(json_path):
        os.remove(json_path)
        deleted.append(f'{name}.json')

    if deleted:
        return jsonify({'success': True, 'deleted': deleted})
    return jsonify({'error': 'Workflow 文件不存在'}), 404


@app.route('/api/workflow/deploy', methods=['POST'])
def workflow_deploy():
    """Deploy a workflow to the deploy service."""
    data = request.json or {}
    project_name = data.get('project')
    name = data.get('name')
    if not project_name or not name:
        return jsonify({'error': '缺少 project 或 name 参数'}), 400

    # Find the workflow file
    workflows_dir = os.path.join(get_project_path(project_name), 'workflows')
    workflow_path = None
    for ext in ('.yaml', '.yml'):
        candidate = os.path.join(workflows_dir, f'{name}{ext}')
        if os.path.exists(candidate):
            workflow_path = candidate
            break

    if not workflow_path:
        return jsonify({'error': 'Workflow 文件不存在'}), 404

    # Call deploy service
    deploy_url = os.environ.get('DEPLOY_SERVER_URL', 'http://127.0.0.1:8000')
    try:
        resp = requests.post(
            f'{deploy_url}/pipeline/load',
            json={'workflow_path': workflow_path},
            timeout=30,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({'error': f'部署失败: {str(e)}'}), 502


@app.route('/api/workflow/undeploy', methods=['POST'])
def workflow_undeploy():
    """Undeploy a workflow from the deploy service."""
    data = request.json or {}
    project_name = data.get('project')
    name = data.get('name', '')
    if not project_name or not name:
        return jsonify({'error': '缺少 project 或 name 参数'}), 400

    deploy_url = os.environ.get('DEPLOY_SERVER_URL', 'http://127.0.0.1:8000')
    try:
        resp = requests.post(
            f'{deploy_url}/pipeline/unload',
            json={'workflow_id': name},
            timeout=10,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        # Fallback: try legacy /unload endpoint
        try:
            resp = requests.post(
                f'{deploy_url}/unload',
                json={'engine_id': name},
                timeout=10,
            )
            resp.raise_for_status()
            return jsonify(resp.json())
        except requests.RequestException:
            return jsonify({'error': f'取消部署失败: {str(e)}'}), 502


# ========================== Project-Independent Workflow API (可单独部署) ==========================

import json as json_lib
import yaml as yaml_lib
WORKFLOW_DIR = os.environ.get('WORKFLOW_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'workflows'))


def _wf_path(name: str) -> str:
    """Get JSON path for a workflow."""
    return os.path.join(WORKFLOW_DIR, f'{name}.json')


def _wf_yaml_path(name: str) -> str:
    """Get YAML path for a workflow."""
    return os.path.join(WORKFLOW_DIR, f'{name}.yaml')


def _wf_safe_name(name: str) -> str:
    """Sanitize workflow name to a safe filesystem name."""
    safe = ''.join(c for c in name.strip() if c.isalnum() or c in '_-.').strip()
    return safe or 'untitled'


@app.route('/api/wf/list')
def wf_list():
    """List all independent workflows (project-independent)."""
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    workflows = []
    for f in sorted(os.listdir(WORKFLOW_DIR)):
        if f.endswith('.json'):
            wf_name = f[:-5]
            try:
                with open(os.path.join(WORKFLOW_DIR, f), 'r', encoding='utf-8') as fh:
                    data = json_lib.load(fh)
                node_count = len(data.get('nodes', []))
            except Exception:
                node_count = 0
            workflows.append({'name': wf_name, 'nodes': node_count, 'file': f})
    return jsonify({'workflows': workflows})


@app.route('/api/wf/get')
def wf_get():
    """Get a workflow's serialized graph JSON."""
    name = request.args.get('name', '')
    if not name:
        return jsonify({'error': '缺少 name 参数'}), 400
    path = _wf_path(name)
    if not os.path.exists(path):
        return jsonify({'error': '工作流不存在'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        graph_data = json_lib.load(f)
        _derive_vllm_conditions(graph_data)
        return jsonify({'name': name, 'content': graph_data})


def _derive_vllm_conditions(graph_data: dict):
    """Patch VLLM node properties.condition from incoming links.

    The front-end ``condition`` property may be empty when the graph was
    last saved, but the YAML derive step already reads it from link
    topology.  This function brings the JSON property in sync so the
    property panel shows the correct value.
    """
    if not graph_data or 'links' not in graph_data or 'nodes' not in graph_data:
        return
    node_map = {n.get('id'): n for n in graph_data['nodes']}
    for link in graph_data['links']:
        if isinstance(link, (list, tuple)):
            link = {'origin_id': link[1], 'target_id': link[3]}
        dst = node_map.get(link.get('target_id'))
        if dst and 'vllm' in (dst.get('type', '') or '').lower():
            src = node_map.get(link.get('origin_id'))
            if src and 'condition' in (src.get('type', '') or '').lower():
                dst.setdefault('properties', {})['condition'] = str(src['id'])


@app.route('/api/wf/save', methods=['POST'])
def wf_save():
    """Save a workflow (LiteGraph JSON + YAML, no project dependency)."""
    data = request.json or {}
    graph_data = data.get('graph')
    name = _wf_safe_name(data.get('name', 'untitled'))
    if not graph_data:
        return jsonify({'error': '缺少 graph 数据'}), 400

    os.makedirs(WORKFLOW_DIR, exist_ok=True)

    _derive_vllm_conditions(graph_data)

    # Save raw LiteGraph JSON (for re-editing)
    json_path = _wf_path(name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json_lib.dump(graph_data, f, ensure_ascii=False, indent=2)

    # Convert and save YAML (for deploy engine)
    try:
        workflow_dict = _litegraph_to_workflow(graph_data)
        workflow_dict['name'] = name
        yaml_path = _wf_yaml_path(name)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml_lib.dump(workflow_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        # YAML conversion is best-effort; JSON save always succeeds
        pass

    return jsonify({'success': True, 'name': name, 'path': json_path})


@app.route('/api/wf/yaml')
def wf_yaml():
    """Serve the workflow YAML content (for deploy service download)."""
    name = request.args.get('name', '')
    if not name:
        return jsonify({'error': '缺少 name 参数'}), 400
    # Ensure YAML exists (generate from JSON if needed)
    yaml_path = _wf_yaml_path(name)
    json_path = _wf_path(name)
    if not os.path.exists(yaml_path) and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                graph_data = json_lib.load(f)
            workflow_dict = _litegraph_to_workflow(graph_data)
            workflow_dict['name'] = name
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml_lib.dump(workflow_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception:
            return jsonify({'error': '工作流文件解析失败'}), 500
    if not os.path.exists(yaml_path):
        return jsonify({'error': '工作流不存在'}), 404
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()
    return yaml_content, 200, {'Content-Type': 'text/yaml; charset=utf-8'}


@app.route('/api/wf/delete', methods=['POST'])
def wf_delete():
    """Delete a workflow."""
    data = request.json or {}
    name = _wf_safe_name(data.get('name', ''))
    if not name:
        return jsonify({'error': '缺少 name 参数'}), 400

    deleted = []
    for path in [_wf_path(name), _wf_yaml_path(name)]:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(os.path.basename(path))

    if deleted:
        return jsonify({'success': True, 'deleted': deleted})
    return jsonify({'error': '工作流不存在'}), 404


@app.route('/api/wf/deploy', methods=['POST'])
def wf_deploy():
    """Deploy a workflow to the deploy service."""
    data = request.json or {}
    name = _wf_safe_name(data.get('name', ''))
    if not name:
        return jsonify({'error': '缺少 name 参数'}), 400

    yaml_path = _wf_yaml_path(name)
    if not os.path.exists(yaml_path):
        # Try converting from JSON
        json_path = _wf_path(name)
        if not os.path.exists(json_path):
            return jsonify({'error': '工作流不存在，请先保存'}), 404
        with open(json_path, 'r', encoding='utf-8') as f:
            graph_data = json_lib.load(f)
        workflow_dict = _litegraph_to_workflow(graph_data)
        workflow_dict['name'] = name
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml_lib.dump(workflow_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    deploy_url = os.environ.get('DEPLOY_SERVER_URL', 'http://127.0.0.1:8000')
    try:
        resp = requests.post(
            f'{deploy_url}/pipeline/load',
            json={'workflow_path': os.path.abspath(yaml_path)},
            timeout=30,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({'error': f'部署失败: {str(e)}'}), 502


@app.route('/api/wf/undeploy', methods=['POST'])
def wf_undeploy():
    """Undeploy a workflow from the deploy service."""
    data = request.json or {}
    name = _wf_safe_name(data.get('name', ''))
    if not name:
        return jsonify({'error': '缺少 name 参数'}), 400

    deploy_url = os.environ.get('DEPLOY_SERVER_URL', 'http://127.0.0.1:8000')
    try:
        resp = requests.post(
            f'{deploy_url}/pipeline/unload',
            json={'workflow_id': name},
            timeout=10,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({'error': f'取消部署失败: {str(e)}'}), 502


@app.route('/api/wf/load_model', methods=['POST'])
def wf_load_model():
    """Load a YOLO model into the local ML engine pool."""
    data = request.json or {}
    engine_id = data.get('engine_id', '')
    model_dir = data.get('model_dir', '')
    task_type = data.get('task_type', 'detect')

    if not engine_id or not model_dir:
        return jsonify({'error': '缺少 engine_id 或 model_dir 参数'}), 400

    try:
        pool = _get_ml_engine_pool()
        yolo = _get_ml_yolo_adapter()

        # Check if already loaded
        import asyncio
        existing = asyncio.run(pool.get(engine_id))
        if existing and existing.engine is not None:
            return jsonify({'engine_id': engine_id, 'status': 'already_loaded'})

        # Find model file
        model_path = None
        for ext in ['.engine', '.pt', '.onnx']:
            for f in os.listdir(model_dir):
                if f.endswith(ext):
                    model_path = os.path.join(model_dir, f)
                    break
            if model_path:
                break

        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': f'在 {model_dir} 中未找到模型文件'}), 404

        model = yolo.load_model(model_path)
        metadata = {'task_type': task_type, 'model_file': model_path}

        from deploy.engine_pool import Engine
        engine = Engine(
            engine_id=engine_id,
            engine_type='yolo_model',
            project_id=data.get('project_id', ''),
            engine=model,
            metadata=metadata,
        )
        import asyncio
        asyncio.run(pool.add(engine))

        return jsonify({'engine_id': engine_id, 'status': 'loaded', 'metadata': metadata})
    except ImportError as e:
        return jsonify({'error': f'加载模型失败: {str(e)}。需要 ultralytics 库'}), 503
    except Exception as e:
        return jsonify({'error': f'加载模型失败: {str(e)}'}), 500


@app.route('/api/wf/execute', methods=['POST'])
def wf_execute():
    """Execute a workflow locally (no separate deploy service needed)."""
    data = request.json or {}
    name = data.get('workflow_id', '')
    exec_mode = data.get('exec_mode', 'auto')  # 'auto', 'local', 'remote'
    if not name:
        return jsonify({'error': '缺少 workflow_id 参数'}), 400

    # 1. Try external deploy service (unless forced local)
    if exec_mode != 'local':
        deploy_url = os.environ.get('DEPLOY_SERVER_URL', 'http://127.0.0.1:8000')
        try:
            resp = requests.post(
                f'{deploy_url}/pipeline/execute',
                json=data,
                timeout=8,
            )
            resp.raise_for_status()
            result = resp.json()
            result['mode'] = 'remote_deploy'
            return jsonify(result)
        except requests.RequestException:
            if exec_mode == 'remote':
                return jsonify({'error': '远程 deploy 服务不可用，请检查 DEPLOY_SERVER_URL'}), 502
            pass  # fall back to local engine

    # 2. Local execution with integrated ML engine
    import yaml as _yaml
    yaml_path = _wf_yaml_path(name)
    json_path = _wf_path(name)

    if not os.path.exists(yaml_path) and not os.path.exists(json_path):
        return jsonify({'error': f'工作流 "{name}" 不存在，请先保存'}), 404

    # Ensure YAML exists
    if not os.path.exists(yaml_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                graph_data = json_lib.load(f)
            workflow_dict = _litegraph_to_workflow(graph_data)
            workflow_dict['name'] = name
            with open(yaml_path, 'w', encoding='utf-8') as f:
                _yaml.dump(workflow_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception:
            return jsonify({'error': '工作流文件解析失败'}), 500

    # Execute pipeline with integrated engine
    try:
        import io as _io, base64 as _b64
        mgr = PipelineManager(yaml_path)
        image = None
        image_warning = None
        if data.get('image'):
            try:
                image = _io.BytesIO(_b64.b64decode(data['image']))
            except Exception:
                image_warning = 'base64 图片解码失败'
        elif data.get('image_url'):
            try:
                import urllib.request
                resp = urllib.request.urlopen(data['image_url'], timeout=15)
                image = _io.BytesIO(resp.read())
            except Exception as url_err:
                image_warning = f'图片下载失败: {url_err}'

        import asyncio
        engine_pool = _get_ml_engine_pool()
        yolo = _get_ml_yolo_adapter()
        vllm = _get_ml_vllm_client()

        async def _run_with_autoload():
            """单次事件循环内完成模型加载 + 工作流执行。"""
            # 自动加载 YOLO 节点所需的模型到引擎池
            for node in mgr.config.pipeline:
                if node.type.value == 'yolo' and node.model:
                    existing = await engine_pool.get(node.model)
                    if existing and existing.engine is not None:
                        continue
                    parts = node.model.split('/', 1)
                    if len(parts) != 2:
                        continue
                    p_name, p_version = parts
                    p_dir = os.path.join(get_project_path(p_name), 'models', p_version)
                    if not os.path.isdir(p_dir):
                        continue
                    model_path = None
                    for ext in ('.engine', '.pt', '.onnx'):
                        for f in os.listdir(p_dir):
                            if f.endswith(ext) and os.path.isfile(os.path.join(p_dir, f)):
                                model_path = os.path.join(p_dir, f)
                                break
                        if model_path:
                            break
                    if not model_path:
                        continue
                    try:
                        loaded = yolo.load_model(model_path)
                        meta = {'task_type': node.task or 'detect', 'model_file': model_path}
                        eng = Engine(
                            engine_id=node.model, engine_type='yolo_model',
                            project_id=p_name, engine=loaded, metadata=meta,
                        )
                        await engine_pool.add(eng)
                        print(f'Auto-loaded model {node.model} from {model_path}')
                    except Exception as load_err:
                        print(f'Failed to load model {node.model}: {load_err}')

            return await mgr.execute(
                image=image, engine_pool=engine_pool,
                yolo_adapter=yolo, vllm_client=vllm,
            )

        result = asyncio.run(_run_with_autoload())
        result['workflow_id'] = name
        result['mode'] = 'local_engine'
        if image_warning:
            result.setdefault('warnings', []).append(image_warning)
        return jsonify(result)
    except ImportError as e:
        return jsonify({'error': f'本地执行需要 deploy 模块: {str(e)}'}), 503
    except Exception as e:
        return jsonify({'error': f'执行失败: {str(e)}', 'workflow_id': name}), 500
if __name__ == '__main__':
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='xclabel图像标注工具')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='绑定的IP地址，默认0.0.0.0')
    parser.add_argument('--port', type=int, default=5000, help='绑定的端口，默认5000')
    parser.add_argument('--debug', action='store_true', default=True, help='启用调试模式，默认开启')
    args = parser.parse_args()

    # 使用SocketIO运行应用，使用命令行参数
    socketio.run(app, debug=args.debug, host=args.host, port=args.port)


def process_content_data(content_data, annotations):
    """处理内容数据并提取标注"""
    print(f"处理内容数据: {content_data}")
    # TODO: 在这里添加您的自定义处理代码

def process_list_data(data_list, annotations):
    """处理列表数据并提取标注"""
    print(f"处理列表数据: {data_list}")
    # TODO: 在这里添加您的自定义处理代码
