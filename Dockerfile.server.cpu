# ============================================================
# Stage 1: Builder - 使用 uv 构建 Python 虚拟环境
# ============================================================
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed

# 安装指定 Python 版本（缓存层）
RUN uv python install 3.12

# 安装系统 Python（用于创建 YOLO venv）
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 仅复制依赖定义文件，利用 Docker 缓存
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# 复制全部源码后再次同步（包含本地包自身）
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
RUN uv add opencv-python-headless

# 创建 YOLO 插件虚拟环境并安装对应版本 ultralytics
RUN python3 -m venv plugins/yolo8/venv \
    && plugins/yolo8/venv/bin/pip install --upgrade pip \
    && plugins/yolo8/venv/bin/pip install ultralytics==8.0.196 \
    && echo '{"is_installed": true, "has_cuda": false, "hardware": "CPU"}' > plugins/yolo8/install_info.json \
    && mkdir -p plugins/yolo8/models

RUN python3 -m venv plugins/yolo11/venv \
    && plugins/yolo11/venv/bin/pip install --upgrade pip \
    && plugins/yolo11/venv/bin/pip install ultralytics==8.4.41 \
    && echo '{"is_installed": true, "has_cuda": false, "hardware": "CPU"}' > plugins/yolo11/install_info.json \
    && mkdir -p plugins/yolo11/models

RUN python3 -m venv plugins/yolo26/venv \
    && plugins/yolo26/venv/bin/pip install --upgrade pip \
    && plugins/yolo26/venv/bin/pip install "ultralytics>=8.4.0" \
    && echo '{"is_installed": true, "has_cuda": false, "hardware": "CPU"}' > plugins/yolo26/install_info.json \
    && mkdir -p plugins/yolo26/models

# 下载 yolo8n/yolo11n 预训练模型（ultralytics 自动下载到 ~/.ultralytics/）
RUN plugins/yolo8/venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && cp /root/.ultralytics/models/yolov8n.pt plugins/yolo8/models/ \
    && plugins/yolo11/venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && cp /root/.ultralytics/models/yolo11n.pt plugins/yolo11/models/ \
    && plugins/yolo26/venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" 2>/dev/null \
    && cp /root/.ultralytics/models/yolo26n.pt plugins/yolo26/models/ 2>/dev/null || echo "yolo26n.pt will use local file if present"

# ============================================================
# Stage 2: Runtime - 最小化运行时镜像
# ============================================================
FROM debian:bookworm-slim

# 安装 OpenCV 运行时系统依赖（不使用 root 用户）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# 从 builder 阶段复制 Python 和虚拟环境
COPY --from=builder --chown=appuser:appgroup /python /python
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# 复制应用源码
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup AiUtils.py .
COPY --chown=appuser:appgroup templates/ ./templates/
COPY --chown=appuser:appgroup static/ ./static/

# 复制 YOLO 插件（包含 venv 和预训练模型）
COPY --from=builder --chown=appuser:appgroup /app/plugins/ ./plugins/

# 创建运行时数据目录并授权
RUN mkdir -p uploads static/annotations \
    && chown -R appuser:appgroup uploads plugins static/annotations

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=false

# 切换到非 root 用户
USER appuser

# 暴露服务端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
