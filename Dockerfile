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

# 创建运行时数据目录并授权
RUN mkdir -p uploads plugins static/annotations \
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
