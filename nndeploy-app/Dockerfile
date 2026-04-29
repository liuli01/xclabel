# ============================================
# xclabel-deploy-workflows Workflow Execution Container
# ============================================
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update \&\& apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    \&\& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install nndeploy only
COPY deploy_workflows/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Create resources directories
RUN mkdir -p /app/resources/{models,workflow,images,videos,audios,db,others}

# Run as non-root user
RUN useradd -m -u 1000 deployuser \&\& \
    chown -R deployuser:deployuser /app
USER deployuser

EXPOSE 8002

CMD ["nndeploy-app", "--port", "8002", "--resources", "/app/resources"]
