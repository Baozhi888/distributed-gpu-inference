FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# 安装Python和系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建缓存目录
RUN mkdir -p /root/.cache/huggingface

# 暴露直连端口
EXPOSE 8080

# 启动命令
CMD ["python", "main.py"]
