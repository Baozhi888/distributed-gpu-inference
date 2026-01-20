# 分布式GPU推理平台 - API使用与模型部署指南

## 目录
1. [推理服务API](#推理服务api)
2. [Worker部署指南](#worker部署指南)
3. [开源模型配置](#开源模型配置)
4. [完整示例](#完整示例)

---

## 推理服务API

### 基础信息
- **基础URL**: `http://your-server:8000/api/v1/jobs`
- **认证方式**: API Key (Header: `X-API-Key`)
- **内容类型**: `application/json`

---

### 1. 创建异步任务

创建推理任务并立即返回，适用于长时间运行的任务。

```http
POST /api/v1/jobs
```

**请求参数**:
```json
{
  "type": "llm",                    // 任务类型: llm, image_gen, vision, whisper
  "params": {                       // 任务参数（见下方各类型详情）
    "messages": [...],
    "max_tokens": 2048
  },
  "priority": 0,                    // 优先级 (越高越优先)
  "region": "asia-east",            // 首选区域 (可选)
  "allow_cross_region": true,       // 是否允许跨区域执行
  "timeout_seconds": 300            // 超时时间
}
```

**响应**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2024-12-18T10:00:00Z",
  "queue_position": 3,
  "estimated_wait_seconds": 15
}
```

---

### 2. 创建同步任务（等待结果）

创建任务并等待执行完成返回结果。

```http
POST /api/v1/jobs/sync?timeout=60&wait_for_worker=true
```

**请求参数**: 同异步任务

**响应**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "response": "你好！有什么我可以帮助你的吗？",
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 15,
      "total_tokens": 25
    }
  },
  "region": "asia-east",
  "worker_id": "worker-123",
  "created_at": "2024-12-18T10:00:00Z",
  "started_at": "2024-12-18T10:00:01Z",
  "completed_at": "2024-12-18T10:00:03Z"
}
```

---

### 3. 查询任务状态

```http
GET /api/v1/jobs/{job_id}
```

**响应**: 同同步任务响应结构

**状态值**:
- `queued`: 排队中
- `running`: 执行中
- `completed`: 已完成
- `failed`: 失败
- `cancelled`: 已取消
- `timeout`: 超时

---

### 4. 取消任务

```http
DELETE /api/v1/jobs/{job_id}
```

**响应**:
```json
{
  "message": "Job cancelled",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### 5. 获取队列状态

```http
GET /api/v1/jobs/stats/queue?region=asia-east
```

**响应**:
```json
{
  "total_queued": 5,
  "available_workers": 3,
  "estimated_wait_seconds": 10,
  "by_type": {
    "llm": 3,
    "image_gen": 2
  }
}
```

---

### 6. 获取最近的直连Worker

```http
GET /api/v1/jobs/direct/nearest?job_type=llm
```

**响应**:
```json
{
  "worker_id": "worker-123",
  "direct_url": "http://192.168.1.100:8080",
  "region": "asia-east",
  "client_region": "asia-east",
  "gpu_model": "NVIDIA RTX 4090",
  "reliability_score": 0.95
}
```

---

## 各任务类型参数详情

### LLM 文本生成

```json
{
  "type": "llm",
  "params": {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "max_tokens": 2048,      // 最大生成Token数
    "temperature": 0.7,      // 温度 (0-2)
    "top_p": 0.9             // Top-p采样
  }
}
```

**响应结果**:
```json
{
  "response": "你好！我是一个AI助手...",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 150,
    "total_tokens": 170
  }
}
```

---

### 图像生成 (Z-Image-Turbo)

```json
{
  "type": "image_gen",
  "params": {
    "prompt": "a beautiful sunset over the ocean, highly detailed",
    "negative_prompt": "blurry, low quality",  // 负面提示词
    "width": 1024,           // 宽度
    "height": 1024,          // 高度
    "steps": 4,              // 推理步数
    "seed": 12345            // 随机种子 (可选，用于复现)
  }
}
```

**响应结果**:
```json
{
  "image_base64": "iVBORw0KGgo...",  // Base64编码的PNG图片
  "width": 1024,
  "height": 1024,
  "seed": 12345,
  "format": "png"
}
```

---

### 图像识别/视觉问答 (GLM-4V)

```json
{
  "type": "vision",
  "params": {
    "image_base64": "iVBORw0KGgo...",  // Base64编码的图片
    "task": "image_qa",                  // 任务类型: image_qa, image_caption, ocr
    "question": "这张图片里有什么？",    // 问题 (image_qa时使用)
    "max_tokens": 1024                   // 最大生成Token数
  }
}
```

**任务类型说明**:
- `image_qa`: 图像问答 - 根据图片回答问题
- `image_caption`: 图像描述 - 自动生成图片详细描述
- `ocr`: 文字识别 - 识别图片中的文字

**响应结果**:
```json
{
  "response": "这张图片显示的是一只橙色的猫正在阳台上晒太阳...",
  "task": "image_qa",
  "usage": {
    "prompt_tokens": 256,
    "completion_tokens": 50,
    "total_tokens": 306
  }
}
```

---

### 语音识别 (Whisper)

```json
{
  "type": "whisper",
  "params": {
    "audio_base64": "UklGRi...",    // Base64编码的音频
    "language": "zh",                // 语言 (可选，自动检测)
    "task": "transcribe"             // transcribe或translate
  }
}
```

**响应结果**:
```json
{
  "text": "这是识别出来的文本内容",
  "language": "zh",
  "duration": 10.5,
  "segments": [...]
}
```

---

## Worker部署指南

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | GTX 1080 (8GB) | RTX 4090 (24GB) |
| RAM | 16GB | 32GB |
| 存储 | 50GB SSD | 200GB NVMe |
| Python | 3.10+ | 3.11 |
| CUDA | 11.8+ | 12.1 |

### 快速部署

#### 方式一：使用 Docker (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/your-org/distributed-gpu-inference.git
cd distributed-gpu-inference/worker

# 2. 创建配置文件
cp config.example.yaml config.yaml

# 3. 编辑配置
nano config.yaml

# 4. 使用 Docker 运行
docker build -t gpu-worker .
docker run --gpus all -v $(pwd)/config.yaml:/app/config.yaml gpu-worker
```

#### 方式二：本地安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量 (或使用config.yaml)
export GPU_SERVER_URL="http://your-server:8000"
export GPU_REGION="asia-east"
export GPU_SUPPORTED_TYPES="llm,image_gen"
export GPU_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"

# 4. 启动Worker
python main.py
```

---

### 配置文件详解

创建 `config.yaml`:

```yaml
# Worker基本信息
name: "My-GPU-Worker"         # Worker名称
region: "asia-east"           # 区域: asia-east, asia-south, europe-west, america-north
country: "China"              # 国家
city: "Shanghai"              # 城市

# 服务器连接
server:
  url: "http://your-server:8000"   # 调度服务器地址
  timeout: 30                       # 请求超时时间(秒)
  verify_ssl: true                  # 是否验证SSL证书

# GPU配置
gpu:
  enable_cpu_offload: true    # 启用CPU内存卸载 (显存不足时自动使用)
  max_memory_gb: null         # 最大显存限制 (null=自动)
  device_id: 0                # 使用的GPU编号

# 支持的任务类型
supported_types:
  - llm
  - image_gen
  - vision
  # - whisper
  # - embedding

# 引擎配置 (各任务类型的模型)
engines:
  llm:
    model_id: "Qwen/Qwen2.5-7B-Instruct"
  image_gen:
    model_id: "Zhihu-ai/Z-Image-Turbo"
  vision:
    model_id: "THUDM/glm-4v-9b"

# 轮询配置
heartbeat_interval: 30        # 心跳间隔(秒)
poll_interval: 2              # 任务轮询间隔(秒)

# 直连模式 (可选，提升性能)
direct:
  enabled: false              # 是否启用直连
  host: "0.0.0.0"
  port: 8080
  public_url: null            # 公网可访问的URL

# 负载控制
load_control:
  acceptance_rate: 1.0        # 接受任务的比例 (0-1)
  max_concurrent_jobs: 1      # 最大并发任务数
  working_hours_start: null   # 工作时间开始 (0-23)
  working_hours_end: null     # 工作时间结束 (0-23)
```

---

### 使用环境变量配置

所有配置项均可通过环境变量设置，优先级：环境变量 > config.yaml > 默认值

```bash
# .env 文件或环境变量

# 服务器
GPU_SERVER_URL=http://your-server:8000
GPU_SERVER_TIMEOUT=30

# 区域
GPU_REGION=asia-east
GPU_COUNTRY=China
GPU_CITY=Shanghai

# 任务类型
GPU_SUPPORTED_TYPES=llm,image_gen,vision

# 模型配置
GPU_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
GPU_IMAGE_MODEL=Zhihu-ai/Z-Image-Turbo
GPU_VISION_MODEL=THUDM/glm-4v-9b

# GPU配置
GPU_ENABLE_CPU_OFFLOAD=true
GPU_DEVICE_ID=0

# 负载控制
GPU_ACCEPTANCE_RATE=1.0
GPU_WORKING_HOURS_START=9
GPU_WORKING_HOURS_END=22
```

---

## 开源模型配置

### 支持的LLM模型

| 模型 | 显存要求 | model_id |
|------|---------|----------|
| Qwen2.5-7B-Instruct | 16GB | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen2.5-14B-Instruct | 32GB | `Qwen/Qwen2.5-14B-Instruct` |
| Qwen2.5-72B-Instruct | 4xA100 | `Qwen/Qwen2.5-72B-Instruct` |
| Llama-3.1-8B-Instruct | 18GB | `meta-llama/Llama-3.1-8B-Instruct` |
| Llama-3.1-70B-Instruct | 4xA100 | `meta-llama/Llama-3.1-70B-Instruct` |
| Yi-34B-Chat | 48GB | `01-ai/Yi-34B-Chat` |
| DeepSeek-V2-Lite | 24GB | `deepseek-ai/DeepSeek-V2-Lite` |
| GLM-4-9B-Chat | 20GB | `THUDM/glm-4-9b-chat` |

### 支持的图像生成模型

| 模型 | 显存要求 | model_id | 说明 |
|------|---------|----------|------|
| **Z-Image-Turbo** | 16GB | `Zhihu-ai/Z-Image-Turbo` | **推荐** 知乎高速图像生成 |
| FLUX.1-schnell | 24GB | `black-forest-labs/FLUX.1-schnell` | 高质量快速生成 |
| FLUX.1-dev | 32GB | `black-forest-labs/FLUX.1-dev` | 开发版本 |
| Stable Diffusion XL | 12GB | `stabilityai/stable-diffusion-xl-base-1.0` | 经典模型 |
| Stable Diffusion 3 | 16GB | `stabilityai/stable-diffusion-3-medium` | SD3中等版本 |

### 支持的视觉理解模型

| 模型 | 显存要求 | model_id | 说明 |
|------|---------|----------|------|
| **GLM-4V-9B** | 20GB | `THUDM/glm-4v-9b` | **推荐** 图像问答/OCR/描述 |
| Qwen2-VL-7B | 18GB | `Qwen/Qwen2-VL-7B-Instruct` | 通义千问视觉版 |
| InternVL2-8B | 18GB | `OpenGVLab/InternVL2-8B` | 书生视觉大模型 |
| LLaVA-1.6-34B | 48GB | `liuhaotian/llava-v1.6-34b` | 高精度视觉模型 |

### 支持的语音识别模型

| 模型 | 显存要求 | model_id |
|------|---------|----------|
| Whisper Large V3 | 10GB | `openai/whisper-large-v3` |
| Whisper Medium | 5GB | `openai/whisper-medium` |
| Whisper Small | 2GB | `openai/whisper-small` |

### 支持的Embedding模型

| 模型 | 显存要求 | model_id |
|------|---------|----------|
| BGE-Large-ZH | 2GB | `BAAI/bge-large-zh-v1.5` |
| BGE-M3 | 3GB | `BAAI/bge-m3` |
| E5-Large-V2 | 2GB | `intfloat/e5-large-v2` |

---

## 完整示例

### Python SDK 示例

```python
import requests
import base64
import time

class GPUInferenceClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers['X-API-Key'] = api_key

    def chat(self, messages: list, **kwargs) -> dict:
        """LLM对话"""
        response = self.session.post(
            f"{self.base_url}/api/v1/jobs/sync",
            json={
                "type": "llm",
                "params": {
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 2048),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            },
            params={"timeout": kwargs.get("timeout", 60)}
        )
        response.raise_for_status()
        return response.json()

    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """图像生成"""
        response = self.session.post(
            f"{self.base_url}/api/v1/jobs/sync",
            json={
                "type": "image_gen",
                "params": {
                    "prompt": prompt,
                    "negative_prompt": kwargs.get("negative_prompt", ""),
                    "width": kwargs.get("width", 1024),
                    "height": kwargs.get("height", 1024),
                    "steps": kwargs.get("steps", 4)
                }
            },
            params={"timeout": kwargs.get("timeout", 120)}
        )
        response.raise_for_status()
        result = response.json()
        return base64.b64decode(result["result"]["image_base64"])

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """语音识别"""
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()

        response = self.session.post(
            f"{self.base_url}/api/v1/jobs/sync",
            json={
                "type": "whisper",
                "params": {
                    "audio_base64": audio_base64,
                    "language": kwargs.get("language")
                }
            },
            params={"timeout": kwargs.get("timeout", 120)}
        )
        response.raise_for_status()
        return response.json()["result"]["text"]

    def analyze_image(self, image_path: str, question: str = None, task: str = "image_qa", **kwargs) -> str:
        """图像识别/视觉问答"""
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()

        params = {
            "image_base64": image_base64,
            "task": task,
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        if question:
            params["question"] = question

        response = self.session.post(
            f"{self.base_url}/api/v1/jobs/sync",
            json={"type": "vision", "params": params},
            params={"timeout": kwargs.get("timeout", 60)}
        )
        response.raise_for_status()
        return response.json()["result"]["response"]


# 使用示例
if __name__ == "__main__":
    client = GPUInferenceClient("http://localhost:8000")

    # LLM对话
    result = client.chat([
        {"role": "user", "content": "请用Python写一个快速排序算法"}
    ])
    print(result["result"]["response"])

    # 图像生成 (Z-Image-Turbo)
    image_data = client.generate_image(
        "a cute cat sitting on a desk, digital art",
        width=1024,
        height=1024
    )
    with open("output.png", "wb") as f:
        f.write(image_data)
    print("Image saved to output.png")

    # 图像识别/问答 (GLM-4V)
    answer = client.analyze_image("photo.jpg", "这张图片里有什么？")
    print(f"图像分析: {answer}")

    # OCR文字识别
    text = client.analyze_image("document.png", task="ocr")
    print(f"识别文字: {text}")
```

### cURL 示例

```bash
# LLM对话
curl -X POST http://localhost:8000/api/v1/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{
    "type": "llm",
    "params": {
      "messages": [
        {"role": "user", "content": "你好"}
      ],
      "max_tokens": 100
    }
  }'

# 图像生成 (Z-Image-Turbo)
curl -X POST http://localhost:8000/api/v1/jobs/sync?timeout=120 \
  -H "Content-Type: application/json" \
  -d '{
    "type": "image_gen",
    "params": {
      "prompt": "a beautiful landscape",
      "width": 1024,
      "height": 1024
    }
  }'

# 图像识别/问答 (GLM-4V)
curl -X POST http://localhost:8000/api/v1/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{
    "type": "vision",
    "params": {
      "image_base64": "'$(base64 -w0 photo.jpg)'",
      "task": "image_qa",
      "question": "这张图片里有什么？"
    }
  }'

# OCR文字识别
curl -X POST http://localhost:8000/api/v1/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{
    "type": "vision",
    "params": {
      "image_base64": "'$(base64 -w0 document.png)'",
      "task": "ocr"
    }
  }'

# 查询队列状态
curl http://localhost:8000/api/v1/jobs/stats/queue
```

---

## 故障排查

### 常见问题

**1. Worker无法连接服务器**
```bash
# 检查网络连接
curl http://your-server:8000/health

# 检查防火墙
sudo ufw allow 8000
```

**2. GPU内存不足**
```yaml
# 启用CPU Offload
gpu:
  enable_cpu_offload: true
```

**3. 模型加载失败**
```bash
# 手动下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 设置镜像源（中国用户）
export HF_ENDPOINT=https://hf-mirror.com
```

**4. 任务超时**
```python
# 增加超时时间
client.chat(messages, timeout=120)
```

---

## 联系与支持

- **问题反馈**: GitHub Issues
- **技术文档**: `/docs` 目录
- **API文档**: `http://your-server:8000/docs` (Swagger UI)
