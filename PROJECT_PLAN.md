# 全球分布式GPU推理平台

## 项目概述

整合全球各地闲置GPU资源，构建统一的AI推理服务平台。

```
                        ┌─────────────────────┐
                        │   中央调度服务器     │
                        │   (云服务器部署)     │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  亚洲节点      │          │  欧洲节点      │          │  美洲节点      │
│  Worker集群   │          │  Worker集群   │          │  Worker集群   │
│  ───────────  │          │  ───────────  │          │  ───────────  │
│  - 中国       │          │  - 德国       │          │  - 美国       │
│  - 日本       │          │  - 英国       │          │  - 加拿大     │
│  - 新加坡     │          │  - 法国       │          │  - 巴西       │
└───────────────┘          └───────────────┘          └───────────────┘
```

---

## 核心特性

| 特性 | 描述 |
|------|------|
| **全球调度** | 就近分配任务，降低延迟 |
| **异构支持** | 支持不同型号GPU（RTX/A100/等） |
| **动态扩缩** | Worker随时加入/退出 |
| **容错机制** | 任务失败自动重试 |
| **安全传输** | TLS加密 + Token认证 |

---

## 技术架构

### 1. 服务端 (Server)

```
server/
├── app/
│   ├── main.py              # FastAPI入口
│   ├── config.py            # 配置管理
│   ├── api/
│   │   ├── jobs.py          # 任务接口
│   │   ├── workers.py       # Worker管理
│   │   └── regions.py       # 区域管理
│   ├── models/
│   │   ├── job.py           # 任务模型
│   │   ├── worker.py        # Worker模型
│   │   └── region.py        # 区域模型
│   ├── services/
│   │   ├── scheduler.py     # 智能调度器
│   │   ├── monitor.py       # 监控服务
│   │   └── geo_router.py    # 地理路由
│   └── db/
│       └── database.py      # 数据库连接
├── requirements.txt
└── Dockerfile
```

**技术选型：**
- 框架：FastAPI (Python 3.11+)
- 数据库：PostgreSQL + Redis
- 部署：Docker + 云服务器

### 2. Worker端 (Worker)

```
worker/
├── main.py                  # Worker入口
├── config.py                # 配置
├── api_client.py            # 与服务器通信
├── engines/
│   ├── base.py              # 引擎基类
│   ├── llm.py               # LLM推理
│   ├── image_gen.py         # 图像生成
│   └── whisper.py           # 语音识别
├── requirements.txt
└── Dockerfile
```

**技术选型：**
- PyTorch 2.0+
- Transformers / Diffusers
- 支持CPU Offload

### 3. 客户端SDK

```
sdk/python/
├── inference_client.py      # Python SDK
└── examples/
    ├── llm_chat.py
    └── image_gen.py
```

---

## 数据库设计

### 任务表 (jobs)

```sql
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,           -- 任务类型: llm, image_gen, whisper
    status VARCHAR(20) DEFAULT 'queued', -- queued, running, completed, failed
    priority INTEGER DEFAULT 0,
    region VARCHAR(20),                  -- 指定区域(可选)

    params JSONB NOT NULL,               -- 任务参数
    result JSONB,                        -- 执行结果
    error TEXT,                          -- 错误信息

    worker_id UUID REFERENCES workers(id),

    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- 来源追踪
    client_id VARCHAR(100),
    client_ip INET
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_region ON jobs(region);
CREATE INDEX idx_jobs_priority ON jobs(priority DESC, created_at ASC);
```

### Worker表 (workers)

```sql
CREATE TABLE workers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'offline', -- online, busy, offline

    -- 地理信息
    region VARCHAR(20) NOT NULL,          -- asia, europe, america
    country VARCHAR(50),
    city VARCHAR(50),

    -- GPU信息
    gpu_model VARCHAR(100),
    gpu_memory_gb FLOAT,
    supported_types TEXT[],               -- ['llm', 'image_gen']

    -- 性能指标
    avg_latency_ms INTEGER,
    success_rate FLOAT DEFAULT 1.0,
    total_jobs INTEGER DEFAULT 0,

    last_heartbeat TIMESTAMP,
    registered_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_workers_status ON workers(status);
CREATE INDEX idx_workers_region ON workers(region);
```

---

## API设计

### 客户端API

```yaml
# 提交任务
POST /api/v1/jobs
Request:
  type: "llm"
  params:
    messages: [{"role": "user", "content": "Hello"}]
    max_tokens: 1000
  region: "asia"  # 可选，不指定则自动路由
Response:
  job_id: "uuid"
  status: "queued"
  estimated_wait: 5  # 预估等待秒数

# 查询结果
GET /api/v1/jobs/{job_id}
Response:
  status: "completed"
  result:
    response: "Hello! How can I help you?"
    usage: {prompt_tokens: 10, completion_tokens: 20}

# 同步调用（等待结果）
POST /api/v1/jobs/sync
Request: (同上)
Response: (直接返回结果，超时60秒)
```

### Worker API

```yaml
# 心跳
POST /api/v1/workers/{worker_id}/heartbeat
Request:
  status: "online"
  gpu_info: {memory_used: 4.5, memory_total: 24}
  supported_types: ["llm", "image_gen"]

# 获取任务
GET /api/v1/workers/{worker_id}/next-job
Response:
  job_id: "uuid"
  type: "llm"
  params: {...}

# 提交结果
POST /api/v1/jobs/{job_id}/complete
Request:
  success: true
  result: {...}
  processing_time: 2.5
```

---

## 智能调度策略

### 1. 地理就近原则

```python
def select_worker(job, available_workers):
    """选择最优Worker"""

    # 1. 按区域过滤
    if job.region:
        workers = [w for w in available_workers if w.region == job.region]
    else:
        # 根据客户端IP选择最近区域
        client_region = get_region_by_ip(job.client_ip)
        workers = sorted(available_workers,
                        key=lambda w: region_distance(w.region, client_region))

    # 2. 按能力过滤
    workers = [w for w in workers if job.type in w.supported_types]

    # 3. 按性能排序
    workers = sorted(workers, key=lambda w: (
        -w.success_rate,      # 成功率高优先
        w.avg_latency_ms,     # 延迟低优先
        -w.gpu_memory_gb      # 显存大优先
    ))

    return workers[0] if workers else None
```

### 2. 负载均衡

```python
# 基于权重的任务分配
def calculate_worker_weight(worker):
    """计算Worker权重，权重越高越可能被分配任务"""

    base_weight = 100

    # GPU性能加成
    if worker.gpu_memory_gb >= 24:
        base_weight += 50
    elif worker.gpu_memory_gb >= 12:
        base_weight += 20

    # 成功率加成
    base_weight *= worker.success_rate

    # 当前负载惩罚
    if worker.status == "busy":
        base_weight *= 0.1

    return base_weight
```

---

## 安全设计

### 1. Worker认证

```python
# Worker注册时生成Token
@router.post("/workers/register")
async def register_worker(payload: WorkerRegisterPayload):
    worker_id = uuid.uuid4()
    token = generate_secure_token(worker_id)  # JWT or API Key

    # 存储Worker信息
    await db.execute(
        insert(Worker).values(
            id=worker_id,
            name=payload.name,
            region=payload.region,
            gpu_model=payload.gpu_model,
            auth_token_hash=hash_token(token)
        )
    )

    return {"worker_id": worker_id, "token": token}

# 所有Worker请求需携带Token
@router.get("/workers/{worker_id}/next-job")
async def get_next_job(
    worker_id: str,
    token: str = Header(..., alias="X-Worker-Token")
):
    if not verify_token(worker_id, token):
        raise HTTPException(401, "Invalid token")
    ...
```

### 2. 传输加密

```yaml
# 所有通信强制HTTPS
# Worker配置示例
server:
  url: "https://api.your-domain.com"
  verify_ssl: true

# 敏感数据额外加密
encryption:
  enabled: true
  algorithm: "AES-256-GCM"
```

### 3. 任务隔离

```python
# Worker在隔离环境执行任务
class SecureExecutor:
    def execute(self, job):
        # 资源限制
        resource_limits = {
            "max_memory_gb": 32,
            "max_time_seconds": 300,
            "max_output_size_mb": 100
        }

        # 在子进程中执行
        with ProcessPoolExecutor() as executor:
            future = executor.submit(
                self._run_inference,
                job.params,
                resource_limits
            )
            result = future.result(timeout=resource_limits["max_time_seconds"])

        return result
```

---

## 实施计划

### Phase 1: 核心框架 (Week 1-2)

- [ ] 服务端基础框架
  - [ ] FastAPI项目结构
  - [ ] 数据库模型和迁移
  - [ ] 任务CRUD API
  - [ ] Worker管理API

- [ ] Worker基础框架
  - [ ] 心跳机制
  - [ ] 任务拉取和执行
  - [ ] 结果上报

### Phase 2: 推理引擎 (Week 3-4)

- [ ] LLM推理引擎
  - [ ] 支持Qwen/Llama等模型
  - [ ] CPU Offload支持
  - [ ] 流式输出

- [ ] 图像生成引擎
  - [ ] 支持FLUX/SD模型
  - [ ] 多种尺寸支持

### Phase 3: 分布式特性 (Week 5-6)

- [ ] 智能调度
  - [ ] 地理路由
  - [ ] 负载均衡
  - [ ] 任务优先级

- [ ] 容错机制
  - [ ] 心跳超时检测
  - [ ] 任务失败重试
  - [ ] Worker故障转移

### Phase 4: 生产化 (Week 7-8)

- [ ] 安全加固
  - [ ] Token认证
  - [ ] TLS配置
  - [ ] 速率限制

- [ ] 监控运维
  - [ ] 指标收集
  - [ ] 日志聚合
  - [ ] 告警配置

- [ ] 部署上线
  - [ ] Docker镜像
  - [ ] CI/CD流程
  - [ ] 文档完善

---

## 部署架构

```
                         ┌─────────────────────────────┐
                         │       云服务商 (推荐)        │
                         │  - AWS / Azure / 阿里云     │
                         │  - 选择中心位置的区域        │
                         └─────────────┬───────────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │      中央服务器              │
                         │  ┌─────────────────────┐   │
                         │  │   Nginx (反向代理)   │   │
                         │  └──────────┬──────────┘   │
                         │             │              │
                         │  ┌──────────▼──────────┐   │
                         │  │   FastAPI Server    │   │
                         │  └──────────┬──────────┘   │
                         │             │              │
                         │  ┌──────────▼──────────┐   │
                         │  │ PostgreSQL + Redis  │   │
                         │  └─────────────────────┘   │
                         └─────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
   ┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
   │   家庭电脑 A     │       │   家庭电脑 B     │       │   云GPU实例     │
   │   RTX 4090      │       │   RTX 3080      │       │   A100          │
   │   ──────────    │       │   ──────────    │       │   ──────────    │
   │   Worker进程    │       │   Worker进程    │       │   Worker进程    │
   │   (Docker)      │       │   (Docker)      │       │   (Docker)      │
   └─────────────────┘       └─────────────────┘       └─────────────────┘
          中国                      日本                      美国
```

---

## 快速开始

### 1. 启动服务端

```bash
cd server
cp .env.example .env
# 编辑 .env 配置数据库等

docker-compose up -d
```

### 2. 注册Worker

```bash
cd worker
cp config.example.yaml config.yaml
# 编辑 config.yaml

# 首次运行会自动注册
python main.py
```

### 3. 提交任务

```python
from sdk.python import InferenceClient

client = InferenceClient("https://api.your-domain.com", api_key="xxx")

# LLM推理
result = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    model="qwen2.5-7b"
)
print(result.response)

# 图像生成
image = client.generate_image(
    prompt="a cute cat",
    size="1024x1024"
)
image.save("output.png")
```

---

## 下一步

1. 确认技术选型是否符合需求
2. 开始实现服务端核心代码
3. 实现Worker端核心代码
4. 集成测试
