# 环境变量配置说明

## 配置文件位置

### 1. 主目录 `.env` (推荐)
**位置**: `distributed-gpu-inference/.env`

**用途**: 
- Docker Compose 部署
- 统一管理所有服务的环境变量

**示例**:
```bash
# Docker Compose 部署配置
DB_PASSWORD=inference_password
SECRET_KEY=your-secret-key-min-32-chars
REGION=asia-east
WORKER_NAME=Docker-Local-Worker
```

### 2. 服务器 `.env.example` (仅作参考)
**位置**: `server/.env.example`

**用途**:
- 本地开发参考（不使用 Docker）
- 直接运行 `uvicorn app.main:app` 时使用

**使用方法**:
```bash
cd server
cp .env.example .env
# 编辑 .env
uvicorn app.main:app --reload
```

## 推荐使用方式

### Docker 部署（推荐）
只需配置主目录的 `.env` 文件：

```bash
# 1. 编辑主目录 .env
nano .env

# 2. 启动服务
docker-compose up -d
```

### 本地开发
如果不使用 Docker，需要配置 `server/.env`：

```bash
# 1. 复制示例配置
cd server
cp .env.example .env

# 2. 编辑配置
nano .env

# 3. 启动服务
uvicorn app.main:app --reload
```

## 配置项说明

### Docker Compose 配置 (主目录 .env)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DB_PASSWORD` | PostgreSQL 密码 | `inference_password` |
| `SECRET_KEY` | API 密钥 | `change-me-in-production` |
| `REGION` | 服务器区域 | `asia-east` |
| `WORKER_NAME` | Worker 名称 | `Docker-Local-Worker` |

### 本地开发配置 (server/.env)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接字符串 | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis 连接字符串 | `redis://localhost:6379/0` |
| `SECRET_KEY` | API 密钥 | `your-secret-key...` |
| `DEBUG` | 调试模式 | `false` |
| `REGION` | 服务器区域 | `asia-east` |

## 最佳实践

### 生产环境
1. 只使用主目录 `.env`
2. 使用 Docker Compose 部署
3. 修改所有默认密码和密钥

### 开发环境
1. 使用 Docker: 配置主目录 `.env`
2. 不使用 Docker: 配置 `server/.env`

### 安全建议
- ✅ 将 `.env` 添加到 `.gitignore`
- ✅ 使用强密码（至少16字符）
- ✅ 定期轮换密钥
- ✅ 不要提交 `.env` 到版本控制

## 配置优先级

1. **Docker Compose**: 主目录 `.env` → `docker-compose.yml` 环境变量
2. **本地开发**: `server/.env` → `server/app/config.py` 默认值

## 常见问题

### Q: 为什么有两个 .env 文件？
A: 
- 主目录 `.env`: Docker 部署用
- `server/.env`: 本地开发用（可选）

### Q: 我应该用哪个？
A: 
- 使用 Docker → 主目录 `.env`
- 不使用 Docker → `server/.env`

### Q: 两个文件内容要一致吗？
A: 不需要。它们服务于不同的部署方式。
