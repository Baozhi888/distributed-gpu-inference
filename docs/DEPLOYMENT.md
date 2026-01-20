# 服务端部署指南

## 快速开始

### 0. 一键部署脚本（可选）

```bash
# Linux/macOS
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

```cmd
:: Windows
scripts\\deploy.bat
```

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
nano .env
```

### 2. 数据库初始化

```bash
# 初始化 Alembic
alembic revision --autogenerate -m "Initial migration"

# 执行迁移
alembic upgrade head
```

### 3. 启动服务

```bash
# 开发模式
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Docker 部署

### 单机部署

```bash
# 构建镜像
docker build -t gpu-inference-server .

# 启动服务（包含数据库）
docker-compose up -d
```

### 多区域部署

每个区域部署一个独立的服务器实例：

```bash
# 区域1: 东亚
export REGION=asia-east
export DB_PASSWORD=your_password
export SECRET_KEY=your_secret_key
docker-compose up -d

# 区域2: 欧洲
export REGION=europe-west
docker-compose -f docker-compose.yml up -d
```

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | PostgreSQL 连接字符串 | - |
| `REDIS_URL` | Redis 连接字符串 | - |
| `SECRET_KEY` | 加密密钥 | - |
| `REGION` | 服务器所在区域 | `asia-east` |
| `DEBUG` | 调试模式 | `false` |
| `HEARTBEAT_TIMEOUT_SECONDS` | Worker 心跳超时 | `90` |
| `JOB_TIMEOUT_SECONDS` | 任务执行超时 | `300` |

### 区域配置

支持的区域代码：
- `asia-east` - 东亚
- `asia-south` - 东南亚
- `europe-west` - 西欧
- `europe-east` - 东欧
- `america-north` - 北美
- `america-south` - 南美
- `oceania` - 大洋洲

## API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 管理后台: http://localhost:8000/admin

## 监控

### Prometheus 指标

```bash
# 访问指标端点
curl http://localhost:8000/metrics
```

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8000/health

# 详细健康检查
curl http://localhost:8000/api/v1/admin/health/detailed
```

## 数据库管理

### 创建迁移

```bash
# 自动生成迁移脚本
alembic revision --autogenerate -m "描述变更"

# 手动创建迁移
alembic revision -m "描述变更"
```

### 执行迁移

```bash
# 升级到最新版本
alembic upgrade head

# 升级到指定版本
alembic upgrade <revision>

# 回滚一个版本
alembic downgrade -1

# 查看当前版本
alembic current

# 查看历史
alembic history
```

### 数据库备份

```bash
# 备份
pg_dump -h localhost -U inference inference > backup.sql

# 恢复
psql -h localhost -U inference inference < backup.sql
```

## 性能优化

### 数据库连接池

```python
# config.py
database_url = "postgresql+asyncpg://user:pass@host/db?min_size=10&max_size=20"
```

### Redis 缓存

启用 Redis 可以提升性能：
- Worker 状态缓存
- 队列统计缓存
- 会话管理

### 负载均衡

使用 Nginx 进行负载均衡：

```nginx
upstream gpu_inference {
    server server1:8000;
    server server2:8000;
    server server3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://gpu_inference;
    }
}
```

## 故障排查

### 常见问题

1. **数据库连接失败**
   ```bash
   # 检查数据库是否运行
   docker-compose ps postgres
   
   # 查看日志
   docker-compose logs postgres
   ```

2. **Worker 心跳超时**
   - 检查网络连接
   - 增加 `HEARTBEAT_TIMEOUT_SECONDS`
   - 查看 Worker 日志

3. **任务堆积**
   - 检查可用 Worker 数量
   - 查看任务失败原因
   - 调整调度策略

### 日志查看

```bash
# Docker 日志
docker-compose logs -f server

# 应用日志
tail -f /var/log/gpu-inference/app.log
```

## 安全建议

1. **生产环境必须修改**
   - `SECRET_KEY`
   - `DB_PASSWORD`
   - 数据库默认端口

2. **启用 HTTPS**
   ```bash
   # 使用 Let's Encrypt
   certbot --nginx -d your-domain.com
   ```

3. **限制访问**
   - 配置防火墙
   - 使用 VPN
   - 启用 API Key 认证

4. **定期备份**
   - 数据库每日备份
   - 配置文件版本控制

## 开发指南

### 添加新的 API 端点

1. 在 `app/api/` 创建路由文件
2. 在 `app/main.py` 注册路由
3. 添加测试用例

### 添加新的服务

1. 在 `app/services/` 创建服务文件
2. 实现业务逻辑
3. 在需要的地方导入使用

### 代码风格

```bash
# 格式化
black app/
isort app/

# 类型检查
mypy app/

# 测试
pytest
```

## 更新日志

### v1.0.0
- 基础架构
- Worker 管理
- 任务调度
- 可靠性评分
- 多区域支持

### 待实现 (v2.0)
- [ ] Prefill/Decode 分离调度
- [ ] 分布式模型分片
- [ ] gRPC 通信
- [ ] KV-Cache 分布式管理
