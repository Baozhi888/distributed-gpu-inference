# 部署前检查清单

## 快速开始

### Linux/macOS
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### Windows
```cmd
scripts/deploy.bat
```

## 部署前检查

### 1. 系统要求

- [ ] Docker 20.10+
- [ ] Docker Compose 2.0+
- [ ] 至少 4GB 可用内存
- [ ] 至少 10GB 可用磁盘空间

### 2. 配置文件

- [ ] 复制 `.env.example` 到 `.env`
- [ ] 修改 `SECRET_KEY`（至少32字符）
- [ ] 修改 `DB_PASSWORD`
- [ ] 设置 `REGION`（区域代码）

### 3. 端口检查

确保以下端口未被占用：
- [ ] 8880 (API服务)
- [ ] 5432 (PostgreSQL)
- [ ] 6380 (Redis)

检查命令：
```bash
# Linux/macOS
netstat -tuln | grep -E '8880|5432|6380'

# Windows
netstat -ano | findstr "8880 5432 6380"
```

### 4. GPU支持（可选）

如果需要运行 Worker：
- [ ] 安装 NVIDIA 驱动
- [ ] 安装 NVIDIA Container Toolkit
- [ ] 验证 GPU 可用：`docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## 部署模式

### 模式1: 仅服务器（推荐）

适用于：
- 中央服务器部署
- 没有 GPU 的服务器
- 分离部署架构

启动命令：
```bash
docker-compose up -d postgres redis server
```

### 模式2: 服务器 + Worker

适用于：
- 单机部署
- 有 GPU 的服务器
- 测试环境

启动命令：
```bash
docker-compose --profile with-worker up -d
```

## 验证部署

### 1. 检查服务状态
```bash
docker-compose ps
```

所有服务应该显示 `Up` 状态。

### 2. 健康检查
```bash
curl http://localhost:8880/health
```

应该返回：
```json
{
  "status": "healthy",
  "region": "asia-east"
}
```

### 3. 查看日志
```bash
# 查看所有服务日志
docker-compose logs -f

# 只查看服务器日志
docker-compose logs -f server

# 查看最近100行
docker-compose logs --tail=100 server
```

### 4. 访问API文档
浏览器打开：http://localhost:8880/docs

## 常见问题

### 问题1: 端口被占用
```bash
# 修改 docker-compose.yml 中的端口映射
# 例如将 8880:8000 改为 9000:8000
```

### 问题2: 数据库连接失败
```bash
# 检查 PostgreSQL 是否启动
docker-compose logs postgres

# 重启数据库
docker-compose restart postgres
```

### 问题3: Worker 无法启动
```bash
# 检查 GPU 是否可用
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 检查 Worker 日志
docker-compose logs worker
```

### 问题4: 内存不足
```bash
# 减少服务数量，只启动必需服务
docker-compose up -d postgres redis server
```

## 停止和清理

### 停止服务
```bash
docker-compose stop
```

### 停止并删除容器
```bash
docker-compose down
```

### 完全清理（包括数据）
```bash
docker-compose down -v
```

⚠️ 警告：`-v` 参数会删除所有数据，包括数据库！

## 生产环境建议

### 1. 安全配置
- [ ] 修改所有默认密码
- [ ] 使用强密码（至少16字符）
- [ ] 限制端口访问（使用防火墙）
- [ ] 启用 HTTPS（使用 Nginx 反向代理）

### 2. 数据备份
```bash
# 备份数据库
docker-compose exec postgres pg_dump -U inference inference > backup.sql

# 恢复数据库
docker-compose exec -T postgres psql -U inference inference < backup.sql
```

### 3. 监控
- [ ] 配置日志收集
- [ ] 设置告警规则
- [ ] 监控资源使用

### 4. 扩展性
- [ ] 使用外部数据库（RDS）
- [ ] 使用外部缓存（ElastiCache）
- [ ] 配置负载均衡

## 下一步

1. 阅读 [API文档](http://localhost:8880/docs)
2. 部署 Worker 节点（参考 `worker/README.md`）
3. 提交第一个推理任务
4. 配置监控和告警

## 获取帮助

- 查看日志：`docker-compose logs -f`
- 查看文档：`docs/API_AND_DEPLOYMENT_GUIDE.md`
- 提交 Issue：GitHub Issues
