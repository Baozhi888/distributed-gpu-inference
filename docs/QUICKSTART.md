# 🚀 快速开始 - 5分钟部署指南

## 前置要求

- ✅ Docker 20.10+
- ✅ Docker Compose 2.0+
- ✅ 4GB+ 内存
- ✅ 10GB+ 磁盘空间

## 一键部署

### Windows
```cmd
scripts/deploy.bat
```

### Linux/macOS
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## 手动部署

### 1. 配置环境变量（可选）
```bash
# 复制配置文件
cp .env.example .env

# 编辑配置（建议修改密码和密钥）
nano .env
```

### 2. 启动服务

**仅服务器模式**（推荐）：
```bash
docker-compose up -d postgres redis server
```

**服务器 + Worker 模式**（需要GPU）：
```bash
docker-compose --profile with-worker up -d
```

### 3. 验证部署

**检查服务状态**：
```bash
docker-compose ps
```

**健康检查**：
```bash
curl http://localhost:8880/health
```

**运行测试**：
```bash
# Linux/macOS
chmod +x scripts/test-deployment.sh
./scripts/test-deployment.sh

# Windows
scripts/test-deployment.bat
```

## 访问服务

- 🌐 **API服务**: http://localhost:8880
- 📚 **API文档**: http://localhost:8880/docs
- ❤️ **健康检查**: http://localhost:8880/health

## 查看日志

```bash
# 所有服务
docker-compose logs -f

# 仅服务器
docker-compose logs -f server

# 最近100行
docker-compose logs --tail=100 server
```

## 停止服务

```bash
# 停止服务
docker-compose stop

# 停止并删除容器
docker-compose down

# 完全清理（包括数据）
docker-compose down -v
```

## 下一步

1. ✅ 部署完成
2. 📖 阅读 [API文档](http://localhost:8880/docs)
3. 🖥️ 部署 Worker 节点（参考 `worker/README.md`）
4. 🧪 提交第一个推理任务

## 常见问题

### 端口被占用？
修改 `docker-compose.yml` 中的端口映射：
```yaml
ports:
  - "9000:8000"  # 改为其他端口
```

### 数据库连接失败？
```bash
# 检查日志
docker-compose logs postgres

# 重启数据库
docker-compose restart postgres
```

### 内存不足？
只启动必需服务：
```bash
docker-compose up -d postgres redis server
```

## 获取帮助

- 📖 完整文档: `DEPLOYMENT_CHECKLIST.md`
- 🐛 问题反馈: GitHub Issues
- 📝 查看日志: `docker-compose logs -f`
