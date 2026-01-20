# 部署脚本说明

## 问题修复

### 原问题
批处理文件使用了中文字符，导致在某些 Windows 系统上出现乱码。

### 解决方案
重新创建了纯英文版本的批处理文件，避免编码问题。

## 可用的部署脚本

### Windows

#### 方式1: 使用批处理文件（推荐）
```cmd
scripts/deploy.bat
```

#### 方式2: 直接使用 Docker Compose
```cmd
docker-compose up -d postgres redis server
```

### Linux/macOS

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## 测试脚本

### Windows
```cmd
scripts/test-deployment.bat
```

### Linux/macOS
```bash
chmod +x scripts/test-deployment.sh
./scripts/test-deployment.sh
```

## 手动部署步骤

如果脚本无法运行，可以手动执行：

```cmd
# 1. 检查 Docker
docker --version
docker-compose --version

# 2. 启动服务
docker-compose up -d postgres redis server

# 3. 等待服务启动（约30秒）
timeout /t 30

# 4. 检查状态
docker-compose ps

# 5. 测试 API
docker-compose exec server curl http://localhost:8000/health
```

## 常见问题

### Q: 批处理文件显示乱码？
A: 使用新的英文版 `scripts/deploy.bat`

### Q: 如何查看日志？
A: `docker-compose logs -f server`

### Q: 如何停止服务？
A: `docker-compose down`

### Q: 如何完全清理？
A: `docker-compose down -v`

## 访问地址

部署成功后：
- API 服务: http://localhost:8880
- 健康检查: http://localhost:8880/health
- API 文档: http://localhost:8880/docs
