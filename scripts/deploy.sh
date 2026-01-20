#!/bin/bash
# 一键部署脚本

set -e

echo "=========================================="
echo "  分布式GPU推理平台 - 一键部署"
echo "=========================================="
echo ""

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未安装 Docker"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误: 未安装 Docker Compose"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件，使用默认配置"
    echo "建议: 复制 .env.example 并修改配置"
fi

echo "✅ 环境检查通过"
echo ""

# 选择部署模式
echo "请选择部署模式:"
echo "1) 仅服务器 (Server + Database + Redis)"
echo "2) 服务器 + Worker (需要 NVIDIA GPU)"
echo ""
read -p "请输入选项 [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动服务器模式..."
        docker-compose up -d postgres redis server
        ;;
    2)
        echo ""
        echo "🚀 启动服务器 + Worker 模式..."
        docker-compose --profile with-worker up -d
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo "📊 服务状态:"
docker-compose ps

echo ""
echo "=========================================="
echo "  ✅ 部署完成！"
echo "=========================================="
echo ""
echo "🌐 访问地址:"
echo "   - API服务: http://localhost:8880"
echo "   - 健康检查: http://localhost:8880/health"
echo "   - API文档: http://localhost:8880/docs"
echo ""
echo "📝 查看日志:"
echo "   docker-compose logs -f server"
echo ""
echo "🛑 停止服务:"
echo "   docker-compose down"
echo ""
