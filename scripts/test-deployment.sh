#!/bin/bash
# 部署后测试脚本

echo "=========================================="
echo "  部署验证测试"
echo "=========================================="
echo ""

API_URL="http://localhost:8880"

# 测试1: 健康检查
echo "1️⃣  测试健康检查..."
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)
if [ "$response" = "200" ]; then
    echo "   ✅ 健康检查通过"
    curl -s $API_URL/health | python3 -m json.tool
else
    echo "   ❌ 健康检查失败 (HTTP $response)"
    exit 1
fi
echo ""

# 测试2: 根路径
echo "2️⃣  测试根路径..."
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/)
if [ "$response" = "200" ]; then
    echo "   ✅ 根路径访问成功"
    curl -s $API_URL/ | python3 -m json.tool
else
    echo "   ❌ 根路径访问失败 (HTTP $response)"
fi
echo ""

# 测试3: 区域信息
echo "3️⃣  测试区域信息..."
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/regions)
if [ "$response" = "200" ]; then
    echo "   ✅ 区域信息获取成功"
    curl -s $API_URL/regions | python3 -m json.tool | head -20
else
    echo "   ❌ 区域信息获取失败 (HTTP $response)"
fi
echo ""

# 测试4: API文档
echo "4️⃣  测试API文档..."
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/docs)
if [ "$response" = "200" ]; then
    echo "   ✅ API文档可访问"
else
    echo "   ❌ API文档不可访问 (HTTP $response)"
fi
echo ""

# 测试5: 数据库连接
echo "5️⃣  测试数据库连接..."
if docker-compose exec -T postgres psql -U inference -d inference -c "SELECT 1;" > /dev/null 2>&1; then
    echo "   ✅ 数据库连接正常"
else
    echo "   ❌ 数据库连接失败"
fi
echo ""

# 测试6: Redis连接
echo "6️⃣  测试Redis连接..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "   ✅ Redis连接正常"
else
    echo "   ❌ Redis连接失败"
fi
echo ""

echo "=========================================="
echo "  ✅ 所有测试完成！"
echo "=========================================="
echo ""
echo "🌐 访问地址:"
echo "   - API服务: $API_URL"
echo "   - API文档: $API_URL/docs"
echo "   - 健康检查: $API_URL/health"
echo ""
