#!/bin/bash
# MaxKB 密码诊断脚本
# 检查密码重置状态并提供解决方案

echo ""
echo "============================================"
echo "    MaxKB 密码诊断工具"
echo "============================================"
echo ""

# 检查容器是否运行
echo "[1/4] 检查 MaxKB 容器状态..."
CONTAINER_RUNNING=$(docker ps --filter "name=openvista-maxkb" --format "{{.Names}}" 2>/dev/null)
if [ -n "$CONTAINER_RUNNING" ]; then
    echo "✓ 容器正在运行"
else
    echo "✗ 容器未运行，请先启动容器"
    echo "  启动命令: docker start openvista-maxkb"
    exit 1
fi

# 检查数据库连接
echo "[2/4] 检查数据库连接..."
if docker exec openvista-maxkb psql -U root -d maxkb -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ 数据库连接正常"
else
    echo "✗ 数据库连接失败"
    exit 1
fi

# 检查 admin 用户是否存在
echo "[3/4] 检查 admin 用户..."
ADMIN_CHECK=$(docker exec openvista-maxkb psql -U root -d maxkb -t -c "SELECT username FROM \"user\" WHERE username = 'admin';" 2>&1 | tr -d '[:space:]')

if [ "$ADMIN_CHECK" = "admin" ]; then
    echo "✓ Admin 用户存在"
    
    # 检查当前密码
    echo "[4/4] 检查当前密码..."
    CURRENT_PASSWORD=$(docker exec openvista-maxkb psql -U root -d maxkb -t -c "SELECT password FROM \"user\" WHERE username = 'admin';" 2>&1 | tr -d '[:space:]')
    
    EXPECTED_PASSWORD="0df6c52f03e1c75504c7bb9a09c2a016"  # MaxKB@123456 的 MD5
    
    echo "  当前密码哈希: $CURRENT_PASSWORD"
    echo "  期望密码哈希: $EXPECTED_PASSWORD"
    
    if [ "$CURRENT_PASSWORD" = "$EXPECTED_PASSWORD" ]; then
        echo "✓ 密码已正确设置为 MaxKB@123456"
    else
        echo "✗ 密码不匹配！"
        echo ""
        echo "正在重置密码..."
        
        SQL="UPDATE \"user\" SET password = '$EXPECTED_PASSWORD' WHERE username = 'admin';"
        echo "$SQL" | docker exec -i openvista-maxkb psql -U root -d maxkb > /dev/null 2>&1
        
        # 验证重置结果
        VERIFY_RESULT=$(docker exec openvista-maxkb psql -U root -d maxkb -t -c "SELECT COUNT(*) FROM \"user\" WHERE username = 'admin' AND password = '$EXPECTED_PASSWORD';" 2>&1 | tr -d '[:space:]')
        
        if [ "$VERIFY_RESULT" = "1" ]; then
            echo "✓ 密码重置成功！"
            echo ""
            echo "请重启 MaxKB 容器使更改生效:"
            echo "  docker restart openvista-maxkb"
        else
            echo "✗ 密码重置失败"
            echo "  请手动执行以下命令:"
            echo "  echo \"UPDATE \\\"user\\\" SET password = '0df6c52f03e1c75504c7bb9a09c2a016' WHERE username = 'admin';\" | docker exec -i openvista-maxkb psql -U root -d maxkb"
        fi
    fi
else
    echo "✗ Admin 用户不存在！"
    echo "  这可能意味着数据库恢复失败"
    echo "  请检查数据库备份文件是否存在"
    exit 1
fi

echo ""
echo "============================================"
echo "    诊断完成"
echo "============================================"
echo ""
echo "登录信息:"
echo "  URL:      http://localhost:8080"
echo "  用户名:   admin"
echo "  密码:     MaxKB@123456"
echo ""
echo "如果仍然无法登录，请尝试:"
echo "  1. 重启容器: docker restart openvista-maxkb"
echo "  2. 等待 30 秒后重试登录"
echo "  3. 检查容器日志: docker logs openvista-maxkb"
echo ""

