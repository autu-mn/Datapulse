# MaxKB 密码诊断脚本
# 检查密码重置状态并提供解决方案

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "    MaxKB 密码诊断工具" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 检查容器是否运行
Write-Host "[1/4] 检查 MaxKB 容器状态..." -ForegroundColor Yellow
$containerRunning = docker ps --filter "name=openvista-maxkb" --format "{{.Names}}" 2>$null
if ($containerRunning) {
    Write-Host "✓ 容器正在运行" -ForegroundColor Green
} else {
    Write-Host "✗ 容器未运行，请先启动容器" -ForegroundColor Red
    Write-Host "  启动命令: docker start openvista-maxkb" -ForegroundColor Yellow
    exit 1
}

# 检查数据库连接
Write-Host "[2/4] 检查数据库连接..." -ForegroundColor Yellow
$dbCheck = docker exec openvista-maxkb psql -U root -d maxkb -c "SELECT 1;" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ 数据库连接正常" -ForegroundColor Green
} else {
    Write-Host "✗ 数据库连接失败" -ForegroundColor Red
    Write-Host "  错误信息: $dbCheck" -ForegroundColor Red
    exit 1
}

# 检查 admin 用户是否存在
Write-Host "[3/4] 检查 admin 用户..." -ForegroundColor Yellow

# Use SQL file method to avoid PowerShell quoting issues
$checkUserSqlFile = [System.IO.Path]::GetTempFileName()
$checkUserSql = "SELECT username FROM `"user`" WHERE username = 'admin';"
$checkUserSql | Out-File -FilePath $checkUserSqlFile -Encoding utf8 -NoNewline

docker cp $checkUserSqlFile openvista-maxkb:/tmp/check_user.sql 2>&1 | Out-Null
$adminCheck = docker exec openvista-maxkb psql -U root -d maxkb -t -f /tmp/check_user.sql 2>&1
docker exec openvista-maxkb rm -f /tmp/check_user.sql 2>&1 | Out-Null
Remove-Item $checkUserSqlFile -Force | Out-Null

$adminExists = ($adminCheck | Out-String).Trim()

if ($adminExists -eq "admin") {
    Write-Host "✓ Admin 用户存在" -ForegroundColor Green
    
    # 检查当前密码
    Write-Host "[4/4] 检查当前密码..." -ForegroundColor Yellow
    
    $checkPasswordSqlFile = [System.IO.Path]::GetTempFileName()
    $checkPasswordSql = "SELECT password FROM `"user`" WHERE username = 'admin';"
    $checkPasswordSql | Out-File -FilePath $checkPasswordSqlFile -Encoding utf8 -NoNewline
    
    docker cp $checkPasswordSqlFile openvista-maxkb:/tmp/check_password.sql 2>&1 | Out-Null
    $currentPassword = docker exec openvista-maxkb psql -U root -d maxkb -t -f /tmp/check_password.sql 2>&1
    docker exec openvista-maxkb rm -f /tmp/check_password.sql 2>&1 | Out-Null
    Remove-Item $checkPasswordSqlFile -Force | Out-Null
    
    $currentPassword = ($currentPassword | Out-String).Trim()
    $expectedPassword = "0df6c52f03e1c75504c7bb9a09c2a016"  # MaxKB@123456 的 MD5
    
    Write-Host "  当前密码哈希: $currentPassword" -ForegroundColor Cyan
    Write-Host "  期望密码哈希: $expectedPassword" -ForegroundColor Cyan
    
    if ($currentPassword -eq $expectedPassword) {
        Write-Host "✓ 密码已正确设置为 MaxKB@123456" -ForegroundColor Green
    } else {
        Write-Host "✗ 密码不匹配！" -ForegroundColor Red
        Write-Host ""
        Write-Host "正在重置密码..." -ForegroundColor Yellow
        
        # Use SQL file method
        $resetSqlFile = [System.IO.Path]::GetTempFileName()
        $resetSql = "UPDATE `"user`" SET password = '$expectedPassword' WHERE username = 'admin';"
        $resetSql | Out-File -FilePath $resetSqlFile -Encoding utf8 -NoNewline
        
        docker cp $resetSqlFile openvista-maxkb:/tmp/reset_password.sql 2>&1 | Out-Null
        docker exec openvista-maxkb psql -U root -d maxkb -f /tmp/reset_password.sql 2>&1 | Out-Null
        docker exec openvista-maxkb rm -f /tmp/reset_password.sql 2>&1 | Out-Null
        Remove-Item $resetSqlFile -Force | Out-Null
        
        # 验证重置结果
        $verifySqlFile = [System.IO.Path]::GetTempFileName()
        $verifySql = "SELECT COUNT(*) FROM `"user`" WHERE username = 'admin' AND password = '$expectedPassword';"
        $verifySql | Out-File -FilePath $verifySqlFile -Encoding utf8 -NoNewline
        
        docker cp $verifySqlFile openvista-maxkb:/tmp/verify_password.sql 2>&1 | Out-Null
        $verifyResult = docker exec openvista-maxkb psql -U root -d maxkb -t -f /tmp/verify_password.sql 2>&1
        docker exec openvista-maxkb rm -f /tmp/verify_password.sql 2>&1 | Out-Null
        Remove-Item $verifySqlFile -Force | Out-Null
        
        $verifyCount = ($verifyResult | Out-String).Trim()
        
        if ($verifyCount -match "^\s*1\s*$") {
            Write-Host "✓ 密码重置成功！" -ForegroundColor Green
            Write-Host ""
            Write-Host "请重启 MaxKB 容器使更改生效:" -ForegroundColor Yellow
            Write-Host "  docker restart openvista-maxkb" -ForegroundColor Cyan
        } else {
            Write-Host "✗ 密码重置失败" -ForegroundColor Red
            Write-Host "  请运行手动重置脚本:" -ForegroundColor Yellow
            Write-Host "  .\maxkb-export\reset_password.ps1" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "✗ Admin 用户不存在！" -ForegroundColor Red
    Write-Host "  这可能意味着数据库恢复失败" -ForegroundColor Yellow
    Write-Host "  请检查数据库备份文件是否存在" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "    诊断完成" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "登录信息:" -ForegroundColor Cyan
Write-Host "  URL:      http://localhost:8080" -ForegroundColor White
Write-Host "  用户名:   admin" -ForegroundColor White
Write-Host "  密码:     MaxKB@123456" -ForegroundColor White
Write-Host ""
Write-Host "如果仍然无法登录，请尝试:" -ForegroundColor Yellow
Write-Host "  1. 重启容器: docker restart openvista-maxkb" -ForegroundColor Cyan
Write-Host "  2. 等待 30 秒后重试登录" -ForegroundColor Cyan
Write-Host "  3. 检查容器日志: docker logs openvista-maxkb" -ForegroundColor Cyan
Write-Host ""

