# 手动重置 MaxKB 管理员密码

Write-Host "正在重置 MaxKB 管理员密码..." -ForegroundColor Yellow

# MaxKB@123456 的 MD5 哈希值
$passwordMd5 = "0df6c52f03e1c75504c7bb9a09c2a016"
$sql = "UPDATE `"user`" SET password = '$passwordMd5' WHERE username = 'admin';"

# Use SQL file method to avoid PowerShell quoting issues
$tempSqlFile = [System.IO.Path]::GetTempFileName()
$sql | Out-File -FilePath $tempSqlFile -Encoding utf8 -NoNewline

# Copy SQL file to container and execute
docker cp $tempSqlFile openvista-maxkb:/tmp/reset_password.sql 2>&1 | Out-Null
docker exec openvista-maxkb psql -U root -d maxkb -f /tmp/reset_password.sql 2>&1 | Out-Null
docker exec openvista-maxkb rm -f /tmp/reset_password.sql 2>&1 | Out-Null
Remove-Item $tempSqlFile -Force | Out-Null

# Verify password was set using SQL file
$verifySqlFile = [System.IO.Path]::GetTempFileName()
$verifySql = "SELECT COUNT(*) FROM `"user`" WHERE username = 'admin' AND password = '$passwordMd5';"
$verifySql | Out-File -FilePath $verifySqlFile -Encoding utf8 -NoNewline

docker cp $verifySqlFile openvista-maxkb:/tmp/verify_password.sql 2>&1 | Out-Null
$verifyResult = docker exec openvista-maxkb psql -U root -d maxkb -t -f /tmp/verify_password.sql 2>&1
docker exec openvista-maxkb rm -f /tmp/verify_password.sql 2>&1 | Out-Null
Remove-Item $verifySqlFile -Force | Out-Null

$verifyCount = ($verifyResult | Out-String).Trim()

if ($verifyCount -match "^\s*1\s*$") {
    Write-Host ""
    Write-Host "✓ 密码重置成功！" -ForegroundColor Green
    Write-Host ""
    Write-Host "请使用以下凭据登录："
    Write-Host "  用户名: admin"
    Write-Host "  密码:   MaxKB@123456"
    Write-Host ""
    Write-Host "请重启容器使更改生效:" -ForegroundColor Yellow
    Write-Host "  docker restart openvista-maxkb" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "✗ 密码重置失败，请检查容器是否正常运行" -ForegroundColor Red
    Write-Host "  验证结果: $verifyCount" -ForegroundColor Red
}
