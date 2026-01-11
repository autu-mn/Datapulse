# OpenVista MaxKB 一键部署

## 📋 前提条件

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 已安装并运行
- [DeepSeek API Key](https://platform.deepseek.com/)（用于 AI 问答）

---

## 🚀 一键安装

### Linux / macOS

```bash
cd maxkb-export
chmod +x install.sh
./install.sh
```

### Windows PowerShell

```powershell
cd maxkb-export
.\install.ps1
```

---

## ⚙️ 配置 DeepSeek API Key

安装完成后，需要配置 API Key 才能使用 AI 功能：

1. 打开浏览器访问 **http://localhost:8080**
2. 登录账号：
   - 用户名：`admin`
   - 密码：`MaxKB@123456`
3. 点击左侧「**系统设置**」
4. 点击「**模型管理**」
5. 找到「**OpenRank-1**」模型，点击编辑
6. 在 API Key 栏填入你的 DeepSeek API Key
7. 点击保存

---

## 📦 包含内容

| 内容 | 说明 |
|------|------|
| 知识库 | Git 基础、仓库文本资料 |
| AI 应用 | OpenPulse 数据分析助手 |
| 向量数据 | 已预处理的文档向量 |

---

## 🛠️ 常用命令

```bash
# 查看日志
docker logs -f openvista-maxkb

# 停止服务
docker stop openvista-maxkb

# 启动服务
docker start openvista-maxkb

# 重启服务
docker restart openvista-maxkb

# 完全卸载
docker stop openvista-maxkb 2>/dev/null || true
docker rm -f openvista-maxkb 2>/dev/null || true
docker volume rm -f openvista_maxkb_data 2>/dev/null || true
```

---

## ❓ 问题排查

| 问题 | 解决方案 |
|------|----------|
| 登录失败 | 密码为 `MaxKB@123456` |
| AI 无响应 | 检查「模型管理」中的 API Key 是否正确 |
| 端口冲突 | 修改 `install.sh` 中的端口映射，或使用根目录的 `docker-compose.maxkb.yml` |
