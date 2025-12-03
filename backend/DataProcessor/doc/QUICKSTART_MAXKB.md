# MaxKB自动上传 - 快速开始

## 🎯 功能说明

在爬取GitHub仓库数据并处理完成后，自动将 `text_data_for_training.txt` 上传到MaxKB知识库。

## ⚡ 快速配置（3步）

### 第1步：获取知识库ID

1. 打开MaxKB：`http://localhost:8080`
2. 登录后进入你要上传文档的知识库
3. 查看浏览器地址栏，找到类似这样的URL：
   ```
   http://localhost:8080/admin/knowledge/document/upload/default?id=019ae417-c380-7790-92e6-2fc017ed1652
   ```
4. 复制 `id=` 后面的部分（例如：`019ae417-c380-7790-92e6-2fc017ed1652`）

### 第2步：配置环境变量

在 `Datapulse/backend/DataProcessor/` 目录下创建或编辑 `.env` 文件：

```env
# MaxKB配置
MAXKB_URL=http://localhost:8080
MAXKB_USERNAME=admin
MAXKB_PASSWORD=你的密码
MAXKB_KNOWLEDGE_ID=019ae417-c380-7790-92e6-2fc017ed1652
MAXKB_CHUNK_SIZE=500
```

### 第3步：启用自动上传

修改你的处理代码，添加 `enable_maxkb_upload=True`：

```python
from DataProcessor.data_processor import DataProcessor

processor = DataProcessor(
    json_file_path="你的json文件路径",
    enable_maxkb_upload=True  # 启用自动上传
)

processor.process_all()
```

## 📝 完整示例

```python
from DataProcessor.data_processor import DataProcessor

# 处理数据并自动上传到MaxKB
processor = DataProcessor(
    json_file_path="microsoft_vscode_text_data_20251128_193435.json",
    enable_maxkb_upload=True,
    maxkb_config={
        'base_url': 'http://localhost:8080',
        'username': 'admin',
        'password': 'admin',
        'knowledge_id': '019ae417-c380-7790-92e6-2fc017ed1652',
        'chunk_size': 500
    }
)

processor.process_all()
```

## 🔧 如果自动登录失败

如果MaxKB的登录API不同，可以从浏览器获取token：

1. 打开MaxKB并登录
2. 按F12 → Network标签
3. 执行任意操作（如上传文档）
4. 查看请求的 `Authorization` header，复制Bearer token
5. 在代码中使用：

```python
from DataProcessor.maxkb_uploader import MaxKBUploader

uploader = MaxKBUploader(
    base_url="http://localhost:8080",
    username="admin",
    password="admin",
    knowledge_id="你的知识库ID"
)

# 手动设置token（从浏览器获取）
uploader.set_token("你的token")
uploader.upload_text_file("text_data_for_training.txt")
```

## ✅ 验证上传

处理完成后，检查输出：

```
✓ MaxKB登录成功
✓ 文件上传成功：text_data_for_training.txt
  处理文档数：XX
```

然后在MaxKB知识库中查看，应该能看到上传的文档。

## 🐛 常见问题

**Q: 登录失败怎么办？**  
A: 检查MaxKB是否运行，用户名密码是否正确。如果还不行，使用浏览器获取token的方法。

**Q: 知识库ID在哪里？**  
A: 进入知识库后，查看浏览器地址栏URL中的 `id=` 参数。

**Q: 上传失败怎么办？**  
A: 检查网络连接、知识库ID是否正确，查看MaxKB日志：`docker logs maxkb`

## 📚 更多信息

详细文档请查看：`MAXKB_UPLOAD_README.md`

