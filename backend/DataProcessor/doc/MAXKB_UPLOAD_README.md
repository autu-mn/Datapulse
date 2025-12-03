# MaxKB 自动上传功能使用说明

## 功能说明

在爬取并处理完GitHub仓库数据后，自动将 `text_data_for_training.txt` 文件上传到MaxKB知识库。

## 配置方法

### 方法1：使用环境变量（推荐）

在项目根目录或 `DataProcessor` 目录下创建或编辑 `.env` 文件：

```env
# MaxKB配置
MAXKB_URL=http://localhost:8080
MAXKB_USERNAME=your_username
MAXKB_PASSWORD=your_password_here
MAXKB_KNOWLEDGE_ID=your_knowledge_id_here
MAXKB_CHUNK_SIZE=500
```

### 方法2：在代码中配置

```python
from DataProcessor.data_processor import DataProcessor

# 创建处理器并启用MaxKB上传
processor = DataProcessor(
    json_file_path="microsoft_vscode_text_data_20251128_193435.json",
    enable_maxkb_upload=True,
    maxkb_config={
        'base_url': 'http://localhost:8080',
        'username': 'your_username',
        'password': 'your_password',
        'knowledge_id': 'your_knowledge_id',
        'chunk_size': 500
    }
)

processor.process_all()
```

## 获取知识库ID

1. 登录MaxKB：`http://localhost:8080`
2. 进入你要上传文档的知识库
3. 查看浏览器地址栏，URL格式类似：
   ```
   http://localhost:8080/admin/knowledge/document/upload/default?id=019ae417-c380-7790-92e6-2fc017ed1652
   ```
   其中 `019ae417-c380-7790-92e6-2fc017ed1652` 就是知识库ID

## 使用流程

### 1. 爬取数据

```bash
cd Datapulse/backend/DataProcessor
python github_text_crawler.py
```

### 2. 处理数据（自动上传）

```bash
python data_processor.py <json文件路径>
```

如果配置了MaxKB，处理完成后会自动上传文本文件。

## 手动上传（测试用）

如果需要单独测试上传功能：

```python
from DataProcessor.maxkb_uploader import MaxKBUploader

uploader = MaxKBUploader(
    base_url="http://localhost:8080",
    username="admin",
    password="admin",
    knowledge_id="019ae417-c380-7790-92e6-2fc017ed1652"
)

if uploader.login():
    uploader.upload_text_file("text_data_for_training.txt", chunk_size=500)
```

## 故障排查

### 1. 登录失败

**问题**：`✗ MaxKB登录失败`

**解决方案**：
- 检查MaxKB服务是否运行：访问 `http://localhost:8080`
- 确认用户名和密码正确
- 如果MaxKB使用了自定义登录API，可能需要修改 `maxkb_uploader.py` 中的登录端点

### 2. 知识库ID未设置

**问题**：`⚠ MaxKB知识库ID未配置`

**解决方案**：
- 设置环境变量 `MAXKB_KNOWLEDGE_ID`
- 或在代码中提供 `knowledge_id` 参数

### 3. 上传失败

**问题**：`✗ 文件上传失败`

**解决方案**：
- 检查网络连接
- 确认知识库ID正确
- 查看MaxKB日志：`docker logs maxkb`
- 检查文件大小（如果文件过大可能需要调整chunk_size）

### 4. 从浏览器获取Token（高级）

如果自动登录失败，可以从浏览器获取token：

1. 打开MaxKB并登录
2. 按F12打开开发者工具
3. 进入Network标签
4. 执行一个操作（如上传文档）
5. 查看请求的Authorization header，复制Bearer token
6. 在代码中直接设置token：

```python
uploader = MaxKBUploader(...)
uploader.token = "你的token"
uploader.session.headers.update({"Authorization": f"Bearer {uploader.token}"})
uploader.upload_text_file("file.txt")
```

## 注意事项

1. **首次使用**：建议先用小文件测试上传功能
2. **文件大小**：如果文本文件很大，可能需要调整 `chunk_size` 参数
3. **网络延迟**：上传大文件可能需要一些时间，请耐心等待
4. **重复上传**：如果多次运行处理脚本，会重复上传相同内容到知识库

## API说明

上传使用的API端点：
```
POST /admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document/split
```

参数：
- `file`: 文件内容（multipart/form-data）
- `chunk_size`: 文档分块大小（字符数）
- `chunk_overlap`: 分块重叠大小（默认50）

