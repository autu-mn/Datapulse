# DataProcessor 文档目录

## 📚 文档列表

### 使用指南
- **[使用说明.md](./使用说明.md)** - 数据爬取和处理的基本使用说明
- **[README.md](./README.md)** - 数据处理层详细说明

### MaxKB自动上传功能
- **[QUICKSTART_MAXKB.md](./QUICKSTART_MAXKB.md)** - MaxKB自动上传快速开始指南（推荐先看这个）
- **[MAXKB_UPLOAD_README.md](./MAXKB_UPLOAD_README.md)** - MaxKB自动上传详细文档

### 配置模板
- **[env_template.txt](./env_template.txt)** - `.env` 文件配置模板

### 示例代码
- **[example_maxkb_upload.py](./example_maxkb_upload.py)** - MaxKB上传功能使用示例

## 🚀 快速开始

### 1. 数据爬取和处理

参考：[使用说明.md](./使用说明.md)

```bash
cd Datapulse/backend/DataProcessor
python github_text_crawler.py
python data_processor.py <json文件>
```

### 2. MaxKB自动上传

参考：[QUICKSTART_MAXKB.md](./QUICKSTART_MAXKB.md)

1. 配置 `.env` 文件（参考 `env_template.txt`）
2. 启用自动上传：

```python
from DataProcessor.data_processor import DataProcessor

processor = DataProcessor(
    json_file_path="你的json文件",
    enable_maxkb_upload=True
)
processor.process_all()
```

## 📖 文档结构

```
doc/
├── README.md                    # 本文档（文档索引）
├── 使用说明.md                  # 基本使用说明
├── README.md                    # 数据处理层说明
├── QUICKSTART_MAXKB.md          # MaxKB快速开始
├── MAXKB_UPLOAD_README.md       # MaxKB详细文档
├── env_template.txt             # 环境变量模板
└── example_maxkb_upload.py      # 示例代码
```

## 🔍 查找文档

- **想快速开始MaxKB上传？** → [QUICKSTART_MAXKB.md](./QUICKSTART_MAXKB.md)
- **想了解详细配置？** → [MAXKB_UPLOAD_README.md](./MAXKB_UPLOAD_README.md)
- **想了解数据处理流程？** → [README.md](./README.md)
- **想查看使用示例？** → [example_maxkb_upload.py](./example_maxkb_upload.py)
