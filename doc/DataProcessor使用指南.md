# DataProcessor 使用指南

## 概述

DataProcessor 用于处理 GitHub 仓库爬取数据，将原始 JSON 数据转换为时序数据和文本数据，并可选择性地将文本数据自动上传至 MaxKB 知识库。

## 基本使用方法

### 初始化

```python
from data_processor import DataProcessor

processor = DataProcessor(
    json_file_path="数据文件路径.json",
    enable_maxkb_upload=False,
    maxkb_config=None
)
```

参数说明：
- `json_file_path`: 必需参数，JSON 数据文件路径
- `enable_maxkb_upload`: 可选参数，是否启用 MaxKB 自动上传，默认为 False
- `maxkb_config`: 可选参数，MaxKB 配置字典，如为 None 则从环境变量读取

### 执行处理

```python
processor.process_all()
```

`process_all()` 方法会自动完成所有数据处理步骤，包括时序数据处理、文本数据格式化、生成处理摘要。如果启用了 MaxKB 上传，会在文本数据处理完成后自动上传。

## MaxKB 自动上传配置

### 环境变量配置

在项目根目录创建 `.env` 文件，配置以下变量：

```
MAXKB_URL=http://localhost:8080
MAXKB_USERNAME=admin
MAXKB_PASSWORD=你的密码
MAXKB_KNOWLEDGE_ID=你的知识库ID
MAXKB_CHUNK_SIZE=500
```

知识库 ID 获取方法：登录 MaxKB，进入目标知识库，从浏览器地址栏的 URL 中获取 `id=` 后面的部分。

### 启用上传功能

在初始化 DataProcessor 时设置 `enable_maxkb_upload=True`：

```python
processor = DataProcessor(
    json_file_path="数据文件路径.json",
    enable_maxkb_upload=True
)
processor.process_all()
```

如果环境变量已正确配置，DataProcessor 会自动读取配置并执行上传。处理完成后，文本数据文件 `text_data_for_training.txt` 会自动上传到指定的 MaxKB 知识库。

### 代码中直接配置

如果不想使用环境变量，可以在代码中直接提供配置：

```python
processor = DataProcessor(
    json_file_path="数据文件路径.json",
    enable_maxkb_upload=True,
    maxkb_config={
        'base_url': 'http://localhost:8080',
        'username': 'admin',
        'password': '你的密码',
        'knowledge_id': '你的知识库ID',
        'chunk_size': 500
    }
)
processor.process_all()
```

## 运行步骤

### 步骤一：准备数据文件

确保已有爬虫生成的 JSON 数据文件。

### 步骤二：配置 MaxKB

在 `.env` 文件中配置 MaxKB 相关参数，确保 MaxKB 服务正在运行。

### 步骤三：运行处理

```python
from data_processor import DataProcessor

processor = DataProcessor(
    json_file_path="你的数据文件.json",
    enable_maxkb_upload=True  # 如需上传则设置为 True
)
processor.process_all()
```



运行后，DataProcessor 会依次执行以下操作：
1. 加载 JSON 数据文件
2. 处理时序数据，生成年度和季度汇总
3. 格式化文本数据
4. 保存所有输出文件
5. 如果启用了 MaxKB 上传，自动上传文本数据到知识库

### 步骤四：查看结果

处理完成后，输出文件保存在 `data/{owner}_{repo}/{owner}_{repo}_text_data_{timestamp}_processed/` 目录下。如果启用了 MaxKB 上传，可在 MaxKB 知识库中查看上传的文档。

## 输出文件说明

处理完成后，所有输出文件保存在 `data/{owner}_{repo}/{owner}_{repo}_text_data_{timestamp}_processed/` 目录下。

时序数据文件：
- `timeseries_data.json`: 完整的时序数据，JSON 格式
- `timeseries_by_year.xlsx`: 按年度汇总的时序数据，Excel 格式
- `timeseries_by_quarter.xlsx`: 按季度汇总的时序数据，Excel 格式

文本数据文件：
- `text_data_structured.json`: 结构化的文本数据，JSON 格式
- `text_data_for_training.txt`: 用于训练的纯文本文件，该文件会被上传到 MaxKB
- `text_data_overview.xlsx`: 文本数据概览，Excel 格式

其他文件：
- `processing_summary.json`: 处理摘要信息，包含文档数量、指标数量等统计信息

## 使用示例

### 基本处理示例（不上传 MaxKB）

```python
from data_processor import DataProcessor

processor = DataProcessor("microsoft_vscode_text_data_20251128_193435.json")
processor.process_all()
```

### 处理并上传到 MaxKB

```python
from data_processor import DataProcessor

processor = DataProcessor(
    json_file_path="microsoft_vscode_text_data_20251128_193435.json",
    enable_maxkb_upload=True
)
processor.process_all()
```

运行后，`text_data_for_training.txt` 文件会自动上传到 MaxKB 知识库。

### 在爬虫流程中使用

```python
from github_text_crawler import GithubTextCrawler
from data_processor import DataProcessor

crawler = GithubTextCrawler()
crawler.crawl_repo("microsoft", "vscode")
json_file = crawler.save_to_json()

processor = DataProcessor(
    json_file_path=json_file,
    enable_maxkb_upload=True
)
processor.process_all()
```

## 自动功能说明

### OpenDigger 数据自动获取

如果输入的 JSON 文件中不包含 OpenDigger 时序数据，DataProcessor 会自动从 OpenDigger API 获取相关指标数据。该过程无需额外配置。

### 输出目录自动创建

输出目录会根据仓库名称和时间戳自动创建，格式为 `data/{owner}_{repo}/{owner}_{repo}_text_data_{timestamp}_processed/`。每次处理都会创建新的时间戳目录，不会覆盖已有数据。
