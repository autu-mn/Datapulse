# 数据处理层

## 用途

从 GitHub 仓库爬取数据，然后整理成两种格式：
- **时序数据**：按年、季度组织的指标数据（比如 Star 数、Issue 数等），用于时序建模
- **文本数据**：README、Issue、PR 等文本内容，整理成结构化格式，用于 LLM 训练

## 两个文件

### `github_text_crawler.py` - 数据爬取

**功能**：
- 优先从 OpenDigger API 获取基础指标（不消耗 GitHub Token）
- 从 GitHub API 爬取文本内容（Issue、PR、README、Commits 等，消耗 Token）
- 如果 OpenDigger 没有某些指标，就用 GitHub API 数据计算备用指标

**输出**：JSON 文件 + Excel 文件

**使用**：
```python
from Processor.github_text_crawler import GitHubTextCrawler

crawler = GitHubTextCrawler()
data = crawler.crawl_all("owner", "repo")
```

### `data_processor.py` - 数据处理

**功能**：
- 读取爬取的 JSON 数据
- 分离时序数据和文本数据
- 时序数据按年、季度组织
- 文本数据整理成结构化格式（方便投喂给 LLM）

**输出**：处理后的文件夹，包含：
- `timeseries_data.json` - 时序数据（年、季度分开）
- `text_data_for_training.txt` - 结构化文本数据
- 各种 Excel 表格

**使用**：
```python
from Processor.data_processor import DataProcessor

processor = DataProcessor("xxx_text_data_xxx.json")
processor.process_all()
```

## 配置

在项目根目录创建 `.env` 文件，配置 GitHub Token：
```
GITHUB_TOKEN=your_token_here
```

## 工作流程

1. 运行 `github_text_crawler.py` 爬取数据 → 得到原始 JSON
2. 运行 `data_processor.py` 处理数据 → 得到时序数据和文本数据
3. 时序数据用于时序建模，文本数据用于 LLM 训练

## 注意事项

- GitHub API 有速率限制，脚本会自动处理
- OpenDigger API 不消耗 Token，优先使用
- 如果 OpenDigger 没有数据，会自动用 GitHub API 计算（会消耗 Token）

