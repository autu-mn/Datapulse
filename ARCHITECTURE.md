# DataPulse 项目架构文档

## 📋 项目概述

**DataPulse** 是一个 GitHub 仓库生态画像分析平台，专注于时序数据可视化与归因分析。通过爬取 GitHub 仓库数据、OpenDigger 指标，结合 AI 技术进行智能分析和预测。

---

## 🏗️ 整体架构

```
DataPulse/
├── backend/                    # 后端服务 (Python Flask)
│   ├── app.py                  # Flask API 入口
│   ├── data_service.py         # 数据服务层
│   │
│   ├── Agent/                  # AI/MaxKB 层
│   │   ├── deepseek_client.py  # DeepSeek API 客户端
│   │   └── qa_agent.py         # 问答 Agent
│   │
│   ├── DataProcessor/          # 数据采集层
│   │   ├── crawl_monthly_data.py      # 主爬虫入口（单个仓库爬取）
│   │   ├── batch_crawl_opendigger.py   # 批量爬取工具（批量爬取多个仓库）
│   │   ├── monthly_crawler.py         # 月度数据爬虫
│   │   ├── github_text_crawler.py     # GitHub 文本爬虫
│   │   ├── github_api_metrics.py      # GitHub API 指标爬取
│   │   ├── github_graphql_crawler.py  # GraphQL 爬虫
│   │   ├── github_metrics_supplement.py # 指标补充
│   │   ├── monthly_data_processor.py  # 数据处理器
│   │   ├── maxkb_uploader.py          # MaxKB 上传
│   │   └── data/                      # 爬取的数据存储
│   │
│   └── LLM2TSA/                # 时序分析层
│       ├── predictor.py        # LLM 辅助时序预测
│       ├── enhancer.py         # 时序增强器
│       └── cache/              # 预测缓存
│
└── frontend/                   # 前端 (React + TypeScript + Vite)
    ├── src/
    │   ├── App.tsx             # 主应用组件
    │   ├── main.tsx            # 入口文件
    │   ├── types.ts            # TypeScript 类型定义
    │   │
    │   ├── components/         # UI 组件
    │   │   ├── HomePage.tsx           # 首页（项目搜索和爬取）
    │   │   ├── Header.tsx             # 顶部导航栏
    │   │   ├── RepoHeader.tsx         # 仓库信息头部
    │   │   ├── ProjectSearch.tsx      # 项目搜索组件
    │   │   ├── StatsCard.tsx          # 统计卡片
    │   │   ├── ProgressIndicator.tsx  # 进度指示器（SSE）
    │   │   ├── GroupedTimeSeriesChart.tsx  # 分组时序图表
    │   │   ├── TimeSeriesChart.tsx    # 时序图表
    │   │   ├── PredictionChart.tsx    # 预测图表
    │   │   ├── IssueAnalysis.tsx      # Issue 分析组件
    │   │   └── AIChat.tsx             # AI 聊天组件
    │   │
    │   └── pages/              # 页面组件
    │       └── AIChatPage.tsx  # AI 问答页面
    │
    ├── vite.config.ts          # Vite 配置（代理到后端）
    └── package.json            # 前端依赖
```

---

## 🔄 数据流程

### 1. 数据采集流程

```
用户输入仓库 → HomePage.tsx
    ↓
/api/crawl (SSE) → app.py
    ↓
crawl_project_monthly() → crawl_monthly_data.py
    ↓
┌─────────────────────────────────────────┐
│ 步骤1: 爬取指标数据                      │
│ - monthly_crawler.py                    │
│ - github_api_metrics.py                 │
│ - github_graphql_crawler.py             │
│ - batch_crawl_opendigger.py             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤2: 爬取文本数据                      │
│ - github_text_crawler.py                │
│ - 获取 README、LICENSE、docs 等          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤3: 爬取 Issue/PR 时序文本            │
│ - monthly_crawler.py                    │
│ - 按月组织 Issue/PR 数据                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤4: 数据处理和存储                    │
│ - monthly_data_processor.py             │
│ - 时序对齐、关键词提取、分类              │
│ - 存储到 DataProcessor/data/             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤5: 上传到 MaxKB                     │
│ - maxkb_uploader.py                     │
│ - 上传文档供 RAG 检索                    │
└─────────────────────────────────────────┘
```

### 2. 数据展示流程

```
前端请求 → /api/timeseries/grouped/<repo>
    ↓
data_service.py → 读取 DataProcessor/data/
    ↓
分组时序数据 → 前端 GroupedTimeSeriesChart
    ↓
用户交互 → 点击月份 → IssueAnalysis 展示详情
```

### 3. AI 预测流程

```
用户请求预测 → /api/predict/<repo>
    ↓
LLM2TSA/predictor.py
    ↓
┌─────────────────────────────────────────┐
│ 1. 提取历史时序数据                      │
│ 2. Prophet 基础预测                      │
│ 3. LLM 分析趋势和归因                    │
│ 4. 生成预测结果和解释                    │
└─────────────────────────────────────────┘
    ↓
返回预测数据 → 前端 PredictionChart 展示
```

---

## 🎯 核心模块详解

### 后端模块

#### 1. **API 层 (app.py)**

**职责：** Flask RESTful API 入口，处理所有 HTTP 请求

**主要端点：**

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/repos` | GET | 获取已加载的仓库列表 |
| `/api/repo/<repo_key>/summary` | GET | 获取仓库摘要（含 AI 摘要） |
| `/api/timeseries/grouped/<repo_key>` | GET | 获取分组时序数据 |
| `/api/issues/<repo_key>` | GET | 获取 Issue 分析数据 |
| `/api/crawl` | GET/POST | 爬取仓库（SSE 实时进度） |
| `/api/check_project` | GET | 检查项目数据是否存在 |
| `/api/predict/<repo_key>` | POST | 预测时序指标 |
| `/api/predict/<repo_key>/multiple` | POST | 批量预测多个指标 |
| `/api/qa` | POST | AI 问答接口 |
| `/api/reload` | POST | 重新加载数据 |

**技术栈：**
- Flask + Flask-CORS
- Server-Sent Events (SSE) 用于实时进度推送

#### 2. **数据服务层 (data_service.py)**

**职责：** 数据处理、时间对齐、关键词提取、波动分析

**核心功能：**
- 自动加载 `DataProcessor/data/` 目录下的数据
- 时序数据分组（热度、开发活动、Issue 活动、贡献者、统计指标）
- Issue 分类（功能需求、Bug修复、社区咨询）
- 关键词提取（使用 jieba）
- 波动分析（计算变化率）

**数据分组：**
```python
metric_groups = {
    'popularity': ['Star数', 'Fork数', '活跃度', 'OpenRank'],
    'development': ['PR接受数', '变更请求', 'PR审查', '代码变更'],
    'issues': ['新增Issue', '关闭Issue', 'Issue评论'],
    'contributors': ['参与者数', '贡献者', '新增贡献者', '总线因子'],
    'statistics': ['关注度']
}
```

#### 3. **数据采集层 (DataProcessor/)**

##### 3.1 **crawl_monthly_data.py** - 主爬虫入口（单个仓库）
- 整合所有数据源
- 按月组织数据
- 协调爬取流程
- 用于爬取单个仓库的数据

##### 3.2 **batch_crawl_opendigger.py** - 批量爬取工具
- **功能**：批量爬取多个 GitHub 仓库数据，用于制作数据集
- **特点**：
  - 支持断点续传（中断后可继续）
  - 预定义 180+ 个热门开源仓库列表
  - 进度记录（已完成/失败列表）
  - 可配置延迟避免 API 限制
- **使用方法**：
  ```bash
  cd backend/DataProcessor
  python batch_crawl_opendigger.py --count 100 --max-per-month 50
  python batch_crawl_opendigger.py --resume  # 从上次中断处继续
  python batch_crawl_opendigger.py --status  # 查看进度
  ```
- **进度文件**：`DataProcessor/data/batch_crawl_progress.json`

##### 3.3 **monthly_crawler.py** - 月度数据爬虫
- 爬取 GitHub Issues、PRs、Commits、Releases
- 按月组织数据

##### 3.4 **github_text_crawler.py** - 文本爬虫
- 爬取 README、LICENSE、文档
- 获取仓库信息和标签
- OpenDigger 指标爬取

##### 3.5 **monthly_data_processor.py** - 数据处理器
- 时序对齐
- Issue 分类
- 关键词提取
- 生成项目摘要

##### 3.6 **maxkb_uploader.py** - MaxKB 上传
- 上传文档到 MaxKB 知识库
- 支持 RAG 检索

#### 4. **AI/MaxKB 层 (Agent/)**

##### 4.1 **deepseek_client.py** - DeepSeek API 客户端
- 封装 DeepSeek API 调用
- 处理 API 密钥和请求

##### 4.2 **qa_agent.py** - 问答 Agent
- 基于项目数据提供智能问答
- 结合 MaxKB 知识库检索
- 使用 DeepSeek 生成回答

#### 5. **时序分析层 (LLM2TSA/)**

##### 5.1 **predictor.py** - LLM 时序预测器
- 使用 Prophet 进行基础预测
- LLM 分析趋势和归因
- 生成预测结果和解释
- 支持缓存机制

##### 5.2 **enhancer.py** - 时序增强器
- 时序数据增强
- 特征工程

---

### 前端模块

#### 1. **主应用 (App.tsx)**

**职责：** 应用主入口，管理路由和状态

**核心状态：**
- `currentProject`: 当前选中的项目
- `showHomePage`: 是否显示首页
- `data`: 项目数据（时序、Issue、摘要）
- `activeTab`: 当前标签页（时序分析/Issue分析）

**主要功能：**
- 项目选择和切换
- 数据加载和展示
- 标签页切换

#### 2. **组件库 (components/)**

##### 2.1 **HomePage.tsx** - 首页
- 项目搜索和输入
- 触发爬取流程
- SSE 实时进度展示
- 项目选择

##### 2.2 **GroupedTimeSeriesChart.tsx** - 分组时序图表
- 展示分组时序数据
- 支持多指标对比
- 月份点击交互

##### 2.3 **IssueAnalysis.tsx** - Issue 分析
- 按月展示 Issue 分类
- 关键词云图
- 月份筛选

##### 2.4 **PredictionChart.tsx** - 预测图表
- 展示预测结果
- 历史数据对比
- LLM 归因解释

##### 2.5 **AIChat.tsx** - AI 聊天
- 项目相关问答
- 基于 MaxKB 知识库

#### 3. **技术栈**

**核心框架：**
- React 18.2
- TypeScript 5.3
- Vite 5.0

**UI 库：**
- Tailwind CSS 3.4
- Framer Motion 10.16（动画）
- Lucide React 0.303（图标）

**图表库：**
- Chart.js 4.5 + react-chartjs-2 5.3
- Recharts 2.10

**路由：**
- React Router DOM 7.10

---

## 🔌 API 接口详细说明

### 数据获取接口

#### GET `/api/repos`
获取已加载的仓库列表

**响应：**
```json
{
  "repos": ["facebook/react", "microsoft/vscode"],
  "summaries": [...]
}
```

#### GET `/api/repo/<repo_key>/summary`
获取仓库摘要（包含 AI 摘要）

**响应：**
```json
{
  "repoInfo": {
    "name": "react",
    "owner": "facebook",
    "description": "...",
    "stars": 12345,
    "openrank": 123.45
  },
  "projectSummary": {
    "aiSummary": "AI 生成的摘要...",
    "dataRange": {
      "start": "2020-01",
      "end": "2024-12",
      "months_count": 60
    },
    "issueStats": {
      "feature": 100,
      "bug": 50,
      "question": 30,
      "total": 180
    }
  }
}
```

#### GET `/api/timeseries/grouped/<repo_key>`
获取分组时序数据

**响应：**
```json
{
  "startMonth": "2020-01",
  "endMonth": "2024-12",
  "timeAxis": ["2020-01", "2020-02", ...],
  "groups": {
    "popularity": {
      "name": "项目热度",
      "metrics": {
        "opendigger_Star数": {
          "data": [100, 150, 200, ...],
          "color": "#FFD700"
        }
      }
    }
  }
}
```

#### GET `/api/issues/<repo_key>`
获取 Issue 分析数据

**响应：**
```json
{
  "categories": [
    {
      "month": "2024-01",
      "feature": 10,
      "bug": 5,
      "question": 3
    }
  ],
  "monthlyKeywords": {
    "2024-01": ["feature", "bug", ...]
  }
}
```

### 爬取接口

#### GET `/api/crawl?owner=<owner>&repo=<repo>`
爬取 GitHub 仓库（SSE 实时进度）

**SSE 事件格式：**
```json
{
  "type": "start|progress|metrics_ready|complete|error",
  "step": 1,
  "stepName": "爬取指标数据",
  "message": "正在爬取...",
  "progress": 25,
  "projectName": "facebook_react"
}
```

### 预测接口

#### POST `/api/predict/<repo_key>`
预测时序指标

**请求体：**
```json
{
  "metric_name": "opendigger_Star数",
  "forecast_months": 6,
  "include_reasoning": true
}
```

**响应：**
```json
{
  "metric_name": "opendigger_Star数",
  "historical": [100, 150, 200, ...],
  "forecast": [250, 300, 350, ...],
  "reasoning": "LLM 生成的归因分析...",
  "confidence": 0.85
}
```

### AI 问答接口

#### POST `/api/qa`
AI 问答

**请求体：**
```json
{
  "question": "这个项目如何使用？",
  "project": "facebook/react"
}
```

**响应：**
```json
{
  "answer": "基于项目文档的回答...",
  "sources": [...]
}
```

---

## 📊 数据存储结构

### 数据目录结构

```
DataProcessor/data/
└── <owner>_<repo>/
    └── monthly_data_<timestamp>/
        ├── monthly_data.json          # 月度时序数据
        ├── text_for_maxkb/            # 上传到 MaxKB 的文档
        │   ├── README.md
        │   ├── LICENSE
        │   └── docs/
        └── issue_classification.json   # Issue 分类数据
```

### 数据格式

#### monthly_data.json
```json
{
  "2024-01": {
    "opendigger_Star数": 1000,
    "opendigger_Fork数": 500,
    "opendigger_活跃度": 0.85,
    "github_issues_count": 10,
    "github_prs_count": 5
  },
  "2024-02": { ... }
}
```

---

## 🔐 配置说明

### 后端配置

**环境变量 (.env)：**
```env
GITHUB_TOKEN=your_github_token
DEEPSEEK_KEY=your_deepseek_api_key
MAXKB_URL=your_maxkb_url
MAXKB_API_KEY=your_maxkb_api_key
```

**端口配置：**
- Flask 后端：`5000` (app.py 第 725 行)

### 前端配置

**代理配置 (vite.config.ts)：**
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:5000',
    changeOrigin: true
  }
}
```

**端口配置：**
- Vite 前端：`3000`

---

## 🚀 部署架构

### 开发环境

```
前端 (Vite) :3000
    ↓ (代理)
后端 (Flask) :5000
    ↓
数据存储: DataProcessor/data/
    ↓
外部服务:
  - GitHub API
  - OpenDigger API
  - DeepSeek API
  - MaxKB API
```

### 生产环境建议

```
Nginx (反向代理)
    ↓
前端 (静态文件)
    ↓
后端 (Flask + Gunicorn)
    ↓
数据存储 (PostgreSQL/MongoDB)
    ↓
外部服务 (同上)
```

---

## 📈 性能优化

### 后端优化
- 数据缓存（内存缓存已加载的数据）
- 预测结果缓存（LLM2TSA/cache/）
- SSE 流式传输（避免超时）

### 前端优化
- React 组件懒加载
- 图表数据虚拟化
- 防抖和节流

---

## 🔍 监控和日志

### 日志位置
- 后端：控制台输出
- 前端：浏览器控制台

### 健康检查
- `/api/health` 端点用于服务健康检查

---

## 📝 开发指南

### 添加新指标
1. 在 `DataProcessor/monthly_crawler.py` 中添加爬取逻辑
2. 在 `data_service.py` 的 `metric_groups` 中添加分组配置
3. 前端自动展示（无需修改）

### 添加新 API 端点
1. 在 `app.py` 中添加路由
2. 在 `data_service.py` 中添加数据处理逻辑
3. 前端调用新接口

### 添加新前端组件
1. 在 `src/components/` 创建组件
2. 在 `App.tsx` 中引入和使用
3. 添加 TypeScript 类型定义

---

## 🎯 未来扩展方向

1. **数据源扩展**
   - 支持更多数据源（GitLab、Bitbucket）
   - 支持自定义指标

2. **AI 能力增强**
   - 多模型支持（GPT、Claude）
   - 更智能的归因分析

3. **可视化增强**
   - 3D 可视化
   - 交互式图表

4. **性能优化**
   - 数据库支持
   - 分布式爬取

---

## 📚 相关文档

- [README.md](./README.md) - 项目说明
- [backend/LLM2TSA/doc/使用说明.md](./backend/LLM2TSA/doc/使用说明.md) - 时序预测使用说明

---

**最后更新：** 2024-12-23

