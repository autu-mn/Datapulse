<div align="center">

# 🔮 OpenVista

### 基于多模态时序预测的 GitHub 仓库生态画像分析平台

<img src="image/首页.png" alt="OpenVista 仪表盘" width="800"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61dafb?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**[English](README.md)** · **[中文文档](README_CN.md)** · **[使用指南](使用指南（Q&A）.md)**

</div>

---

## 🌟 项目概述

**OpenVista** 是新一代开源项目健康度分析与预测平台。平台集成两大核心能力：

1. **🤖 MaxKB 智能问答系统** — 基于 RAG 技术的项目知识库问答
2. **🔮 GitPulse 多模态预测模型** — 融合时序指标与文本信息的智能预测

通过这两大核心模块，OpenVista 能够全方位分析开源项目的历史、现状与未来。

---

## 📑 目录

- [🌟 项目概述](#-项目概述)
- [🛠️ 技术架构](#️-技术架构)
- [📁 项目结构](#-项目结构)
- [🤖 MaxKB 智能问答系统](#-maxkb-智能问答系统)
  - [系统架构](#系统架构)
  - [知识库内容](#知识库内容)
  - [部署与配置](#部署与配置)
- [🔬 GitPulse 预测模型](#-gitpulse-预测模型)
  - [模型性能](#模型性能)
  - [模型概述](#模型概述)
  - [架构亮点](#架构亮点)
  - [两阶段训练](#两阶段训练)
  - [复现 GitPulse 模型](#复现-gitpulse-模型)
- [✨ 功能亮点](#-功能亮点)
- [🚀 快速开始](#-快速开始)
- [📖 使用指南](#-使用指南)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)
- [🙏 致谢](#-致谢)

---

## 🛠️ 技术架构

<div align="center">
<img src="image/技术架构.png" alt="技术架构" width="700"/>
</div>

<table>
<tr>
<td width="50%">

### 后端技术
- **框架**: Flask (Python)
- **深度学习**: PyTorch 2.0+
- **NLP**: Transformers (DistilBERT)
- **数据处理**: Pandas, NumPy

</td>
<td width="50%">

### 前端技术
- **框架**: React 18+ (TypeScript)
- **样式**: Tailwind CSS
- **图表**: Recharts + 自定义 SVG
- **动画**: Framer Motion

</td>
</tr>
<tr>
<td>

### AI 与知识库
- **RAG 系统**: MaxKB
- **LLM 备用**: DeepSeek API
- **文本编码**: DistilBERT

</td>
<td>

### 数据来源
- **GitHub API**: Issues、PRs、Commits
- **OpenDigger**: 16 个时序指标

</td>
</tr>
</table>

---

## 📁 项目结构

```
OpenVista/
├── 🔧 backend/                     # Flask 后端服务
│   ├── Agent/                      # AI 与 MaxKB 集成
│   │   ├── maxkb_client.py         # MaxKB 知识库客户端
│   │   ├── prediction_explainer.py # AI 预测解释器
│   │   └── qa_agent.py             # 智能问答 Agent
│   │
│   ├── DataProcessor/              # 数据爬取与处理
│   │   ├── crawl_monthly_data.py   # 主爬虫入口
│   │   ├── github_text_crawler.py  # GitHub 文本爬虫
│   │   ├── maxkb_uploader.py       # MaxKB 文档上传器
│   │   └── monthly_crawler.py      # OpenDigger 数据爬虫
│   │
│   ├── GitPulse/                   # GitPulse 预测模型
│   │   ├── model.py                # 模型架构定义
│   │   ├── prediction_service.py   # 预测服务
│   │   └── gitpulse_weights.pt     # 训练好的模型权重 (LFS)
│   │
│   ├── CHAOSSEvaluation/           # 社区健康度评分
│   │   └── chaoss_calculator.py    # CHAOSS 指标计算器
│   │
│   └── app.py                      # Flask API 入口
│
├── 🎨 frontend/                    # React 前端
│
├── 📊 get-dataset/                 # 训练数据集生成器
│
├── 🔬 GitPulse-Training/          # GitPulse 模型训练与复现
│   ├── model/                     # 模型架构定义
│   ├── training/                  # 训练脚本
│   ├── Fine-tuning/               # 微调实验
│   ├── predict/                   # 预测脚本
│   ├── ablation-test/             # 消融实验
│   ├── baseline-test/             # 基线对比实验
│   └── Pretrain-data/             # 训练数据集
│
├── 🐳 maxkb-export/                # MaxKB 部署配置
│   ├── install.sh                  # 一键安装脚本
│   ├── docker-compose.yml          # Docker 编排文件
│   └── db/                         # 数据库备份
│
└── 📄 README.md
```

---

## 🤖 MaxKB 智能问答系统

<div align="center">
<img src="image/MaxKB知识库.png" alt="MaxKB 知识库" width="700"/>
</div>

### 系统架构

MaxKB 是 OpenVista 的 **AI 问答核心**，采用 **RAG（检索增强生成）** 技术，让用户可以自然语言询问关于项目的任何问题。

```
用户问题 → MaxKB 检索知识库 → LLM 生成回答 → 返回结果
```

### 知识库内容

系统自动为每个分析的仓库构建知识库，包含：

| 文档类型 | 内容说明 |
|----------|----------|
| 📄 **README** | 项目介绍、安装指南、使用说明 |
| 📜 **LICENSE** | 开源许可证信息 |
| 📁 **docs/** | 项目文档目录下的所有文档 |
| 📊 **项目摘要** | AI 生成的项目分析报告 |
| 🐛 **Issue 汇总** | 抽样 Issue 数据（每月 30 条 × 最多 50 个月） |

### 技术栈与工具

| 组件 | 工具/技术 | 说明 |
|------|-----------|------|
| **知识库平台** | [MaxKB](https://github.com/1Panel-dev/MaxKB) | 开源 RAG 知识库系统 |
| **部署方式** | Docker Compose | 一键部署，支持数据持久化 |
| **向量数据库** | PostgreSQL + pgvector | 高效向量相似度检索 |
| **LLM 后端** | 可配置（DeepSeek/OpenAI 等） | 支持多种大模型 |

### 部署与配置

#### 方式一：使用预配置知识库（推荐）

```bash
cd maxkb-export

# 一键安装（含数据库备份恢复）
chmod +x install.sh
./install.sh
```

安装脚本会自动：
- 拉取 MaxKB Docker 镜像
- 创建数据卷并恢复预配置数据
- 启动服务在 `http://localhost:8080`

#### 方式二：全新安装

```bash
# 使用 Docker Compose 启动
docker-compose -f docker-compose.maxkb.yml up -d
```

#### 配置 .env 文件

```env
# MaxKB 服务配置
MAXKB_URL=http://localhost:8080
MAXKB_USERNAME=admin
MAXKB_PASSWORD=your_password
MAXKB_KNOWLEDGE_ID=your_knowledge_id

# MaxKB AI API（用于问答）
MAXKB_AI_URL=http://localhost:8080/api/application/{app_id}/chat/completions
MAXKB_API_KEY=your_maxkb_api_key
```

### 使用方式

1. **自动文档上传**：爬取仓库时自动将文档上传到 MaxKB
2. **智能问答**：在平台的 AI 问答模块中提问
3. **预测解释**：MaxKB 为预测结果生成可解释性分析

<div align="center">
<img src="image/Agent.png" alt="AI Agent" width="600"/>
</div>

---

## 🔬 GitPulse 预测模型

### 模型性能

<div align="center">
<img src="image/不同方法在测试集上的性能对比.png" alt="性能对比" width="800"/>
</div>

在 **4,232 个 GitHub 项目** 的 **636 个测试样本** 上评估（两阶段训练：预训练 + 微调）：

<div align="center">

| 模型 | MSE ↓ | MAE ↓ | R² ↑ | DA ↑ | TA@0.2 ↑ |
|:-----|:-----:|:-----:|:----:|:----:|:--------:|
| **GitPulse (Transformer+Text)** | **0.0712** | **0.1075** | **0.77** | **73.00%** | **81.75%** |
| CondGRU+Text | 0.0949 | 0.1243 | 0.69 | 68.56% | 79.55% |
| GRU+Text | 0.1084 | 0.1297 | 0.65 | 68.28% | 79.12% |
| Transformer | 0.1693 | 0.1667 | 0.46 | 62.22% | 75.97% |
| CondGRU | 0.1961 | 0.1872 | 0.44 | 61.49% | 74.39% |
| LSTM | 0.2142 | 0.1914 | 0.46 | 56.00% | 75.00% |
| MLP | 0.2280 | 0.2025 | 0.34 | 56.00% | 73.00% |
| Linear | 0.2261 | 0.1896 | 0.34 | 53.00% | 74.00% |

</div>

> **文本贡献**: 加入文本特征后，R² 从 0.46 提升到 0.77（**+66.7%**）

### 模型概述

**GitPulse** 是 OpenVista 的核心多模态时序预测模型，能够同时预测 16 个 OpenDigger 指标的未来走势。

<div align="center">
<img src="image/预测模型.png" alt="GitPulse 预测界面" width="800"/>
</div>

### 架构亮点

| 组件 | 技术 | 作用 |
|------|------|------|
| **时序编码器** | Transformer（2 层，4 头） | 捕捉 16 个指标的时序模式 |
| **文本编码器** | DistilBERT（冻结）+ 注意力池化 | 提取项目描述文本特征 |
| **融合层** | 交叉注意力 + 动态门控（0.1-0.3） | 文本引导时序特征融合 |
| **辅助任务** | 对比学习 + 匹配任务 | 提升文本-时序对齐 |
| **预测头** | MLP + 时间投影 | 输出未来 32 个月的预测值 |

<details>
<summary>📈 点击查看文本贡献效果</summary>

<div align="center">
<img src="image/时序与文本的结合效果.png" alt="GitPulse 模型效果" width="700"/>
</div>

</details>

### 模型参数

| 参数 | 数值 | 说明 |
|------|------|------|
| d_model | 128 | 模型隐藏维度 |
| n_heads | 4 | 多头注意力头数 |
| n_layers | 2 | Transformer 编码器层数 |
| hist_len | 128 个月 | 历史输入长度 |
| pred_len | 32 个月 | 预测时长 |
| n_vars | 16 | 指标数量 |
| text_weight | 0.1-0.3 | 动态文本贡献权重 |

### 两阶段训练

1. **预训练**: 多任务学习，包括 MSE + 对比学习损失（λ=0.1）+ 匹配任务损失（λ=0.05）
2. **微调**: 全参数微调，使用较低学习率（1e-5）

### 支持的指标（共 16 个）

| 类别 | 指标 |
|------|------|
| **影响力** | OpenRank、Star 数、Fork 数、关注度 |
| **活跃度** | 活跃度、参与者数、新增贡献者 |
| **贡献者** | 贡献者数、不活跃贡献者、总线因子 |
| **Issue** | 新增 Issue、关闭 Issue、Issue 评论 |
| **PR** | 变更请求、PR 接受数、PR 审查 |

### 训练自己的模型

```bash
cd get-dataset

# 生成数据集（默认：10,000 个仓库）
python generate_training_dataset.py --count 10000

# 从中断处继续
python generate_training_dataset.py --resume
```

详细说明请参考 [get-dataset/README.md](get-dataset/README.md)。

### 复现 GitPulse 模型

我们提供了完整的训练仓库 `GitPulse-Training/`，用于从零开始复现 GitPulse 模型。

#### 快速开始

```bash
cd GitPulse-Training

# 安装依赖
pip install -r requirements.txt

# 训练模型（两阶段训练）
cd training
python train_multimodal_v4_1.py --epochs 100 --batch_size 8

# 微调（获得最佳性能）
cd ../Fine-tuning
python finetune_all_v4_1.py \
    --pretrained_checkpoint ../training/checkpoints/best_model_transformer_mm.pt \
    --strategy full \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5
```

#### 目录结构

```
GitPulse-Training/
├── model/                          # 模型架构定义
│   └── multimodal_ts_v4_1.py      # Transformer+Text 模型
├── training/                       # 训练脚本
│   ├── train_multimodal_v4_1.py   # 主训练脚本
│   └── checkpoints/               # 保存的模型权重
├── Fine-tuning/                   # 微调实验
│   └── results/                   # 微调后的模型
├── predict/                       # 预测脚本
│   └── predict_single_repo.py    # 单仓库预测脚本
├── ablation-test/                 # 消融实验
├── baseline-test/                 # 基线对比实验
├── merge-test/                    # 融合方法对比实验
├── Pretrain-data/                 # 训练数据集
│   └── github_multivar.json      # 多变量时序数据
├── evaluate_all_models.py         # 统一评估脚本
└── README.md                      # 详细训练指南
```

#### 核心特性

- **两阶段训练**：多任务学习预训练 + 全参数微调
- **完整实验**：消融实验、基线对比、融合方法测试
- **统一评估**：单一脚本评估所有模型，包含完整指标
- **生产就绪**：包含实际使用的预测脚本

详细的训练说明、模型架构和实验结果，请参考 [GitPulse-Training/README.md](GitPulse-Training/README.md)。

---

## ✨ 功能亮点

### 核心特性

| 功能 | 描述 |
|------|------|
| 🤖 **MaxKB 智能问答** | 基于项目文档的 RAG 知识库问答系统 |
| 🔮 **GitPulse 预测** | 时序指标 + 文本嵌入融合，预测未来 32 个月 |
| 📊 **CHAOSS 评估** | 社区健康度六维雷达图可视化评估 |
| 🔍 **相似仓库发现** | 基于 GitHub API 的多维度相似项目推荐 |
| 📈 **交互式可视化** | 精美图表，60+ 个月历史数据对比分析 |
| ⚡ **实时数据爬取** | 按需爬取任意 GitHub 仓库数据 |

<details>
<summary><b>🔮 智能趋势预测</b> - 12 个月预测与历史数据对比</summary>

<div align="center">
<img src="image/预测模型.png" alt="预测模型" width="800"/>
</div>

**AI 预测归因解释：**

<div align="center">
<img src="image/issue预测解释图.png" alt="AI 预测解释" width="800"/>
</div>

</details>

<details>
<summary><b>📊 时序可视化</b> - 多维度指标分析仪表盘</summary>

<div align="center">
<img src="image/可视化图.png" alt="可视化仪表盘" width="800"/>
</div>

</details>

<details>
<summary><b>🏥 CHAOSS 健康评价</b> - 六维雷达图分析</summary>

<div align="center">
<img src="image/CHAOSS健康评价.png" alt="CHAOSS 评价" width="800"/>
</div>

</details>

<details>
<summary><b>🤖 AI 智能摘要</b> - 项目分析与相似仓库推荐</summary>

<div align="center">
<img src="image/项目摘要.png" alt="AI 摘要" width="800"/>
</div>

</details>

<details>
<summary><b>🐛 Issue 智能分析</b> - 分类统计与趋势分析</summary>

<div align="center">
<img src="image/issue分析（2）.png" alt="Issue 分析" width="800"/>
</div>

**分类统计饼图：**

<div align="center">
<img src="image/issue分析（1）.png" alt="Issue 分类统计" width="800"/>
</div>

</details>

<details>
<summary><b>📖 内置技术文档</b> - 技术文档与 API 参考</summary>

<div align="center">
<img src="image/技术文档.png" alt="技术文档" width="800"/>
</div>

</details>

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+
- Docker（用于 MaxKB）
- Git LFS（用于模型权重）

### 1️⃣ 克隆与初始化

```bash
# 克隆仓库
git clone https://github.com/your-username/OpenVista.git
cd OpenVista

# 初始化 Git LFS 并拉取模型权重
# Windows:
setup.bat
# Linux/Mac:
chmod +x setup.sh && ./setup.sh
```

### 2️⃣ 部署 MaxKB（可选但推荐）

```bash
cd maxkb-export
chmod +x install.sh
./install.sh
```

访问 `http://localhost:8080` 验证 MaxKB 运行正常。

### 3️⃣ 环境配置

在项目根目录创建 `.env` 文件：

```env
# 必需：GitHub API Token
GITHUB_TOKEN=your_github_token

# MaxKB 配置（如已部署）
MAXKB_URL=http://localhost:8080
MAXKB_USERNAME=admin
MAXKB_PASSWORD=your_password
MAXKB_KNOWLEDGE_ID=your_knowledge_id
MAXKB_AI_URL=http://localhost:8080/api/application/{app_id}/chat/completions
MAXKB_API_KEY=your_maxkb_api_key

# 可选：DeepSeek 作为 LLM 备用
DEEPSEEK_API_KEY=your_deepseek_key
```

### 4️⃣ 安装依赖

```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../frontend
npm install
```

### 5️⃣ 启动服务

```bash
# 终端 1：启动后端（端口 5000）
cd backend
python app.py

# 终端 2：启动前端（端口 3000）
cd frontend
npm run dev
```

### 6️⃣ 访问平台

打开浏览器访问 **http://localhost:3000**

---

## 📖 使用指南

### 基本流程

1. **🔍 搜索仓库** — 输入 `owner/repo`（如 `facebook/react`）
2. **⏳ 等待爬取** — 从 GitHub API 和 OpenDigger 获取数据
3. **📊 探索分析** — 查看时序图表、Issue 分析
4. **🔮 查看预测** — 查看 12 个月预测及 AI 解释
5. **📈 CHAOSS 评估** — 评估社区健康度评分
6. **🤖 AI 问答** — 使用 MaxKB 询问关于仓库的问题

---

## 🤝 贡献指南

欢迎贡献代码！请按以下步骤操作：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 发起 Pull Request

---

## 📄 许可证

本项目采用 **MIT 许可证**。详情请参阅 [LICENSE](LICENSE)。

---

## 🙏 致谢

- [MaxKB](https://github.com/1Panel-dev/MaxKB) — RAG 知识库系统
- [OpenDigger](https://github.com/X-lab2017/open-digger) — 时序指标数据来源
- [CHAOSS](https://chaoss.community/) — 社区健康度指标框架
- [GitHub API](https://docs.github.com/en/rest) — 仓库数据来源

---

<div align="center">

### ⭐ 如果这个项目对你有帮助，请给个 Star！⭐

<br/>

**Made with ❤️ by OpenVista Team**

*用预测智能赋能开源社区*

</div>
