<style>
input[name="lang"] { display: none; }
#lang-en-content { display: block; }
#lang-cn-content { display: none; }
#lang-en:checked ~ #lang-en-content { display: block !important; }
#lang-en:checked ~ #lang-cn-content { display: none !important; }
#lang-cn:checked ~ #lang-cn-content { display: block !important; }
#lang-cn:checked ~ #lang-en-content { display: none !important; }
#lang-en-content label[for="lang-en"] { color: #0969da; text-decoration: underline; }
#lang-en-content label[for="lang-cn"] { color: #656d76; text-decoration: none; }
#lang-cn-content label[for="lang-cn"] { color: #0969da; text-decoration: underline; }
#lang-cn-content label[for="lang-en"] { color: #656d76; text-decoration: none; }
#lang-en:checked ~ #lang-en-content label[for="lang-en"] { color: #0969da !important; text-decoration: underline !important; }
#lang-en:checked ~ #lang-en-content label[for="lang-cn"] { color: #656d76 !important; text-decoration: none !important; }
#lang-cn:checked ~ #lang-cn-content label[for="lang-cn"] { color: #0969da !important; text-decoration: underline !important; }
#lang-cn:checked ~ #lang-cn-content label[for="lang-en"] { color: #656d76 !important; text-decoration: none !important; }
</style>

<!-- 语言切换 radio buttons -->
<input type="radio" id="lang-en" name="lang" checked>
<input type="radio" id="lang-cn" name="lang">

<!-- 英文内容 -->
<div id="lang-en-content" class="lang-content">

<div align="center">

# 🔮 OpenVista

### Multimodal Time-Series Prediction Platform for GitHub Repository Health

<img src="image/首页.png" alt="OpenVista Dashboard" width="800"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61dafb?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

<label for="lang-en" style="color: #0969da; text-decoration: underline; cursor: pointer; margin: 0 10px;">English</label>
<span style="color: #656d76;">·</span>
<label for="lang-cn" style="color: #656d76; cursor: pointer; margin: 0 10px;">中文文档</label>

</div>

---

## 🌟 Overview

**OpenVista** is a next-generation platform for analyzing and predicting the health of open-source GitHub repositories. The platform integrates two core capabilities:

1. **🤖 MaxKB Intelligent Q&A System** — RAG-based knowledge base for project documentation
2. **🔮 GitPulse Multimodal Prediction Model** — Intelligent forecasting combining time-series and text

Together, these modules provide comprehensive analysis of open-source projects: past, present, and future.

---

### 💡 The Problem We're Solving

Open-source projects face numerous challenges in maintaining long-term health and sustainability. Our research, based on **600+ GitHub repositories**, reveals critical gaps in how we understand and predict project health:

#### 🔴 Current Pain Points

1. **Fragmented Health Assessment**
   - Traditional metrics (Stars, Commits) provide only partial insights
   - No unified framework to evaluate project health holistically
   - Hard to identify early warning signs of declining projects

2. **Incomplete Prediction Models**
   - Existing approaches rely solely on time-series metrics (R² ≈ 0.46)
   - Ignore rich textual information (README, Issues, documentation)
   - Cannot capture semantic signals about project direction and community engagement

3. **High Collaboration Friction**
   - Poorly written Issues increase maintenance burden
   - Incomplete documentation barriers new contributors
   - Lack of structured guidance for improving project health

4. **Limited Actionable Insights**
   - Raw metrics don't translate to actionable recommendations
   - No systematic way to understand "why" a project is struggling
   - Difficult to discover similar projects for learning and collaboration

#### ✅ Our Solution

**OpenVista** addresses these challenges through three innovations:

1. **Multimodal Prediction** — Combining time-series metrics with textual features improves prediction accuracy by **66.7%** (R²: 0.46 → 0.77)
2. **CHAOSS-Based Health Scoring** — Six-dimensional framework providing comprehensive health assessment
3. **Intelligent Q&A** — RAG-powered knowledge base enabling natural language queries about any repository

Our platform transforms raw data into actionable intelligence, helping maintainers, contributors, and organizations make data-driven decisions about open-source projects.

---

## 📑 Table of Content

- [🌟 Overview](#-overview)
  - [💡 The Problem We're Solving](#-the-problem-were-solving)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🤖 MaxKB Intelligent Q&A System](#-maxkb-intelligent-qa-system)
  - [System Architecture](#system-architecture)
  - [Knowledge Base Contents](#knowledge-base-contents)
  - [Deployment & Configuration](#deployment--configuration)
- [🔬 GitPulse Prediction Model](#-gitpulse-prediction-model)
  - [Model Performance](#model-performance)
  - [Model Overview](#model-overview)
  - [Architecture Highlights](#architecture-highlights)
  - [Two-Stage Training](#two-stage-training)
  - [Reproducing GitPulse Model](#reproducing-gitpulse-model)
- [✨ Feature Gallery](#-feature-gallery)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage Guide](#-usage-guide)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📢 Community Initiative](#-community-initiative)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🛠️ Tech Stack

<div align="center">
<img src="image/技术架构.png" alt="Tech Architecture" width="700"/>
</div>

<table>
<tr>
<td width="50%">

### Backend
- **Framework**: Flask (Python)
- **Deep Learning**: PyTorch 2.0+
- **NLP**: Transformers (DistilBERT)
- **Data Processing**: Pandas, NumPy

</td>
<td width="50%">

### Frontend
- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts + Custom SVG
- **Animation**: Framer Motion

</td>
</tr>
<tr>
<td>

### AI & Knowledge Base
- **RAG System**: MaxKB
- **LLM Backup**: DeepSeek API
- **Text Encoding**: DistilBERT

</td>
<td>

### Data Sources
- **GitHub API**: Issues, PRs, Commits
- **OpenDigger**: 16 time-series metrics

</td>
</tr>
</table>

---

## 📁 Project Structure

```
OpenVista/
├── 🔧 backend/                     # Flask Backend
│   ├── Agent/                      # AI & MaxKB Integration
│   │   ├── maxkb_client.py         # MaxKB Knowledge Base Client
│   │   ├── prediction_explainer.py # AI Prediction Explainer
│   │   └── qa_agent.py             # Intelligent Q&A Agent
│   │
│   ├── DataProcessor/              # Data Crawling & Processing
│   │   ├── crawl_monthly_data.py   # Main Crawler Entry
│   │   ├── github_text_crawler.py  # GitHub Text Crawler
│   │   ├── maxkb_uploader.py       # MaxKB Document Uploader
│   │   └── monthly_crawler.py      # OpenDigger Data Crawler
│   │
│   ├── GitPulse/                   # GitPulse Prediction Model
│   │   ├── model.py                # Model Architecture
│   │   ├── prediction_service.py   # Prediction Service
│   │   └── gitpulse_weights.pt     # Trained Model Weights (LFS)
│   │
│   ├── CHAOSSEvaluation/           # Community Health Scoring
│   │   └── chaoss_calculator.py    # CHAOSS Metric Calculator
│   │
│   └── app.py                      # Flask API Entry Point
│
├── 🎨 frontend/                    # React Frontend
│
├── 📊 get-dataset/                 # Training Dataset Generator
│
├── 🔬 GitPulse-Training/          # GitPulse Model Training & Reproduction
│   ├── model/                     # Model Architecture Definitions
│   ├── training/                  # Training Scripts
│   ├── Fine-tuning/               # Fine-tuning Experiments
│   ├── predict/                   # Prediction Scripts
│   ├── ablation-test/             # Ablation Studies
│   ├── baseline-test/             # Baseline Comparisons
│   └── Pretrain-data/             # Training Dataset
│
├── 🐳 maxkb-export/                # MaxKB Deployment Config
│   ├── install.sh                  # One-click Install Script
│   ├── docker-compose.yml          # Docker Compose File
│   └── db/                         # Database Backup
│
└── 📄 README.md
```

---

## 🤖 MaxKB Intelligent Q&A System

<div align="center">
<img src="image/MaxKB知识库.png" alt="MaxKB Knowledge Base" width="700"/>
</div>

### System Architecture

MaxKB is the **AI Q&A core** of OpenVista, using **RAG (Retrieval-Augmented Generation)** technology to enable natural language questions about any analyzed repository.

```
User Question → MaxKB Retrieves from Knowledge Base → LLM Generates Answer → Response
```

### Knowledge Base Contents

The system automatically builds a knowledge base for each analyzed repository:

| Document Type | Description |
|---------------|-------------|
| 📄 **README** | Project introduction, installation guide, usage instructions |
| 📜 **LICENSE** | Open source license information |
| 📁 **docs/** | All documents in the project's docs directory |
| 📊 **Project Summary** | AI-generated project analysis report |
| 🐛 **Issue Summary** | Sampled issue data (30 issues/month × 50 months max) |

### Tech Stack & Tools

| Component | Tool/Technology | Description |
|-----------|-----------------|-------------|
| **Knowledge Base Platform** | [MaxKB](https://github.com/1Panel-dev/MaxKB) | Open-source RAG knowledge base system |
| **Deployment** | Docker Compose | One-click deployment with data persistence |
| **Vector Database** | PostgreSQL + pgvector | Efficient vector similarity search |
| **LLM Backend** | Configurable (DeepSeek/OpenAI etc.) | Supports multiple LLM providers |

### Deployment & Configuration

#### Option 1: Use Pre-configured Knowledge Base (Recommended)

```bash
cd maxkb-export

# One-click install (includes database backup restoration)
chmod +x install.sh
./install.sh
```

The installation script will automatically:
- Pull MaxKB Docker image
- Create data volumes and restore pre-configured data
- Start service at `http://localhost:8080`

#### Option 2: Fresh Installation

```bash
# Start with Docker Compose
docker-compose -f docker-compose.maxkb.yml up -d
```

#### Configure .env File

```env
# MaxKB Service Configuration
MAXKB_URL=http://localhost:8080
MAXKB_USERNAME=admin
MAXKB_PASSWORD=your_password
MAXKB_KNOWLEDGE_ID=your_knowledge_id

# MaxKB AI API (for Q&A)
MAXKB_AI_URL=http://localhost:8080/api/application/{app_id}/chat/completions
MAXKB_API_KEY=your_maxkb_api_key
```

### Usage

1. **Automatic Document Upload**: Documents are automatically uploaded to MaxKB during repository crawling
2. **Intelligent Q&A**: Ask questions in the platform's AI Q&A module
3. **Prediction Explanations**: MaxKB generates interpretability analysis for predictions

<div align="center">
<img src="image/Agent.png" alt="AI Agent" width="600"/>
</div>

---

## 🔬 GitPulse Prediction Model

### Model Performance

<div align="center">
<img src="image/不同方法在测试集上的性能对比.png" alt="Performance Comparison" width="800"/>
</div>

Evaluated on **636 test samples** from **4,232 samples（Generated from 600+ projects）** (Two-stage training: Pretrain + Fine-tune):

<div align="center">

| Model | MSE ↓ | MAE ↓ | R² ↑ | DA ↑ | TA@0.2 ↑ |
|:------|:-----:|:-----:|:----:|:----:|:--------:|
| **GitPulse (Transformer+Text)** | **0.0712** | **0.1075** | **0.77** | **73.00%** | **81.75%** |
| CondGRU+Text | 0.0949 | 0.1243 | 0.69 | 68.56% | 79.55% |
| GRU+Text | 0.1084 | 0.1297 | 0.65 | 68.28% | 79.12% |
| Transformer | 0.1693 | 0.1667 | 0.46 | 62.22% | 75.97% |
| CondGRU | 0.1961 | 0.1872 | 0.44 | 61.49% | 74.39% |
| LSTM | 0.2142 | 0.1914 | 0.46 | 56.00% | 75.00% |
| MLP | 0.2280 | 0.2025 | 0.34 | 56.00% | 73.00% |
| Linear | 0.2261 | 0.1896 | 0.34 | 53.00% | 74.00% |

</div>

> **Text Contribution**: Adding text features improves R² from 0.46 → 0.77 (**+66.7%**)

### Model Overview

**GitPulse** is OpenVista's core multimodal time-series prediction model, capable of simultaneously forecasting 16 OpenDigger metrics.

<div align="center">
<img src="image/预测模型.png" alt="GitPulse Prediction Interface" width="800"/>
</div>

### Architecture Highlights

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Time-Series Encoder** | Transformer (2 layers, 4 heads) | Captures temporal patterns across 16 metrics |
| **Text Encoder** | DistilBERT (frozen) + Attention Pooling | Extracts features from project descriptions |
| **Fusion Layer** | Cross-Attention + Dynamic Gating (0.1-0.3) | Text-guided temporal feature fusion |
| **Auxiliary Tasks** | Contrastive Learning + Matching | Improves text-timeseries alignment |
| **Prediction Head** | MLP + Time Projection | Outputs predictions for 32 months ahead |

<details>
<summary>📈 Click to see text contribution effect</summary>

<div align="center">
<img src="image/时序与文本的结合效果.png" alt="GitPulse Model Effect" width="700"/>
</div>

</details>

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 128 | Model hidden dimension |
| n_heads | 4 | Multi-head attention heads |
| n_layers | 2 | Transformer encoder layers |
| hist_len | 128 months | Historical input length |
| pred_len | 32 months | Prediction horizon |
| n_vars | 16 | Number of metrics |
| text_weight | 0.1-0.3 | Dynamic text contribution |

### Two-Stage Training

1. **Pretraining**: Multi-task learning with MSE + Contrastive Loss (λ=0.1) + Matching Loss (λ=0.05)
2. **Fine-tuning**: Full parameter fine-tuning with lower learning rate (1e-5)

### Supported Metrics (16 total)

| Category | Metrics |
|----------|---------|
| **Popularity** | OpenRank, Stars, Forks, Attention |
| **Activity** | Activity, Participants, New Contributors |
| **Contributors** | Contributors, Inactive Contributors, Bus Factor |
| **Issues** | New Issues, Closed Issues, Issue Comments |
| **Pull Requests** | Change Requests, PR Accepted, PR Reviews |

### Training Your Own Model

```bash
cd get-dataset

# Generate dataset (default: 10,000 repos)
python generate_training_dataset.py --count 10000

# Resume from interruption
python generate_training_dataset.py --resume
```

See [get-dataset/README.md](get-dataset/README.md) for detailed options.

### Reproducing GitPulse Model

We provide a complete training repository `GitPulse-Training/` for reproducing the GitPulse model from scratch.

#### Quick Start

```bash
cd GitPulse-Training

# Install dependencies
pip install -r requirements.txt

# Train the model (two-stage training)
cd training
python train_multimodal_v4_1.py --epochs 100 --batch_size 8

# Fine-tuning (for best performance)
cd ../Fine-tuning
python finetune_all_v4_1.py \
    --pretrained_checkpoint ../training/checkpoints/best_model_transformer_mm.pt \
    --strategy full \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5
```

#### Directory Structure

```
GitPulse-Training/
├── model/                          # Model architecture definitions
│   └── multimodal_ts_v4_1.py      # Transformer+Text model
├── training/                       # Training scripts
│   ├── train_multimodal_v4_1.py   # Main training script
│   └── checkpoints/               # Saved model weights
├── Fine-tuning/                   # Fine-tuning experiments
│   └── results/                   # Fine-tuned models
├── predict/                       # Prediction scripts
│   └── predict_single_repo.py    # Single repository prediction
├── ablation-test/                 # Ablation studies
├── baseline-test/                 # Baseline comparisons
├── merge-test/                    # Fusion method comparisons
├── Pretrain-data/                 # Training dataset
│   └── github_multivar.json      # Multi-variable time-series data
├── evaluate_all_models.py         # Unified evaluation script
└── README.md                      # Detailed training guide
```

#### Key Features

- **Two-Stage Training**: Pretraining with multi-task learning + full parameter fine-tuning
- **Complete Experiments**: Ablation studies, baseline comparisons, fusion method tests
- **Unified Evaluation**: Single script to evaluate all models with comprehensive metrics
- **Production Ready**: Includes prediction scripts for real-world usage

For detailed training instructions, model architecture, and experiment results, see [GitPulse-Training/README.md](GitPulse-Training/README.md).

---

## ✨ Feature Gallery

### Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **MaxKB AI Q&A** | RAG-powered knowledge base Q&A for project documentation |
| 🔮 **GitPulse Prediction** | Time-series + text embeddings, forecasting up to 32 months |
| 📊 **CHAOSS Evaluation** | Community health assessment with 6-dimension radar visualization |
| 🔍 **Similar Repo Discovery** | Find related projects via GitHub API-based similarity matching |
| 📈 **Interactive Visualization** | Beautiful charts with 60+ months historical data comparison |
| ⚡ **Real-time Crawling** | Fetch any GitHub repository data on demand |

<details>
<summary><b>🔮 Intelligent Prediction</b> - 12-month forecasting with historical comparison</summary>

<div align="center">
<img src="image/预测模型.png" alt="Prediction Model" width="800"/>
</div>

**AI-Powered Prediction Explanation:**

<div align="center">
<img src="image/issue预测解释图.png" alt="AI Prediction Explanation" width="800"/>
</div>

</details>

<details>
<summary><b>📊 Time-Series Visualization</b> - Multi-dimensional metric analysis</summary>

<div align="center">
<img src="image/可视化图.png" alt="Visualization Dashboard" width="800"/>
</div>

</details>

<details>
<summary><b>🏥 CHAOSS Health Evaluation</b> - 6-dimension radar chart analysis</summary>

<div align="center">
<img src="image/CHAOSS健康评价.png" alt="CHAOSS Evaluation" width="800"/>
</div>

</details>

<details>
<summary><b>🤖 AI Smart Summary</b> - Project analysis with similar repo recommendations</summary>

<div align="center">
<img src="image/项目摘要.png" alt="AI Summary" width="800"/>
</div>

</details>

<details>
<summary><b>🐛 Issue Analysis</b> - Intelligent classification and trend analysis</summary>

<div align="center">
<img src="image/issue分析（2）.png" alt="Issue Analysis" width="800"/>
</div>

**Classification Statistics:**

<div align="center">
<img src="image/issue分析（1）.png" alt="Issue Classification" width="800"/>
</div>

</details>

<details>
<summary><b>📖 Built-in Documentation</b> - Technical documentation and API reference</summary>

<div align="center">
<img src="image/技术文档.png" alt="Technical Documentation" width="800"/>
</div>

</details>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker Desktop (for MaxKB)
- Git (Git LFS auto-installed)

### 🎯 One-Click Setup (Recommended)

We provide unified setup scripts that automate all configuration:

**Windows PowerShell:**
```powershell
git clone https://github.com/your-username/OpenVista.git
cd OpenVista
.\setup.ps1
```

**Linux / macOS:**
```bash
git clone https://github.com/your-username/OpenVista.git
cd OpenVista
chmod +x setup.sh && ./setup.sh
```

The setup script will automatically:

| Step | Description |
|------|-------------|
| 📦 Git LFS | Pull model weights, training data, knowledge base |
| 🐳 Docker | Detect installation, guide setup if needed |
| 🤖 MaxKB | Deploy knowledge base, auto-restore data |
| 🔑 API Keys | Configure GitHub Token & DeepSeek API Key |
| 📚 Dependencies | Optional Python/Node.js installation |
| 🚀 Launch Services | Auto-start backend, frontend, open browser |

---

### 📖 Manual Installation (Advanced)

<details>
<summary>Click to expand manual steps</summary>

#### 1️⃣ Clone & Initialize

```bash
git clone https://github.com/your-username/OpenVista.git
cd OpenVista

# Pull large files (model weights, training data)
git lfs install
git lfs pull
```

#### 2️⃣ Deploy MaxKB

```bash
cd maxkb-export
chmod +x install.sh
./install.sh  # Windows: .\install.ps1
```

Visit `http://localhost:8080` to verify MaxKB is running.

#### 3️⃣ Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
# GitHub API Token (required)
GITHUB_TOKEN=your_github_token

# DeepSeek API Key (for AI features)
DEEPSEEK_API_KEY=your_deepseek_key
```

#### 4️⃣ Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

</details>

---

### 🚀 Launch Services

```bash
# Terminal 1: Start Backend (port 5001)
cd backend
python app.py

# Terminal 2: Start Frontend (port 5173)
cd frontend
npm run dev
```

### 🌐 Access the Platform

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:5001 |
| MaxKB Knowledge Base | http://localhost:8080 |

---

## 📖 Usage Guide

### Basic Workflow

1. **🔍 Search Repository** — Enter `owner/repo` (e.g., `facebook/react`)
2. **⏳ Wait for Crawling** — Data fetched from GitHub API & OpenDigger
3. **📊 Explore Analytics** — View time-series charts, Issue analysis
4. **🔮 Check Predictions** — See 12-month forecasts with AI explanations
5. **📈 CHAOSS Evaluation** — Assess community health scores
6. **🤖 AI Q&A** — Use MaxKB to ask questions about the repository

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📢 Community Initiative

Based on our research findings, we have published a **Community Practice Initiative** that calls for the open-source ecosystem to prioritize text information quality (documentation, Issue descriptions, etc.) as a core component of project governance.

**Key Message**: Well-written documentation and structured Issue templates are not optional—they are essential for project sustainability and, in the AI era, critical for lowering contribution barriers. When combined with AI technologies (RAG-based Q&A, intelligent code assistants), high-quality text information can dramatically reduce the onboarding cost for new contributors.

📖 **Read the full initiative**: [doc/倡议书/倡议书.md](doc/倡议书/倡议书.md) (Chinese)

The initiative includes:
- Empirical findings from 600+ GitHub projects
- Practical recommendations for maintainers, contributors, organizations, and platform developers
- Emphasis on AI-assisted collaboration and knowledge management

---

## 🙏 Acknowledgments

- [MaxKB](https://github.com/1Panel-dev/MaxKB) — RAG Knowledge Base System
- [OpenDigger](https://github.com/X-lab2017/open-digger) — Time-series metrics data
- [CHAOSS](https://chaoss.community/) — Community health metrics framework
- [GitHub API](https://docs.github.com/en/rest) — Repository data source

---

<div align="center">

### ⭐ Star this repo if you find it useful! ⭐

<br/>

**Made with ❤️ by the OpenVista Team**

*Empowering open-source with predictive intelligence*

</div>

</div>
<!-- 英文内容结束 -->

<!-- 中文内容开始 -->
<div id="lang-cn-content" class="lang-content">

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

<br/>

<label for="lang-en" style="color: #656d76; cursor: pointer; margin: 0 10px;">English</label>
<span style="color: #656d76;">·</span>
<label for="lang-cn" style="color: #0969da; text-decoration: underline; cursor: pointer; margin: 0 10px;">中文文档</label>

</div>

---

## 🌟 项目概述

**OpenVista** 是新一代开源项目健康度分析与预测平台。平台集成两大核心能力：

1. **🤖 MaxKB 智能问答系统** — 基于 RAG 技术的项目知识库问答
2. **🔮 GitPulse 多模态预测模型** — 融合时序指标与文本信息的智能预测

通过这两大核心模块，OpenVista 能够全方位分析开源项目的历史、现状与未来。

---

### 💡 我们解决什么问题

开源项目在长期维护和可持续发展中面临诸多挑战。基于 **600+ 个 GitHub 仓库** 的实证研究，我们发现了项目健康度理解与预测中的关键盲点：

#### 🔴 当前痛点

1. **碎片化的健康评估**
   - 传统指标（Star 数、Commit 数）只能反映局部信息
   - 缺乏统一的框架全面评估项目健康状况
   - 难以识别项目衰退的早期预警信号

2. **不完整的预测模型**
   - 现有方法仅依赖时序指标（R² ≈ 0.46）
   - 忽略了丰富的文本信息（README、Issue、文档等）
   - 无法捕捉项目方向和社区参与的语义信号

3. **高协作摩擦成本**
   - Issue 描述不清晰增加维护负担
   - 文档不完善阻碍新贡献者参与
   - 缺乏结构化的健康度改进指导

4. **可执行洞察有限**
   - 原始指标难以转化为可执行的改进建议
   - 缺乏系统性的方法理解项目"为什么"表现不佳
   - 难以发现相似项目进行学习协作

#### ✅ 我们的解决方案

**OpenVista** 通过三大创新解决上述问题：

1. **多模态预测** — 融合时序指标与文本特征，预测准确率提升 **66.7%**（R²: 0.46 → 0.77）
2. **CHAOSS 健康评分** — 六维度评价框架提供全面的健康度评估
3. **智能问答** — 基于 RAG 的知识库，支持自然语言查询任意仓库

我们的平台将原始数据转化为可执行的洞察，帮助维护者、贡献者和组织做出数据驱动的开源项目决策。

---

## 📑 目录

- [🌟 项目概述](#-项目概述)
  - [💡 我们解决什么问题](#-我们解决什么问题)
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
- [📢 社区倡议](#-社区倡议)
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

在 **4,232 个 数据（600+仓库滑动窗口生成）** 的 **636 个测试样本** 上评估（两阶段训练：预训练 + 微调）：

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
- Docker Desktop（用于 MaxKB）
- Git（自动安装 Git LFS）

### 🎯 一键安装（推荐）

我们提供了统一的安装脚本，自动完成所有配置：

**Windows PowerShell:**
```powershell
git clone https://github.com/your-username/OpenVista.git
cd OpenVista
.\setup.ps1
```

**Linux / macOS:**
```bash
git clone https://github.com/your-username/OpenVista.git
cd OpenVista
chmod +x setup.sh && ./setup.sh
```

安装脚本将自动完成：

| 步骤 | 说明 |
|------|------|
| 📦 Git LFS | 拉取模型权重、训练数据、知识库数据 |
| 🐳 Docker | 检测安装状态，引导安装 |
| 🤖 MaxKB | 一键部署知识库系统，自动恢复数据 |
| 🔑 GitHub Token | 交互式配置，自动验证有效性 |
| 📚 依赖安装 | Python/Node.js 依赖可选安装 |

---

### 📖 手动安装（高级用户）

<details>
<summary>点击展开手动安装步骤</summary>

#### 1️⃣ 克隆与初始化

```bash
git clone https://github.com/your-username/OpenVista.git
cd OpenVista

# 拉取大文件（模型权重、训练数据）
git lfs install
git lfs pull
```

#### 2️⃣ 部署 MaxKB

```bash
cd maxkb-export
chmod +x install.sh
./install.sh  # 或 Windows: .\install.ps1
```

访问 `http://localhost:8080` 验证 MaxKB 运行正常。

#### 3️⃣ 环境配置

在 `backend/` 目录创建 `.env` 文件：

```env
# GitHub API Token（必需）
GITHUB_TOKEN=your_github_token

# DeepSeek API Key（AI 功能）
DEEPSEEK_API_KEY=your_deepseek_key
```

#### 4️⃣ 安装依赖

```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../frontend
npm install
```

</details>

---

### 🚀 启动服务

```bash
# 终端 1：启动后端（端口 5001）
cd backend
python app.py

# 终端 2：启动前端（端口 5173）
cd frontend
npm run dev
```

### 🌐 访问平台

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:5173 |
| 后端 API | http://localhost:5001 |
| MaxKB 知识库 | http://localhost:8080 |

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

## 📢 社区倡议

基于我们的研究发现，我们发布了一份**面向开源生态的实践倡议**，呼吁整个开源生态将文本信息质量（文档、Issue 描述等）提升到项目治理的核心位置。

**核心观点**：写好文档、规范 Issue、完善描述，不是可选项，而是项目可持续发展的必要保障。在 AI 时代，这些结构化的文本信息正在成为降低贡献门槛的关键工具。当文档清晰完整时，AI 代码助手可以更好地理解项目上下文；当 Issue 描述规范详细时，AI 可以自动分类、提取关键信息、推荐解决方案；当文档体系完善时，基于 RAG 的智能问答系统可以让任何人通过自然语言快速获取项目知识。

📖 **阅读完整倡议书**：[doc/倡议书/倡议书.md](doc/倡议书/倡议书.md)

倡议书包含：
- 基于 600+ 个 GitHub 项目的实证发现
- 面向维护者、贡献者、组织机构和平台开发者的实践建议
- 强调 AI 辅助协作和知识管理的重要性

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
<!-- 中文内容结束 -->
