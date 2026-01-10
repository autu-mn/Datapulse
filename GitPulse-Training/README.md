# GitPulse

**GitHub 开源项目健康度多模态时序预测模型**

## 简介

GitPulse 是一个基于 Transformer + 文本的多模态时序预测模型，用于预测 GitHub 开源项目的发展趋势。模型通过 Transformer 编码时序信息，并结合项目描述、Issue、Commit 等文本信息，实现文本与时序信息的深度融合。

**最佳模型通过两阶段训练获得**：首先在完整数据集上进行预训练，然后使用全参数微调策略进一步优化，最终达到 R² = 0.7699 的最佳性能。

### 核心创新

- **Transformer 时序编码**：使用标准 Transformer 编码器处理多变量时序数据
- **文本注意力引导融合**：文本特征通过交叉注意力机制引导时序特征融合
- **多任务学习**：结合对比学习和匹配任务提升文本-时序对齐
- **动态门控权重**：自适应调整文本贡献权重（0.1-0.3）
- **多变量时序**：同时预测 16 个指标的未来走势

### 性能表现

**最佳模型：Transformer+Text（经过微调）**（在测试集上的表现）：

| 指标 | 值 | 说明 |
|------|-----|------|
| MSE | **0.0712** | 均方误差（越低越好） |
| MAE | **0.1075** | 平均绝对误差（越低越好） |
| RMSE | **0.2668** | 均方根误差（越低越好） |
| R² | **0.7699** | 决定系数（越高越好，最高1.0） |
| DA | **73.00%** | 方向预测准确率 |
| TA@0.2 | **81.75%** | 阈值准确率（误差<0.2的比例） |

**重要说明**：最佳性能是通过**两阶段训练**获得的：
1. **预训练阶段**：在完整数据集上训练 Transformer+Text 模型
2. **微调阶段**：使用全参数微调策略进一步优化模型

相比纯时序 Transformer 模型，Transformer+Text 在 MSE 上提升了 **57.9%**，在 R² 上提升了 **66.7%**，证明了文本信息的显著价值。微调策略进一步提升了模型性能，R² 从 0.7566（零样本）提升到 **0.7699**（全参数微调）。

### 16 个输入指标

| 维度 | 指标 | 说明 |
|------|------|------|
| 0 | OpenRank | 项目影响力指数 |
| 1 | 活跃度 | 综合活跃度评分 |
| 2 | Star数 | 当月新增 Star |
| 3 | Fork数 | 当月新增 Fork |
| 4 | 关注度 | 关注者增量 |
| 5 | 参与者数 | 活跃参与者 |
| 6 | 新增贡献者 | 新加入的贡献者 |
| 7 | 贡献者 | 活跃贡献者 |
| 8 | 不活跃贡献者 | 流失贡献者 |
| 9 | 总线因子 | 项目风险指标 |
| 10 | 新增Issue | 新开 Issue |
| 11 | 关闭Issue | 关闭的 Issue |
| 12 | Issue评论 | Issue 讨论数 |
| 13 | 变更请求 | PR 数量 |
| 14 | PR接受数 | 合并的 PR |
| 15 | PR审查 | PR 审查数 |

## 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd GitPulse

# 安装依赖
pip install torch transformers tqdm pandas numpy matplotlib
```

### 2. 数据准备

数据已经准备好，位于 `Pretrain-data/github_multivar.json`

如果需要重新生成：
```bash
python convert_github_data.py
```

### 3. 训练模型

**训练 Transformer+Text（最佳模型）需要两阶段训练**：

#### 阶段一：预训练

```bash
cd training

# 基础预训练命令
python train_multimodal_v4_1.py --epochs 100 --batch_size 8

# 完整参数示例
python train_multimodal_v4_1.py \
    --data_path ../Pretrain-data/github_multivar.json \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 8 \
    --lr 5e-4 \
    --hist_len 128 \
    --pred_len 32 \
    --d_model 128 \
    --lambda_cl 0.1 \
    --lambda_ml 0.05 \
    --min_text_weight 0.1 \
    --max_text_weight 0.3 \
    --device cuda
```

预训练完成后，模型会保存在 `training/checkpoints/best_model_transformer_mm.pt`

#### 阶段二：微调（获得最佳性能）

```bash
cd Fine-tuning

# 使用全参数微调策略
python finetune_all_v4_1.py \
    --pretrained_checkpoint ../training/checkpoints/best_model_transformer_mm.pt \
    --strategy full \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5
```

微调完成后，最佳模型会保存在 `Fine-tuning/results/Transformer_full_best.pt`

**训练参数说明**：

- `--epochs`: 训练轮数（默认100，建议100-150）
- `--batch_size`: 批次大小（默认8，根据GPU显存调整）
- `--lr`: 学习率（默认5e-4）
- `--hist_len`: 历史时间步长度（默认128个月）
- `--pred_len`: 预测时间步长度（默认32个月）
- `--d_model`: 模型维度（默认128）
- `--lambda_cl`: 对比学习损失权重（默认0.1）
- `--lambda_ml`: 匹配任务损失权重（默认0.05）
- `--min_text_weight`: 最小文本权重（默认0.1）
- `--max_text_weight`: 最大文本权重（默认0.3）
- `--device`: 设备（cuda/cpu）

### 4. 训练方法详解

Transformer+Text 模型采用**两阶段训练**策略，包括预训练和微调两个阶段：

#### 4.1 预训练阶段

预训练阶段采用**多任务学习**策略，训练过程包括：

#### 4.1 损失函数

总损失 = 预测损失 + λ_cl × 对比学习损失 + λ_ml × 匹配任务损失

- **预测损失（MSE）**：主任务，预测未来时序值
- **对比学习损失（λ_cl=0.1）**：拉近时序特征和文本特征的距离
- **匹配任务损失（λ_ml=0.05）**：判断时序-文本对是否匹配

#### 4.2 优化器配置

- **优化器**：AdamW
- **学习率**：5e-4（多模态模型），1e-3（纯时序模型）
- **权重衰减**：0.01
- **梯度裁剪**：max_norm=1.0
- **学习率调度**：ReduceLROnPlateau
  - mode='min'（监控验证MSE）
  - factor=0.5（每次减半）
  - patience=10（10个epoch无改善则降低学习率）
  - min_lr=1e-5（最小学习率）

#### 4.3 训练策略

- **Early Stopping**：patience=20（验证集MSE连续20个epoch无改善则停止）
- **数据划分**：80% 训练集，20% 验证集（随机种子42）
- **批次处理**：训练时drop_last=True，确保批次大小一致
- **验证频率**：每个epoch结束后在验证集上评估

#### 4.2 微调阶段

预训练完成后，使用**全参数微调**策略进一步优化模型：

**微调策略对比**：
- **Zero-shot**：零样本（不微调），R² = 0.7566
- **Freeze**：冻结预训练模型，R² = 0.7577
- **Full**：全参数微调（**推荐**），R² = **0.7699** ⭐
- **Layerwise**：逐层微调，R² = 0.7575

**微调配置**：
- **学习率**：1e-5（比预训练低一个数量级）
- **训练轮数**：50 epochs（通常比预训练少）
- **批次大小**：8（与预训练保持一致）
- **优化器**：AdamW（权重衰减 0.01）
- **Early Stopping**：patience=15

**微调效果**：
- R² 从 0.7566（零样本）提升到 **0.7699**（全参数微调）
- MSE 进一步降低，模型泛化能力增强

#### 4.3 模型保存

**预训练阶段**：
- 保存路径：`training/checkpoints/best_model_transformer_mm.pt`
- 保存内容：epoch、model_state_dict、val_mse

**微调阶段**：
- 保存路径：`Fine-tuning/results/Transformer_full_best.pt`
- 保存内容：微调后的最佳模型权重

### 5. 评估模型

使用统一评估脚本评估所有模型：

```bash
# 评估所有模型
python evaluate_all_models.py

# 评估指定模型
python evaluate_all_models.py --models "Transformer+Text"

# 生成论文图表
python evaluate_all_models.py --generate_figures
```

评估指标包括：MSE、MAE、RMSE、R²、DA（方向准确率）、TA@0.2（阈值准确率）

### 6. 使用模型预测

**方法一：使用预测脚本（推荐）**

```bash
# 使用微调后的最佳模型进行预测（推荐）
python predict/predict_single_repo.py \
    --timeseries repo_data.json \
    --text repo_description.txt \
    --output prediction.json \
    --checkpoint Fine-tuning/results/Transformer_full_best.pt

# 或使用预训练模型（性能略低）
python predict/predict_single_repo.py \
    --timeseries repo_data.json \
    --text repo_description.txt \
    --output prediction.json \
    --checkpoint training/checkpoints/best_model_transformer_mm.pt
```

**方法二：Python API**

```python
import torch
from model.multimodal_ts_v4_1 import MultimodalTransformerV4_1
from transformers import DistilBertTokenizer

# 加载模型
model = MultimodalTransformerV4_1(
    n_vars=16, 
    hist_len=128, 
    pred_len=32, 
    d_model=128,
    min_text_weight=0.1,
    max_text_weight=0.3
)

# 推荐使用微调后的最佳模型
checkpoint = torch.load('Fine-tuning/results/Transformer_full_best.pt', map_location='cpu')
# 或使用预训练模型
# checkpoint = torch.load('training/checkpoints/best_model_transformer_mm.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 准备输入
ts_history = torch.randn(1, 128, 16)  # [batch, 时间步, 16个指标]
text = "React is a popular frontend framework with growing community."

# Tokenize 文本
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_encoded = tokenizer(text, return_tensors='pt', max_length=256, 
                         padding='max_length', truncation=True)

# 预测
with torch.no_grad():
    prediction = model(
        ts_history, 
        text_encoded['input_ids'], 
        text_encoded['attention_mask'],
        return_auxiliary=False
    )
print(prediction.shape)  # [1, 32, 16] - 预测未来 32 个月的 16 个指标
```

## 目录结构

```
GitPulse/
├── model/
│   ├── __init__.py
│   └── multimodal_ts_v4_1.py     # Transformer+Text 模型定义
├── training/
│   ├── train_multimodal_v4_1.py  # Transformer+Text 训练脚本
│   └── checkpoints/              # 模型保存目录
├── predict/
│   ├── predict_single_repo.py    # 单仓库预测脚本
│   └── models/
│       └── README.md
├── ablation-test/                # 消融实验
│   ├── README.md
│   └── results/                  # 消融实验结果
├── baseline-test/                # 基线测试
│   └── README.md
├── Fine-tuning/                  # 微调实验
│   ├── README.md
│   └── results/                  # 微调实验结果
├── merge-test/                   # 融合方法对比实验
│   ├── README.md
│   └── results/                  # 融合实验结果
├── paper/                        # 论文 LaTeX 源文件
├── Pretrain-data/
│   ├── github_multivar.json      # 多变量数据集
│   └── github_multivar_summary.csv # 数据摘要
├── evaluate_all_models.py        # 统一评估脚本
├── convert_github_data.py        # 数据转换脚本
├── evaluation_results.json       # 评估结果
├── requirements.txt
└── README.md
```

### 实验目录说明

- **ablation-test/**：消融实验，分析不同组件对模型性能的贡献
- **baseline-test/**：基线测试，对比 GitPulse 与基线方法的性能
- **Fine-tuning/**：微调实验，分析不同微调策略（零样本、冻结、全参数、逐层）的效果
- **merge-test/**：融合方法对比实验，对比不同文本-时序融合策略的效果

## 模型架构

Transformer+Text 采用**文本注意力引导融合**策略：

```
输入
├── 时序: [batch, 128, 16] ─────→ Transformer 编码器
│                                        ↓
│                                   时序特征 [batch, 128, 128]
│                                        ↓
└── 文本: "project description..." ──→ DistilBERT (冻结)
                                        ↓
                                   注意力池化 + 投影
                                        ↓
                                   文本特征 [batch, 128]
                                        ↓
                               ┌────────┴────────┐
                               ↓                 ↓
                        交叉注意力融合      辅助任务
                        (文本引导时序)    (对比学习+匹配)
                               ↓                 ↓
                               └────────┬────────┘
                                        ↓
                                  预测头 (MLP + 时间投影)
                                        ↓
输出: [batch, 32, 16] (预测未来 32 个月)
```

**关键组件**：

1. **时序编码器（TransformerTSEncoder）**
   - 输入投影：16维 → 128维
   - 位置编码：正弦位置编码
   - Transformer 编码器：2层，4头注意力，512维FFN
   - 输出：时序特征序列 [batch, 128, 128]

2. **文本编码器（TextEncoderV4）**
   - DistilBERT（冻结参数）
   - 投影层：768维 → 128维
   - 注意力池化：生成全局文本特征
   - 输出：文本序列特征 [batch, text_len, 128] 和全局特征 [batch, 128]

3. **融合层（TransformerTextFusion）**
   - 自注意力：时序特征内部交互
   - 交叉注意力：时序 query，文本 key/value
   - 动态门控：自适应文本权重（0.1-0.3）
   - FFN：前馈网络
   - 2层融合层堆叠

4. **辅助任务**
   - 对比学习（ContrastiveLoss）：拉近匹配的时序-文本对
   - 匹配任务（MatchingLoss）：判断时序-文本对是否匹配

5. **预测头**
   - MLP：128 → 256 → 16
   - 时间投影：128时间步 → 32时间步

## 配置需求

| 项目 | 需求 |
|------|------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU 显存 | 8GB（训练） / 4GB（推理） |
| 训练时间 | ~40-60 分钟（100 epochs，batch_size=8） |

## 数据集统计

- **数据集大小**：4232 个样本
- **数据划分**：80% 训练集，20% 验证集（训练时）
- **测试集划分**：70% 训练，15% 验证，15% 测试（评估时）
- **时间跨度**：128 个月历史数据，预测未来 32 个月
- **输入维度**：16 个指标

数据集包含多种类型的开源项目，涵盖前端框架、后端框架、工具库等。

## 扩展数据集

```bash
# 1. 爬取更多仓库
cd ../backend/DataProcessor
python batch_crawl_opendigger.py --count 100

# 2. 重新转换数据
cd ../../GitPulse
python convert_github_data.py
```

## License

MIT
