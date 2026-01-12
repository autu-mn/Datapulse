# GitHub 多变量时序数据集处理文档

## 数据集概述

本数据集包含 4232 个样本，每个样本包含：
- **历史时序数据**：48 个月（4年）的 16 个指标
- **预测目标**：未来 12 个月（1年）的 16 个指标
- **文本描述**：项目信息、Issue、Commit 等结构化文本

数据集通过滑动窗口技术从原始 GitHub 仓库数据生成，大幅增加了训练样本数量。

## 数据来源

原始数据来自 **DataPulse** 爬取的 GitHub 开源项目时序数据，包含：
- OpenDigger 指标（OpenRank、活跃度、Star、Fork 等）
- Issue 和 PR 统计信息
- Commit 记录
- 项目元数据（描述、标签等）

## 数据处理流程

### 1. 数据加载

从每个仓库的 `timeseries_for_model/all_months.json` 文件中加载时序数据，包含：
- 每个月的 OpenDigger 指标
- Issue 分类信息（feature、bug、question）
- Commit 文本数据
- 项目摘要信息

### 2. 多变量时序提取

从原始数据中提取 16 个指标，构建 `[T, 16]` 的多变量时序矩阵：

```python
# 16 个指标（按顺序）
METRICS = [
    "OpenRank",       # 0: 项目影响力指数
    "活跃度",         # 1: 综合活跃度评分
    "Star数",         # 2: 当月新增 Star
    "Fork数",         # 3: 当月新增 Fork
    "关注度",         # 4: 关注者增量
    "参与者数",       # 5: 活跃参与者
    "新增贡献者",     # 6: 新加入的贡献者
    "贡献者",         # 7: 活跃贡献者
    "不活跃贡献者",   # 8: 流失贡献者
    "总线因子",       # 9: 项目风险指标
    "新增Issue",      # 10: 新开 Issue
    "关闭Issue",      # 11: 关闭的 Issue
    "Issue评论",      # 12: Issue 讨论数
    "变更请求",       # 13: PR 数量
    "PR接受数",       # 14: 合并的 PR
    "PR审查",         # 15: PR 审查数
]
```

### 3. 数据质量检查

对每个仓库的数据进行质量检查：

**检查项**：
1. **最小月份数**：至少需要 60 个月的数据（48 个月历史 + 12 个月预测）
2. **核心指标非零率**：OpenRank 和活跃度的非零率必须 > 50%
3. **数据完整性**：检查缺失值和异常值

**过滤规则**：
- 核心指标（OpenRank、活跃度）非零率 < 50% 的样本被过滤
- 数据不足 60 个月的仓库被跳过
- 空文件或损坏的数据被跳过

### 4. Z-score 标准化

对每个仓库的时序数据进行 **Z-score 标准化**（按列）：

```python
def normalize_data(data: np.ndarray):
    mean = np.mean(data, axis=0)  # 计算每个指标的均值
    std = np.std(data, axis=0)    # 计算每个指标的标准差
    std[std == 0] = 1  # 避免除零
    normalized = (data - mean) / std  # Z-score标准化
    return normalized, mean, std
```

**标准化原因**：
- 不同指标的量级差异很大（如 OpenRank 可能是几十，Star数可能是几千）
- 标准化后所有指标均值为 0，标准差为 1，便于模型学习
- 加速训练收敛，提高模型性能

**反标准化**：
每个样本保存了标准化参数（`NormMean` 和 `NormStd`），可用于将预测结果转换回原始尺度：
```python
original_value = normalized_value * norm_std + norm_mean
```

### 5. 滑动窗口生成

使用滑动窗口技术从每个仓库生成多个样本：

**窗口参数**：
- **历史长度（hist_len）**：48 个月（4年）
- **预测长度（pred_len）**：12 个月（1年）
- **滑动步长（stride）**：6 个月（每半年滑动一次）

**示例**：
```
仓库有 72 个月的数据：
- 窗口1: 月份 0-59  (历史: 0-47, 预测: 48-59)
- 窗口2: 月份 6-65  (历史: 6-53, 预测: 54-65)
- 窗口3: 月份 12-71 (历史: 12-59, 预测: 60-71)
```

**优势**：
- 大幅增加训练样本数量（平均每个仓库生成 3-5 个样本）
- 增加数据多样性，提高模型泛化能力
- 充分利用长期数据

### 6. 文本上下文生成

为每个样本生成结构化文本描述，包含：

**文本结构**：
1. **项目基本信息**
   - 项目名称和时间窗口

2. **活跃度趋势**（基于时序数据）
   - OpenRank 当前值和趋势（rising/falling/stable/volatile）
   - 活跃度评分和趋势
   - Star/Fork 增长统计

3. **贡献者统计**
   - 活跃贡献者数量
   - 新增贡献者统计
   - 流失率（Churn rate）
   - 总线因子（项目风险）

4. **Issue 统计**
   - 打开/关闭数量
   - 解决率（Resolution rate）
   - 讨论活跃度（评论数）
   - Issue 趋势

5. **代码贡献统计**
   - PR 提交和合并数量
   - 合并率
   - 审查文化（reviews/PR）

6. **Issue 分类统计**（最近 12 个月）
   - Bug、Feature、Question 的数量和比例

7. **项目描述**（如果有）
   - README 中的项目描述（截断到 150 字符）

8. **关键 Issue 标题**（最近 12 个月）
   - 最多 8-10 个 Issue 标题
   - 包含分类标签（[bug]、[feature]、[question]）

9. **热门标签/主题**
   - 最常见的 Issue 标签（最多 8 个）

10. **主要 Commit 摘要**（最近 6 个月）
    - 最多 5-6 个 Commit 消息摘要
    - 截断到 80 字符

**文本长度限制**：
- 总长度限制在 2500 字符以内
- 超过限制时截断，保留最重要的信息

**文本提取策略**：
- Issue：从最近 12 个月中提取，按时间顺序
- Commit：从最近 6 个月中提取，按时间顺序
- 优先保留高参与度、关键标签的 Issue/Commit

## 数据格式

### JSON 文件结构

```json
{
  "metrics": ["OpenRank", "活跃度", ...],  // 16个指标名称
  "n_dims": 16,                            // 指标数量
  "hist_len": 48,                          // 历史长度（月）
  "pred_len": 12,                          // 预测长度（月）
  "stride": 6,                             // 滑动步长（月）
  "samples": [
    {
      "Repo": "1024pix/pix",               // 仓库名称
      "WindowStart": "2018-03",            // 窗口开始时间
      "WindowEnd": "2023-02",              // 窗口结束时间
      "HistLen": 48,                       // 历史长度
      "PredLen": 12,                       // 预测长度
      "Hist": [[...], [...], ...],         // 历史数据 [48, 16]，已标准化
      "Pred": [[...], [...], ...],         // 预测目标 [12, 16]，已标准化
      "Text": "...",                       // 文本描述（结构化）
      "NormMean": [...],                   // 标准化均值 [16]
      "NormStd": [...]                     // 标准化标准差 [16]
    },
    ...
  ]
}
```

### 单个样本结构

**时序数据**：
- `Hist`: `[48, 16]` - 48 个时间步 × 16 个指标（已标准化）
- `Pred`: `[12, 16]` - 12 个时间步 × 16 个指标（已标准化）

**文本数据**：
- `Text`: 结构化文本描述，包含统计信息、趋势分析和关键 Issue/Commit

**标准化参数**：
- `NormMean`: `[16]` - 每个指标的均值
- `NormStd`: `[16]` - 每个指标的标准差

## 16 个指标详细说明

| 维度 | 指标名称 | 说明 | 典型范围 |
|------|----------|------|----------|
| 0 | OpenRank | 项目影响力指数，综合评估项目在开源社区的影响力 | 0-1000+ |
| 1 | 活跃度 | 综合活跃度评分，反映项目的开发活跃程度 | 0-500+ |
| 2 | Star数 | 当月新增的 Star 数量 | 0-10000+ |
| 3 | Fork数 | 当月新增的 Fork 数量 | 0-1000+ |
| 4 | 关注度 | 关注者增量，反映项目受关注程度 | 0-1000+ |
| 5 | 参与者数 | 活跃参与者数量，参与代码贡献的人数 | 0-100+ |
| 6 | 新增贡献者 | 新加入的贡献者数量 | 0-50+ |
| 7 | 贡献者 | 活跃贡献者数量（当月有提交） | 0-100+ |
| 8 | 不活跃贡献者 | 流失的贡献者数量 | 0-50+ |
| 9 | 总线因子 | 项目风险指标，反映项目对关键贡献者的依赖程度 | 1-20+ |
| 10 | 新增Issue | 新打开的 Issue 数量 | 0-1000+ |
| 11 | 关闭Issue | 关闭的 Issue 数量 | 0-1000+ |
| 12 | Issue评论 | Issue 讨论的评论总数 | 0-10000+ |
| 13 | 变更请求 | Pull Request 数量 | 0-1000+ |
| 14 | PR接受数 | 合并的 PR 数量 | 0-1000+ |
| 15 | PR审查 | PR 审查评论数量 | 0-5000+ |

## 数据集统计

### 基本统计

- **总样本数**：4232 个
- **仓库数量**：约 800-1000 个（通过滑动窗口扩展）
- **平均每个仓库样本数**：约 4-5 个
- **时间跨度**：2015-2023 年
- **数据来源**：GitHub 开源项目（Web框架、ML库、工具库等）

### 数据分布

- **历史长度**：48 个月（固定）
- **预测长度**：12 个月（固定）
- **滑动步长**：6 个月
- **最小数据要求**：60 个月

### 文本统计

- **平均文本长度**：约 1156 字符
- **最大文本长度**：2500 字符（限制）
- **Issue 标题数量**：平均 8 个/样本
- **Commit 摘要数量**：平均 5 个/样本

## 使用方法

### 1. 加载数据集

```python
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# 加载 JSON 文件
with open('Pretrain-data/github_multivar.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取样本
samples = data['samples']
print(f"总样本数: {len(samples)}")

# 查看第一个样本
sample = samples[0]
print(f"仓库: {sample['Repo']}")
print(f"时间窗口: {sample['WindowStart']} 到 {sample['WindowEnd']}")
print(f"历史数据形状: {np.array(sample['Hist']).shape}")  # (48, 16)
print(f"预测数据形状: {np.array(sample['Pred']).shape}")  # (12, 16)
```

### 2. 反标准化预测结果

```python
# 获取标准化参数
norm_mean = np.array(sample['NormMean'])
norm_std = np.array(sample['NormStd'])

# 假设模型预测结果（标准化后的值）
predicted_normalized = np.array([...])  # [12, 16]

# 反标准化到原始尺度
predicted_original = predicted_normalized * norm_std + norm_mean
```

### 3. 数据划分

```python
from torch.utils.data import random_split

# 数据集划分（70% 训练，15% 验证，15% 测试）
n = len(samples)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    samples, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```
