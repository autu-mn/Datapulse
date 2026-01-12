# CHAOSS 社区健康评价模块

## 📋 概述

本模块实现了基于 **CHAOSS (Community Health Analytics for Open Source Software)** 框架的社区健康度评估系统。将 OpenDigger 的 16 个时序指标映射到 CHAOSS 的 6 个评价维度，通过多维加权评分、异常值处理、数据质量评估等方法，输出 0-100 的健康评分和具体改进建议。

## 🎯 核心功能

- **6 维度评估**：活跃度、贡献者结构、响应效率、社区多样性、项目治理、技术健康
- **智能评分算法**：基于百分位分布、数据质量加权、异常值降权
- **改进建议生成**：针对每个维度的具体优化建议
- **可视化支持**：生成雷达图数据，支持前端展示

## 📊 CHAOSS 维度与指标映射

| 维度 | OpenDigger 指标 | 权重 | 说明 |
|------|----------------|------|------|
| **活跃度** (Activity) | Activity, Participants, New Contributors | 0.2 | 项目整体活跃程度 |
| **贡献者结构** (Contributor Diversity) | Contributors, Inactive Contributors, Bus Factor | 0.2 | 贡献者分布与依赖度 |
| **响应效率** (Response Efficiency) | Issues New, Issues Closed, Issue Comments | 0.15 | Issue 处理效率 |
| **社区多样性** (Community Diversity) | New Contributors, Contributors, Participants | 0.15 | 社区参与多样性 |
| **项目治理** (Governance) | Change Requests, Change Requests Accepted, Change Requests Reviews | 0.15 | PR 处理流程健康度 |
| **技术健康** (Technical Health) | OpenRank, Stars, Forks, Attention | 0.15 | 项目技术影响力 |

## 🏗️ 架构设计

```
CHAOSSEvaluation/
├── chaoss_calculator.py      # 主计算器，评估逻辑
├── chaoss_mapper.py          # 维度映射器
├── chaoss_metric_config.py   # 指标配置（权重、归一化参数）
├── distribution_aligner.py   # 百分位分布对齐器
└── quality_utils.py          # 数据质量评估工具
```

## 📖 使用方法

### 基本用法

```python
from CHAOSSEvaluation import CHAOSSEvaluator

# 初始化评估器（需要 DataService 实例）
evaluator = CHAOSSEvaluator(data_service=data_service)

# 评估仓库
result = evaluator.evaluate_repo('facebook/react')

# 返回结果结构
{
    'repo_key': 'facebook/react',
    'time_range': {
        'start': '2020-01',
        'end': '2024-12',
        'total_months': 48
    },
    'dimension_scores': {
        'Activity': 75.5,
        'Contributor Diversity': 68.2,
        'Response Efficiency': 82.1,
        'Community Diversity': 71.3,
        'Governance': 79.4,
        'Technical Health': 88.7
    },
    'final_scores': {
        'overall_score': 77.5,
        'weighted_average': 77.5
    },
    'recommendations': [
        {
            'dimension': 'Contributor Diversity',
            'score': 68.2,
            'suggestion': '建议增加新贡献者参与，降低核心贡献者依赖'
        },
        ...
    ]
}
```

### API 端点

#### 获取 CHAOSS 评估结果

```http
GET /api/repo/{repo_key}/chaoss
```

**参数：**
- `repo_key`: 仓库标识（格式：`owner/repo` 或 `owner_repo`）

**响应示例：**

```json
{
  "repo_key": "facebook/react",
  "time_range": {
    "start": "2020-01",
    "end": "2024-12",
    "total_months": 48
  },
  "dimension_scores": {
    "Activity": 75.5,
    "Contributor Diversity": 68.2,
    "Response Efficiency": 82.1,
    "Community Diversity": 71.3,
    "Governance": 79.4,
    "Technical Health": 88.7
  },
  "final_scores": {
    "overall_score": 77.5,
    "weighted_average": 77.5
  },
  "recommendations": [...]
}
```

#### 获取维度映射信息

```http
GET /api/chaoss/dimensions
```

返回所有维度的详细映射信息。

## 🔬 算法说明

### 评分流程

1. **数据预处理**
   - 从 DataService 获取时序数据（16 个 OpenDigger 指标）
   - 按月份组织数据，去除无效值

2. **按月计算原始分数**
   - 对每个维度，从相关指标提取月度数据
   - 使用 `chaoss_mapper.py` 的映射规则聚合指标
   - 应用归一化函数（百分位/基准/回退）

3. **异常值处理**
   - 使用 IQR（四分位距）方法识别异常值
   - 按指标类型设置不同的 IQR 倍数阈值
   - 对异常值进行降权而非删除

4. **数据质量加权**
   - 评估每个月份的数据完整性
   - 根据数据质量调整权重
   - 高质量数据权重更高

5. **最终聚合**
   - 对处理后的月度分数取加权平均
   - 应用维度权重
   - 输出 0-100 的最终分数

### 百分位分布对齐

为了提高评分的一致性和可比性，系统使用 `PercentileDistributionAligner` 对分数进行分布对齐：

- 维护最近 500 个项目的分数分布
- 将新项目的原始分数映射到标准分布
- 避免因数据分布变化导致的评分偏差

### 改进建议生成

系统会根据各维度得分自动生成改进建议：

- **高分维度（>80）**：保持优势，持续优化
- **中等维度（60-80）**：识别提升空间，给出具体建议
- **低分维度（<60）**：重点关注，提供详细改进方案

## ⚙️ 配置说明

### 指标配置 (`chaoss_metric_config.py`)

```python
METRIC_CONFIG = {
    'Activity': {
        'weight': 0.2,
        'metrics': ['Activity', 'Participants', 'New Contributors'],
        'normalization': 'percentile',
        'iqr_multiplier': 1.5  # 异常值检测阈值
    },
    ...
}
```

### 自定义配置

可以通过修改 `chaoss_metric_config.py` 来调整：

- 维度权重
- 指标映射关系
- 归一化方法
- 异常值检测参数

## 📈 性能优化

- **缓存机制**：评估结果可缓存，避免重复计算
- **批量处理**：支持批量评估多个仓库
- **异步处理**：对于大型仓库，可考虑异步评估

## 🔗 相关资源

- [CHAOSS 官方文档](https://chaoss.community/)
- [OpenDigger 指标说明](https://github.com/X-lab2017/open-digger)
- [前端展示组件](../../frontend/src/components/CHAOSSEvaluation.tsx)

## 📝 开发说明

### 添加新维度

1. 在 `chaoss_metric_config.py` 中添加维度配置
2. 在 `chaoss_mapper.py` 中实现映射逻辑
3. 更新前端展示组件（如需要）

### 调试

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

评估器会在关键步骤输出日志，便于调试。

## 🐛 已知问题

- 对于创建时间不足 3 个月的仓库，评分可能不够准确
- 数据缺失较多的仓库，评分会相应降权

## 📄 许可证

本项目遵循 MIT 许可证。

