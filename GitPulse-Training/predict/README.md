# 单个仓库预测模块

## 功能

`predict_single_repo.py` 用于对单个 GitHub 仓库进行健康度预测，预测未来 32 个月的 16 维活动指标。

## 使用方法

### 基本用法

```bash
# 使用 JSON 格式的时序数据
python predict/predict_single_repo.py --timeseries data/repo_timeseries.json

# 同时提供文本数据
python predict/predict_single_repo.py \
    --timeseries data/repo_timeseries.json \
    --text data/repo_description.txt

# 直接提供文本字符串
python predict/predict_single_repo.py \
    --timeseries data/repo_timeseries.json \
    --text_string "A machine learning library for Python"

# 指定输出文件
python predict/predict_single_repo.py \
    --timeseries data/repo_timeseries.json \
    --output my_prediction.json
```

### 数据格式

#### 时序数据格式（JSON）

```json
{
  "timeseries": [
    [0.1, 0.2, 0.3, ...],  // 第1个月的16维指标
    [0.2, 0.3, 0.4, ...],  // 第2个月的16维指标
    ...
  ]
}
```

或者直接是数组：

```json
[
  [0.1, 0.2, 0.3, ...],
  [0.2, 0.3, 0.4, ...],
  ...
]
```

#### 时序数据格式（CSV）

CSV 文件应包含 16 列，每行代表一个时间步的数据。

#### 文本数据格式

文本文件应为纯文本格式，包含项目描述、技术栈等信息。

## 输出格式

预测结果保存为 JSON 格式：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "model": "CondGRU+Text",
  "checkpoint": "training/checkpoints/best_model_cond_gru_mm.pt",
  "prediction": {
    "shape": [32, 16],
    "data": [[...], [...], ...],
    "description": "预测未来32个月的16维活动指标"
  },
  "statistics": {
    "mean": [...],
    "std": [...],
    "prediction_normalized": [[...], ...],
    "prediction_denormalized": [[...], ...]
  },
  "metrics": {
    "prediction_mean": [...],
    "prediction_std": [...],
    "trend": "increasing"
  }
}
```

## 16 维指标说明

1. OpenRank（项目影响力指数）
2. 活跃度（综合活跃度评分）
3. Star数（当月新增 Star）
4. Fork数（当月新增 Fork）
5. 关注度（关注者增量）
6. 参与者数（活跃参与者）
7. 新增贡献者（新加入的贡献者）
8. 贡献者（活跃贡献者）
9. 不活跃贡献者（流失贡献者）
10. 总线因子（项目风险指标）
11. 新增Issue（新开 Issue）
12. 关闭Issue（关闭的 Issue）
13. Issue评论（Issue 讨论数）
14. 变更请求（PR 数量）
15. PR接受数（合并的 PR）
16. PR审查（PR 审查数）

## 示例

```python
from predict.predict_single_repo import RepoPredictor

# 初始化预测器
predictor = RepoPredictor('training/checkpoints/best_model_cond_gru_mm.pt')

# 准备数据
timeseries_data = [[0.1, 0.2, ...], [0.2, 0.3, ...], ...]  # [T, 16]
text_data = "A Python machine learning library"

# 预测
prediction, stats = predictor.predict(timeseries_data, text_data)

print(f"预测形状: {prediction.shape}")  # (32, 16)
print(f"预测结果: {prediction}")
```

## 模型文件

预测脚本默认使用 `predict/models/best_model.pt`，这是训练好的最优模型（CondGRU+Text）。

如果需要使用其他模型，可以通过 `--checkpoint` 参数指定：

```bash
python predict/predict_single_repo.py \
    --timeseries data.json \
    --checkpoint training/checkpoints/best_model_v4.pt
```

## 注意事项

1. 时序数据至少需要 128 个月的历史数据，如果不足会自动填充零
2. 如果数据超过 128 个月，只使用最后 128 个月
3. 文本数据是可选的，但提供文本数据可以提高预测精度
4. 预测结果会自动反标准化，返回原始尺度
5. 首次使用时，DistilBERT tokenizer 会自动下载（约 250 MB）
