# 模型权重文件

本目录包含训练好的模型权重文件，用于预测单个仓库的健康度。

## 文件说明

- `best_model.pt` - **最优模型权重（GitPulse）**
  - 模型架构：MultimodalConditionalGRUV4_1（条件 GRU + 文本融合）
  - 性能：MSE=0.0886, DA=67.28%, TA@0.2=81.41%, R²=0.70
  - 这是所有实验中表现最好的模型

## 模型信息

- **架构**：GitPulse（条件 GRU + 文本融合）
- **输入**：128 个月历史数据（16 维）+ 文本描述
- **输出**：32 个月未来预测（16 维）
- **参数量**：约 1.2M

## 使用方法

预测脚本会自动使用此目录下的模型：

```bash
# 使用默认模型（predict/models/best_model.pt）
python predict/predict_single_repo.py --timeseries data.json

# 或指定其他模型
python predict/predict_single_repo.py \
    --timeseries data.json \
    --checkpoint training/checkpoints/best_model_v4.pt
```

## 模型来源

此模型权重来自训练脚本 `training/train_multimodal_v4_1.py`，使用条件 GRU 初始化策略训练得到。GitPulse 是 CondGRU+Text 的正式命名。

## 注意事项

- 确保模型文件完整，如果文件损坏可能导致预测失败
- 模型文件大小约为 5-10 MB（取决于 PyTorch 版本）
- 首次使用时，DistilBERT tokenizer 会自动下载（约 250 MB）

