# 微调实验 (Fine-tuning)

本目录包含模型微调实验的代码和结果，用于分析不同微调策略的效果。

## 微调策略

- **Zero-shot**：零样本（不微调）
- **Freeze**：冻结预训练模型
- **Full**：全参数微调
- **Layerwise**：逐层微调

## 模型

- **CondGRU**：条件 GRU 模型
- **Transformer**：Transformer 模型

## 文件说明

- `finetune_all_v4_1.py` - 完整微调实验脚本
- `finetune_v4_1.py` - 微调实验脚本
- `results/` - 微调结果目录
  - `*_freeze_best.pt` - 冻结策略最佳模型
  - `*_full_best.pt` - 全参数微调最佳模型
  - `*_layerwise_best.pt` - 逐层微调最佳模型
  - `*_result.json` - 各策略结果
  - `finetune_summary.json` - 微调总结

## 使用方法

```bash
# 运行完整微调实验
python finetune_all_v4_1.py

# 运行单个微调实验
python finetune_v4_1.py --strategy freeze
```

## 结果

微调实验结果会对比不同策略的性能，帮助选择最优微调方法。

