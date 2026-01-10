# 消融实验 (Ablation Test)

本目录包含消融实验的代码和结果，用于分析不同组件对模型性能的贡献。

## 实验内容

- **指标消融**：分析不同评估指标的重要性
- **架构消融**：分析不同架构组件的作用
- **融合策略消融**：分析不同文本融合策略的效果

## 文件说明

- `comprehensive_metric_ablation.py` - 综合指标消融实验
- `metric_ablation.py` - 指标消融实验
- `results/` - 实验结果目录
  - `all_architecture_results.json` - 所有架构结果
  - `per_metric_results.json` - 每个指标的结果
  - `*.pdf`, `*.png` - 可视化图表

## 使用方法

```bash
# 运行综合指标消融实验
python comprehensive_metric_ablation.py

# 运行指标消融实验
python metric_ablation.py
```

