# 基线测试 (Baseline Test)

本目录包含基线模型的测试代码和结果，用于对比 GitPulse 与基线方法的性能。

## 基线模型

- **Persistence**：持久化模型（使用最后值预测）
- **Linear**：线性回归
- **MLP**：多层感知机
- **LSTM**：长短期记忆网络
- **Transformer**：标准 Transformer（纯时序）

## 使用方法

```bash
# 运行基线测试
python baseline_test.py
```

## 结果

基线测试结果会与 GitPulse 模型进行对比，展示文本信息的价值。

