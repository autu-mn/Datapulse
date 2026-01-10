# GitPulse 项目结构说明

## 核心文件

### 模型代码
- `model/multimodal_ts_v4.py` - PatchTST+Text（对比模型，原 v4）
- `model/multimodal_ts_v4_1.py` - **GitPulse**（条件 GRU + 文本，最优模型）及对比架构

### 训练脚本（可复现）
- `training/train_multimodal_v4.py` - 训练 PatchTST+Text（对比模型）
- `training/train_multimodal_v4_1.py` - **训练 GitPulse 及架构对比实验**（推荐）

### 评估脚本（可复现）
- `evaluate_all_models.py` - **统一模型评估脚本**（推荐使用）
  - 支持评估所有模型或指定模型
  - 计算完整指标：MSE, MAE, RMSE, DA, TA@0.2, R²
  - 使用方法：`python evaluate_all_models.py --models GRU "GRU+Text"`

- `final_evaluation.py` - 基线对比评估脚本
  - 评估所有基线模型和 GitPulse
  - 生成最终对比结果

### 预测脚本（生产使用）
- `predict/predict_single_repo.py` - **单个仓库预测脚本**
  - 用于对单个 GitHub 仓库进行健康度预测
  - 使用方法：`python predict/predict_single_repo.py --timeseries data.json --text description.txt`
  - 默认使用 `predict/models/best_model.pt`（最优模型 CondGRU+Text）
  - 详见 `predict/README.md`

- `predict/models/best_model.pt` - **训练好的最优模型权重（GitPulse）**
  - 模型架构：MultimodalConditionalGRUV4_1（条件 GRU + 文本）
  - 性能：MSE=0.0886, DA=67.28%, TA@0.2=81.41%, R²=0.70
  - 详见 `predict/models/README.md`

### 数据预处理
- `convert_github_data.py` - GitHub 数据转换脚本
  - 将原始数据转换为模型训练格式

## 数据文件

- `Pretrain-data/github_multivar.json` - 主要数据集（训练/评估用）
- `Pretrain-data/github_multivar_summary.csv` - 数据摘要

## 结果文件

- `final_evaluation_results.json` - 基线对比评估结果
- `missing_metrics_results.json` - 补充的评估指标结果
- `training/checkpoints/*.pt` - 训练好的模型权重
- `training/checkpoints/*.json` - 训练过程结果

## 论文相关

- `paper/main.tex` - LaTeX 主文档
- `paper/experiment.tex` - 实验部分（已更新）
- `paper/figures/` - 实验图表
- `paper/编译说明.md` - LaTeX 编译说明

## 快速开始

### 1. 训练模型

```bash
# 训练 GitPulse（条件 GRU + 文本，最优模型）
cd training
python train_multimodal_v4_1.py --epochs 100 --batch_size 8

# 训练所有架构对比实验
python train_multimodal_v4_1.py --epochs 100 --batch_size 8 --compare

# 训练 PatchTST+Text（对比模型）
python train_multimodal_v4.py --epochs 100 --batch_size 8
```

### 2. 评估模型

```bash
# 评估所有模型
python evaluate_all_models.py

# 评估指定模型
python evaluate_all_models.py --models GRU "GRU+Text" GitPulse

# 生成图表和 LaTeX 表格
python evaluate_all_models.py --generate_figures
```

### 3. 预测单个仓库

```bash
# 使用预测脚本
python predict/predict_single_repo.py \
    --timeseries repo_data.json \
    --text repo_description.txt \
    --output prediction.json
```

## 模型检查点位置

### 预测用模型（推荐）
- `predict/models/best_model.pt` - **GitPulse（最优模型）**，用于生产预测
  - 已从 `training/checkpoints/best_model_cond_gru_mm.pt` 复制
  - 性能：MSE=0.0886, DA=67.28%, TA@0.2=81.41%, R²=0.70

### 训练检查点
所有训练好的模型权重保存在 `training/checkpoints/` 目录下：
- `best_model_cond_gru_mm.pt` - **GitPulse**（条件 GRU + 文本，最优模型）
- `best_model_v4.pt` - PatchTST+Text（对比模型）
- `best_model_gru_mm.pt` - GRU+Text（门控融合）
- `best_model_gru_ts.pt` - GRU（纯时序）
- `best_model_cond_gru_ts.pt` - CondGRU（纯时序，无文本）
- 等等...

## 注意事项

1. **评估脚本**：推荐使用 `evaluate_all_models.py`，它统一管理所有模型的评估
2. **预测脚本**：使用 `predict/predict_single_repo.py` 进行单个仓库预测，默认使用 `predict/models/best_model.pt`
3. **数据格式**：确保时序数据为 16 维，文本数据为字符串格式
4. **设备**：默认使用 CUDA，如果 GPU 不可用会自动回退到 CPU
5. **模型文件**：预测模块已包含最优模型权重，可直接使用


1. **评估脚本**：推荐使用 `evaluate_all_models.py`，它统一管理所有模型的评估
2. **预测脚本**：使用 `predict/predict_single_repo.py` 进行单个仓库预测，默认使用 `predict/models/best_model.pt`
3. **数据格式**：确保时序数据为 16 维，文本数据为字符串格式
4. **设备**：默认使用 CUDA，如果 GPU 不可用会自动回退到 CPU
5. **模型文件**：预测模块已包含最优模型权重，可直接使用

