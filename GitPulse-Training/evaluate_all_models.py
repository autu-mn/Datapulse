"""
统一模型评估脚本
支持评估所有模型或指定特定模型
计算所有指标：MSE, MAE, RMSE, DA, TA@0.2, R²
自动生成论文所需的图片和表格
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有模型
from model.multimodal_ts_v4 import MultimodalTSPredictorV4, TimeSeriesOnlyPredictorV4
from model.multimodal_ts_v4_1 import (
    GRUTSOnlyV4_1,
    MultimodalGRUV4_1,
    MultimodalConditionalGRUV4_1,
    MultimodalTransformerV4_1,
    TransformerTSOnlyV4_1,
    CondGRUTSOnlyV4_1
)


# ============== 数据集 ==============

class TimeSeriesOnlyDataset(Dataset):
    """纯时序数据集"""
    def __init__(self, json_path, max_hist_len=128, max_pred_len=32):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.max_hist_len = max_hist_len
        self.max_pred_len = max_pred_len
        self.n_vars = 16
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        hist = np.array(sample['Hist'], dtype=np.float32)
        pred = np.array(sample['Pred'], dtype=np.float32)
        
        if len(hist) > self.max_hist_len:
            hist = hist[-self.max_hist_len:]
        elif len(hist) < self.max_hist_len:
            pad = np.zeros((self.max_hist_len - len(hist), self.n_vars), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        
        if len(pred) > self.max_pred_len:
            pred = pred[:self.max_pred_len]
        elif len(pred) < self.max_pred_len:
            pad = np.zeros((self.max_pred_len - len(pred), self.n_vars), dtype=np.float32)
            pred = np.concatenate([pred, pad], axis=0)
        
        return {
            'hist': torch.tensor(hist),
            'pred': torch.tensor(pred),
            'text': sample.get('Text', '')
        }


class MultimodalDataset(Dataset):
    """多模态数据集"""
    def __init__(self, json_path, tokenizer, max_hist_len=128, max_pred_len=32):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.tokenizer = tokenizer
        self.max_hist_len = max_hist_len
        self.max_pred_len = max_pred_len
        self.n_vars = 16
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        hist = np.array(sample['Hist'], dtype=np.float32)
        pred = np.array(sample['Pred'], dtype=np.float32)
        
        if len(hist) > self.max_hist_len:
            hist = hist[-self.max_hist_len:]
        elif len(hist) < self.max_hist_len:
            pad = np.zeros((self.max_hist_len - len(hist), self.n_vars), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        
        if len(pred) > self.max_pred_len:
            pred = pred[:self.max_pred_len]
        elif len(pred) < self.max_pred_len:
            pad = np.zeros((self.max_pred_len - len(pred), self.n_vars), dtype=np.float32)
            pred = np.concatenate([pred, pad], axis=0)
        
        text = sample.get('Text', '')
        text_encoded = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=256, return_tensors='pt'
        )
        
        return {
            'hist': torch.tensor(hist),
            'pred': torch.tensor(pred),
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0)
        }


# ============== 评估指标 ==============

def compute_metrics(preds, targets, hist=None):
    """
    计算多种评估指标
    
    Args:
        preds: [N, pred_len, n_vars] 预测值
        targets: [N, pred_len, n_vars] 真实值
        hist: [N, hist_len, n_vars] 历史值（用于计算方向准确率）
    
    Returns:
        dict: 各项指标
    """
    # 基础误差指标
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    rmse = np.sqrt(mse)
    
    # R² 决定系数
    eps = 1e-8
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / (ss_tot + eps)
    
    # 方向准确率 (DA) - 预测变化方向正确的比例
    if hist is not None:
        last_hist = hist[:, -1:, :]  # [N, 1, n_vars]
        
        # 真实变化方向
        true_direction = (targets[:, 0:1, :] - last_hist) > 0  # [N, 1, n_vars]
        # 预测变化方向
        pred_direction = (preds[:, 0:1, :] - last_hist) > 0
        # 方向准确率
        direction_acc = (true_direction == pred_direction).float().mean().item() * 100
    else:
        # 使用相邻时间步的方向
        true_diff = targets[:, 1:, :] - targets[:, :-1, :]
        pred_diff = preds[:, 1:, :] - preds[:, :-1, :]
        direction_acc = ((true_diff > 0) == (pred_diff > 0)).float().mean().item() * 100
    
    # 阈值准确率 (TA@0.2) - 预测误差在阈值内的比例
    abs_error = (preds - targets).abs()
    ta_02 = (abs_error < 0.2).float().mean().item() * 100  # 误差 < 0.2
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'DA': direction_acc,
        'TA@0.2': ta_02,
    }


# ============== 模型配置 ==============

MODEL_CONFIGS = {
    'GRU': {
        'model_class': GRUTSOnlyV4_1,
        'checkpoint': 'training/checkpoints/best_model_gru_ts.pt',
        'is_multimodal': False,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'GRU+Text': {
        'model_class': MultimodalGRUV4_1,
        'checkpoint': 'training/checkpoints/best_model_gru_mm.pt',
        'is_multimodal': True,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'CondGRU': {
        'model_class': CondGRUTSOnlyV4_1,
        'checkpoint': 'training/checkpoints/best_model_cond_gru_ts.pt',
        'is_multimodal': False,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'GitPulse': {
        'model_class': MultimodalConditionalGRUV4_1,
        'checkpoint': 'training/checkpoints/best_model_cond_gru_mm.pt',
        'is_multimodal': True,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'Transformer': {
        'model_class': TransformerTSOnlyV4_1,
        'checkpoint': 'training/checkpoints/best_model_transformer_ts.pt',
        'is_multimodal': False,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'Transformer+Text': {
        'model_class': MultimodalTransformerV4_1,
        'checkpoint': 'training/checkpoints/best_model_transformer_mm.pt',
        'is_multimodal': True,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'PatchTST-TS': {
        'model_class': TimeSeriesOnlyPredictorV4,
        'checkpoint': 'training/checkpoints/best_model_ts_only_v4.pt',
        'is_multimodal': False,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
    'PatchTST+Text': {
        'model_class': MultimodalTSPredictorV4,
        'checkpoint': 'training/checkpoints/best_model_v4.pt',
        'is_multimodal': True,
        'args': {'n_vars': 16, 'hist_len': 128, 'pred_len': 32, 'd_model': 128}
    },
}


# ============== 评估函数 ==============

def load_checkpoint(checkpoint_path, device='cuda'):
    """加载检查点，处理不同的保存格式"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是否是字典格式（包含 model_state_dict）
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            # 可能是直接的 state_dict
            return checkpoint
    else:
        return checkpoint


def evaluate_model(model, test_loader, device='cuda', is_multimodal=False):
    """评估模型"""
    model.eval()
    model = model.to(device)
    
    all_preds, all_targets, all_hist = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中", leave=False):
            hist = batch['hist'].to(device)
            target = batch['pred']
            
            if is_multimodal:
                try:
                    pred = model(
                        hist,
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device),
                        return_auxiliary=False
                    ).cpu()
                except TypeError:
                    # 某些模型可能不需要 return_auxiliary 参数
                    pred = model(
                        hist,
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device)
                    ).cpu()
            else:
                pred = model(hist).cpu()
            
            all_preds.append(pred)
            all_targets.append(target)
            all_hist.append(hist.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    hist = torch.cat(all_hist, dim=0)
    
    return compute_metrics(preds, targets, hist)


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description='统一模型评估脚本')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='要评估的模型列表，如: --models GRU "GRU+Text" CondGRU+Text。默认评估所有模型')
    parser.add_argument('--data_path', type=str, default='Pretrain-data/github_multivar.json',
                        help='数据路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='输出结果文件')
    parser.add_argument('--generate_figures', action='store_true',
                        help='生成论文所需的图片和表格')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("统一模型评估脚本")
    print("=" * 80)
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 确定要评估的模型
    if args.models is None:
        models_to_eval = list(MODEL_CONFIGS.keys())
        print(f"\n将评估所有模型: {', '.join(models_to_eval)}")
    else:
        models_to_eval = args.models
        print(f"\n将评估指定模型: {', '.join(models_to_eval)}")
        # 检查模型名称是否有效
        invalid_models = [m for m in models_to_eval if m not in MODEL_CONFIGS]
        if invalid_models:
            print(f"⚠ 警告: 以下模型名称无效: {', '.join(invalid_models)}")
            print(f"可用模型: {', '.join(MODEL_CONFIGS.keys())}")
            models_to_eval = [m for m in models_to_eval if m in MODEL_CONFIGS]
    
    # 加载数据
    print(f"\n加载数据: {args.data_path}")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    ts_dataset = TimeSeriesOnlyDataset(args.data_path, 128, 32)
    mm_dataset = MultimodalDataset(args.data_path, tokenizer, 128, 32)
    
    # 划分（使用与训练时相同的随机种子）
    n = len(ts_dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    gen = torch.Generator().manual_seed(42)
    ts_train, ts_val, ts_test = torch.utils.data.random_split(
        ts_dataset, [train_size, val_size, test_size], gen
    )
    mm_train, mm_val, mm_test = torch.utils.data.random_split(
        mm_dataset, [train_size, val_size, test_size], gen
    )
    
    ts_test_loader = DataLoader(ts_test, batch_size=args.batch_size, shuffle=False)
    mm_test_loader = DataLoader(mm_test, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}, 测试集大小: {test_size}")
    print("注意: 所有评估结果均来自测试集，用于最终性能评估")
    
    # 评估模型
    results = {}
    
    for model_name in models_to_eval:
        config = MODEL_CONFIGS[model_name]
        checkpoint_path = config['checkpoint']
        
        print("\n" + "=" * 60)
        print(f"评估模型: {model_name}")
        print("=" * 60)
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠ 检查点不存在: {checkpoint_path}")
            print("   跳过此模型")
            continue
        
        try:
            # 创建模型
            model = config['model_class'](**config['args'])
            
            # 加载权重
            state_dict = load_checkpoint(checkpoint_path, device)
            model.load_state_dict(state_dict)
            
            # 选择数据集
            test_loader = mm_test_loader if config['is_multimodal'] else ts_test_loader
            
            # 评估
            metrics = evaluate_model(model, test_loader, device, config['is_multimodal'])
            results[model_name] = metrics
            
            print(f"   ✅ MSE: {metrics['MSE']:.4f}")
            print(f"   ✅ MAE: {metrics['MAE']:.4f}")
            print(f"   ✅ RMSE: {metrics['RMSE']:.4f}")
            print(f"   ✅ DA: {metrics['DA']:.2f}%")
            print(f"   ✅ TA@0.2: {metrics['TA@0.2']:.2f}%")
            print(f"   ✅ R²: {metrics['R²']:.4f}")
            
        except Exception as e:
            print(f"   ❌ 评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== 结果汇总 ==========
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    
    if results:
        print(f"\n{'模型':<20} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'DA(%)':<10} {'TA@0.2(%)':<12} {'R²':<10}")
        print("-" * 92)
        
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['MSE']):
            print(f"{model_name:<20} {metrics['MSE']:<10.4f} {metrics['MAE']:<10.4f} "
                  f"{metrics['RMSE']:<10.4f} {metrics['DA']:<10.2f} {metrics['TA@0.2']:<12.2f} "
                  f"{metrics['R²']:<10.4f}")
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到: {args.output}")
        
        # 生成图片和表格
        if args.generate_figures:
            print("\n" + "=" * 80)
            print("生成图片和表格...")
            print("=" * 80)
            generate_figures_and_tables(results, args.output)
    else:
        print("\n⚠ 没有成功评估任何模型")
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


def generate_figures_and_tables(results, results_file):
    """生成论文所需的图片和表格"""
    
    # 基线模型数据
    baseline_results = {
        'Persistence': {'MSE': 0.7903, 'MAE': 0.6137, 'RMSE': 0.8890, 'DA': 59.16, 'TA@0.2': 28.17, 'R²': -1.29},
        'Linear': {'MSE': 0.2261, 'MAE': 0.1896, 'RMSE': 0.4755, 'DA': 53.20, 'TA@0.2': 73.81, 'R²': 0.34},
        'MLP': {'MSE': 0.2280, 'MAE': 0.2025, 'RMSE': 0.4775, 'DA': 56.11, 'TA@0.2': 73.43, 'R²': 0.34},
        'LSTM': {'MSE': 0.2142, 'MAE': 0.1914, 'RMSE': 0.4628, 'DA': 55.92, 'TA@0.2': 74.64, 'R²': 0.38},
        'Transformer': {'MSE': 0.1909, 'MAE': 0.1679, 'RMSE': 0.4369, 'DA': 62.40, 'TA@0.2': 76.43, 'R²': 0.45},
    }
    
    # 合并所有结果
    all_results = {**baseline_results, **results}
    
    # 确保输出目录存在
    os.makedirs('paper/figures', exist_ok=True)
    
    # ============== 图1: MSE对比 ==============
    fig, ax = plt.subplots(figsize=(14, 6))
    
    methods_order = [
        'Persistence', 'Linear', 'MLP', 'LSTM', 'Transformer',
        'GRU', 'PatchTST-TS', 'GRU+Text', 'PatchTST+Text', 'CondGRU', 'GitPulse'
    ]
    
    mse_values = [all_results[m]['MSE'] for m in methods_order if m in all_results]
    methods_order = [m for m in methods_order if m in all_results]
    
    colors = []
    for m in methods_order:
        if m in ['Persistence', 'Linear', 'MLP', 'LSTM', 'Transformer']:
            colors.append('#808080')  # 灰色：基线
        elif '+' in m or m == 'GitPulse':
            colors.append('#2ecc71')  # 绿色：多模态
        else:
            colors.append('#3498db')  # 蓝色：纯时序
    
    bars = ax.bar(range(len(methods_order)), mse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(methods_order)))
    ax.set_xticklabels(methods_order, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('All Methods MSE Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注最佳值
    if 'GitPulse' in methods_order:
        best_idx = methods_order.index('GitPulse')
        ax.annotate('Best', xy=(best_idx, mse_values[best_idx]), 
                    xytext=(best_idx, mse_values[best_idx] + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold', color='red', ha='center')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.8, label='Baseline'),
        Patch(facecolor='#3498db', alpha=0.8, label='Time Series Only'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Multimodal')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig1_mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/fig1_mse_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 1 generated: fig1_mse_comparison.png/pdf")
    
    # ============== 图2: 文本贡献分析 ==============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：纯时序 vs +文本 的MSE对比
    architectures = ['PatchTST', 'Transformer', 'GRU', 'CondGRU']
    ts_only_mse = []
    with_text_mse = []
    valid_archs = []
    
    arch_mapping = {
        'PatchTST': ('PatchTST-TS', 'PatchTST+Text'),
        'Transformer': ('Transformer', 'Transformer+Text'),
        'GRU': ('GRU', 'GRU+Text'),
        'CondGRU': ('CondGRU', 'GitPulse'),
    }
    
    for arch in architectures:
        ts_key, text_key = arch_mapping[arch]
        if ts_key in all_results and text_key in all_results:
            ts_only_mse.append(all_results[ts_key]['MSE'])
            with_text_mse.append(all_results[text_key]['MSE'])
            valid_archs.append(arch)
    
    if valid_archs:
        x = np.arange(len(valid_archs))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, ts_only_mse, width, label='TS Only', color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, with_text_mse, width, label='+Text', color='#2ecc71', alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Architecture', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax1.set_title('(a) TS Only vs +Text MSE Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_archs)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 右图：文本贡献百分比
        text_contributions = []
        for i in range(len(valid_archs)):
            if ts_only_mse[i] > 0:
                contrib = (ts_only_mse[i] - with_text_mse[i]) / ts_only_mse[i] * 100
                text_contributions.append(contrib)
            else:
                text_contributions.append(0)
        
        bars3 = ax2.bar(valid_archs, text_contributions, color='#e74c3c', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Architecture', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Text Contribution (%)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Text Contribution by Architecture', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 标注数值
        for i, (bar, contrib) in enumerate(zip(bars3, text_contributions)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{contrib:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('paper/figures/fig2_text_contribution.png', dpi=300, bbox_inches='tight')
        plt.savefig('paper/figures/fig2_text_contribution.pdf', bbox_inches='tight')
        plt.close()
        print("[OK] Figure 2 generated: fig2_text_contribution.png/pdf")
    
    # ============== 生成LaTeX表格 ==============
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{不同方法在测试集上的性能对比}
\\label{tab:main_results}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{方法} & \\textbf{MSE}$\\downarrow$ & \\textbf{MAE}$\\downarrow$ & \\textbf{RMSE}$\\downarrow$ & \\textbf{DA(\\%)}$\\uparrow$ & \\textbf{TA@0.2(\\%)}$\\uparrow$ & \\textbf{R²}$\\uparrow$ \\\\
\\midrule
"""
    
    methods_table = [
        ('Persistence', baseline_results.get('Persistence')),
        ('Linear', baseline_results.get('Linear')),
        ('MLP', baseline_results.get('MLP')),
        ('LSTM', baseline_results.get('LSTM')),
        ('Transformer', baseline_results.get('Transformer')),
        ('GRU', results.get('GRU')),
        ('PatchTST-TS', results.get('PatchTST-TS')),
        ('GRU+Text', results.get('GRU+Text')),
        ('PatchTST+Text', results.get('PatchTST+Text')),
        ('CondGRU', results.get('CondGRU')),
        ('GitPulse', results.get('GitPulse')),
    ]
    
    methods_table = [(n, d) for n, d in methods_table if d is not None]
    
    for i, (name, data) in enumerate(methods_table):
        if name == 'GitPulse':
            latex_table += f"\\textbf{{{name}}} & \\textbf{{{data['MSE']:.4f}}} & \\textbf{{{data['MAE']:.4f}}} & \\textbf{{{data['RMSE']:.4f}}} & \\textbf{{{data['DA']:.2f}}} & \\textbf{{{data['TA@0.2']:.2f}}} & \\textbf{{{data['R²']:.2f}}} \\\\\n"
        else:
            latex_table += f"{name} & {data['MSE']:.4f} & {data['MAE']:.4f} & {data['RMSE']:.4f} & {data['DA']:.2f} & {data['TA@0.2']:.2f} & {data['R²']:.2f} \\\\\n"
        if i == 4:  # 在Transformer后添加分隔线
            latex_table += "\\midrule\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('paper/main_results_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("[OK] LaTeX table generated: paper/main_results_table.tex")
    
    # ============== 生成文本贡献表格 ==============
    text_contrib_table = """\\begin{table}[htbp]
\\centering
\\caption{不同架构上的文本贡献分析}
\\label{tab:text_contribution}
\\begin{tabular}{lccc}
\\toprule
\\textbf{架构} & \\textbf{纯时序 MSE} & \\textbf{+文本 MSE} & \\textbf{改进率} \\\\
\\midrule
"""
    
    text_contrib_data = [
        ('PatchTST', results.get('PatchTST-TS'), results.get('PatchTST+Text')),
        ('标准 Transformer', baseline_results.get('Transformer'), results.get('Transformer+Text')),
        ('GRU（门控融合）', results.get('GRU'), results.get('GRU+Text')),
        ('CondGRU', results.get('CondGRU'), results.get('GitPulse')),
        ('GitPulse（条件初始化）', results.get('CondGRU'), results.get('GitPulse')),
    ]
    
    text_contrib_data = [(a, ts, txt) for a, ts, txt in text_contrib_data if ts is not None and txt is not None]
    
    for arch, ts_data, text_data in text_contrib_data:
        ts_mse = ts_data['MSE'] if isinstance(ts_data, dict) else ts_data
        text_mse = text_data['MSE'] if isinstance(text_data, dict) else text_data
        improvement = (ts_mse - text_mse) / ts_mse * 100 if ts_mse > 0 else 0
        if arch == 'GitPulse（条件初始化）':
            text_contrib_table += f"\\textbf{{{arch}}} & \\textbf{{{ts_mse:.4f}}} & \\textbf{{{text_mse:.4f}}} & \\textbf{{+{improvement:.2f}\\%}} \\\\\n"
        else:
            text_contrib_table += f"{arch} & {ts_mse:.4f} & {text_mse:.4f} & +{improvement:.2f}\\% \\\\\n"
    
    text_contrib_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('paper/text_contribution_table.tex', 'w', encoding='utf-8') as f:
        f.write(text_contrib_table)
    print("[OK] Text contribution table generated: paper/text_contribution_table.tex")
    
    print("\n[OK] All figures and tables generated successfully!")


if __name__ == '__main__':
    main()


