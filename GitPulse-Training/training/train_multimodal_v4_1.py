"""
GitPulse v4.1 训练脚本 - 时序底座对比实验

目的：证明文本在不同时序架构上的普适性价值

对比实验：
1. Transformer + 文本 vs 纯 Transformer
2. GRU + 文本 vs 纯 GRU

判定标准：
- text_contribution_pct > 0 即证明文本有正向贡献
- 不需要达到 PatchTST 的 10.67%，只要稳定正向增益

使用:
    python train_multimodal_v4_1.py --epochs 100
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import DistilBertTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.multimodal_ts_v4_1 import (
    MultimodalTransformerV4_1, TransformerTSOnlyV4_1,
    MultimodalGRUV4_1, GRUTSOnlyV4_1,
    MultimodalConditionalGRUV4_1,
    count_parameters
)


class GitHubDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_hist_len=128, max_pred_len=32):
        self.tokenizer = tokenizer
        self.max_hist_len = max_hist_len
        self.max_pred_len = max_pred_len
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data['samples']
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        hist = np.array(sample['Hist'], dtype=np.float32)
        pred = np.array(sample['Pred'], dtype=np.float32)
        n_vars = hist.shape[1] if len(hist.shape) > 1 else 16
        
        if len(hist) > self.max_hist_len:
            hist = hist[-self.max_hist_len:]
        elif len(hist) < self.max_hist_len:
            pad = np.zeros((self.max_hist_len - len(hist), n_vars), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        
        if len(pred) > self.max_pred_len:
            pred = pred[:self.max_pred_len]
        elif len(pred) < self.max_pred_len:
            pad = np.zeros((self.max_pred_len - len(pred), n_vars), dtype=np.float32)
            pred = np.concatenate([pred, pad], axis=0)
        
        text = sample.get('Text', '')
        text_encoded = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=256, return_tensors='pt'
        )
        
        return {
            'hist': torch.tensor(hist, dtype=torch.float32),
            'pred': torch.tensor(pred, dtype=torch.float32),
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0)
        }


def train_multimodal(model, train_loader, val_loader, device, epochs, patience, 
                     model_name, output_dir, lr=5e-4, lambda_cl=0.1, lambda_ml=0.05):
    """训练多模态模型"""
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
    
    best_mse = float('inf')
    best_mae = None
    best_rmse = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_pred = 0
        total_cl_acc = 0
        total_ml_acc = 0
        total_tw = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}")
        for batch in pbar:
            hist = batch['hist'].to(device)
            targets = batch['pred'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            pred, cl_loss, ml_loss, metrics = model(hist, input_ids, attention_mask, return_auxiliary=True)
            
            pred_loss = criterion(pred, targets)
            total_loss = pred_loss + lambda_cl * cl_loss + lambda_ml * ml_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_pred += pred_loss.item()
            total_cl_acc += metrics['cl_acc']
            total_ml_acc += metrics['ml_acc']
            total_tw += metrics['text_weight']
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{pred_loss.item():.4f}',
                'cl': f'{metrics["cl_acc"]:.1%}',
                'tw': f'{metrics["text_weight"]:.2f}'
            })
        
        # 验证
        model.eval()
        val_mse = 0
        val_mae = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hist = batch['hist'].to(device)
                targets = batch['pred'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                pred = model(hist, input_ids, attention_mask, return_auxiliary=False)
                
                val_mse += nn.MSELoss(reduction='sum')(pred, targets).item()
                val_mae += torch.abs(pred - targets).sum().item()
                val_samples += pred.numel()
        
        val_mse /= val_samples
        val_mae /= val_samples
        val_rmse = np.sqrt(val_mse)
        
        scheduler.step(val_mse)
        
        print(f"[{model_name}] Epoch {epoch}: loss={total_pred/n_batches:.4f}, "
              f"val_mse={val_mse:.4f}, cl={total_cl_acc/n_batches:.1%}, tw={total_tw/n_batches:.2f}")
        
        if val_mse < best_mse:
            best_mse = val_mse
            best_mae = val_mae
            best_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mse': val_mse
            }, os.path.join(output_dir, f'best_model_{model_name}.pt'))
            print(f"  -> Saved (MSE={val_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_mse, best_mae, best_rmse, best_epoch


def train_ts_only(model, train_loader, val_loader, device, epochs, patience, 
                  model_name, output_dir, lr=1e-3):
    """训练纯时序模型"""
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
    
    best_mse = float('inf')
    best_mae = None
    best_rmse = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}")
        for batch in pbar:
            hist = batch['hist'].to(device)
            targets = batch['pred'].to(device)
            
            optimizer.zero_grad()
            pred = model(hist)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 验证
        model.eval()
        val_mse = 0
        val_mae = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hist = batch['hist'].to(device)
                targets = batch['pred'].to(device)
                pred = model(hist)
                
                val_mse += nn.MSELoss(reduction='sum')(pred, targets).item()
                val_mae += torch.abs(pred - targets).sum().item()
                val_samples += pred.numel()
        
        val_mse /= val_samples
        val_mae /= val_samples
        val_rmse = np.sqrt(val_mse)
        
        scheduler.step(val_mse)
        
        print(f"[{model_name}] Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, val_mse={val_mse:.4f}")
        
        if val_mse < best_mse:
            best_mse = val_mse
            best_mae = val_mae
            best_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mse': val_mse
            }, os.path.join(output_dir, f'best_model_{model_name}.pt'))
            print(f"  -> Saved (MSE={val_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_mse, best_mae, best_rmse, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../Pretrain-data/github_multivar.json')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hist_len', type=int, default=128)
    parser.add_argument('--pred_len', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--lambda_ml', type=float, default=0.05)
    parser.add_argument('--min_text_weight', type=float, default=0.1)
    parser.add_argument('--max_text_weight', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GitPulse v4.1 - 时序底座对比实验")
    print("目的：证明文本在不同架构上的普适性价值")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Text weight: [{args.min_text_weight}, {args.max_text_weight}]")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = GitHubDataset(data_path, tokenizer, args.hist_len, args.pred_len)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    patience = 20
    results = {}
    
    # ==================== 1. Transformer + 文本 ====================
    print("\n" + "=" * 70)
    print("1. Training Transformer + Text")
    print("=" * 70)
    
    transformer_mm = MultimodalTransformerV4_1(
        n_vars=16, hist_len=args.hist_len, pred_len=args.pred_len,
        d_model=args.d_model, min_text_weight=args.min_text_weight, max_text_weight=args.max_text_weight
    ).to(args.device)
    
    print(f"参数量: {count_parameters(transformer_mm) / 1e6:.3f}M")
    
    mse, mae, rmse, epoch = train_multimodal(
        transformer_mm, train_loader, val_loader, args.device,
        args.epochs, patience, 'transformer_mm', args.output_dir,
        args.lr, args.lambda_cl, args.lambda_ml
    )
    results['Transformer+Text'] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'epoch': epoch}
    
    # ==================== 2. 纯 Transformer ====================
    print("\n" + "=" * 70)
    print("2. Training Transformer (TS-only)")
    print("=" * 70)
    
    transformer_ts = TransformerTSOnlyV4_1(
        n_vars=16, hist_len=args.hist_len, pred_len=args.pred_len, d_model=args.d_model
    ).to(args.device)
    
    print(f"参数量: {count_parameters(transformer_ts) / 1e6:.3f}M")
    
    mse, mae, rmse, epoch = train_ts_only(
        transformer_ts, train_loader, val_loader, args.device,
        args.epochs, patience, 'transformer_ts', args.output_dir, 1e-3
    )
    results['Transformer'] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'epoch': epoch}
    
    # ==================== 3. GRU + 文本 ====================
    print("\n" + "=" * 70)
    print("3. Training GRU + Text")
    print("=" * 70)
    
    gru_mm = MultimodalGRUV4_1(
        n_vars=16, hist_len=args.hist_len, pred_len=args.pred_len,
        d_model=args.d_model, min_text_weight=args.min_text_weight, max_text_weight=args.max_text_weight
    ).to(args.device)
    
    print(f"参数量: {count_parameters(gru_mm) / 1e6:.3f}M")
    
    mse, mae, rmse, epoch = train_multimodal(
        gru_mm, train_loader, val_loader, args.device,
        args.epochs, patience, 'gru_mm', args.output_dir,
        args.lr, args.lambda_cl, args.lambda_ml
    )
    results['GRU+Text'] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'epoch': epoch}
    
    # ==================== 4. 纯 GRU ====================
    print("\n" + "=" * 70)
    print("4. Training GRU (TS-only)")
    print("=" * 70)
    
    gru_ts = GRUTSOnlyV4_1(
        n_vars=16, hist_len=args.hist_len, pred_len=args.pred_len, d_model=args.d_model
    ).to(args.device)
    
    print(f"参数量: {count_parameters(gru_ts) / 1e6:.3f}M")
    
    mse, mae, rmse, epoch = train_ts_only(
        gru_ts, train_loader, val_loader, args.device,
        args.epochs, patience, 'gru_ts', args.output_dir, 1e-3
    )
    results['GRU'] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'epoch': epoch}
    
    # ==================== 5. Conditional GRU + 文本（v6最优策略） ====================
    print("\n" + "=" * 70)
    print("5. Training Conditional GRU + Text (Best Strategy from v6)")
    print("=" * 70)
    
    cond_gru_mm = MultimodalConditionalGRUV4_1(
        n_vars=16, hist_len=args.hist_len, pred_len=args.pred_len, d_model=args.d_model
    ).to(args.device)
    
    print(f"参数量: {count_parameters(cond_gru_mm) / 1e6:.3f}M")
    
    mse, mae, rmse, epoch = train_multimodal(
        cond_gru_mm, train_loader, val_loader, args.device,
        args.epochs, patience, 'cond_gru_mm', args.output_dir,
        args.lr, args.lambda_cl, args.lambda_ml
    )
    results['CondGRU+Text'] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'epoch': epoch}
    
    # ==================== 结果汇总 ====================
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'Epoch'}")
    print("-" * 65)
    
    for name, r in results.items():
        print(f"{name:<25} {r['mse']:<10.4f} {r['mae']:<10.4f} {r['rmse']:<10.4f} {r['epoch']}")
    
    # ==================== 文本贡献分析 ====================
    print("\n" + "=" * 70)
    print("文本贡献分析 (text_contribution_pct)")
    print("=" * 70)
    
    # Transformer
    transformer_contrib = (results['Transformer']['mse'] - results['Transformer+Text']['mse']) / results['Transformer']['mse'] * 100
    print(f"\nTransformer:")
    print(f"  纯时序 MSE: {results['Transformer']['mse']:.4f}")
    print(f"  +文本 MSE: {results['Transformer+Text']['mse']:.4f}")
    print(f"  文本贡献: {transformer_contrib:+.2f}%")
    
    if transformer_contrib > 0:
        print(f"  ✅ 文本对 Transformer 有 {transformer_contrib:.2f}% 的正向贡献")
    else:
        print(f"  ⚠ 文本对 Transformer 贡献为负 ({transformer_contrib:.2f}%)")
    
    # GRU（普通融合）
    gru_contrib = (results['GRU']['mse'] - results['GRU+Text']['mse']) / results['GRU']['mse'] * 100
    print(f"\nGRU (普通融合):")
    print(f"  纯时序 MSE: {results['GRU']['mse']:.4f}")
    print(f"  +文本 MSE: {results['GRU+Text']['mse']:.4f}")
    print(f"  文本贡献: {gru_contrib:+.2f}%")
    
    if gru_contrib > 0:
        print(f"  ✅ 文本对 GRU 有 {gru_contrib:.2f}% 的正向贡献")
    else:
        print(f"  ⚠ 文本对 GRU 贡献为负 ({gru_contrib:.2f}%)")
    
    # Conditional GRU（v6最优策略）
    cond_gru_contrib = (results['GRU']['mse'] - results['CondGRU+Text']['mse']) / results['GRU']['mse'] * 100
    print(f"\nConditional GRU (v6最优策略):")
    print(f"  纯时序 MSE: {results['GRU']['mse']:.4f}")
    print(f"  +文本 MSE: {results['CondGRU+Text']['mse']:.4f}")
    print(f"  文本贡献: {cond_gru_contrib:+.2f}%")
    
    if cond_gru_contrib > 0:
        print(f"  ✅ 文本对 Conditional GRU 有 {cond_gru_contrib:.2f}% 的正向贡献")
    else:
        print(f"  ⚠ 文本对 Conditional GRU 贡献为负 ({cond_gru_contrib:.2f}%)")
    
    # ==================== 结论 ====================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    positive_count = sum([transformer_contrib > 0, gru_contrib > 0, cond_gru_contrib > 0])
    
    if positive_count >= 2:
        print("\n🏆 文本在多数时序架构上都有正向贡献！")
        print("   → 证明了文本信息的普适性价值")
    elif positive_count == 1:
        print("\n✓ 文本在部分架构上有正向贡献")
    else:
        print("\n⚠ 当前实验中文本贡献有限")
    
    # 对比 PatchTST (v4) 的 10.67%
    print(f"\n📊 文本贡献汇总:")
    print(f"   PatchTST (v4 baseline): +10.67%")
    print(f"   Transformer: {transformer_contrib:+.2f}%")
    print(f"   GRU (普通融合): {gru_contrib:+.2f}%")
    print(f"   Conditional GRU (v6最优): {cond_gru_contrib:+.2f}%")
    
    # 找最优
    best_contrib = max(transformer_contrib, gru_contrib, cond_gru_contrib)
    if cond_gru_contrib == best_contrib:
        print(f"\n🏆 Conditional GRU 是最优融合策略 ({cond_gru_contrib:+.2f}%)")
    
    # 保存结果
    final_results = {
        'results': results,
        'text_contribution': {
            'Transformer': transformer_contrib,
            'GRU': gru_contrib,
            'CondGRU': cond_gru_contrib,
            'PatchTST_v4_reference': 10.67
        },
        'conclusion': {
            'transformer_positive': transformer_contrib > 0,
            'gru_positive': gru_contrib > 0,
            'cond_gru_positive': cond_gru_contrib > 0,
            'best_strategy': 'CondGRU' if cond_gru_contrib == best_contrib else ('Transformer' if transformer_contrib == best_contrib else 'GRU')
        }
    }
    
    with open(os.path.join(args.output_dir, 'v4_1_comparison_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📁 结果已保存到: {args.output_dir}/v4_1_comparison_results.json")
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

