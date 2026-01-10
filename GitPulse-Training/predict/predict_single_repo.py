"""
单个仓库预测脚本
用于对单个 GitHub 仓库进行健康度预测
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
from model.multimodal_ts_v4_1 import MultimodalConditionalGRUV4_1


class RepoPredictor:
    """单个仓库预测器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        初始化预测器
        
        Args:
            checkpoint_path: 模型检查点路径（支持相对路径和绝对路径）
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 处理相对路径：如果是相对路径，从项目根目录查找
        if not os.path.isabs(checkpoint_path):
            # 获取项目根目录（predict 目录的父目录）
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(script_dir, checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
        
        # 加载模型
        self.model = MultimodalConditionalGRUV4_1(
            n_vars=16,
            hist_len=128,
            pred_len=32,
            d_model=128
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"[OK] 模型已加载: {checkpoint_path}")
        print(f"[OK] 使用设备: {self.device}")
    
    def preprocess_timeseries(self, timeseries_data, hist_len=128):
        """
        预处理时序数据
        
        Args:
            timeseries_data: 时序数据列表，每个元素为 [16] 维向量
            hist_len: 历史长度
        
        Returns:
            preprocessed_data: 预处理后的数据 [hist_len, 16]
        """
        # 转换为 numpy 数组
        if isinstance(timeseries_data, list):
            data = np.array(timeseries_data, dtype=np.float32)
        else:
            data = timeseries_data.astype(np.float32)
        
        # 如果数据长度不足，用零填充
        if len(data) < hist_len:
            pad = np.zeros((hist_len - len(data), 16), dtype=np.float32)
            data = np.concatenate([pad, data], axis=0)
        elif len(data) > hist_len:
            # 取最后 hist_len 个时间步
            data = data[-hist_len:]
        
        # Z-score 标准化
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std == 0] = 1  # 避免除零
        normalized = (data - mean) / std
        
        return normalized, mean.flatten(), std.flatten()
    
    def preprocess_text(self, text):
        """
        预处理文本数据
        
        Args:
            text: 文本字符串
        
        Returns:
            input_ids: token IDs
            attention_mask: attention mask
        """
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)
    
    def predict(self, timeseries_data, text_data):
        """
        预测未来趋势
        
        Args:
            timeseries_data: 时序数据，形状为 [T, 16] 或列表
            text_data: 文本数据（字符串）
        
        Returns:
            prediction: 预测结果 [pred_len, 16]
            stats: 统计信息（均值、标准差等）
        """
        # 预处理时序数据
        ts_normalized, mean, std = self.preprocess_timeseries(timeseries_data)
        
        # 预处理文本数据
        input_ids, attention_mask = self.preprocess_text(text_data)
        
        # 转换为 tensor
        ts_tensor = torch.tensor(ts_normalized).unsqueeze(0).to(self.device)  # [1, hist_len, 16]
        input_ids = input_ids.unsqueeze(0).to(self.device)  # [1, text_len]
        attention_mask = attention_mask.unsqueeze(0).to(self.device)  # [1, text_len]
        
        # 预测
        with torch.no_grad():
            prediction = self.model(
                ts_tensor,
                input_ids,
                attention_mask,
                return_auxiliary=False
            ).cpu().numpy()[0]  # [pred_len, 16]
        
        # 反标准化
        prediction_denorm = prediction * std + mean
        
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'prediction_normalized': prediction.tolist(),
            'prediction_denormalized': prediction_denorm.tolist()
        }
        
        return prediction_denorm, stats
    
    def predict_from_file(self, timeseries_file, text_file=None, text_string=None):
        """
        从文件加载数据并预测
        
        Args:
            timeseries_file: 时序数据文件路径（JSON 或 CSV）
            text_file: 文本文件路径（可选）
            text_string: 文本字符串（可选，如果提供则优先使用）
        
        Returns:
            prediction: 预测结果
            stats: 统计信息
        """
        # 加载时序数据
        if timeseries_file.endswith('.json'):
            with open(timeseries_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'timeseries' in data:
                    timeseries_data = data['timeseries']
                elif isinstance(data, list):
                    timeseries_data = data
                else:
                    raise ValueError("无法解析时序数据格式")
        elif timeseries_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(timeseries_file)
            timeseries_data = df.values.tolist()
        else:
            raise ValueError(f"不支持的文件格式: {timeseries_file}")
        
        # 加载文本数据
        if text_string:
            text_data = text_string
        elif text_file:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_data = f.read()
        else:
            text_data = ""  # 如果没有文本，使用空字符串
        
        return self.predict(timeseries_data, text_data)


def main():
    parser = argparse.ArgumentParser(description='单个仓库健康度预测')
    parser.add_argument('--checkpoint', type=str, 
                        default='predict/models/best_model.pt',
                        help='模型检查点路径（默认使用 predict/models/best_model.pt）')
    parser.add_argument('--timeseries', type=str, required=True,
                        help='时序数据文件路径（JSON 或 CSV）')
    parser.add_argument('--text', type=str, default=None,
                        help='文本数据文件路径（可选）')
    parser.add_argument('--text_string', type=str, default=None,
                        help='文本字符串（可选，如果提供则优先使用）')
    parser.add_argument('--output', type=str, default='prediction_result.json',
                        help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GitHub 仓库健康度预测")
    print("=" * 80)
    
    # 初始化预测器
    predictor = RepoPredictor(args.checkpoint, args.device)
    
    # 预测
    print(f"\n加载数据: {args.timeseries}")
    prediction, stats = predictor.predict_from_file(
        args.timeseries,
        args.text,
        args.text_string
    )
    
    # 保存结果
    result = {
        'timestamp': datetime.now().isoformat(),
        'model': 'GitPulse (MultimodalConditionalGRUV4_1)',
        'checkpoint': args.checkpoint,
        'prediction': {
            'shape': list(prediction.shape),
            'data': prediction.tolist(),
            'description': '预测未来32个月的16维活动指标'
        },
        'statistics': stats,
        'metrics': {
            'prediction_mean': np.mean(prediction, axis=0).tolist(),
            'prediction_std': np.std(prediction, axis=0).tolist(),
            'trend': 'increasing' if np.mean(prediction[-1] - prediction[0]) > 0 else 'decreasing'
        }
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 预测完成！")
    print(f"   预测形状: {prediction.shape}")
    print(f"   预测均值: {np.mean(prediction, axis=0)[:3]}... (前3维)")
    print(f"   趋势: {result['metrics']['trend']}")
    print(f"   结果已保存到: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()

