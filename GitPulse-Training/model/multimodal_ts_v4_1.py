"""
GitPulse v4.1: 时序底座对比实验

目的：证明文本在不同时序架构上的普适性价值

底座对比：
1. PatchTST（原 v4，baseline）
2. 标准 Transformer（无 Patch 池化，136 时间步）
3. GRU（文本作为门控偏置）

文本融合策略：完全复用 v4 的最优配置
- 文本注意力引导
- 动态门控（min_weight=0.1, max_weight=0.3）
- 对比学习 + 匹配任务
- 冻结 BERT + 投影层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import math


# ==================== 共享文本编码器（复用 v4） ====================

class TextEncoderV4(nn.Module):
    """
    文本编码器 - 复用 v4 的设计
    - 冻结 BERT
    - 投影层
    - 注意力池化
    """
    def __init__(self, model_name="distilbert-base-uncased", d_model=128):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.proj = nn.Sequential(
            nn.Linear(768, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 注意力池化
        self.attn_pool = nn.Linear(d_model, 1)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden = outputs.last_hidden_state  # [batch, seq, 768]
        seq_feat = self.proj(hidden)  # [batch, seq, d_model]
        
        # 注意力池化得到全局特征
        attn_weights = self.attn_pool(seq_feat)
        attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        global_feat = (seq_feat * attn_weights).sum(dim=1)  # [batch, d_model]
        
        return seq_feat, global_feat


# ==================== 共享融合模块（复用 v4） ====================

class AdaptiveFusion(nn.Module):
    """
    自适应融合模块 - 复用 v4
    - 动态门控权重
    - min_weight 保证文本贡献
    """
    def __init__(self, d_model=128, min_weight=0.1, max_weight=0.3):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, ts_global, text_global):
        combined = torch.cat([ts_global, text_global], dim=-1)
        weight = self.gate(combined)
        weight = self.min_weight + (self.max_weight - self.min_weight) * weight
        return weight


class ContrastiveLoss(nn.Module):
    """对比学习损失 - 复用 v4"""
    def __init__(self, d_model=128):
        super().__init__()
        self.ts_proj = nn.Linear(d_model, 64)
        self.text_proj = nn.Linear(d_model, 64)
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, ts_global, text_global):
        ts_emb = F.normalize(self.ts_proj(ts_global), dim=-1)
        text_emb = F.normalize(self.text_proj(text_global), dim=-1)
        
        batch_size = ts_emb.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=ts_emb.device), 1.0
        
        logits = torch.matmul(ts_emb, text_emb.T) / self.temperature.clamp(min=0.01)
        labels = torch.arange(batch_size, device=ts_emb.device)
        
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        
        return loss, acc


class MatchingLoss(nn.Module):
    """匹配任务损失 - 复用 v4"""
    def __init__(self, d_model=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, ts_global, text_global):
        batch_size = ts_global.size(0)
        device = ts_global.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device), 1.0
        
        pos_pairs = torch.cat([ts_global, text_global], dim=-1)
        pos_labels = torch.ones(batch_size, device=device)
        
        perm = torch.randperm(batch_size, device=device)
        neg_pairs = torch.cat([ts_global, text_global[perm]], dim=-1)
        neg_labels = torch.zeros(batch_size, device=device)
        
        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        logits = self.classifier(pairs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        with torch.no_grad():
            acc = ((logits > 0).float() == labels).float().mean().item()
        
        return loss, acc


# ==================== 底座1: 标准 Transformer ====================

class TransformerTSEncoder(nn.Module):
    """
    标准 Transformer 时序编码器
    - 无 Patch 池化，保留所有时间步
    - 与 v4 对齐的参数规模
    """
    def __init__(self, n_vars=16, d_model=128, n_heads=4, n_layers=2, 
                 ffn_dim=512, dropout=0.1, max_len=256):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [batch, seq_len, n_vars]
        seq_len = x.size(1)
        
        x = self.input_proj(x)
        x = x + self.pe[:, :seq_len, :]
        x = self.encoder(x)
        x = self.norm(x)
        
        global_feat = x.mean(dim=1)
        
        return x, global_feat


class TransformerTextFusion(nn.Module):
    """
    Transformer + 文本注意力引导融合
    
    适配方法：
    - 文本特征生成 [batch, seq_len, 1] 的注意力权重
    - 乘到自注意力权重上（Patch→时间步级）
    - 约束文本权重 β∈[0.1,0.3]
    """
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, min_weight=0.1, max_weight=0.3):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 文本生成时间步级注意力掩码
        self.text_to_temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # 动态门控
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, ts_feat, text_seq, text_global, text_mask=None):
        """
        ts_feat: [batch, seq_len, d_model]
        text_seq: [batch, text_len, d_model]
        text_global: [batch, d_model]
        """
        batch_size, seq_len, d_model = ts_feat.shape
        
        # 1. 文本引导的时间步注意力权重
        temporal_weights = self.text_to_temporal_attn(text_global)  # [batch, 1]
        
        # 2. 自注意力（可选：用文本权重调制）
        self_out, _ = self.self_attn(ts_feat, ts_feat, ts_feat)
        ts_feat = self.norm1(ts_feat + self_out)
        
        # 3. 交叉注意力（时序 query，文本 key/value）
        key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        cross_out, _ = self.cross_attn(ts_feat, text_seq, text_seq, key_padding_mask=key_padding_mask)
        
        # 4. 动态门控权重
        ts_global = ts_feat.mean(dim=1)
        combined = torch.cat([ts_global, text_global], dim=-1)
        gate_weight = self.gate(combined)  # [batch, 1]
        gate_weight = self.min_weight + (self.max_weight - self.min_weight) * gate_weight
        
        # 5. 门控融合
        ts_feat = self.norm2(ts_feat + gate_weight.unsqueeze(-1) * cross_out)
        
        # 6. FFN
        ts_feat = self.norm3(ts_feat + self.ffn(ts_feat))
        
        return ts_feat, gate_weight.mean()


# ==================== 底座2: GRU ====================

class GRUTSEncoder(nn.Module):
    """
    GRU 时序编码器
    """
    def __init__(self, n_vars=16, d_model=128, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            d_model, d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, gate_bias=None):
        # x: [batch, seq_len, n_vars]
        x = self.input_proj(x)
        
        # 如果有门控偏置，加到输入上
        if gate_bias is not None:
            x = x + gate_bias
        
        x, hidden = self.gru(x)
        x = self.norm(x)
        
        # 全局特征：最后时刻的隐状态
        global_feat = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [batch, d_model]
        
        return x, global_feat


class GRUTextFusion(nn.Module):
    """
    GRU + 文本融合（原始方式）
    """
    def __init__(self, d_model=128, seq_len=128, min_weight=0.1, max_weight=0.3):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.seq_len = seq_len
        
        self.text_to_gate_bias = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
    
    def forward(self, ts_feat, ts_global, text_global):
        combined = torch.cat([ts_global, text_global], dim=-1)
        weight = self.gate(combined)
        weight = self.min_weight + (self.max_weight - self.min_weight) * weight
        
        text_contrib = self.text_to_gate_bias(text_global)
        text_contrib = text_contrib.unsqueeze(1)
        
        ts_feat = ts_feat + weight.unsqueeze(-1) * text_contrib
        fused_global = self.fusion(torch.cat([ts_global, text_global * weight], dim=-1))
        
        return ts_feat, fused_global, weight.mean()


# ==================== Conditional GRU（v6 最优策略） ====================

class ConditionalGRUEncoder(nn.Module):
    """
    Conditional GRU - 文本作为初始隐藏状态
    
    这是 v6 实验中表现最好的策略（+20.85%）
    """
    def __init__(self, n_vars=16, d_model=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.input_proj = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            d_model, d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, init_hidden=None):
        x = self.input_proj(x)
        
        if init_hidden is not None:
            init_hidden = init_hidden.unsqueeze(0).expand(self.n_layers, -1, -1).contiguous()
            x, hidden = self.gru(x, init_hidden)
        else:
            x, hidden = self.gru(x)
        
        x = self.norm(x)
        global_feat = hidden[-1]
        
        return x, global_feat


class MultimodalConditionalGRUV4_1(nn.Module):
    """
    v4.1 - Conditional GRU（v6 最优策略）
    
    核心思想：文本特征作为 GRU 初始隐藏状态
    """
    def __init__(
        self,
        n_vars=16,
        hist_len=128,
        pred_len=32,
        d_model=128,
        n_gru_layers=2,
        dropout=0.2,
        text_model="distilbert-base-uncased"
    ):
        super().__init__()
        self.n_vars = n_vars
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 文本编码器
        self.text_encoder = TextEncoderV4(text_model, d_model)
        
        # 文本到隐藏状态
        self.text_to_hidden = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        # Conditional GRU
        self.gru_encoder = ConditionalGRUEncoder(n_vars, d_model, n_gru_layers, dropout)
        
        # 辅助任务
        self.contrastive = ContrastiveLoss(d_model)
        self.matching = MatchingLoss(d_model)
        
        # 条件强度
        self.cond_strength = nn.Parameter(torch.tensor(0.5))
        
        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars)
        )
        self.temporal_proj = nn.Linear(hist_len, pred_len)
        
        self._tokenizer = None
        self._text_model = text_model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = DistilBertTokenizer.from_pretrained(self._text_model)
        return self._tokenizer
    
    def forward(self, ts_input, text_input_ids, text_attention_mask, return_auxiliary=True):
        # 文本编码
        text_seq, text_global = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 生成条件隐藏状态
        cond_hidden = self.text_to_hidden(text_global)
        strength = torch.sigmoid(self.cond_strength)
        init_hidden = strength * cond_hidden
        
        # Conditional GRU 编码
        ts_feat, ts_global = self.gru_encoder(ts_input, init_hidden)
        
        # 辅助损失
        if return_auxiliary and self.training:
            cl_loss, cl_acc = self.contrastive(ts_global, text_global)
            ml_loss, ml_acc = self.matching(ts_global, text_global)
        else:
            cl_loss = torch.tensor(0.0, device=ts_input.device)
            ml_loss = torch.tensor(0.0, device=ts_input.device)
            cl_acc, ml_acc = 0.0, 0.0
        
        # 预测
        pred_feat = self.pred_head(ts_feat)
        pred_feat = pred_feat.transpose(1, 2)
        prediction = self.temporal_proj(pred_feat).transpose(1, 2)
        
        if return_auxiliary:
            metrics = {
                'cl_acc': cl_acc,
                'ml_acc': ml_acc,
                'text_weight': strength.item()
            }
            return prediction, cl_loss, ml_loss, metrics
        
        return prediction


# ==================== 完整模型 ====================

class MultimodalTransformerV4_1(nn.Module):
    """
    v4.1 - Transformer 底座 + v4 文本融合
    """
    def __init__(
        self,
        n_vars=16,
        hist_len=128,
        pred_len=32,
        d_model=128,
        n_heads=4,
        n_layers=2,
        ffn_dim=512,
        dropout=0.1,
        text_model="distilbert-base-uncased",
        min_text_weight=0.1,
        max_text_weight=0.3
    ):
        super().__init__()
        self.n_vars = n_vars
        self.hist_len = hist_len
        self.pred_len = pred_len
        
        # 时序编码器
        self.ts_encoder = TransformerTSEncoder(
            n_vars, d_model, n_heads, n_layers, ffn_dim, dropout
        )
        
        # 文本编码器（复用 v4）
        self.text_encoder = TextEncoderV4(text_model, d_model)
        
        # 融合层（2层）
        self.fusion_layers = nn.ModuleList([
            TransformerTextFusion(d_model, n_heads, dropout, min_text_weight, max_text_weight)
            for _ in range(2)
        ])
        
        # 辅助任务（复用 v4）
        self.contrastive = ContrastiveLoss(d_model)
        self.matching = MatchingLoss(d_model)
        
        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars)
        )
        self.temporal_proj = nn.Linear(hist_len, pred_len)
        
        self._tokenizer = None
        self._text_model = text_model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = DistilBertTokenizer.from_pretrained(self._text_model)
        return self._tokenizer
    
    def forward(self, ts_input, text_input_ids, text_attention_mask, return_auxiliary=True):
        # 编码
        ts_feat, ts_global = self.ts_encoder(ts_input)
        text_seq, text_global = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 辅助损失
        if return_auxiliary and self.training:
            cl_loss, cl_acc = self.contrastive(ts_global, text_global)
            ml_loss, ml_acc = self.matching(ts_global, text_global)
        else:
            cl_loss = torch.tensor(0.0, device=ts_input.device)
            ml_loss = torch.tensor(0.0, device=ts_input.device)
            cl_acc, ml_acc = 0.0, 0.0
        
        # 融合
        text_weights = []
        for fusion in self.fusion_layers:
            ts_feat, tw = fusion(ts_feat, text_seq, text_global, text_attention_mask)
            text_weights.append(tw)
        
        # 预测
        pred_feat = self.pred_head(ts_feat)
        pred_feat = pred_feat.transpose(1, 2)
        prediction = self.temporal_proj(pred_feat).transpose(1, 2)
        
        if return_auxiliary:
            metrics = {
                'cl_acc': cl_acc,
                'ml_acc': ml_acc,
                'text_weight': sum(tw.item() if isinstance(tw, torch.Tensor) else tw 
                                   for tw in text_weights) / len(text_weights)
            }
            return prediction, cl_loss, ml_loss, metrics
        
        return prediction


class MultimodalGRUV4_1(nn.Module):
    """
    v4.1 - GRU 底座 + v4 文本融合
    """
    def __init__(
        self,
        n_vars=16,
        hist_len=128,
        pred_len=32,
        d_model=128,
        n_gru_layers=2,
        dropout=0.2,
        text_model="distilbert-base-uncased",
        min_text_weight=0.1,
        max_text_weight=0.3
    ):
        super().__init__()
        self.n_vars = n_vars
        self.hist_len = hist_len
        self.pred_len = pred_len
        
        # GRU 编码器
        self.ts_encoder = GRUTSEncoder(n_vars, d_model, n_gru_layers, dropout)
        
        # 文本编码器（复用 v4）
        self.text_encoder = TextEncoderV4(text_model, d_model)
        
        # 融合模块
        self.fusion = GRUTextFusion(d_model, hist_len, min_text_weight, max_text_weight)
        
        # 辅助任务（复用 v4）
        self.contrastive = ContrastiveLoss(d_model)
        self.matching = MatchingLoss(d_model)
        
        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars)
        )
        self.temporal_proj = nn.Linear(hist_len, pred_len)
        
        self._tokenizer = None
        self._text_model = text_model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = DistilBertTokenizer.from_pretrained(self._text_model)
        return self._tokenizer
    
    def forward(self, ts_input, text_input_ids, text_attention_mask, return_auxiliary=True):
        # 文本编码
        text_seq, text_global = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 时序编码
        ts_feat, ts_global = self.ts_encoder(ts_input)
        
        # 辅助损失
        if return_auxiliary and self.training:
            cl_loss, cl_acc = self.contrastive(ts_global, text_global)
            ml_loss, ml_acc = self.matching(ts_global, text_global)
        else:
            cl_loss = torch.tensor(0.0, device=ts_input.device)
            ml_loss = torch.tensor(0.0, device=ts_input.device)
            cl_acc, ml_acc = 0.0, 0.0
        
        # 融合
        ts_feat, fused_global, text_weight = self.fusion(ts_feat, ts_global, text_global)
        
        # 预测
        pred_feat = self.pred_head(ts_feat)
        pred_feat = pred_feat.transpose(1, 2)
        prediction = self.temporal_proj(pred_feat).transpose(1, 2)
        
        if return_auxiliary:
            metrics = {
                'cl_acc': cl_acc,
                'ml_acc': ml_acc,
                'text_weight': text_weight.item() if isinstance(text_weight, torch.Tensor) else text_weight
            }
            return prediction, cl_loss, ml_loss, metrics
        
        return prediction


# ==================== 纯时序模型（对照组） ====================

class TransformerTSOnlyV4_1(nn.Module):
    """纯 Transformer 时序模型"""
    def __init__(self, n_vars=16, hist_len=128, pred_len=32, d_model=128,
                 n_heads=4, n_layers=2, ffn_dim=512, dropout=0.1):
        super().__init__()
        
        self.ts_encoder = TransformerTSEncoder(n_vars, d_model, n_heads, n_layers, ffn_dim, dropout)
        
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars)
        )
        self.temporal_proj = nn.Linear(hist_len, pred_len)
    
    def forward(self, ts_input):
        ts_feat, _ = self.ts_encoder(ts_input)
        pred_feat = self.pred_head(ts_feat)
        pred_feat = pred_feat.transpose(1, 2)
        prediction = self.temporal_proj(pred_feat).transpose(1, 2)
        return prediction


class GRUTSOnlyV4_1(nn.Module):
    """纯 GRU 时序模型"""
    def __init__(self, n_vars=16, hist_len=128, pred_len=32, d_model=128,
                 n_gru_layers=2, dropout=0.2):
        super().__init__()
        
        self.ts_encoder = GRUTSEncoder(n_vars, d_model, n_gru_layers, dropout)
        
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars)
        )
        self.temporal_proj = nn.Linear(hist_len, pred_len)
    
    def forward(self, ts_input):
        ts_feat, _ = self.ts_encoder(ts_input)
        pred_feat = self.pred_head(ts_feat)
        pred_feat = pred_feat.transpose(1, 2)
        prediction = self.temporal_proj(pred_feat).transpose(1, 2)
        return prediction


class CondGRUTSOnlyV4_1(nn.Module):
    """纯 CondGRU 时序模型（使用 ConditionalGRUEncoder，但不使用文本）"""
    def __init__(self, n_vars=16, hist_len=128, pred_len=32, d_model=128,
                 n_gru_layers=2, dropout=0.2):
        super().__init__()
        
        # 使用 ConditionalGRUEncoder，但不传入 init_hidden
        self.ts_encoder = ConditionalGRUEncoder(n_vars, d_model, n_gru_layers, dropout)
        
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars)
        )
        self.temporal_proj = nn.Linear(hist_len, pred_len)
    
    def forward(self, ts_input):
        ts_feat, _ = self.ts_encoder(ts_input, init_hidden=None)
        pred_feat = self.pred_head(ts_feat)
        pred_feat = pred_feat.transpose(1, 2)
        prediction = self.temporal_proj(pred_feat).transpose(1, 2)
        return prediction


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("GitPulse v4.1 - 时序底座对比测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Transformer
    transformer_mm = MultimodalTransformerV4_1().to(device)
    transformer_ts = TransformerTSOnlyV4_1().to(device)
    
    # GRU
    gru_mm = MultimodalGRUV4_1().to(device)
    gru_ts = GRUTSOnlyV4_1().to(device)
    
    print(f"\n参数量对比:")
    print(f"  Transformer 多模态: {count_parameters(transformer_mm) / 1e6:.3f}M")
    print(f"  Transformer 纯时序: {count_parameters(transformer_ts) / 1e6:.3f}M")
    print(f"  GRU 多模态: {count_parameters(gru_mm) / 1e6:.3f}M")
    print(f"  GRU 纯时序: {count_parameters(gru_ts) / 1e6:.3f}M")
    
    # 测试
    ts = torch.randn(4, 128, 16).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    enc = tokenizer(["Test project"] * 4, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    transformer_mm.train()
    pred, cl, ml, metrics = transformer_mm(ts, enc["input_ids"].to(device), enc["attention_mask"].to(device))
    print(f"\nTransformer: pred={pred.shape}, tw={metrics['text_weight']:.3f}")
    
    gru_mm.train()
    pred, cl, ml, metrics = gru_mm(ts, enc["input_ids"].to(device), enc["attention_mask"].to(device))
    print(f"GRU: pred={pred.shape}, tw={metrics['text_weight']:.3f}")
    
    print("\n✓ 测试通过!")

