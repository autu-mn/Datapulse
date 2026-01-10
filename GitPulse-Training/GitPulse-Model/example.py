"""
GitPulse 使用示例

安装: pip install gitpulse
"""

import torch
from gitpulse import GitPulseModel
from transformers import DistilBertTokenizer

def main():
    print("=" * 60)
    print("GitPulse 使用示例")
    print("=" * 60)
    
    # 1. 从 HuggingFace Hub 加载模型
    print("\n1. 加载模型...")
    model = GitPulseModel.from_pretrained("Patronum-ZJ/GitPulse", device='cpu')
    print("✓ 模型加载成功")
    
    # 2. 准备 tokenizer
    print("\n2. 加载 tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✓ Tokenizer 加载成功")
    
    # 3. 准备输入数据
    print("\n3. 准备输入数据...")
    # 历史时序数据: [batch, hist_len=128, n_vars=16]
    time_series = torch.randn(1, 128, 16)
    
    # 项目描述文本
    text = "A Python library for machine learning and data science"
    
    print(f"   - 时序数据形状: {time_series.shape}")
    print(f"   - 文本: {text}")
    
    # 4. 进行预测
    print("\n4. 进行预测...")
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            time_series=time_series,
            text=text,
            tokenizer=tokenizer
        )
    
    print(f"✓ 预测完成")
    print(f"   - 预测结果形状: {predictions.shape}")  # [1, 32, 16]
    print(f"   - 预测未来 {predictions.shape[1]} 个月")
    print(f"   - 每个时间步有 {predictions.shape[2]} 个特征")
    
    # 5. 显示模型信息
    print("\n5. 模型信息:")
    from gitpulse import get_model_info
    info = get_model_info()
    print(f"   - 模型名称: {info['name']}")
    print(f"   - 架构: {info['architecture']}")
    print(f"   - R²: {info['metrics']['R2']:.4f}")
    print(f"   - MSE: {info['metrics']['MSE']:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()





