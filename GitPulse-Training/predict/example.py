"""
预测示例脚本
演示如何使用预测模块
"""

import json
import numpy as np
from predict_single_repo import RepoPredictor

# 示例：使用模拟数据预测
def example_with_synthetic_data():
    """使用模拟数据示例"""
    print("=" * 60)
    print("示例1: 使用模拟数据")
    print("=" * 60)
    
    # 初始化预测器
    predictor = RepoPredictor('training/checkpoints/best_model_cond_gru_mm.pt')
    
    # 生成模拟时序数据（150个月，16维）
    np.random.seed(42)
    timeseries_data = np.random.randn(150, 16).astype(np.float32)
    # 添加一些趋势
    for i in range(150):
        timeseries_data[i] += np.linspace(0, 1, 16) * (i / 150)
    
    # 模拟文本数据
    text_data = """
    A modern web framework built with React and TypeScript.
    Features include server-side rendering, API routes, and automatic code splitting.
    Used by thousands of developers worldwide for building production-ready applications.
    """
    
    # 预测
    prediction, stats = predictor.predict(timeseries_data, text_data)
    
    print(f"\n预测结果:")
    print(f"  形状: {prediction.shape}")
    print(f"  前3个月预测（前3维）:")
    for i in range(3):
        print(f"    第{i+1}个月: {prediction[i][:3]}")
    print(f"\n  趋势: {'上升' if np.mean(prediction[-1] - prediction[0]) > 0 else '下降'}")


def example_from_file():
    """从文件加载数据示例"""
    print("\n" + "=" * 60)
    print("示例2: 从文件加载数据")
    print("=" * 60)
    
    # 创建示例数据文件
    example_timeseries = {
        "timeseries": np.random.randn(130, 16).tolist()
    }
    
    with open('example_timeseries.json', 'w') as f:
        json.dump(example_timeseries, f)
    
    example_text = "A machine learning library for Python"
    with open('example_text.txt', 'w') as f:
        f.write(example_text)
    
    # 初始化预测器
    predictor = RepoPredictor('training/checkpoints/best_model_cond_gru_mm.pt')
    
    # 从文件预测
    prediction, stats = predictor.predict_from_file(
        'example_timeseries.json',
        text_file='example_text.txt'
    )
    
    print(f"\n预测完成!")
    print(f"  预测形状: {prediction.shape}")
    print(f"  结果已保存到 prediction_result.json")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'file':
        example_from_file()
    else:
        example_with_synthetic_data()

