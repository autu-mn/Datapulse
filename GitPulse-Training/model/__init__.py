"""
GitPulse - GitHub 开源项目健康度多模态时序预测模型

保留的模型版本：
- v4: 主模型 (PatchTST + 文本融合 + 对比学习)
- v4_1: 架构对比实验 (Transformer/GRU + 文本)，包含最优模型 CondGRU+Text
- v5: 轻量级优化版本（特征增强 + 趋势分解 + 增强损失）
- v5.1: 结构化文本解析 + 多条件门控GRU + 时序对齐注入
- v6: Transformer + Multi-Layer Cross-Attention（大规模数据集优化）
- v6.1: CondGRU-AT（条件注意力增强）
- v7: Patch-FiLM Transformer + 频域增强（创新融合策略）
- v7.1: Enhanced Transformer + Text（基于 v4.1 逻辑 + 创新改进）
"""

from .multimodal_ts_v4 import (
    MultimodalTSPredictorV4,
    TimeSeriesOnlyPredictorV4,
    ContrastiveHead,
    MatchingHead,
    AdaptiveFusion,
    count_parameters,
)

from .multimodal_ts_v4_1 import (
    MultimodalTransformerV4_1,
    TransformerTSOnlyV4_1,
    MultimodalGRUV4_1,
    GRUTSOnlyV4_1,
    MultimodalConditionalGRUV4_1,
    CondGRUTSOnlyV4_1,
)

from .multimodal_ts_v5 import (
    GitPulseV5,
    GitPulseV5TSOnly,
    get_model_info as get_v5_info,
    count_parameters as count_v5_parameters,
)

from .multimodal_ts_v5_1 import (
    GitPulseV5_1,
    GitPulseV5_1TSOnly,
    get_model_info as get_v5_1_info,
    count_parameters as count_v5_1_parameters,
    StructuredTextParser,
)

from .multimodal_ts_v6 import (
    GitPulseV6,
    GitPulseV6TSOnly,
    get_model_info as get_v6_info,
    count_parameters as count_v6_parameters,
)

from .multimodal_ts_v6_1 import (
    GitPulseV6_1,
    GitPulseV6_1TSOnly,
    get_model_info as get_v6_1_info,
    count_parameters as count_v6_1_parameters,
)

from .multimodal_ts_v7 import (
    GitPulseV7,
    GitPulseV7TSOnly,
    count_parameters as count_v7_parameters,
)

from .multimodal_ts_v7_1 import (
    MultimodalTransformerV7_1,
    TransformerTSOnlyV7_1,
    count_parameters as count_v7_1_parameters,
)

from .multimodal_ts_v8_1 import (
    MultimodalCondGRUV8_1,
    MultimodalTransformerV8_1,
    MultimodalGRUV8_1,
    CondGRUTSOnlyV8_1,
    TransformerTSOnlyV8_1,
    GRUTSOnlyV8_1,
    load_finetuned_model,
    count_parameters as count_v8_1_parameters,
)

__all__ = [
    # v4 - 主模型 (PatchTST + Text)
    'MultimodalTSPredictorV4',
    'TimeSeriesOnlyPredictorV4',
    'ContrastiveHead',
    'MatchingHead',
    'AdaptiveFusion',
    'count_parameters',
    
    # v4.1 - 架构对比 (Transformer/GRU)
    'MultimodalTransformerV4_1',
    'TransformerTSOnlyV4_1',
    'MultimodalGRUV4_1',
    'GRUTSOnlyV4_1',
    'MultimodalConditionalGRUV4_1',
    'CondGRUTSOnlyV4_1',
    
    # v5-Lite - 轻量级优化版本
    'GitPulseV5',
    'GitPulseV5TSOnly',
    'get_v5_info',
    
    # v5.1 - 结构化文本 + 多条件门控GRU
    'GitPulseV5_1',
    'GitPulseV5_1TSOnly',
    'get_v5_1_info',
    'StructuredTextParser',
    
    # v6 - Transformer + Multi-Layer Cross-Attention
    'GitPulseV6',
    'GitPulseV6TSOnly',
    'get_v6_info',
    
    # v6.1 - CondGRU-AT (条件注意力增强)
    'GitPulseV6_1',
    'GitPulseV6_1TSOnly',
    'get_v6_1_info',
    
    # v7 - Patch-FiLM Transformer + 频域增强
    'GitPulseV7',
    'GitPulseV7TSOnly',
    'count_v7_parameters',
    
    # v7.1 - Enhanced Transformer + Text（基于 v4.1）
    'MultimodalTransformerV7_1',
    'TransformerTSOnlyV7_1',
    'count_v7_1_parameters',
    
    # v8.1 - 微调整合版本（小数据集预训练 + 大数据集微调）
    'MultimodalCondGRUV8_1',
    'MultimodalTransformerV8_1',
    'MultimodalGRUV8_1',
    'CondGRUTSOnlyV8_1',
    'TransformerTSOnlyV8_1',
    'GRUTSOnlyV8_1',
    'load_finetuned_model',
    'count_v8_1_parameters',
]

__version__ = '1.6.0'
