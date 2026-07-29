"""1D-CNN 模型定义：K 线模糊形态识别。

设计要点：
1. 输入: (B, 5, 20)  # batch, channels=5特征(open/high/low/close/vol), length=20根K线
2. 输出: (B, 1) logit；推理时 sigmoid 得 0~1 置信度
3. 轻量级: 2 层 Conv1d + 全局平均池化，参数量 ~4.4K，远低于 50 万上限
4. 推理速度: CPU <0.1ms/样本，满足 <1ms 要求
5. 设备自适应: MPS(Mac) > CUDA > CPU 自动降级

依赖说明:
    本模块设计为「无 torch 也可导入」——常量 SEQ_LEN/NUM_FEATURES/FEATURE_NAMES 独立定义，
    被 sample_collector 等非 torch 模块复用。
    实际构建/训练 CNN 时调用 build_candle_cnn()，此时才会触发 torch 导入。
"""
from __future__ import annotations

# ============================================================================
# 模型输入规格（常量，供 sample_collector / inference / features 复用）
# 这些常量不依赖 torch，独立定义以便无 torch 环境下也能导入本模块
# ============================================================================
SEQ_LEN = 20          # 输入 K 线根数
NUM_FEATURES = 5       # 每根 K 线特征数：open, high, low, close, vol
FEATURE_NAMES = ('open', 'high', 'low', 'close', 'vol')


def _import_torch():
    """延迟导入 torch，仅在实际构建/训练 CNN 模型时才需要。"""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError(
            'CNN 模型需要 PyTorch。安装: pip install torch\n'
            '若只用 XGBoost 阶段1，可忽略此错误（sample_collector / features 均不依赖 torch）。'
        ) from e


def build_candle_cnn(num_features: int = NUM_FEATURES, seq_len: int = SEQ_LEN):
    """构建 CandleCNN 模型实例（延迟导入 torch）。

    Args:
        num_features: 输入特征通道数（默认 5: open/high/low/close/vol）
        seq_len: 输入序列长度（默认 20 根 K 线）

    Returns:
        CandleCNN 实例（nn.Module）
    """
    torch, nn = _import_torch()

    class CandleCNN(nn.Module):
        """轻量级 1D-CNN，对 20×5 K 线序列做二分类（是否为有效形态信号）。

        结构:
            Conv1d(5→32, k=3, pad=1) → BN → ReLU       # 感受野 3
            Conv1d(32→32, k=3, pad=1) → BN → ReLU      # 感受野 5
            AdaptiveAvgPool1d(1) → Flatten             # 全局平均池化，固定输出尺寸
            Linear(32→16) → ReLU → Dropout(0.3)
            Linear(16→1)                               # 输出 logit

        参数量约 4.4K；训练用 BCEWithLogitsLoss，推理用 sigmoid 得 0~1 置信度。
        """

        def __init__(self, num_features: int = num_features, seq_len: int = seq_len):
            super().__init__()
            self.num_features = num_features
            self.seq_len = seq_len

            # 卷积特征提取
            self.features = nn.Sequential(
                nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),  # (B, 32, 1) - 全局平均池化
            )

            # 分类头
            self.classifier = nn.Sequential(
                nn.Flatten(),              # (B, 32)
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(16, 1),          # 输出 logit，不用 Sigmoid（BCEWithLogitsLoss 内置）
            )

        def forward(self, x):
            """前向传播。

            Args:
                x: 形状 (B, 5, 20) 的 K 线序列张量。
                    维度约定遵循 PyTorch Conv1d: (batch, channels, length)。
                    channels=5 对应 open/high/low/close/vol。

            Returns:
                形状 (B, 1) 的 logit 张量。推理时需 torch.sigmoid() 得 0~1 置信度。
            """
            x = self.features(x)
            return self.classifier(x)

        def forward_confidence(self, x):
            """推理便捷接口：直接返回 0~1 置信度。

            Args:
                x: 形状 (B, 5, 20) 或 (5, 20) 的 K 线序列张量。

            Returns:
                形状 (B,) 的 0~1 置信度张量。
            """
            if x.dim() == 2:
                x = x.unsqueeze(0)  # (5,20) -> (1,5,20)
            with torch.no_grad():
                logit = self.forward(x)
            return torch.sigmoid(logit).squeeze(-1)

    return CandleCNN()


# 向后兼容：允许 `from cnn_model import CandleCNN` 的旧代码继续工作
# 实际使用时通过 build_candle_cnn() 创建实例
def __getattr__(name: str):
    """模块级 __getattr__，仅在访问未定义符号时触发。"""
    if name == 'CandleCNN':
        # 返回一个代理类，实例化时才真正构建模型
        class _CandleCNNProxy:
            def __init__(self, *args, **kwargs):
                self._model = build_candle_cnn(
                    num_features=kwargs.get('num_features', NUM_FEATURES),
                    seq_len=kwargs.get('seq_len', SEQ_LEN),
                )
                # 代理所有属性访问
                for attr in dir(self._model):
                    if not attr.startswith('_'):
                        setattr(self, attr, getattr(self._model, attr))

            def __getattr__(self, item):
                return getattr(self._model, item)

            def __call__(self, *args, **kwargs):
                return self._model(*args, **kwargs)
        return _CandleCNNProxy
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def count_parameters(model) -> int:
    """统计模型可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_mps: bool = True):
    """设备自适应选择：MPS(Mac) > CUDA > CPU。"""
    torch, _ = _import_torch()
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


if __name__ == '__main__':
    # 自测：模型结构、参数量、推理速度
    model = build_candle_cnn()
    print(model)
    print(f'\n可训练参数量: {count_parameters(model):,}')

    # 模拟输入 (batch=8, 5, 20)
    x = torch.randn(8, NUM_FEATURES, SEQ_LEN)
    logit = model(x)
    print(f'输入形状: {x.shape} -> 输出 logit 形状: {logit.shape}')
    print(f'置信度样本: {model.forward_confidence(x).tolist()}')

    # 推理速度测试
    import time
    model.eval()
    single = torch.randn(1, NUM_FEATURES, SEQ_LEN)
    # warmup
    for _ in range(10):
        _ = model.forward_confidence(single)
    t0 = time.perf_counter()
    n = 1000
    for _ in range(n):
        _ = model.forward_confidence(single)
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f'单样本推理耗时: {elapsed:.3f} ms')
