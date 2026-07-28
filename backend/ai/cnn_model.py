"""1D-CNN 模型定义：K 线模糊形态识别。

设计要点：
1. 输入: (B, 5, 20)  # batch, channels=5特征(open/high/low/close/vol), length=20根K线
2. 输出: (B, 1) logit；推理时 sigmoid 得 0~1 置信度
3. 轻量级: 2 层 Conv1d + 全局平均池化，参数量 ~4.4K，远低于 50 万上限
4. 推理速度: CPU <0.1ms/样本，满足 <1ms 要求
5. 设备自适应: MPS(Mac) > CUDA > CPU 自动降级
"""
from __future__ import annotations

import torch
import torch.nn as nn


# 模型输入规格（常量，供 sample_collector / inference 复用）
SEQ_LEN = 20          # 输入 K 线根数
NUM_FEATURES = 5       # 每根 K 线特征数：open, high, low, close, vol
FEATURE_NAMES = ('open', 'high', 'low', 'close', 'vol')


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

    def __init__(self, num_features: int = NUM_FEATURES, seq_len: int = SEQ_LEN):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_confidence(self, x: torch.Tensor) -> torch.Tensor:
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


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_mps: bool = True) -> torch.device:
    """设备自适应选择：MPS(Mac) > CUDA > CPU。

    Args:
        prefer_mps: 是否优先尝试 MPS（Mac M 系列芯片 GPU）。

    Returns:
        torch.device，自动降级到可用设备。
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


if __name__ == '__main__':
    # 自测：模型结构、参数量、推理速度
    model = CandleCNN()
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
