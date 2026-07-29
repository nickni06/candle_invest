"""阶段1 XGBoost 推理：加载训练好的模型，对单点/批量样本预测买卖信号概率。

用法:
    infer = XGBInference('backend/ai/outputs/xgb_model.json')
    prob = infer.predict_single(df_kline)  # df_kline 含完整 OHLCV，预测末根
    # prob: {'buy_prob': 0.72, 'sell_prob': 0.08}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.features import extract_features, FEATURE_NAMES  # noqa: E402


class XGBInference:
    """XGBoost 推理器。

    模型文件为 xgb_model.json（XGBoost 原生 JSON 格式，跨平台兼容）。
    """

    def __init__(self, model_path: str):
        """加载 XGBoost 模型。

        Args:
            model_path: xgb_model.json 路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'XGBoost 模型不存在: {model_path}')

        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.model_path = model_path
        print(f'[xgb_infer] 模型已加载: {model_path}', flush=True)

    def predict_features(self, features: np.ndarray) -> dict:
        """对已提取的特征向量预测买卖概率。

        Args:
            features: (N, 131) 特征矩阵 或 (131,) 单样本

        Returns:
            {'buy_prob': float, 'sell_prob': float, 'raw_prob': float}
            raw_prob: 模型原始输出概率（如 0.72 表示"上涨信号概率"）
            buy_prob: raw_prob
            sell_prob: 1 - raw_prob
        """
        import xgboost as xgb
        if features.ndim == 1:
            features = features.reshape(1, -1)
        dmat = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
        prob = self.model.predict(dmat)
        # 模型输出是"正样本概率"，即"未来5日上涨信号概率"
        # 正样本标签=1 表示"未来涨幅 > ATR×1.0 且 回撤>-8%"
        # 所以 prob 高 = 看涨信号强
        buy_prob = float(prob[-1]) if len(prob) == 1 else prob
        if len(prob) == 1:
            return {
                'buy_prob': float(prob[0]),
                'sell_prob': 1.0 - float(prob[0]),
                'raw_prob': float(prob[0]),
            }
        return {
            'buy_prob': prob.tolist(),
            'sell_prob': (1 - prob).tolist(),
            'raw_prob': prob.tolist(),
        }

    def predict_single(self, df: pd.DataFrame) -> dict:
        """对 K 线 DataFrame 预测末根（最后一行）的买卖概率。

        Args:
            df: K 线 DataFrame，含 open/high/low/close/vol，按日期升序。

        Returns:
            {'buy_prob': float, 'sell_prob': float, 'raw_prob': float}
        """
        feats = extract_features(df)  # (N, 131)
        last_feat = feats[-1]         # (131,)
        return self.predict_features(last_feat)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """对 K 线 DataFrame 的所有时点预测买卖概率。

        Returns:
            (N,) 涨信号概率数组
        """
        feats = extract_features(df)
        import xgboost as xgb
        dmat = xgb.DMatrix(feats, feature_names=FEATURE_NAMES)
        return self.model.predict(dmat)


if __name__ == '__main__':
    # 自测：用合成数据验证推理
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'open': np.random.uniform(10, 20, n).cumsum() + 100,
        'high': 0,
        'low': 0,
        'close': 0,
        'vol': np.random.randint(100000, 500000, n),
    }, dtype=float)
    df['close'] = df['open'] + np.random.uniform(-1, 1, n)
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.5, n)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.5, n)

    model_path = 'backend/ai/outputs/xgb_model.json'
    if not os.path.exists(model_path):
        print(f'模型不存在: {model_path}，跳过自测')
        sys.exit(0)

    infer = XGBInference(model_path)
    result = infer.predict_single(df)
    print(f'单点预测: {result}')
