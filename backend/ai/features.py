"""特征工程：118维 TA-Lib 形态置信度 + 手工特征。

用途:
    - 阶段1 XGBoost only 的输入特征
    - 阶段2 融合模型中 XGBoost 的手工特征部分

特征清单（共 118 + 13 = 131 维）:
    1. TA-Lib 形态置信度 × 118（与 strategy_signals.ALL_BUY_PATTERNS + ALL_SELL_PATTERNS 一致）
    2. 手工特征 × 13:
       - ATR(14) / close        波动率
       - ATR(100) / close       中长期波动率
       - RSI(6)                  短期超买超卖
       - RSI(14)                中期超买超卖
       - MA5 乖离率              (close - MA5) / MA5
       - MA10 乖离率             (close - MA10) / MA10
       - MA20 乖离率             (close - MA20) / MA20
       - MA60 乖离率             (close - MA60) / MA60
       - 量比                    vol / MA(vol, 5)
       - 量比                    vol / MA(vol, 20)
       - 收盘价相对位置           (close - low_20) / (high_20 - low_20)
       - 过去5日收益率           close / close_5d_ago - 1
       - 过去20日收益率          close / close_20d_ago - 1

调用方式:
    feat = extract_features(df)  # df 为完整 K 线 DataFrame
    # feat 形状 (len(df), 131)，前 118 列为 TA-Lib，后 13 列为手工特征
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# 导入与 strategy_signals 完全一致的形态列表
try:
    from signal.strategy_signals import ALL_BUY_PATTERNS, ALL_SELL_PATTERNS
except Exception:
    # 兜底：直接定义（与 strategy_signals 保持同步）
    ALL_BUY_PATTERNS = [
        'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLCOUNTERATTACK',
        'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLGAPSIDESIDEWHITE',
        'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
        'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLINVERTEDHAMMER', 'CDLKICKING',
        'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
        'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING',
        'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLUNIQUE3RIVER',
        'CDLXSIDEGAP3METHODS',
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
        'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBREAKAWAY', 'CDLCONCEALBABYSWALL',
        'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHANGINGMAN',
        'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLSHOOTINGSTAR', 'CDLSTALLEDPATTERN',
        'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUPSIDEGAP2CROWS',
    ]
    ALL_SELL_PATTERNS = ALL_BUY_PATTERNS  # 同一批，靠 arr 符号区分方向

import talib


# ============================================================================
# 特征列名（固定顺序，训练和推理必须一致）
# ============================================================================
# TA-Lib 形态去重后的唯一形态名（59 个，买入和卖出共用同一批形态）
TA_PATTERN_NAMES = list(dict.fromkeys(ALL_BUY_PATTERNS + ALL_SELL_PATTERNS))  # 去重保序

# 每个形态拆分为「买入信号强度」和「卖出信号强度」两个特征
# talib 原始输出为整数: 正数=买入信号, 负数=卖出信号
# 拆分后特征数 = 59 × 2 = 118 维
TA_FEATURE_NAMES = []
for p in TA_PATTERN_NAMES:
    TA_FEATURE_NAMES.append(f'ta_buy_{p}')   # 正信号强度（max(arr, 0) / 100）
    TA_FEATURE_NAMES.append(f'ta_sell_{p}')  # 负信号强度（abs(min(arr, 0)) / 100）

HANDCRAFTED_FEATURE_NAMES = [
    'atr_14_ratio',        # ATR(14) / close
    'atr_100_ratio',       # ATR(100) / close
    'rsi_6',               # RSI(6)
    'rsi_14',              # RSI(14)
    'ma5_bias',            # (close - MA5) / MA5
    'ma10_bias',           # (close - MA10) / MA10
    'ma20_bias',            # (close - MA20) / MA20
    'ma60_bias',           # (close - MA60) / MA60
    'vol_ratio_5',         # vol / MA(vol, 5)
    'vol_ratio_20',        # vol / MA(vol, 20)
    'close_position_20',   # (close - low_20) / (high_20 - low_20)
    'return_5d',           # close / close_5d_ago - 1
    'return_20d',          # close / close_20d_ago - 1
]

FEATURE_NAMES = TA_FEATURE_NAMES + HANDCRAFTED_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)


def _compute_talib_patterns(df: pd.DataFrame) -> np.ndarray:
    """计算所有 TA-Lib 形态信号，返回 (N, 118) 的数组。

    每个形态拆分为「买入信号强度」和「卖出信号强度」两个特征:
        talib 原始值 > 0 (如 100) → ta_buy=1.0, ta_sell=0.0
        talib 原始值 < 0 (如 -100) → ta_buy=0.0, ta_sell=1.0
        talib 原始值 = 0 → ta_buy=0.0, ta_sell=0.0
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)

    n = len(df)
    # 59 形态 × 2 方向 = 118 维
    out = np.zeros((n, len(TA_PATTERN_NAMES) * 2), dtype=np.float32)
    for i, p in enumerate(TA_PATTERN_NAMES):
        try:
            func = getattr(talib, p)
            arr = func(o, h, l, c).astype(np.float32)
            # 买入信号强度: 正值归一化到 [0, 1]
            out[:, i * 2] = np.clip(arr / 100.0, 0.0, 1.0)
            # 卖出信号强度: 负值的绝对值归一化到 [0, 1]
            out[:, i * 2 + 1] = np.clip(-arr / 100.0, 0.0, 1.0)
        except Exception:
            out[:, i * 2] = 0.0
            out[:, i * 2 + 1] = 0.0
    return out


def _compute_handcrafted(df: pd.DataFrame) -> np.ndarray:
    """计算 13 个手工特征，返回 (N, 13) 数组。"""
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64) if 'vol' in df.columns else \
        (df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros_like(c))
    n = len(c)

    feat = np.zeros((n, 13), dtype=np.float32)

    # ATR
    atr_14 = talib.ATR(h, l, c, timeperiod=14)
    atr_100 = talib.ATR(h, l, c, timeperiod=100)
    feat[:, 0] = np.nan_to_num(atr_14 / c, nan=0.0)
    feat[:, 1] = np.nan_to_num(atr_100 / c, nan=0.0)

    # RSI
    rsi_6 = talib.RSI(c, timeperiod=6)
    rsi_14 = talib.RSI(c, timeperiod=14)
    feat[:, 2] = np.nan_to_num(rsi_6 / 100.0, nan=0.5)  # 归一化到 [0, 1]
    feat[:, 3] = np.nan_to_num(rsi_14 / 100.0, nan=0.5)

    # 均线乖离率
    for i, period in enumerate([5, 10, 20, 60]):
        ma = talib.SMA(c, timeperiod=period)
        feat[:, 4 + i] = np.nan_to_num((c - ma) / ma, nan=0.0)

    # 量比
    vol_ma5 = pd.Series(v).rolling(5, min_periods=1).mean().values
    vol_ma20 = pd.Series(v).rolling(20, min_periods=1).mean().values
    feat[:, 8] = np.nan_to_num(v / np.maximum(vol_ma5, 1e-9), nan=1.0)
    feat[:, 9] = np.nan_to_num(v / np.maximum(vol_ma20, 1e-9), nan=1.0)

    # 收盘价相对位置（过去 20 日）
    high_20 = pd.Series(h).rolling(20, min_periods=1).max().values
    low_20 = pd.Series(l).rolling(20, min_periods=1).min().values
    rng = np.maximum(high_20 - low_20, 1e-9)
    feat[:, 10] = np.nan_to_num((c - low_20) / rng, nan=0.5)

    # 过去收益率
    s_close = pd.Series(c)
    feat[:, 11] = np.nan_to_num(s_close.pct_change(5).values, nan=0.0)
    feat[:, 12] = np.nan_to_num(s_close.pct_change(20).values, nan=0.0)

    return feat


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """提取完整特征矩阵（118 维 TA-Lib + 13 维手工特征）。

    Args:
        df: K 线 DataFrame，需含 ['open','high','low','close','vol']，按日期升序。

    Returns:
        (N, 131) float32 特征矩阵，N = len(df)。
    """
    ta = _compute_talib_patterns(df)
    hand = _compute_handcrafted(df)
    return np.concatenate([ta, hand], axis=1)


def extract_features_at(df: pd.DataFrame, idx: int) -> np.ndarray:
    """提取 df 在第 idx 行的特征向量（131 维）。

    用于推理时单点预测。
    """
    return extract_features(df)[idx]


if __name__ == '__main__':
    # 自测：用合成数据验证特征提取
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

    feat = extract_features(df)
    print(f'特征矩阵: shape={feat.shape}, dtype={feat.dtype}')
    print(f'特征列数: {N_FEATURES} (TA-Lib {len(TA_PATTERN_NAMES)} + 手工 {len(HANDCRAFTED_FEATURE_NAMES)})')
    print(f'首行前5个特征: {feat[-1, :5]}')
    print(f'首行后5个特征: {feat[-1, -5:]}')
