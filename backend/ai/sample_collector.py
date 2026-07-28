"""样本采集：从盈湖 K 线数据生成训练样本。

流程:
    盈湖 get_kline(code, start, end)
        → 滑动窗口切 20 根 K 线
        → 末根 close 归一化（OHLC 除以末根 close；vol 窗口内 z-score）
        → 用未来 5 日表现 + ATR(100) 打标签
        → 按标的分文件存 .npy

标签定义（N=5, ATR period=100）:
    正样本(1): 未来5日涨幅 > 1.5×ATR(100) 且 期间最大回撤 > -5%
    负样本(0): 未来5日涨幅 < 0 或 期间最大回撤 < -8%
    模糊样本: 丢弃（避免污染标签边界）
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 让本模块既能被 backend/ 内部调用，也能独立运行（python sample_collector.py）
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.cnn_model import SEQ_LEN, NUM_FEATURES, FEATURE_NAMES  # noqa: E402

logger = logging.getLogger('trader_system')

# ============================================================================
# 标签生成参数
# ============================================================================
FORWARD_DAYS = 5              # 未来窗口 N
ATR_PERIOD = 100              # ATR 计算周期
ATR_MULTIPLIER = 1.5          # 正样本涨幅阈值 = ATR × 1.5
MAX_DRAWDOWN_POS = -0.05      # 正样本：期间最大回撤 > -5%
MAX_DRAWDOWN_NEG = -0.08      # 负样本：期间最大回撤 < -8%

# 切窗所需的前置 K 线数：ATR(100) 需要 100 根，窗口 20 根，未来 5 根
MIN_HISTORY = ATR_PERIOD + SEQ_LEN + FORWARD_DAYS  # 125 根


# ============================================================================
# ATR 计算（纯 numpy 实现，无需 TA-Lib C 库）
# ============================================================================
def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = ATR_PERIOD) -> np.ndarray:
    """计算 ATR (Average True Range)。

    用 Wilder 平滑法（等价于 talib.ATR），纯 numpy 实现，避免 TA-Lib C 库依赖。

    True Range 取以下三者最大值:
        - 当日 high - low
        - |当日 high - 昨日 close|
        - |当日 low - 昨日 close|

    ATR = TR 的 Wilder 平滑移动平均（period 日）。

    Args:
        high: 最高价序列 (N,)
        low: 最低价序列 (N,)
        close: 收盘价序列 (N,)
        period: 平滑周期

    Returns:
        ATR 序列 (N,)，前 period-1 个为 NaN。
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder 平滑：前 period-1 个为 NaN；第 period 个为前 period 个 TR 的简单平均；
    # 之后 = (prev_ATR × (period-1) + TR) / period
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ============================================================================
# 归一化：末根 close 归一
# ============================================================================
def normalize_window(window: np.ndarray) -> np.ndarray:
    """对 20×5 窗口做末根 close 归一化。

    - OHLC (前4列): 全部除以窗口最后一根的 close，使末根 close = 1
    - vol (第5列): 窗口内 z-score 标准化（消除量纲差异）

    Args:
        window: 形状 (20, 5) 的原始 K 线窗口，列顺序 [open, high, low, close, vol]。

    Returns:
        归一化后的 (20, 5) float32 数组。
    """
    w = window.astype(np.float64).copy()
    last_close = w[-1, 3]  # 末根 close
    if last_close == 0:
        return np.zeros_like(w, dtype=np.float32)

    # OHLC 除以末根 close
    w[:, :4] = w[:, :4] / last_close

    # vol 窗口内 z-score
    vol = w[:, 4]
    vol_mean = vol.mean()
    vol_std = vol.std()
    if vol_std > 1e-12:
        w[:, 4] = (vol - vol_mean) / vol_std
    else:
        w[:, 4] = 0.0  # vol 全相同（如停牌期），置 0

    return w.astype(np.float32)


# ============================================================================
# 标签生成
# ============================================================================
def compute_label(future_close: np.ndarray, future_high: np.ndarray,
                  future_low: np.ndarray, entry_close: float,
                  atr_value: float) -> Optional[int]:
    """根据未来 N 日表现 + ATR 生成标签。

    Args:
        future_close: 未来 N 日的 close 序列（长度 N）
        future_high: 未来 N 日的 high 序列
        future_low: 未来 N 日的 low 序列
        entry_close: 入场 close（窗口末根的 close）
        atr_value: 入场时的 ATR 值

    Returns:
        1（正样本）/ 0（负样本）/ None（模糊样本，丢弃）
    """
    if atr_value is None or np.isnan(atr_value) or atr_value <= 0:
        return None

    # 未来 N 日涨幅（基于 entry_close）
    future_return = (future_close[-1] - entry_close) / entry_close

    # 期间最大回撤：相对 entry_close 的最大下跌幅度（负数）
    # 用 future_low 计算最坏情况
    max_drawdown = (future_low.min() - entry_close) / entry_close

    # 正样本：涨幅 > 1.5×ATR 占比 且 回撤 > -5%
    # 注意 ATR 是绝对价格单位，需转为相对 entry_close 的比例
    atr_ratio = atr_value / entry_close
    is_positive = (future_return > ATR_MULTIPLIER * atr_ratio
                   and max_drawdown > MAX_DRAWDOWN_POS)

    # 负样本：涨幅 < 0 或 回撤 < -8%
    is_negative = (future_return < 0 or max_drawdown < MAX_DRAWDOWN_NEG)

    if is_positive:
        return 1
    if is_negative:
        return 0
    return None  # 模糊区，丢弃


# ============================================================================
# 单标的样本生成
# ============================================================================
def collect_samples_for_code(df: pd.DataFrame,
                             seq_len: int = SEQ_LEN,
                             forward_days: int = FORWARD_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """从单个标的的 K 线 DataFrame 生成训练样本。

    Args:
        df: K 线 DataFrame，需含列 ['open','high','low','close','vol']，按日期升序。
        seq_len: 输入窗口长度（默认 20）。
        forward_days: 未来窗口 N（默认 5）。

    Returns:
        (samples, labels):
            samples: (M, seq_len, 5) float32
            labels:  (M,) int32，1=正样本，0=负样本
        M 为生成的有效样本数（模糊样本已丢弃）。
    """
    required = ['open', 'high', 'low', 'close', 'vol']
    if not all(c in df.columns for c in required):
        raise ValueError(f'K线数据缺少必要列，需要 {required}，实际 {list(df.columns)}')

    # 清洗：去除停牌（vol=0 或 OHLC 全 0）的行
    df = df.copy()
    df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df.reset_index(drop=True)

    n = len(df)
    if n < MIN_HISTORY:
        return np.empty((0, seq_len, NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int32)

    ohlcv = df[required].values.astype(np.float64)
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    # 计算 ATR(100)
    atr = compute_atr(high, low, close, period=ATR_PERIOD)

    samples = []
    labels = []

    # 滑动窗口：i 是窗口末根索引
    # 起点：i = ATR_PERIOD-1（ATR 开始有效）+ (seq_len-1)？不，ATR 索引和窗口末根对齐即可
    # 窗口 [i-seq_len+1, i]，未来 [i+1, i+forward_days]
    # 需 i >= seq_len-1 且 i >= ATR_PERIOD-1（ATR 有效），且 i+forward_days < n
    start_i = max(seq_len - 1, ATR_PERIOD - 1)
    end_i = n - forward_days  # 不含，保证有完整的未来窗口

    for i in range(start_i, end_i):
        # 切窗口
        window = ohlcv[i - seq_len + 1: i + 1]  # (seq_len, 5)

        # 归一化
        norm_window = normalize_window(window)

        # 取未来 N 日数据
        future_close = close[i + 1: i + 1 + forward_days]
        future_high = high[i + 1: i + 1 + forward_days]
        future_low = low[i + 1: i + 1 + forward_days]
        entry_close = close[i]
        atr_value = atr[i]

        # 打标签
        label = compute_label(future_close, future_high, future_low,
                              entry_close, atr_value)
        if label is None:
            continue

        samples.append(norm_window)
        labels.append(label)

    if not samples:
        return np.empty((0, seq_len, NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(samples, axis=0), np.array(labels, dtype=np.int32)


# ============================================================================
# 批量采集：从盈湖拉数据并存储
# ============================================================================
def collect_and_save(code_list: list[str],
                     start_date: str,
                     end_date: str,
                     output_dir: str,
                     source: str = 'yinghu') -> dict:
    """批量采集样本并按标的分文件存储。

    Args:
        code_list: 标的代码列表。
        start_date: 起始日 YYYYMMDD。
        end_date: 结束日 YYYYMMDD。
        output_dir: 输出目录，每个标的生成 <code>_samples.npy 和 <code>_labels.npy。
        source: 数据源，'yinghu' 走盈湖数据库。

    Returns:
        统计 dict: {'total_codes', 'success', 'failed', 'total_samples',
                    'positive', 'negative', 'failed_codes'}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 延迟导入，避免在没有盈湖依赖的环境（如纯训练环境）报错
    if source == 'yinghu':
        sys.path.insert(0, str(_BACKEND_DIR / 'data'))
        try:
            from data.yinghu_db import get_kline
        except ImportError:
            from yinghu_db import get_kline
    else:
        raise ValueError(f'不支持的数据源: {source}')

    stats = {
        'total_codes': len(code_list), 'success': 0, 'failed': 0,
        'total_samples': 0, 'positive': 0, 'negative': 0,
        'failed_codes': [],
    }

    for idx, code in enumerate(code_list):
        try:
            df = get_kline(code, start_date, end_date)
            if df is None or df.empty:
                stats['failed'] += 1
                stats['failed_codes'].append({'code': code, 'error': '无数据'})
                continue

            samples, labels = collect_samples_for_code(df)

            if len(samples) == 0:
                stats['failed'] += 1
                stats['failed_codes'].append({'code': code, 'error': '样本数不足'})
                continue

            # 按标的分文件存储
            sample_path = os.path.join(output_dir, f'{code}_samples.npy')
            label_path = os.path.join(output_dir, f'{code}_labels.npy')
            np.save(sample_path, samples)
            np.save(label_path, labels)

            stats['success'] += 1
            stats['total_samples'] += len(samples)
            stats['positive'] += int((labels == 1).sum())
            stats['negative'] += int((labels == 0).sum())

            if (idx + 1) % 10 == 0 or idx + 1 == len(code_list):
                print(f'[采集] 进度: {idx+1}/{len(code_list)}，'
                      f'累计样本 {stats["total_samples"]} '
                      f'(正{stats["positive"]}/负{stats["negative"]})', flush=True)
        except Exception as e:
            stats['failed'] += 1
            stats['failed_codes'].append({'code': code, 'error': str(e)})
            print(f'[采集] {code} 失败: {e}', flush=True)

    print(f'[采集] 完成: 成功 {stats["success"]}/{stats["total_codes"]}，'
          f'总样本 {stats["total_samples"]} (正{stats["positive"]}/负{stats["negative"]})',
          flush=True)
    return stats


# ============================================================================
# CLI 入口
# ============================================================================
def main():
    """命令行入口：从环境变量读取参数采集样本。"""
    import json
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # 默认用几只代表性标的做端到端验证
    code_list_str = os.environ.get('CODE_LIST', '000001.SZ,600000.SH,000300.SH')
    code_list = [c.strip() for c in code_list_str.split(',') if c.strip()]
    start_date = os.environ.get('START_DATE', '20200101')
    end_date = os.environ.get('END_DATE', '20241231')

    # 输出目录：优先用 config 中的训练目录，兜底用 backend/ai/data
    try:
        from config import config
        output_dir = str(config.TRAIN_DATA_A_DIR)
    except Exception:
        output_dir = str(_THIS_DIR / 'data' / 'train')

    print(f'[采集] 配置: codes={code_list}, start={start_date}, end={end_date}')
    print(f'[采集] 输出目录: {output_dir}')

    stats = collect_and_save(code_list, start_date, end_date, output_dir)
    print(f'[采集] 统计: {json.dumps(stats, ensure_ascii=False, indent=2)}')


if __name__ == '__main__':
    main()
