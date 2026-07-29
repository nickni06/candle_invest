import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import multiprocessing
import hashlib
import pickle
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import backtrader as bt

# 尝试加载 Numba 加速版回测；失败不影响功能，回退到原版
try:
    import numba_backtest
    _NUMBA_OK = numba_backtest._NUMBA_AVAILABLE
except Exception as _e:
    numba_backtest = None
    _NUMBA_OK = False
    import logging
    logging.getLogger('trader_system').warning(
        f'[pattern_scan] numba_backtest 加载失败，回退原版回测: {_e}'
    )
import talib

from signal_utils import BUY_PATTERNS, SELL_PATTERNS, PATTERN_CN_NAMES, get_pattern_description
from cautious_mode import meets_extra_condition, meets_extra_condition_df
from config import config

# Parquet 支持状态：未安装 pyarrow 时自动回退到 CSV
try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False


def _is_index_code(code):
    """判断 code 是否为指数代码（用于选择数据目录）"""
    if not code:
        return False
    overseas = {'DJI', 'FCHI', 'SPX', 'GDAXI', 'N225'}
    if code in overseas:
        return True
    s = str(code)
    # 沪市指数：代码以 000/880 开头且后缀为 .SH（000xxx.SZ 为深市主板股票，不能误判）
    if s.endswith('.SH'):
        prefix = s.split('.')[0]
        if prefix.startswith(('000', '880')):
            return True
    # 深市指数：代码以 399 开头且后缀为 .SZ
    if s.endswith('.SZ'):
        prefix = s.split('.')[0]
        if prefix.startswith('399'):
            return True
    return False


def _effective_data_path(code, data_dir):
    """返回 code 在指定目录中的有效数据文件路径：优先 Parquet，回退 CSV。

    与 data_source._find_local_data 保持一致的优先级，确保 pattern_scan
    与统一数据源模块使用同一数据源视图。
    """
    data_dir = Path(data_dir)
    pq_path = data_dir / f'{code}_daily.parquet'
    if _PARQUET_AVAILABLE and pq_path.exists() and pq_path.stat().st_size > 0:
        return pq_path, True
    csv_path = data_dir / f'{code}_daily.csv'
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path, False
    return None, False


def _compute_mtime_signature(code, data_folder_dir=None):
    """根据所有候选源数据文件（Parquet 优先，CSV 回退）的 mtime+size 生成签名，用于 lru_cache 失效判断。"""
    candidate_dirs = _get_candidate_dirs(code, data_folder_dir)
    parts = []
    for d in candidate_dirs:
        data_path, is_parquet = _effective_data_path(code, d)
        if data_path:
            st = data_path.stat()
            parts.append(f'{data_path}:{st.st_mtime:.6f}:{st.st_size}:{"pq" if is_parquet else "csv"}')
    return hashlib.md5('|'.join(parts).encode('utf-8')).hexdigest()[:16]


@lru_cache(maxsize=64)
def _load_raw_dataframe_cached(code, data_folder_dir, mtime_signature):
    """实际执行数据加载/合并/去重的函数，由 lru_cache 包装。

    注意：data_folder_dir 与 mtime_signature 共同构成缓存 key。
    任一源数据文件的 mtime/size 变化都会改变 mtime_signature，使旧缓存失效。
    """
    candidate_dirs = _get_candidate_dirs(code, data_folder_dir)

    # 收集所有目录中存在的数据文件（Parquet 优先），合并后去重
    frames = []
    for d in candidate_dirs:
        try:
            data_path, is_parquet = _effective_data_path(code, d)
            if not data_path:
                continue
            if is_parquet:
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(str(data_path), dtype={'ts_code': str})
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        raise ValueError(f"{code} 在所有候选目录均无数据")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.drop_duplicates(subset=['trade_date'], keep='first')
    df = df.sort_values(by=['trade_date'], ascending=True).reset_index(drop=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index('trade_date', drop=True)
    return df


def _load_raw_dataframe(code, data_folder_dir=None):
    """加载标的原始日K数据（合并多目录 Parquet/CSV，去重排序），返回带日期索引的 DataFrame。

    通过进程内 lru_cache + 源文件 mtime/size 签名实现缓存，
    避免同一 code 在单进程内重复读取数据；源数据变更后自动失效。
    """
    mtime_signature = _compute_mtime_signature(code, data_folder_dir)
    return _load_raw_dataframe_cached(code, data_folder_dir, mtime_signature)


# ---------- code 级形态信号缓存 ----------

def _get_candidate_dirs(code, data_folder_dir=None):
    """获取 code 所有候选数据目录（与 _load_raw_dataframe 保持一致）。"""
    candidate_dirs = []
    if _is_index_code(code):
        candidate_dirs.append(config.DAILY_TRACKING_INDEX_DIR)
        candidate_dirs.append(config.TRAIN_DATA_INDEX_DIR)
        candidate_dirs.append(config.TEST_DATA_INDEX_DIR)
    else:
        candidate_dirs.append(config.DAILY_TRACKING_A_DIR)
        candidate_dirs.append(config.TRAIN_DATA_A_DIR)
        candidate_dirs.append(config.TEST_DATA_A_DIR)
    if data_folder_dir:
        try:
            d = Path(data_folder_dir) if not isinstance(data_folder_dir, Path) else data_folder_dir
            if d not in candidate_dirs:
                candidate_dirs.append(d)
        except Exception:
            pass
    return candidate_dirs


def _signal_cache_path(code, start_date, end_date, data_folder_dir=None):
    """生成 code 级形态信号缓存文件路径。

    缓存 key 包含：code、日期范围、源数据文件 mtime 签名、形态列表签名。
    任一源数据文件变更或形态列表变更都会使旧缓存自动失效。
    """
    candidate_dirs = _get_candidate_dirs(code, data_folder_dir)
    mtime_parts = []
    for d in candidate_dirs:
        data_path, is_parquet = _effective_data_path(code, d)
        if data_path:
            st = data_path.stat()
            mtime_parts.append(f'{data_path}:{st.st_mtime:.6f}:{st.st_size}:{"pq" if is_parquet else "csv"}')
    data_sig = hashlib.md5('|'.join(mtime_parts).encode('utf-8')).hexdigest()[:16]

    all_patterns = sorted(BUY_PATTERNS + SELL_PATTERNS)
    patterns_sig = hashlib.md5('|'.join(all_patterns).encode('utf-8')).hexdigest()[:16]

    cache_name = f'{code}_{start_date}_{end_date}_{data_sig}_{patterns_sig}.pkl'
    return config.BASE_DIR / '.cache' / 'pattern_signals' / cache_name


def _load_cached_signals(cache_path):
    """从磁盘加载缓存的形态信号字典（pattern_name -> numpy array）。"""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cached_signals(cache_path, signals):
    """原子写入形态信号缓存（避免多进程竞争损坏）。"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_path.with_suffix('.tmp')
        with open(tmp_file, 'wb') as f:
            pickle.dump(signals, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(cache_path)
    except Exception:
        pass


def _compute_all_pattern_signals(df, pattern_names):
    """一次性向量化计算多个形态的信号序列，返回 {pattern_name: signal_array}。"""
    signals = {}
    for pattern_name in pattern_names:
        signals[pattern_name] = _compute_pattern_signal(df, pattern_name)
    return signals


def _prepare_feed(df, start_date, end_date):
    """从已加载的 DataFrame 过滤日期范围并创建 Backtrader PandasData feed。"""
    filtered_df = df.loc[start_date:end_date].copy()

    if filtered_df is None or filtered_df.empty:
        raise ValueError(f"{start_date}~{end_date} 无数据")

    # 填充 NaN 值：OHLC 等关键列向前填充，避免回测时因 NaN 崩溃
    for col in filtered_df.columns:
        if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            filtered_df[col] = filtered_df[col].ffill().fillna(0)

    # 用列名定位，兼容不同数据源 CSV 的列顺序差异
    cols = filtered_df.columns.tolist()
    vol_col = 'vol' if 'vol' in cols else ('volume' if 'volume' in cols else -1)
    data = bt.feeds.PandasData(
        dataname=filtered_df, datetime=None,
        open='open', high='high', low='low', close='close',
        volume=vol_col, openinterest=-1,
    )
    return data


def get_data(code, start_date, end_date, data_folder_dir):
    """读取标的日K数据，返回 Backtrader PandasData。

    合并 每日跟踪/训练/测试 多个目录的 CSV 数据，按 trade_date 去重排序后再按日期范围过滤，
    避免只读到单目录（如每日跟踪只有最近21天数据）导致回测区间数据缺失。
    """
    df = _load_raw_dataframe(code, data_folder_dir)
    return _prepare_feed(df, start_date, end_date)


def _compute_pattern_signal(df, pattern_name):
    """使用 TA-Lib 向量化计算单个形态信号序列。"""
    func = getattr(talib, pattern_name, None)
    if func is None:
        raise ValueError(f"不支持的形态: {pattern_name}")
    return func(
        df['open'].values.astype(np.float64),
        df['high'].values.astype(np.float64),
        df['low'].values.astype(np.float64),
        df['close'].values.astype(np.float64),
    )


def _backtest_pattern_vectorized(df, signal, pattern_name, pattern_type,
                                 observe_day, cash, cautious):
    """pandas/numpy 向量化轻量回测，语义与 Backtrader 版本保持一致。

    参数：
        df: 已按日期过滤并升序排列的 DataFrame，含 open/high/low/close
        signal: 形态信号序列（numpy array），与 df 等长
        pattern_name: 形态名
        pattern_type: 'buy' 或 'sell'
        observe_day: 持有期（交易日数）
        cash: 初始资金
        cautious: 是否启用谨慎模式

    返回：
        trade_details: 交易明细列表
        equity_curve: 每日净值序列（numpy array）
        holding_days_total: 持仓天数累计

    优先走 Numba 加速路径；失败时回退到原 pandas/numpy 实现。
    """
    # ===== Numba 加速路径（带 try/except 回退） =====
    if _NUMBA_OK and numba_backtest is not None:
        try:
            return numba_backtest.backtest_with_numba(
                df, signal, pattern_name, pattern_type,
                observe_day, cash, cautious,
            )
        except Exception as _e:
            # 静默回退到原版（已在日志记录一次）
            import logging
            logging.getLogger('trader_system').debug(
                f'[pattern_scan] Numba 回测失败，回退原版: pattern={pattern_name}, '
                f'type={pattern_type}, err={_e}'
            )

    # ===== 原版 pandas/numpy 实现（保留作为回退路径） =====
    n = len(df)
    if n == 0:
        return [], np.array([], dtype=np.float64), 0

    dates = df.index
    opens = df['open'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    commission = 0.0001

    equity_curve = np.full(n, float(cash), dtype=np.float64)
    trade_details = []

    if pattern_type == 'buy':
        signal_mask = signal > 0
    else:
        signal_mask = signal < 0

    position_size = 0
    entry_price = 0.0
    entry_idx = -1
    entry_date = None
    hold_peak_value = float(cash)
    hold_max_drawdown = 0.0
    stop_loss_pending = False
    holding_days_total = 0

    for i in range(n):
        # ===== 1. 执行挂起的止损卖出（买入信号特有） =====
        if pattern_type == 'buy' and stop_loss_pending and position_size > 0:
            sell_price = opens[i]
            pnl = (sell_price - entry_price) * position_size
            pnl_pct = (sell_price / entry_price - 1) * 100
            trade_details.append({
                'buy_date': entry_date.strftime('%Y-%m-%d'),
                'buy_price': float(round(entry_price, 3)),
                'sell_date': dates[i].strftime('%Y-%m-%d'),
                'sell_price': float(round(sell_price, 3)),
                'size': int(position_size),
                'pnl': float(round(pnl, 2)),
                'pnl_pct': float(round(pnl_pct, 2)),
                'hold_days': int(i - entry_idx),
                'hold_max_drawdown': float(round(hold_max_drawdown, 2)),
                'exit_reason': '3%止损',
            })
            cash += position_size * sell_price * (1 - commission)
            position_size = 0
            stop_loss_pending = False

        # ===== 2. 开仓：前一根 bar 信号 → 当前 bar 开盘执行 =====
        if i > 0 and signal_mask[i - 1] and position_size == 0:
            if cautious and not meets_extra_condition_df(pattern_name, df, i):
                pass
            else:
                stock_price = opens[i]
                if stock_price > 0 and stock_price == stock_price:
                    size = int(cash * 0.995 // stock_price // 100 * 100)
                    if size > 0:
                        entry_price = stock_price
                        entry_idx = i
                        entry_date = dates[i]
                        if pattern_type == 'buy':
                            position_size = size
                            cash -= size * stock_price * (1 + commission)
                        else:
                            position_size = size
                            cash += size * stock_price * (1 - commission)
                        # 开仓后初始化峰值（考虑手续费后的净值）
                        if pattern_type == 'buy':
                            hold_peak_value = cash + position_size * stock_price
                        else:
                            hold_peak_value = cash - position_size * stock_price
                        hold_max_drawdown = 0.0

        # ===== 3. 计算当日净值与持仓期回撤 =====
        if position_size != 0:
            if pattern_type == 'buy':
                equity = cash + position_size * closes[i]
            else:
                equity = cash - position_size * closes[i]
            equity_curve[i] = equity
            holding_days_total += 1
            if equity > hold_peak_value:
                hold_peak_value = equity
            if hold_peak_value > 0:
                dd = (hold_peak_value - equity) / hold_peak_value * 100
                if dd > hold_max_drawdown:
                    hold_max_drawdown = dd
        else:
            equity_curve[i] = cash
            hold_peak_value = cash
            hold_max_drawdown = 0.0

        # ===== 4. 检查止损条件（买入信号，买入当天不检查） =====
        if (pattern_type == 'buy' and position_size > 0
                and (i - entry_idx) > 0
                and (closes[i] / entry_price - 1) <= -0.03):
            stop_loss_pending = True

        # ===== 5. 持有期满平仓 =====
        if position_size > 0 and (i - entry_idx) >= observe_day and not stop_loss_pending:
            if pattern_type == 'buy':
                sell_price = opens[i]
                pnl = (sell_price - entry_price) * position_size
                pnl_pct = (sell_price / entry_price - 1) * 100
                trade_details.append({
                    'buy_date': entry_date.strftime('%Y-%m-%d'),
                    'buy_price': float(round(entry_price, 3)),
                    'sell_date': dates[i].strftime('%Y-%m-%d'),
                    'sell_price': float(round(sell_price, 3)),
                    'size': int(position_size),
                    'pnl': float(round(pnl, 2)),
                    'pnl_pct': float(round(pnl_pct, 2)),
                    'hold_days': int(i - entry_idx),
                    'hold_max_drawdown': float(round(hold_max_drawdown, 2)),
                })
                cash += position_size * sell_price * (1 - commission)
            else:
                cover_price = opens[i]
                pnl = (entry_price - cover_price) * position_size
                pnl_pct = (entry_price / cover_price - 1) * 100
                trade_details.append({
                    'open_date': entry_date.strftime('%Y-%m-%d'),
                    'open_price': float(round(entry_price, 3)),
                    'close_date': dates[i].strftime('%Y-%m-%d'),
                    'close_price': float(round(cover_price, 3)),
                    'size': int(position_size),
                    'pnl': float(round(pnl, 2)),
                    'pnl_pct': float(round(pnl_pct, 2)),
                    'hold_days': int(i - entry_idx),
                    'hold_max_drawdown': float(round(hold_max_drawdown, 2)),
                })
                cash -= position_size * cover_price * (1 + commission)
            position_size = 0

    # 数据结束仍未平仓，按最后收盘价平仓
    if position_size > 0:
        last_idx = n - 1
        if pattern_type == 'buy':
            sell_price = closes[last_idx]
            pnl = (sell_price - entry_price) * position_size
            pnl_pct = (sell_price / entry_price - 1) * 100
            trade_details.append({
                'buy_date': entry_date.strftime('%Y-%m-%d'),
                'buy_price': float(round(entry_price, 3)),
                'sell_date': dates[last_idx].strftime('%Y-%m-%d'),
                'sell_price': float(round(sell_price, 3)),
                'size': int(position_size),
                'pnl': float(round(pnl, 2)),
                'pnl_pct': float(round(pnl_pct, 2)),
                'hold_days': int(last_idx - entry_idx),
                'hold_max_drawdown': float(round(hold_max_drawdown, 2)),
            })
            cash += position_size * sell_price * (1 - commission)
        else:
            cover_price = closes[last_idx]
            pnl = (entry_price - cover_price) * position_size
            pnl_pct = (entry_price / cover_price - 1) * 100
            trade_details.append({
                'open_date': entry_date.strftime('%Y-%m-%d'),
                'open_price': float(round(entry_price, 3)),
                'close_date': dates[last_idx].strftime('%Y-%m-%d'),
                'close_price': float(round(cover_price, 3)),
                'size': int(position_size),
                'pnl': float(round(pnl, 2)),
                'pnl_pct': float(round(pnl_pct, 2)),
                'hold_days': int(last_idx - entry_idx),
                'hold_max_drawdown': float(round(hold_max_drawdown, 2)),
            })
            cash -= position_size * cover_price * (1 + commission)
        position_size = 0
        equity_curve[last_idx] = cash

    return trade_details, equity_curve, holding_days_total


# ============================================================================
# AI 过滤对照回测
# ============================================================================
_AI_SERVER_URL = 'http://127.0.0.1:8766'
_ai_filter_available = None  # 缓存探测结果


def _ai_filter_health_check():
    """探测 Model Server 是否可达，结果缓存。"""
    global _ai_filter_available
    if _ai_filter_available is not None:
        return _ai_filter_available
    try:
        import urllib.request
        req = urllib.request.Request(f'{_AI_SERVER_URL}/health', method='GET')
        with urllib.request.urlopen(req, timeout=2) as resp:
            _ai_filter_available = (resp.status == 200)
            return _ai_filter_available
    except Exception:
        _ai_filter_available = False
        return False


def _ai_filter_batch_predict(klines):
    """批量调用 Model Server，返回 (xgb_probs, cnn_probs)。"""
    import urllib.request
    import json as _json
    if not _ai_filter_health_check():
        return None, None

    xgb_probs = None
    cnn_probs = None
    payload = _json.dumps({'klines': klines}).encode('utf-8')

    try:
        req = urllib.request.Request(
            f'{_AI_SERVER_URL}/predict/xgb/batch',
            data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
            if data.get('ai_available'):
                xgb_probs = data.get('probs')
    except Exception:
        pass

    try:
        req = urllib.request.Request(
            f'{_AI_SERVER_URL}/predict/cnn/batch',
            data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
            if data.get('ai_available'):
                cnn_probs = data.get('probs')
    except Exception:
        pass

    return xgb_probs, cnn_probs


def _run_ai_filter_backtest(df, signal, pattern_name, pattern_type,
                            observe_day, cash, cautious,
                            xgb_threshold=0.6, cnn_threshold=0.6):
    """AI 过滤对照回测：仅保留 AI 置信度达标的信号，重新跑回测。

    逻辑:
        1. 找出所有信号触发点（signal[i-1] 为 True，i 为执行日）
        2. 对每个触发点的前 20 根 K 线调用 AI 批量推理
        3. 过滤：xgb_prob >= threshold AND cnn_prob >= threshold → 保留
        4. 用过滤后的信号 mask 重新跑回测
        5. 返回过滤后绩效 + 过滤率

    Args:
        df: 已过滤日期的 K 线 DataFrame
        signal: 原始信号 numpy array
        pattern_name/pattern_type: 形态名/类型
        observe_day/cash/cautious: 回测参数
        xgb_threshold/cnn_threshold: AI 过滤阈值

    Returns:
        dict: {
            'enabled': True,
            'xgb_threshold': float,
            'cnn_threshold': float,
            'original_trades': int,
            'trades': int,
            'win_rate': float,
            'return_pct': float,
            'sharpe': float,
            'hold_max_drawdown': float,
            'filter_rate': float,  # 被过滤比例
        }
        若 AI 服务不可用返回 None（不添加 ai_filter 字段）
    """
    if not _ai_filter_health_check():
        return None

    n = len(df)
    if n < 22:  # 至少需要 20 根历史 + 1 触发 + 1 执行
        return None

    # 找出所有信号触发点（signal[i-1] 为 True，i 为执行日）
    # 与 _backtest_pattern_vectorized 的开仓逻辑一致：i>0 且 signal_mask[i-1] 为 True
    trigger_indices = []
    for i in range(1, n):
        if pattern_type == 'buy':
            if signal[i - 1] > 0:
                trigger_indices.append(i)
        else:
            if signal[i - 1] < 0:
                trigger_indices.append(i)

    if not trigger_indices:
        # 原始信号就无交易，AI 过滤也无意义
        return None

    # 收集每个触发点的前 20 根 K 线窗口
    klines = []
    valid_triggers = []
    for idx in trigger_indices:
        if idx < 20:
            # 数据不足 20 根，跳过此触发点（不过滤，保留原始信号）
            valid_triggers.append((idx, None, None))
            continue
        window = df.iloc[idx - 20: idx]
        if len(window) < 20:
            valid_triggers.append((idx, None, None))
            continue
        arr = window[['open', 'high', 'low', 'close', 'vol']].values.astype(np.float64)
        klines.append(arr.tolist())
        valid_triggers.append((idx, None, None))  # 占位，后面填概率

    if not klines:
        return None

    # 批量推理
    xgb_probs, cnn_probs = _ai_filter_batch_predict(klines)
    if xgb_probs is None and cnn_probs is None:
        return None

    # 分发概率到对应触发点
    prob_idx = 0
    for i, (idx, _, _) in enumerate(valid_triggers):
        if idx < 20 or len(df.iloc[idx - 20: idx]) < 20:
            continue  # 数据不足，不过滤
        xgb_p = xgb_probs[prob_idx] if xgb_probs and prob_idx < len(xgb_probs) else None
        cnn_p = cnn_probs[prob_idx] if cnn_probs and prob_idx < len(cnn_probs) else None
        valid_triggers[i] = (idx, xgb_p, cnn_p)
        prob_idx += 1

    # 构建过滤后的信号 mask
    # 信号在 i-1 触发，i 执行。过滤即把 signal[i-1] 置 0
    filtered_signal = signal.copy()
    kept_count = 0
    total_count = 0
    for idx, xgb_p, cnn_p in valid_triggers:
        if idx < 1:
            continue
        total_count += 1
        # AI 过滤：两个模型概率都需达标（若有概率则判断，无概率则保留）
        keep = True
        if xgb_p is not None and xgb_p < xgb_threshold:
            keep = False
        if cnn_p is not None and cnn_p < cnn_threshold:
            keep = False
        if not keep:
            filtered_signal[idx - 1] = 0  # 把触发信号置 0
        else:
            kept_count += 1

    if kept_count == 0:
        # 所有信号被过滤
        return {
            'enabled': True,
            'xgb_threshold': xgb_threshold,
            'cnn_threshold': cnn_threshold,
            'original_trades': len(trigger_indices),
            'trades': 0,
            'win_rate': 0,
            'return_pct': 0,
            'sharpe': 0,
            'hold_max_drawdown': 0,
            'filter_rate': 1.0 if total_count > 0 else 0,
        }

    # 用过滤后信号重新跑回测
    ai_trade_details, ai_equity_curve, _ = _backtest_pattern_vectorized(
        df, filtered_signal, pattern_name, pattern_type,
        observe_day, cash, cautious,
    )

    ai_total_trades = len(ai_trade_details)
    if ai_total_trades > 0:
        ai_won = sum(1 for t in ai_trade_details if t.get('pnl_pct', 0) > 0)
        ai_win_rate = round(ai_won / ai_total_trades * 100)
    else:
        ai_win_rate = 0
    ai_return_pct = round(sum(t.get('pnl_pct', 0) for t in ai_trade_details), 2)

    # 夏普
    ai_sharpe = 0.0
    if len(ai_equity_curve) > 1:
        daily_returns = np.diff(ai_equity_curve) / ai_equity_curve[:-1]
        valid_mask = daily_returns != 0
        if valid_mask.sum() > 1:
            mean_ret = np.mean(daily_returns[valid_mask])
            std_ret = np.std(daily_returns[valid_mask], ddof=1)
            if std_ret > 1e-12:
                ai_sharpe = round((mean_ret / std_ret) * np.sqrt(252), 2)

    # 最大回撤
    ai_max_dd = 0.0
    if len(ai_equity_curve) > 0:
        peak = ai_equity_curve[0]
        for v in ai_equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > ai_max_dd:
                ai_max_dd = dd

    return {
        'enabled': True,
        'xgb_threshold': xgb_threshold,
        'cnn_threshold': cnn_threshold,
        'original_trades': len(trigger_indices),
        'trades': ai_total_trades,
        'win_rate': ai_win_rate,
        'return_pct': ai_return_pct,
        'sharpe': ai_sharpe,
        'hold_max_drawdown': round(ai_max_dd, 2),
        'filter_rate': round(1 - kept_count / max(total_count, 1), 4),
    }


def run_single_pattern(code, pattern_name, pattern_type, start_date, end_date,
                       data_folder_dir, observe_day=2, cash=100000000, cautious=False,
                       cached_df=None, cached_signal=None):
    """向量化回测单个形态，接口与旧版 Backtrader 实现完全兼容。

    参数：
        cached_signal: 可选，已计算好的形态信号序列（numpy array），
                       与 filtered_df 等长。提供时可跳过 TA-Lib 计算。

    结果库集成：
    - 缓存键：code + pattern_name + pattern_type + start_date + end_date + observe_day + cautious
    - 失效：源数据 mtime 变化自动失效；策略逻辑变更通过 RESULT_VERSION 失效
    - 命中时直接返回缓存，跳过回测计算（秒级返回）
    """
    # 结果库缓存查询：相同参数直接读库
    cache_key = None
    try:
        import result_db
        cache_key = result_db.compute_cache_key(
            code, pattern_name, pattern_type, start_date, end_date,
            observe_day, cautious,
        )
        cached = result_db.get_result(cache_key)
        if cached is not None:
            return cached
    except Exception as e:
        import logging
        logging.getLogger('trader_system').debug(f'[pattern_scan] 结果库查询失败: {e}')

    # 加载数据
    if cached_df is not None:
        df = cached_df.copy()
    else:
        df = _load_raw_dataframe(code, data_folder_dir)

    # 日期范围过滤
    filtered_df = df.loc[start_date:end_date].copy()
    if filtered_df is None or filtered_df.empty:
        raise ValueError(f"{code} 在 {start_date}~{end_date} 无数据")

    # 数值列前向填充，避免 NaN 导致信号异常
    for col in filtered_df.columns:
        if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            filtered_df[col] = filtered_df[col].ffill().fillna(0)

    # 向量化计算形态信号（优先使用缓存）
    if cached_signal is not None:
        signal = cached_signal
    else:
        signal = _compute_pattern_signal(filtered_df, pattern_name)

    # 执行回测
    trade_details, equity_curve, holding_days_total = _backtest_pattern_vectorized(
        filtered_df, signal, pattern_name, pattern_type,
        observe_day, cash, cautious,
    )

    total_trades = len(trade_details)
    if total_trades > 0:
        won = sum(1 for t in trade_details if t.get('pnl_pct', 0) > 0)
        win_rate = round(won / total_trades * 100)
    else:
        win_rate = 0

    return_pct = round(sum(t.get('pnl_pct', 0) for t in trade_details), 2)

    # 年化收益率
    start_dt = datetime.strptime(start_date.replace('-', ''), '%Y%m%d')
    end_dt = datetime.strptime(end_date.replace('-', ''), '%Y%m%d')
    years = max((end_dt - start_dt).days / 365.0, 0.01)
    if return_pct > -100:
        annualized_return = round(((1 + return_pct / 100) ** (1 / years) - 1) * 100, 2)
    else:
        annualized_return = -100

    # 资金占用时间占比
    total_bars = len(filtered_df)
    capital_occupation = round(holding_days_total / total_bars * 100, 2) if total_bars > 0 else 0

    # 夏普比率：基于日收益率（仅含持仓日的净值变化）
    sharpe_ratio = 0.0
    if len(equity_curve) > 1:
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        # 过滤掉无变化的日期（未持仓），只对有效交易日计算
        valid_mask = daily_returns != 0
        if valid_mask.sum() > 1:
            mean_ret = np.mean(daily_returns[valid_mask])
            std_ret = np.std(daily_returns[valid_mask], ddof=1)
            if std_ret > 1e-12:
                sharpe_ratio = round((mean_ret / std_ret) * np.sqrt(252), 2)

    # 最大回撤：基于完整净值曲线
    max_drawdown = 0.0
    if len(equity_curve) > 0:
        peak = equity_curve[0]
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

    result = {
        'pattern': pattern_name,
        'pattern_cn': PATTERN_CN_NAMES.get(pattern_name, pattern_name),
        'pattern_desc': get_pattern_description(pattern_name),
        'type': pattern_type,
        'trades': total_trades,
        'win_rate': win_rate,
        'return_pct': return_pct,
        'annualized_return': annualized_return,
        'capital_occupation': capital_occupation,
        'sharpe': sharpe_ratio,
        'hold_max_drawdown': round(max_drawdown, 2),
        'trade_details': trade_details,
    }

    # AI 过滤对照回测（可选，失败静默降级）
    ai_filter_result = _run_ai_filter_backtest(
        filtered_df, signal, pattern_name, pattern_type,
        observe_day, cash, cautious,
        xgb_threshold=getattr(config, 'AI_BUY_THRESHOLD', 0.6),
        cnn_threshold=getattr(config, 'AI_SELL_THRESHOLD', 0.6),
    )
    if ai_filter_result is not None:
        result['ai_filter'] = ai_filter_result

    # 结果库写入：缓存回测结果，下次相同参数直接读库
    if cache_key is not None:
        try:
            import result_db
            result_db.save_result(
                cache_key, code, pattern_name, pattern_type,
                start_date, end_date, observe_day, cautious, result
            )
        except Exception as e:
            import logging
            logging.getLogger('trader_system').debug(f'[pattern_scan] 结果库写入失败: {e}')

    return result


def calc_buy_hold_return(code, start_date, end_date, data_folder_dir, cash=100000000):
    """计算标的买入持有基准（首日开盘满仓买入，末日收盘卖出）。

    使用与策略回测一致的合并数据源（每日跟踪/训练/测试多目录 CSV），
    避免仅读取单一目录导致的数据缺失或口径不一致。
    """
    try:
        df = _load_raw_dataframe(code, data_folder_dir)
        filtered = df.loc[start_date:end_date]
        if filtered.empty:
            return None
        first_open = float(filtered.iloc[0]['open'])
        last_close = float(filtered.iloc[-1]['close'])
        buy_size = int(cash * 0.995 // first_open // 100 * 100)
        if buy_size <= 0:
            return None
        cost = buy_size * first_open
        income = buy_size * last_close
        return_pct = (income / cost - 1) * 100

        start_dt = datetime.strptime(start_date.replace('-', ''), '%Y%m%d')
        end_dt = datetime.strptime(end_date.replace('-', ''), '%Y%m%d')
        years = max((end_dt - start_dt).days / 365.0, 0.01)
        if return_pct > -100:
            annualized = round(((1 + return_pct / 100) ** (1 / years) - 1) * 100, 2)
        else:
            annualized = -100
        return {
            'return_pct': round(return_pct, 2),
            'annualized_return': annualized,
            'start_date': filtered.index[0].strftime('%Y-%m-%d'),
            'end_date': filtered.index[-1].strftime('%Y-%m-%d'),
            'buy_price': round(first_open, 3),
            'sell_price': round(last_close, 3),
        }
    except Exception:
        return None


def _scan_one_pattern(task_args):
    """进程池worker：回测单个形态。模块级函数，可被pickle。"""
    code, pattern, ptype, start_date, end_date, data_folder_dir, observe_day, cash, cautious = task_args
    try:
        r = run_single_pattern(code, pattern, ptype, start_date, end_date,
                               data_folder_dir, observe_day=observe_day, cash=cash, cautious=cautious)
        r['code'] = code
        return r
    except Exception as e:
        return {
            'code': code,
            'pattern': pattern,
            'pattern_cn': PATTERN_CN_NAMES.get(pattern, pattern),
            'type': ptype,
            'trades': 0,
            'win_rate': 0,
            'return_pct': 0,
            'sharpe': 0,
            'hold_max_drawdown': 0,
            'error': str(e),
        }


def _scan_one_code(args):
    """进程池 worker：回测单个 code 的全部形态（任务粒度 = 单 code 全形态）。

    模块级函数可被 pickle；内部一次性加载 DataFrame，所有形态复用，
    并通过 code 级信号缓存避免重复 TA-Lib 计算。
    """
    code, patterns, start_date, end_date, data_folder_dir, observe_day, cash, cautious = args
    try:
        df = _load_raw_dataframe(code, data_folder_dir)

        # 按日期范围预处理（与 run_single_pattern 内部保持一致）
        filtered_df = df.loc[start_date:end_date].copy()
        if filtered_df is None or filtered_df.empty:
            raise ValueError(f"{code} 在 {start_date}~{end_date} 无数据")
        for col in filtered_df.columns:
            if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                filtered_df[col] = filtered_df[col].ffill().fillna(0)

        # code 级形态信号缓存
        unique_patterns = sorted({p for p, _ in patterns})
        cache_path = _signal_cache_path(code, start_date, end_date, data_folder_dir)
        cached_signals = _load_cached_signals(cache_path)
        if cached_signals is None:
            cached_signals = _compute_all_pattern_signals(filtered_df, unique_patterns)
            _save_cached_signals(cache_path, cached_signals)

        results = []
        for pattern_name, pattern_type in patterns:
            r = run_single_pattern(
                code, pattern_name, pattern_type, start_date, end_date,
                data_folder_dir, observe_day=observe_day, cash=cash,
                cautious=cautious, cached_df=df,
                cached_signal=cached_signals.get(pattern_name),
            )
            results.append(r)
        return results
    except Exception as e:
        # 该 code 全部形态失败
        return [
            {
                'code': code,
                'pattern': pattern_name,
                'pattern_cn': PATTERN_CN_NAMES.get(pattern_name, pattern_name),
                'type': pattern_type,
                'trades': 0,
                'win_rate': 0,
                'return_pct': 0,
                'sharpe': 0,
                'hold_max_drawdown': 0,
                'error': str(e),
            }
            for pattern_name, pattern_type in patterns
        ]


def scan_stock(code, start_date, end_date, observe_day=2, cash=100000000,
               data_folder_dir=None, scan_buy=True, scan_sell=True, progress_cb=None,
               cautious=False):
    """扫描单个 code 的全部形态，任务粒度为"单 code 全形态"。

    一次性加载该 code 的 DataFrame 并预计算所有形态信号，避免重复 IO；
    默认在单进程内串行跑完所有形态（向量化后已足够快）。
    """
    if data_folder_dir is None:
        if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
            data_folder_dir = str(config.TRAIN_DATA_INDEX_DIR) + '/'
        else:
            data_folder_dir = str(config.TRAIN_DATA_A_DIR) + '/'

    # 构建该 code 的全部 (pattern, type) 对
    patterns = []
    if scan_buy:
        for pattern in BUY_PATTERNS:
            patterns.append((pattern, 'buy'))
    if scan_sell:
        for pattern in SELL_PATTERNS:
            patterns.append((pattern, 'sell'))

    total = len(patterns)
    results = []

    # 一次性加载 DataFrame，所有形态复用
    cached_df = _load_raw_dataframe(code, data_folder_dir)

    # 按本次扫描的日期范围预处理（与 run_single_pattern 内部过滤/填充保持一致）
    filtered_df = cached_df.loc[start_date:end_date].copy()
    if filtered_df is None or filtered_df.empty:
        raise ValueError(f"{code} 在 {start_date}~{end_date} 无数据")
    for col in filtered_df.columns:
        if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            filtered_df[col] = filtered_df[col].ffill().fillna(0)

    # code 级形态信号缓存：先尝试加载，未命中则一次性计算全部并写入缓存
    unique_patterns = sorted({p for p, _ in patterns})
    cache_path = _signal_cache_path(code, start_date, end_date, data_folder_dir)
    cached_signals = _load_cached_signals(cache_path)
    if cached_signals is None:
        cached_signals = _compute_all_pattern_signals(filtered_df, unique_patterns)
        _save_cached_signals(cache_path, cached_signals)

    # 单 code 全形态串行执行：避免多进程调度开销
    for idx, (pattern_name, pattern_type) in enumerate(patterns):
        r = run_single_pattern(
            code, pattern_name, pattern_type, start_date, end_date,
            data_folder_dir, observe_day=observe_day, cash=cash,
            cautious=cautious, cached_df=cached_df,
            cached_signal=cached_signals.get(pattern_name),
        )
        results.append(r)
        if progress_cb:
            progress_cb(idx + 1, total, code, pattern_name)

    return [r for r in results if r is not None]
