"""多策略叠加回测模块：单标的 + 多形态并行回测，汇总组合指标。

每个形态独立运行回测（全额资金），最后按等权合并结果。
"""

import numpy as np
import pandas as pd
import backtrader as bt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

from config import config
from signal_utils import BUY_PATTERNS, SELL_PATTERNS, PATTERN_CN_NAMES
from cautious_mode import meets_extra_condition

try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False


def _get_candidate_dirs(code, data_folder_dir):
    """获取标的可能的数据目录列表（供缓存失效检查复用）。"""
    candidate_dirs = []
    try:
        from config import config as _cfg
        if _is_index_code(code):
            candidate_dirs.append(_cfg.DAILY_TRACKING_INDEX_DIR)
            candidate_dirs.append(_cfg.TRAIN_DATA_INDEX_DIR)
            candidate_dirs.append(_cfg.TEST_DATA_INDEX_DIR)
        else:
            candidate_dirs.append(_cfg.DAILY_TRACKING_A_DIR)
            candidate_dirs.append(_cfg.TRAIN_DATA_A_DIR)
            candidate_dirs.append(_cfg.TEST_DATA_A_DIR)
    except Exception:
        pass
    if data_folder_dir:
        try:
            d = Path(data_folder_dir) if not isinstance(data_folder_dir, Path) else data_folder_dir
            if d not in candidate_dirs:
                candidate_dirs.append(d)
        except Exception:
            pass
    return candidate_dirs


def _effective_data_path(code, data_dir):
    """返回指定目录中 code 的有效数据文件：优先 Parquet，回退 CSV。"""
    data_dir = Path(data_dir)
    pq_path = data_dir / f'{code}_daily.parquet'
    if _PARQUET_AVAILABLE and pq_path.exists() and pq_path.stat().st_size > 0:
        return pq_path, True
    csv_path = data_dir / f'{code}_daily.csv'
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path, False
    return None, False


def _load_merged_df_uncached(code, data_folder_dir):
    """合并 每日跟踪/训练/测试 多个目录的 Parquet/CSV 数据（无缓存，原始实现）。

    返回按 trade_date 索引的 DataFrame（未按日期范围过滤）。
    """
    import pandas as pd

    candidate_dirs = _get_candidate_dirs(code, data_folder_dir)
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


def _load_merged_df(code, data_folder_dir):
    """合并 Parquet/CSV 数据（带文件缓存，避免重复 I/O）。

    多个 worker（ProcessPoolExecutor 子进程）对同一标的重复调用时，
    第一个进程写入 pickle 缓存，后续进程直接从缓存读取，
    避免每个 worker 独立读取+拼接数据文件的重复开销。
    缓存通过源数据文件（Parquet 优先，CSV 回退）的 mtime 自动失效。
    """
    import pickle

    candidate_dirs = _get_candidate_dirs(code, data_folder_dir)

    # 缓存目录
    cache_dir = Path(config.BASE_DIR) / '.cache' / 'merged_data'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{code}.pkl'

    # 检查缓存是否有效：所有源数据文件的 mtime 均 ≤ 缓存 mtime
    if cache_file.exists():
        try:
            cache_mtime = cache_file.stat().st_mtime
            stale = False
            for d in candidate_dirs:
                data_path, _ = _effective_data_path(code, d)
                if data_path and data_path.stat().st_mtime > cache_mtime:
                    stale = True
                    break
            if not stale:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass  # 缓存损坏或读取失败，回退到原始加载

    # 从源数据文件加载
    df = _load_merged_df_uncached(code, data_folder_dir)

    # 写入缓存（原子写入：先写临时文件，再 rename，避免多进程竞争损坏）
    try:
        tmp_file = cache_file.with_suffix('.tmp')
        with open(tmp_file, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(cache_file)
    except Exception:
        pass  # 缓存写入失败不影响主流程

    return df


def _get_data(code, start_date, end_date, data_folder_dir):
    """读取标的日K数据，返回 Backtrader PandasData。

    合并 每日跟踪/训练/测试 多个目录的 Parquet/CSV 数据，按 trade_date 去重排序后再按日期范围过滤，
    避免只读到单目录（如每日跟踪只有最近21天数据）导致回测区间数据缺失。
    """
    df = _load_merged_df(code, data_folder_dir)
    filtered_df = df.loc[start_date:end_date]

    if filtered_df is None or filtered_df.empty:
        raise ValueError(f"{code} 在 {start_date}~{end_date} 无数据")

    # 用列名定位，兼容不同数据源 CSV 的列顺序差异
    cols = filtered_df.columns.tolist()
    vol_col = 'vol' if 'vol' in cols else ('volume' if 'volume' in cols else -1)
    data = bt.feeds.PandasData(
        dataname=filtered_df, datetime=None,
        open='open', high='high', low='low', close='close',
        volume=vol_col, openinterest=-1,
    )
    return data


def _calc_buy_hold_and_annualized(code, start_date, end_date, data_folder_dir, strategy_return, trade_details=None):
    """计算买入持有收益率、买入持有年化、策略年化收益。

    采用「资金利用率调整」：买入持有收益按策略实际持仓天数占比折算，
    使对比更公平（策略资金大部分时间空仓，不应与全程满仓的买入持有直接比较）。

    Args:
        strategy_return: 策略总收益率（百分比），用于计算策略年化
        trade_details: 策略交易明细列表，用于计算总持仓天数

    Returns:
        {
            'buy_hold_return': float,        # 买入持有收益率 %（原始）
            'buy_hold_annualized': float,    # 买入持有年化 %（原始）
            'adj_buy_hold_return': float,    # 调整后买入持有收益率 %（按资金利用率）
            'adj_buy_hold_annualized': float,# 调整后买入持有年化 %
            'strategy_annualized': float,    # 策略年化 %
            'years': float,                  # 回测年数
            'start_close': float,            # 起始收盘价
            'end_close': float,              # 结束收盘价
            'total_trading_days': int,       # 回测区间总交易日数
            'total_hold_days': int,          # 策略总持仓天数
            'hold_ratio': float,             # 持仓天数占比（0-1）
        }
    """
    from datetime import datetime

    try:
        df = _load_merged_df(code, data_folder_dir)
        filtered = df.loc[start_date:end_date]
        if filtered is None or filtered.empty or 'close' not in filtered.columns:
            return {'buy_hold_return': 0.0, 'buy_hold_annualized': 0.0,
                    'adj_buy_hold_return': 0.0, 'adj_buy_hold_annualized': 0.0,
                    'strategy_annualized': 0.0, 'years': 0.0,
                    'start_close': 0.0, 'end_close': 0.0,
                    'total_trading_days': 0, 'total_hold_days': 0, 'hold_ratio': 0.0}

        start_close = float(filtered['close'].iloc[0])
        end_close = float(filtered['close'].iloc[-1])
        if start_close <= 0:
            buy_hold_return = 0.0
        else:
            buy_hold_return = round((end_close / start_close - 1) * 100, 2)

        # 计算年数（按自然日）
        start_dt = datetime.strptime(start_date.replace('-', ''), '%Y%m%d')
        end_dt = datetime.strptime(end_date.replace('-', ''), '%Y%m%d')
        years = max((end_dt - start_dt).days / 365.0, 0.01)

        # 买入持有年化
        if buy_hold_return > -100:
            buy_hold_annualized = round(((1 + buy_hold_return / 100) ** (1 / years) - 1) * 100, 2)
        else:
            buy_hold_annualized = -100.0

        # 策略年化
        if strategy_return > -100:
            strategy_annualized = round(((1 + strategy_return / 100) ** (1 / years) - 1) * 100, 2)
        else:
            strategy_annualized = -100.0

        # ===== 资金利用率调整 =====
        # 计算策略总持仓天数与回测区间总交易日数
        total_trading_days = len(filtered)
        total_hold_days = sum(int(t.get('hold_days', 0) or 0) for t in (trade_details or []))
        hold_ratio = round(total_hold_days / total_trading_days, 4) if total_trading_days > 0 else 0.0

        # 调整后买入持有收益 = 原始买入持有收益 × 持仓天数占比
        adj_buy_hold_return = round(buy_hold_return * hold_ratio, 2)
        # 调整后买入持有年化
        if adj_buy_hold_return > -100:
            adj_buy_hold_annualized = round(((1 + adj_buy_hold_return / 100) ** (1 / years) - 1) * 100, 2)
        else:
            adj_buy_hold_annualized = -100.0

        return {
            'buy_hold_return': buy_hold_return,
            'buy_hold_annualized': buy_hold_annualized,
            'adj_buy_hold_return': adj_buy_hold_return,
            'adj_buy_hold_annualized': adj_buy_hold_annualized,
            'strategy_annualized': strategy_annualized,
            'years': round(years, 2),
            'start_close': round(start_close, 2),
            'end_close': round(end_close, 2),
            'total_trading_days': total_trading_days,
            'total_hold_days': total_hold_days,
            'hold_ratio': hold_ratio,
        }
    except Exception:
        return {'buy_hold_return': 0.0, 'buy_hold_annualized': 0.0,
                'adj_buy_hold_return': 0.0, 'adj_buy_hold_annualized': 0.0,
                'strategy_annualized': 0.0, 'years': 0.0,
                'start_close': 0.0, 'end_close': 0.0,
                'total_trading_days': 0, 'total_hold_days': 0, 'hold_ratio': 0.0}


def _is_index_code(code):
    """判断 code 是否为指数代码（用于选择数据目录）"""
    if not code:
        return False
    # 海外指数简码
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


def _run_single_pattern_worker(args):
    """子进程工作函数：跑单个形态回测，返回指标。

    统一使用 pattern_scan.run_single_pattern（向量化），确保与信号更新/
    策略表现的计算结果一致。返回结构与原 Backtrader 版本保持兼容。
    """
    import pattern_scan

    code, pattern_name, pattern_type, start_date, end_date, data_folder_dir, observe_day, cash, cautious = args

    # 确定数据目录：优先使用传入目录；未传入时按指数/A股默认
    if data_folder_dir:
        dfd = str(data_folder_dir)
    elif _is_index_code(code):
        dfd = str(config.TRAIN_DATA_INDEX_DIR)
    else:
        dfd = str(config.TRAIN_DATA_A_DIR)

    try:
        res = pattern_scan.run_single_pattern(
            code=code,
            pattern_name=pattern_name,
            pattern_type=pattern_type,
            start_date=start_date,
            end_date=end_date,
            data_folder_dir=dfd,
            observe_day=observe_day,
            cash=cash,
            cautious=cautious,
        )
    except Exception as e:
        cn_name = PATTERN_CN_NAMES.get(pattern_name, pattern_name)
        return {
            'pattern_name': pattern_name,
            'pattern_cn': cn_name,
            'pattern_type': pattern_type,
            'observe_day': observe_day,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'won': 0,
            'win_rate': 0,
            'pnl_total': 0,
            'trade_details': [],
            'error': str(e),
        }

    trade_details = res.get('trade_details', [])
    total_trades = len(trade_details)
    won = sum(1 for t in trade_details if t.get('pnl_pct', 0) > 0)
    win_rate = round(won / total_trades * 100, 0) if total_trades > 0 else 0

    return {
        'pattern_name': pattern_name,
        'pattern_cn': res.get('pattern_cn', PATTERN_CN_NAMES.get(pattern_name, pattern_name)),
        'pattern_type': pattern_type,
        'observe_day': observe_day,
        'total_return': res.get('return_pct', 0),
        'sharpe_ratio': res.get('sharpe', 0),
        'max_drawdown': res.get('hold_max_drawdown', 0),
        'total_trades': total_trades,
        'won': won,
        'win_rate': int(win_rate),
        'pnl_total': 0,
        'trade_details': trade_details,
    }


def _run_combined_strategy_worker(args):
    """子进程工作函数：所有形态合并到一个 Strategy 实例，共享一份资金。

    解决"多信号同日触发，资金被重复计算"的问题：
    - 同一交易日多形态触发买入，只买入1次（全额资金）
    - 持有期内（observe_day 内）忽略任何新信号，不加仓
    - 卖出/空头对称处理
    - 每笔交易明细记录 trigger_patterns（由哪些形态触发）

    Args:
        args: (code, buy_pattern_names, sell_pattern_names, observe_day,
               start_date, end_date, data_folder_dir, cash, cautious)
    """
    (code, buy_pattern_names, sell_pattern_names, observe_day,
     start_date, end_date, data_folder_dir, cash, cautious) = args

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0001)

    data = _get_data(code, start_date, end_date, data_folder_dir)
    cerebro.adddata(data)

    # 启用 Cheat-On-Open：在 next() 中创建的订单以当前 bar 的开盘价成交
    # 配合 signal[-1] 实现：前一根 bar 的信号 → 当前 bar 开盘价执行
    cerebro.broker.set_coc(True)

    # 交易明细收集器（闭包变量，被 Strategy 内部类访问）
    trade_details = []

    class CombinedPatternStrategy(bt.Strategy):
        params = (
            ('buy_names', []),     # 买入形态名列表
            ('sell_names', []),    # 卖出形态名列表
            ('observe_day', 2),
            ('cautious', False),
        )

        def __init__(self):
            # 初始化所有形态的 TA-Lib 信号
            self.buy_signals = {}
            self.sell_signals = {}
            for name in self.p.buy_names:
                func = getattr(bt.talib, name, None)
                if func:
                    self.buy_signals[name] = func(self.data.open, self.data.high,
                                                  self.data.low, self.data.close)
            for name in self.p.sell_names:
                func = getattr(bt.talib, name, None)
                if func:
                    self.sell_signals[name] = func(self.data.open, self.data.high,
                                                   self.data.low, self.data.close)
            self.buyday = 0
            self.have_position = False
            self.position_direction = 0  # 1=多头, -1=空头
            self.entry_price = 0
            self.entry_date = None
            self.trigger_patterns = []  # 本次开仓由哪些形态触发
            self.hold_peak_value = 0
            self.hold_max_drawdown = 0
            self.stop_loss_pending = False  # 止损挂起，下一根 bar 开盘执行

        def next(self):
            stock_price = self.data.open[0]
            # 跳过 NaN 数据
            if not (stock_price == stock_price) or stock_price <= 0:
                return

            # Margin 拒绝后重置状态
            if self.have_position and self.position.size == 0 and self.buyday > 0:
                self.have_position = False
                self.buyday = 0
                self.position_direction = 0
                self.trigger_patterns = []

            # ===== 1. 执行挂起的止损卖出（前一根 bar 收盘触发，本根 bar 开盘执行）=====
            if self.stop_loss_pending and self.have_position and self.position.size != 0:
                if self.position_direction > 0:
                    sell_price = self.data.open[0]
                    pnl_pct = (sell_price / self.entry_price - 1) * 100
                    trade_details.append({
                        'buy_date': self.entry_date.isoformat() if self.entry_date else '',
                        'buy_price': round(self.entry_price, 3),
                        'sell_date': self.datas[0].datetime.date(0).isoformat(),
                        'sell_price': round(sell_price, 3),
                        'size': int(self.position.size),
                        'hold_days': self.buyday,
                        'pnl_pct': round(pnl_pct, 2),
                        'hold_max_drawdown': round(self.hold_max_drawdown, 2),
                        'trigger_patterns': list(self.trigger_patterns),
                        'direction': 'long',
                        'exit_reason': '3%止损',
                    })
                    self.sell(size=self.position.size)
                else:
                    cover_price = self.data.open[0]
                    pnl_pct = (self.entry_price / cover_price - 1) * 100
                    trade_details.append({
                        'open_date': self.entry_date.isoformat() if self.entry_date else '',
                        'open_price': round(self.entry_price, 3),
                        'close_date': self.datas[0].datetime.date(0).isoformat(),
                        'close_price': round(cover_price, 3),
                        'size': int(abs(self.position.size)),
                        'hold_days': self.buyday,
                        'pnl_pct': round(pnl_pct, 2),
                        'hold_max_drawdown': round(self.hold_max_drawdown, 2),
                        'trigger_patterns': list(self.trigger_patterns),
                        'direction': 'short',
                        'exit_reason': '3%止损',
                    })
                    self.buy(size=abs(self.position.size))
                self.have_position = False
                self.buyday = 0
                self.position_direction = 0
                self.trigger_patterns = []
                self.stop_loss_pending = False

            # ===== 2. 正常持有期满平仓（open 执行，COC 以开盘价成交）=====
            if self.have_position and self.buyday == self.p.observe_day and self.position.size != 0:
                if self.position_direction > 0:
                    sell_price = self.data.open[0]
                    pnl_pct = (sell_price / self.entry_price - 1) * 100
                    trade_details.append({
                        'buy_date': self.entry_date.isoformat() if self.entry_date else '',
                        'buy_price': round(self.entry_price, 3),
                        'sell_date': self.datas[0].datetime.date(0).isoformat(),
                        'sell_price': round(sell_price, 3),
                        'size': int(self.position.size),
                        'hold_days': self.buyday,
                        'pnl_pct': round(pnl_pct, 2),
                        'hold_max_drawdown': round(self.hold_max_drawdown, 2),
                        'trigger_patterns': list(self.trigger_patterns),
                        'direction': 'long',
                    })
                    self.sell(size=self.position.size)
                else:
                    cover_price = self.data.open[0]
                    pnl_pct = (self.entry_price / cover_price - 1) * 100
                    trade_details.append({
                        'open_date': self.entry_date.isoformat() if self.entry_date else '',
                        'open_price': round(self.entry_price, 3),
                        'close_date': self.datas[0].datetime.date(0).isoformat(),
                        'close_price': round(cover_price, 3),
                        'size': int(abs(self.position.size)),
                        'hold_days': self.buyday,
                        'pnl_pct': round(pnl_pct, 2),
                        'hold_max_drawdown': round(self.hold_max_drawdown, 2),
                        'trigger_patterns': list(self.trigger_patterns),
                        'direction': 'short',
                    })
                    self.buy(size=abs(self.position.size))
                self.have_position = False
                self.buyday = 0
                self.position_direction = 0
                self.trigger_patterns = []

            # ===== 3. 无持仓：检查前一根 bar 的开仓信号（买入优先）=====
            if not self.have_position and not self.stop_loss_pending:
                cash = self.broker.getcash()
                buy_size = int(cash * 0.995 // stock_price // 100 * 100)
                if buy_size > 0:
                    # 检查前一根 bar 的买入信号
                    triggered_buy = []
                    for name, sig in self.buy_signals.items():
                        if sig is not None and sig[-1] is not None and sig[-1] > 0:
                            if not self.p.cautious or meets_extra_condition(name, self.data):
                                triggered_buy.append(name)
                    if triggered_buy:
                        self.buy(size=buy_size)
                        self.have_position = True
                        self.buyday = 0
                        self.position_direction = 1
                        self.entry_price = self.data.open[0]
                        self.entry_date = self.datas[0].datetime.date(0)
                        self.trigger_patterns = triggered_buy
                        cur_value = self.broker.getvalue()
                        self.hold_peak_value = cur_value
                        self.hold_max_drawdown = 0
                    else:
                        # 检查前一根 bar 的卖出信号
                        triggered_sell = []
                        for name, sig in self.sell_signals.items():
                            if sig is not None and sig[-1] is not None and sig[-1] < 0:
                                if not self.p.cautious or meets_extra_condition(name, self.data):
                                    triggered_sell.append(name)
                        if triggered_sell:
                            self.sell(size=buy_size)
                            self.have_position = True
                            self.buyday = 0
                            self.position_direction = -1
                            self.entry_price = self.data.open[0]
                            self.entry_date = self.datas[0].datetime.date(0)
                            self.trigger_patterns = triggered_sell
                            cur_value = self.broker.getvalue()
                            self.hold_peak_value = cur_value
                            self.hold_max_drawdown = 0

            # ===== 4. 检查3%止损条件（收盘价判断，触发后下一根 bar 开盘执行）=====
            if (self.have_position and self.buyday > 0 and self.position.size != 0
                    and self.entry_price > 0 and not self.stop_loss_pending):
                if (self.position_direction > 0
                        and (self.data.close[0] / self.entry_price - 1) <= -0.03):
                    self.stop_loss_pending = True

            # 持仓中：累加天数 + 更新持有期回撤
            if self.have_position:
                self.buyday += 1
                cur_value = self.broker.getvalue()
                if cur_value > self.hold_peak_value:
                    self.hold_peak_value = cur_value
                if self.hold_peak_value > 0:
                    dd = (self.hold_peak_value - cur_value) / self.hold_peak_value * 100
                    if dd > self.hold_max_drawdown:
                        self.hold_max_drawdown = dd

        def stop(self):
            # 回测结束时若仍有持仓，用最后一根 bar 的收盘价平仓并记录交易明细
            if self.have_position and self.position.size != 0 and self.entry_price > 0:
                if self.position_direction > 0:
                    sell_price = self.data.close[0]
                    pnl_pct = (sell_price / self.entry_price - 1) * 100
                    trade_details.append({
                        'buy_date': self.entry_date.isoformat() if self.entry_date else '',
                        'buy_price': round(self.entry_price, 3),
                        'sell_date': self.datas[0].datetime.date(0).isoformat(),
                        'sell_price': round(sell_price, 3),
                        'size': int(self.position.size),
                        'hold_days': self.buyday,
                        'pnl_pct': round(pnl_pct, 2),
                        'hold_max_drawdown': round(self.hold_max_drawdown, 2),
                        'trigger_patterns': list(self.trigger_patterns),
                        'direction': 'long',
                    })
                    self.sell(size=self.position.size)
                else:
                    cover_price = self.data.close[0]
                    pnl_pct = (self.entry_price / cover_price - 1) * 100
                    trade_details.append({
                        'open_date': self.entry_date.isoformat() if self.entry_date else '',
                        'open_price': round(self.entry_price, 3),
                        'close_date': self.datas[0].datetime.date(0).isoformat(),
                        'close_price': round(cover_price, 3),
                        'size': int(abs(self.position.size)),
                        'hold_days': self.buyday,
                        'pnl_pct': round(pnl_pct, 2),
                        'hold_max_drawdown': round(self.hold_max_drawdown, 2),
                        'trigger_patterns': list(self.trigger_patterns),
                        'direction': 'short',
                    })
                    self.buy(size=abs(self.position.size))

    cerebro.addstrategy(
        CombinedPatternStrategy,
        buy_names=buy_pattern_names,
        sell_names=sell_pattern_names,
        observe_day=observe_day,
        cautious=cautious,
    )
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    result = cerebro.run()
    strat = result[0]

    final_value = cerebro.broker.getvalue()
    # 真实账户收益率（考虑复利，无资金重复）—— 仅作参考，不展示
    real_return = round((final_value / cash - 1) * 100, 2)

    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio')
    if sharpe_ratio is not None and sharpe_ratio == sharpe_ratio:
        sharpe_ratio = round(sharpe_ratio, 2)
    else:
        sharpe_ratio = 0.0

    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get('max', {}).get('drawdown', 0)

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    # 胜率按 pnl_pct > 0 统计（与交易明细的口径一致，不含手续费干扰）
    if total_trades > 0:
        won = sum(1 for t in trade_details if t.get('pnl_pct', 0) > 0)
        win_rate = round(won / total_trades * 100, 0)
    else:
        win_rate = 0

    # 组合总收益 = 各笔交易盈亏率的算术和（与交易明细一致，不考虑复利）
    # 这样 UI 展示的总收益与交易明细的加总完全对得上
    total_return = round(sum(t.get('pnl_pct', 0) for t in trade_details), 2)

    return {
        'total_return': total_return,
        'real_return': real_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': round(max_dd, 2),
        'total_trades': total_trades,
        'win_rate': int(win_rate),
        'trade_details': trade_details,
    }


def run_multi_pattern_backtest(code, patterns, start_date, end_date,
                                data_folder_dir, cash=100000000, cautious=False,
                                max_workers=None):
    """多策略叠加回测：并行运行多个形态，汇总结果。

    组合收益（combined）采用"合并策略实例"算法：所有形态合并到一个 Strategy 实例，
    共享一份资金，同日多信号只买1次，持有期内忽略新信号。这样得到的组合收益是
    真实的账户收益率，无资金重复计算。

    同时并行运行各形态的独立回测（patterns 列表），用于展示单形态单独表现对比。

    Args:
        code: 标的代码，如 '000533.SZ'
        patterns: [{'pattern_name': 'CDLMARUBOZU', 'pattern_type': 'buy', 'observe_day': 2}, ...]
        start_date: '20240101'
        end_date: '20241231'
        data_folder_dir: 数据目录路径
        cash: 总资金
        cautious: 谨慎模式
        max_workers: 并行数（默认 CPU 核心数）

    Returns:
        {
            'code': str,
            'patterns': [{per-pattern metrics}, ...],  # 单形态独立回测结果
            'combined': {total_return, sharpe_ratio, max_drawdown, total_trades,
                         win_rate, trade_details},  # 合并策略真实结果
            'pattern_count': int,
            'valid_count': int,
        }
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(patterns), getattr(config, 'MAX_WORKERS', 4))
    max_workers = max(1, max_workers)

    worker_timeout = getattr(config, 'WORKER_TIMEOUT_SECONDS', 30)

    # 每个形态独立回测（全额资金）—— 用于展示单形态表现对比
    cash_per = cash

    args_list = []
    for p in patterns:
        args_list.append((
            code, p['pattern_name'], p['pattern_type'],
            start_date, end_date, data_folder_dir,
            p.get('observe_day', 2), cash_per, cautious
        ))

    pattern_results = []
    if len(args_list) == 1:
        try:
            pattern_results.append(_run_single_pattern_worker(args_list[0]))
        except Exception as e:
            pattern_results.append({
                'pattern_name': args_list[0][1],
                'pattern_cn': PATTERN_CN_NAMES.get(args_list[0][1], args_list[0][1]),
                'pattern_type': args_list[0][2],
                'observe_day': args_list[0][6],
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                'total_trades': 0, 'win_rate': 0, 'pnl_total': 0,
                'error': str(e),
            })
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_pattern_worker, args): args for args in args_list}
            for future in as_completed(futures):
                args = futures[future]
                try:
                    result = future.result(timeout=worker_timeout)
                    pattern_results.append(result)
                except TimeoutError:
                    pattern_results.append({
                        'pattern_name': args[1],
                        'pattern_cn': PATTERN_CN_NAMES.get(args[1], args[1]),
                        'pattern_type': args[2],
                        'observe_day': args[6],
                        'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                        'total_trades': 0, 'win_rate': 0, 'pnl_total': 0,
                        'error': f'任务超时（{worker_timeout}秒）',
                    })
                except Exception as e:
                    pattern_results.append({
                        'pattern_name': args[1],
                        'pattern_cn': PATTERN_CN_NAMES.get(args[1], args[1]),
                        'pattern_type': args[2],
                        'observe_day': args[6],
                        'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                        'total_trades': 0, 'win_rate': 0, 'pnl_total': 0,
                        'error': str(e),
                    })

    valid = [r for r in pattern_results if not r.get('error')]

    # 合并策略实例：所有形态共享一份资金，获取真实组合收益
    # 按方向分组，并取统一的 observe_day（用第一个有效 pattern 的值）
    buy_pattern_names = [p['pattern_name'] for p in patterns if p.get('pattern_type') == 'buy']
    sell_pattern_names = [p['pattern_name'] for p in patterns if p.get('pattern_type') == 'sell']
    # observe_day 取所有 patterns 中第一个非 None 的值
    combined_observe_day = 2
    for p in patterns:
        od = p.get('observe_day')
        if od is not None:
            combined_observe_day = od
            break

    # 合并策略实例：用于获取真实账户层面的最大回撤/夏普比率/真实收益率
    # 注意：合并策略会丢弃重叠信号（持仓中不重复买入），其 trade_details 与
    # 各形态独立 trade_details 之和会不一致。为保持"组合总收益 = 各形态独立收益之和"
    # 的一致性，组合的 total_return/trade_details/win_rate/total_trades 改为合并
    # 各形态独立的 trade_details；max_drawdown/sharpe_ratio/real_return 仍用合并策略实例。
    combined = {
        'total_return': 0.0,
        'real_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'total_trades': 0,
        'win_rate': 0,
        'trade_details': [],
        'error': '',
    }

    # 只有存在有效形态时才跑合并策略（用于获取回撤/夏普/真实账户收益）
    # 当只选一个形态时，合并结果应严格等于该独立形态结果，避免两套引擎不一致
    if len(patterns) == 1 and valid:
        single = valid[0]
        combined = {
            'total_return': single.get('total_return', 0),
            'real_return': single.get('total_return', 0),
            'sharpe_ratio': single.get('sharpe_ratio', 0),
            'max_drawdown': single.get('max_drawdown', 0),
            'total_trades': single.get('total_trades', 0),
            'win_rate': single.get('win_rate', 0),
            'trade_details': [dict(t, trigger_patterns=[single.get('pattern_cn', single.get('pattern_name', ''))])
                              for t in single.get('trade_details', [])],
            'error': '',
        }
    elif buy_pattern_names or sell_pattern_names:
        combined_args = (
            code, buy_pattern_names, sell_pattern_names, combined_observe_day,
            start_date, end_date, data_folder_dir, cash, cautious
        )
        try:
            combined = _run_combined_strategy_worker(combined_args)
        except Exception as e:
            combined = {
                'total_return': 0.0,
                'real_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0,
                'trade_details': [],
                'error': f'合并策略回测失败: {e}',
            }

    # ===== 统一口径：组合的交易明细/总收益/胜率/笔数 = 各形态独立 trade_details 的合并 =====
    # 这样组合总收益严格等于各形态独立收益之和，胜率与交易明细完全对得上。
    # 风险指标（max_drawdown/sharpe_ratio）保留合并策略实例的真实账户层面结果。
    merged_trade_details = []
    for p_res in valid:
        for t in p_res.get('trade_details', []):
            # 复制并标注来源形态（便于前端展示）
            t_copy = dict(t)
            if not t_copy.get('trigger_patterns'):
                t_copy['trigger_patterns'] = [p_res.get('pattern_cn', p_res.get('pattern_name', ''))]
            merged_trade_details.append(t_copy)
    # 按买入日期升序排序，方便前端时间线展示
    merged_trade_details.sort(key=lambda t: t.get('buy_date', '') or t.get('open_date', ''))

    combined_total_return = round(sum(t.get('pnl_pct', 0) for t in merged_trade_details), 2)
    combined_total_trades = len(merged_trade_details)
    if combined_total_trades > 0:
        # 胜率按 pnl_pct > 0 统计（与交易明细的口径一致，不含手续费干扰）
        combined_won = sum(1 for t in merged_trade_details if t.get('pnl_pct', 0) > 0)
        combined_win_rate = round(combined_won / combined_total_trades * 100, 0)
    else:
        combined_won = 0
        combined_win_rate = 0

    # 覆盖合并策略实例的口径不一致字段（保留 max_drawdown / sharpe_ratio / real_return）
    combined['total_return'] = combined_total_return
    combined['total_trades'] = combined_total_trades
    combined['win_rate'] = int(combined_win_rate)
    combined['trade_details'] = merged_trade_details

    # 计算买入持有收益率 + 买入持有年化 + 策略年化（含资金利用率调整）
    bh = _calc_buy_hold_and_annualized(
        code, start_date, end_date, data_folder_dir,
        combined.get('total_return', 0.0),
        trade_details=combined.get('trade_details', [])
    )
    combined['buy_hold_return'] = bh['buy_hold_return']
    combined['buy_hold_annualized'] = bh['buy_hold_annualized']
    combined['adj_buy_hold_return'] = bh['adj_buy_hold_return']
    combined['adj_buy_hold_annualized'] = bh['adj_buy_hold_annualized']
    combined['strategy_annualized'] = bh['strategy_annualized']
    combined['years'] = bh['years']
    combined['start_close'] = bh['start_close']
    combined['end_close'] = bh['end_close']
    combined['total_trading_days'] = bh['total_trading_days']
    combined['total_hold_days'] = bh['total_hold_days']
    combined['hold_ratio'] = bh['hold_ratio']

    return {
        'code': code,
        'patterns': pattern_results,
        'combined': combined,
        'pattern_count': len(patterns),
        'valid_count': len(valid),
    }


def identify_market_cycles(code, data_folder_dir=None, threshold=0.08,
                            min_days=22, classify_threshold=0.05,
                            start_date=None, end_date=None):
    """识别标的的上涨/下跌/震荡周期（基于 ZigZag 峰谷算法）。

    算法步骤：
    1. 加载全部日K数据，按日期升序排序，并按 start_date/end_date 过滤
    2. 用 ZigZag 算法找出所有显著转折点（峰谷）：
       - 从起点出发，跟踪当前极值
       - 价格从当前极值反向变动超过 threshold（默认8%）时，确认该极值为转折点
       - 重置极值跟踪，从新转折点继续
    3. 相邻转折点之间归为一段
    4. 对每个主段用次级阈值（classify_threshold*0.6，默认3%）做次级 ZigZag 切分，
       得到更细的子段；这样子段涨跌幅可能 < classify_threshold（5%），可归入震荡段
    5. 按子段涨跌幅分类：
       - 涨幅 ≥ +classify_threshold（5%）→ 上涨段
       - 跌幅 ≤ -classify_threshold（-5%）→ 下跌段
       - 涨跌幅在 ±classify_threshold 之间 → 震荡段
    6. 过滤：持续交易日数 ≥ min_days（默认22，约1个月）才返回

    Args:
        code: 标的代码
        data_folder_dir: 数据目录（可选）
        threshold: ZigZag 主转折点阈值（默认0.08=8%）
        min_days: 最小持续交易日数（默认22）
        classify_threshold: 上涨/下跌段分类阈值（默认0.05=5%）
        start_date: 起始日期 'YYYY-MM-DD'（可选，默认不过滤）
        end_date: 结束日期 'YYYY-MM-DD'（可选，默认不过滤）

    Returns:
        {
            'code': str,
            'data_range': {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD', 'days': int},
            'up_cycles': [{start, end, days, change_pct, low_price, high_price}, ...],
            'down_cycles': [...],
            'flat_cycles': [...],
        }
    """
    import pandas as pd

    df = _load_merged_df(code, data_folder_dir)
    if df is None or df.empty or 'close' not in df.columns:
        return {
            'code': code,
            'data_range': None,
            'up_cycles': [], 'down_cycles': [], 'flat_cycles': [],
            'error': '无可用数据',
        }

    # 按日期范围过滤（支持 start_date/end_date 参数）
    if start_date or end_date:
        try:
            mask = pd.Series(True, index=df.index)
            if start_date:
                sd = pd.to_datetime(start_date)
                mask &= df.index >= sd
            if end_date:
                ed = pd.to_datetime(end_date)
                mask &= df.index <= ed
            df = df[mask]
        except Exception:
            pass
        if df.empty:
            return {
                'code': code,
                'data_range': None,
                'up_cycles': [], 'down_cycles': [], 'flat_cycles': [],
                'error': f'指定时间范围 {start_date or "?"}~{end_date or "?"} 内无数据',
            }

    closes = df['close'].values
    dates = df.index

    # ===== ZigZag 转折点识别 =====
    if len(closes) < 5:
        return {
            'code': code,
            'data_range': {
                'start': dates[0].strftime('%Y-%m-%d'),
                'end': dates[-1].strftime('%Y-%m-%d'),
                'days': len(closes),
            },
            'up_cycles': [], 'down_cycles': [], 'flat_cycles': [],
        }

    # 转折点列表：[(date_idx, price, 'H'|'L'), ...]
    pivots = []

    # 初始化：找第一个极值方向
    last_price = closes[0]
    last_idx = 0
    # 当前跟踪极值
    cur_extreme = closes[0]
    cur_extreme_idx = 0
    cur_direction = None  # None / 'up' / 'down'，表示当前正在跟踪的方向

    for i in range(1, len(closes)):
        price = closes[i]
        if cur_direction is None or cur_direction == 'up':
            # 正在跟踪上涨（寻找高点）
            if price > cur_extreme:
                cur_extreme = price
                cur_extreme_idx = i
                cur_direction = 'up'
            # 反向变动超过阈值，确认高点
            if price < cur_extreme * (1 - threshold):
                pivots.append((cur_extreme_idx, cur_extreme, 'H'))
                cur_direction = 'down'
                cur_extreme = price
                cur_extreme_idx = i
        if cur_direction == 'down':
            # 正在跟踪下跌（寻找低点）
            if price < cur_extreme:
                cur_extreme = price
                cur_extreme_idx = i
                cur_direction = 'down'
            # 反向变动超过阈值，确认低点
            if price > cur_extreme * (1 + threshold):
                pivots.append((cur_extreme_idx, cur_extreme, 'L'))
                cur_direction = 'up'
                cur_extreme = price
                cur_extreme_idx = i

    # 处理最后一段：把最后的极值也加入
    if cur_extreme_idx != last_idx and (not pivots or pivots[-1][0] != cur_extreme_idx):
        last_type = 'L' if cur_direction == 'down' else 'H'
        pivots.append((cur_extreme_idx, cur_extreme, last_type))

    # 起点也作为第一个转折点
    if not pivots or pivots[0][0] != 0:
        pivots.insert(0, (0, closes[0], 'L' if pivots and pivots[0][2] == 'H' else 'H'))

    # ===== 相邻转折点之间归为一段 =====
    segments = []
    for j in range(len(pivots) - 1):
        idx1, price1, type1 = pivots[j]
        idx2, price2, type2 = pivots[j + 1]
        seg_start = dates[idx1].strftime('%Y-%m-%d')
        seg_end = dates[idx2].strftime('%Y-%m-%d')
        seg_days = idx2 - idx1
        if price1 > 0:
            change_pct = round((price2 / price1 - 1) * 100, 2)
        else:
            change_pct = 0.0
        segments.append({
            'start': seg_start,
            'end': seg_end,
            'days': int(seg_days),
            'change_pct': change_pct,
            'start_price': round(float(price1), 2),
            'end_price': round(float(price2), 2),
            'pivot_type': f'{type1}->{type2}',
        })

    # ===== 按涨跌幅分类 + 过滤最短持续天数 =====
    # 注意：ZigZag 阈值（8%）大于分类阈值（5%），所以被识别出的段涨跌幅必然 ≥8%，
    # 没有震荡段。为支持用户"震荡也是一种形态"的需求，对每个 ZigZag 转折段再细分：
    # 在该段内部用更小阈值（5%）找次级转折点，被次级转折点切出来的小段如果涨跌幅 < 5% 即为震荡段。
    up_cycles, down_cycles, flat_cycles = [], [], []

    def _classify_and_add(seg):
        """根据涨跌幅分类并加入对应列表（含最短天数过滤）"""
        if seg['days'] < min_days:
            return
        chg = seg['change_pct']
        if chg >= classify_threshold * 100:
            seg['type'] = 'up'
            up_cycles.append(seg)
        elif chg <= -classify_threshold * 100:
            seg['type'] = 'down'
            down_cycles.append(seg)
        else:
            seg['type'] = 'flat'
            flat_cycles.append(seg)

    # 对每个 ZigZag 段，如果段内涨跌幅较大（≥ 2 * classify_threshold * 100），
    # 尝试用次级阈值（classify_threshold * 0.6，默认3%）找内部转折点，切出更小的子段。
    # 次级阈值 < 分类阈值，所以切出的子段涨跌幅可能 < classify_threshold（5%），
    # 从而可识别出震荡段。
    sub_threshold = classify_threshold * 0.6  # 次级转折点阈值（默认3%）
    for seg in segments:
        seg_chg_abs = abs(seg['change_pct'])
        # 段内涨跌幅 < 2倍次级阈值，无法再切分，直接分类
        if seg_chg_abs < 2 * sub_threshold * 100 or seg['days'] < 2 * min_days:
            _classify_and_add(seg)
            continue

        # 在该段范围内用次级阈值找次级转折点
        idx_start = None
        for k, (idx, _, _) in enumerate(pivots):
            if seg['start'] == dates[idx].strftime('%Y-%m-%d'):
                idx_start = k
                break
        if idx_start is None:
            _classify_and_add(seg)
            continue
        idx_end = idx_start + 1

        # 段内 close 序列
        seg_pivot_idx1, seg_price1, _ = pivots[idx_start]
        seg_pivot_idx2, seg_price2, _ = pivots[idx_end]
        seg_closes = closes[seg_pivot_idx1:seg_pivot_idx2 + 1]
        seg_dates = dates[seg_pivot_idx1:seg_pivot_idx2 + 1]

        # 次级 ZigZag
        sub_pivots = [(0, seg_closes[0])]
        cur_extreme = seg_closes[0]
        cur_extreme_idx = 0
        cur_direction = None
        for i in range(1, len(seg_closes)):
            price = seg_closes[i]
            if cur_direction is None or cur_direction == 'up':
                if price > cur_extreme:
                    cur_extreme = price
                    cur_extreme_idx = i
                    cur_direction = 'up'
                if price < cur_extreme * (1 - sub_threshold):
                    sub_pivots.append((cur_extreme_idx, cur_extreme))
                    cur_direction = 'down'
                    cur_extreme = price
                    cur_extreme_idx = i
            if cur_direction == 'down':
                if price < cur_extreme:
                    cur_extreme = price
                    cur_extreme_idx = i
                    cur_direction = 'down'
                if price > cur_extreme * (1 + sub_threshold):
                    sub_pivots.append((cur_extreme_idx, cur_extreme))
                    cur_direction = 'up'
                    cur_extreme = price
                    cur_extreme_idx = i
        if sub_pivots[-1][0] != len(seg_closes) - 1:
            sub_pivots.append((len(seg_closes) - 1, seg_closes[-1]))

        # 切出子段
        for j in range(len(sub_pivots) - 1):
            i1, p1 = sub_pivots[j]
            i2, p2 = sub_pivots[j + 1]
            if p1 > 0:
                chg = round((p2 / p1 - 1) * 100, 2)
            else:
                chg = 0.0
            sub_seg = {
                'start': seg_dates[i1].strftime('%Y-%m-%d'),
                'end': seg_dates[i2].strftime('%Y-%m-%d'),
                'days': int(i2 - i1),
                'change_pct': chg,
                'start_price': round(float(p1), 2),
                'end_price': round(float(p2), 2),
                'pivot_type': seg.get('pivot_type', ''),
            }
            _classify_and_add(sub_seg)

    # 按起始日期升序排序（合并三类后整体排序，便于前端时间线展示）
    all_segments = up_cycles + down_cycles + flat_cycles
    all_segments.sort(key=lambda x: x['start'])

    # 重新分类填充（排序不影响分类）
    up_cycles = [s for s in all_segments if s.get('type') == 'up']
    down_cycles = [s for s in all_segments if s.get('type') == 'down']
    flat_cycles = [s for s in all_segments if s.get('type') == 'flat']

    return {
        'code': code,
        'data_range': {
            'start': dates[0].strftime('%Y-%m-%d'),
            'end': dates[-1].strftime('%Y-%m-%d'),
            'days': int(len(closes)),
        },
        'up_cycles': up_cycles,
        'down_cycles': down_cycles,
        'flat_cycles': flat_cycles,
    }