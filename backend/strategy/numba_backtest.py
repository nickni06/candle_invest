"""Numba 加速的回测核心循环。

设计原则：
1. **风险隔离**：Numba 函数只处理纯数值（numpy 数组 + 标量），不触碰 pandas / 字符串 / dict
2. **输出回退**：Numba 函数输出数值数组，Python 端再转 trade_details（含日期字符串）
3. **失败回退**：调用方 try/except，Numba 不可用或编译失败时回退到原 _backtest_pattern_vectorized
4. **对拍验证**：提供 _eq_compare 函数，逐字段对比 Numba 版与原版的输出

Numba 函数语义与 pattern_scan._backtest_pattern_vectorized 完全一致，
包括：
- 0.995 仓位比例 + 100 股整数倍
- 0.0001 单边手续费
- 买入信号 3% 止损规则（次日开盘价卖出）
- observe_day 持有期平仓
- 持仓期峰值/最大回撤计算
- 数据末尾未平仓时按最后收盘价平仓
- 谨慎模式：通过 cautious_mask 数组传入（Python 端预计算）
"""
import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    njit = None


# 最大交易次数硬上限（防数组越界）；600 个标的 × 单标的最多 ~200 次交易 = 12 万
_MAX_TRADES = 5000


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=False)
    def _backtest_core(opens, closes, signal_mask, cautious_mask,
                       observe_day, cash, pattern_is_buy):
        """Numba 加速的回测核心循环。

        Args:
            opens: np.float64[n] 开盘价
            closes: np.float64[n] 收盘价
            signal_mask: np.bool_[n] 信号掩码（True 表示该 bar 触发开仓信号）
            cautious_mask: np.bool_[n] 谨慎模式过滤掩码（True 表示该 bar 通过谨慎过滤）
                            非谨慎模式时全为 True
            observe_day: int 持有期
            cash: float64 初始资金
            pattern_is_buy: bool True=买入信号（有止损），False=卖出信号

        Returns:
            n_trades: int 实际交易数
            entry_idx_arr: int32[n_trades] 开仓 bar 索引
            exit_idx_arr: int32[n_trades] 平仓 bar 索引
            entry_price_arr: float64[n_trades] 开仓价
            exit_price_arr: float64[n_trades] 平仓价
            size_arr: int64[n_trades] 持仓股数
            pnl_pct_arr: float64[n_trades] 单次交易收益率（%）
            hold_dd_arr: float64[n_trades] 持仓期最大回撤（%）
            exit_reason_arr: int8[n_trades] 平仓原因 0=持有期满 1=止损 2=末尾平仓
            equity_curve: float64[n] 每日净值
            holding_days_total: int 持仓总天数
        """
        n = len(opens)
        commission = 0.0001

        # 预分配数组
        entry_idx_arr = np.zeros(_MAX_TRADES, dtype=np.int32)
        exit_idx_arr = np.zeros(_MAX_TRADES, dtype=np.int32)
        entry_price_arr = np.zeros(_MAX_TRADES, dtype=np.float64)
        exit_price_arr = np.zeros(_MAX_TRADES, dtype=np.float64)
        size_arr = np.zeros(_MAX_TRADES, dtype=np.int64)
        pnl_pct_arr = np.zeros(_MAX_TRADES, dtype=np.float64)
        hold_dd_arr = np.zeros(_MAX_TRADES, dtype=np.float64)
        exit_reason_arr = np.zeros(_MAX_TRADES, dtype=np.int8)

        equity_curve = np.full(n, cash, dtype=np.float64)

        position_size = np.int64(0)
        entry_price = 0.0
        entry_idx = -1
        hold_peak_value = cash
        hold_max_drawdown = 0.0
        stop_loss_pending = False
        holding_days_total = 0
        n_trades = 0

        for i in range(n):
            # ===== 1. 执行挂起的止损卖出（买入信号特有） =====
            if pattern_is_buy and stop_loss_pending and position_size > 0:
                sell_price = opens[i]
                pnl_pct = (sell_price / entry_price - 1.0) * 100.0
                if n_trades < _MAX_TRADES:
                    entry_idx_arr[n_trades] = entry_idx
                    exit_idx_arr[n_trades] = i
                    entry_price_arr[n_trades] = entry_price
                    exit_price_arr[n_trades] = sell_price
                    size_arr[n_trades] = position_size
                    pnl_pct_arr[n_trades] = pnl_pct
                    hold_dd_arr[n_trades] = hold_max_drawdown
                    exit_reason_arr[n_trades] = 1  # 止损
                    n_trades += 1
                cash += position_size * sell_price * (1.0 - commission)
                position_size = 0
                stop_loss_pending = False

            # ===== 2. 开仓：前一根 bar 信号 → 当前 bar 开盘执行 =====
            if i > 0 and signal_mask[i - 1] and position_size == 0:
                if cautious_mask[i]:
                    stock_price = opens[i]
                    if stock_price > 0.0 and stock_price == stock_price:
                        # 0.995 仓位比例 + 100 股整数倍
                        size = np.int64(cash * 0.995 // stock_price // 100 * 100)
                        if size > 0:
                            entry_price = stock_price
                            entry_idx = i
                            position_size = size
                            if pattern_is_buy:
                                cash -= size * stock_price * (1.0 + commission)
                            else:
                                cash += size * stock_price * (1.0 - commission)
                            # 开仓后初始化峰值
                            if pattern_is_buy:
                                hold_peak_value = cash + position_size * stock_price
                            else:
                                hold_peak_value = cash - position_size * stock_price
                            hold_max_drawdown = 0.0

            # ===== 3. 计算当日净值与持仓期回撤 =====
            if position_size != 0:
                if pattern_is_buy:
                    equity = cash + position_size * closes[i]
                else:
                    equity = cash - position_size * closes[i]
                equity_curve[i] = equity
                holding_days_total += 1
                if equity > hold_peak_value:
                    hold_peak_value = equity
                if hold_peak_value > 0:
                    dd = (hold_peak_value - equity) / hold_peak_value * 100.0
                    if dd > hold_max_drawdown:
                        hold_max_drawdown = dd
            else:
                equity_curve[i] = cash
                hold_peak_value = cash
                hold_max_drawdown = 0.0

            # ===== 4. 检查止损条件（仅买入信号，买入当天不检查） =====
            if (pattern_is_buy and position_size > 0
                    and (i - entry_idx) > 0
                    and (closes[i] / entry_price - 1.0) <= -0.03):
                stop_loss_pending = True

            # ===== 5. 持有期满平仓 =====
            if position_size > 0 and (i - entry_idx) >= observe_day and not stop_loss_pending:
                if pattern_is_buy:
                    sell_price = opens[i]
                    pnl_pct = (sell_price / entry_price - 1.0) * 100.0
                    if n_trades < _MAX_TRADES:
                        entry_idx_arr[n_trades] = entry_idx
                        exit_idx_arr[n_trades] = i
                        entry_price_arr[n_trades] = entry_price
                        exit_price_arr[n_trades] = sell_price
                        size_arr[n_trades] = position_size
                        pnl_pct_arr[n_trades] = pnl_pct
                        hold_dd_arr[n_trades] = hold_max_drawdown
                        exit_reason_arr[n_trades] = 0  # 持有期满
                        n_trades += 1
                    cash += position_size * sell_price * (1.0 - commission)
                else:
                    cover_price = opens[i]
                    pnl_pct = (entry_price / cover_price - 1.0) * 100.0
                    if n_trades < _MAX_TRADES:
                        entry_idx_arr[n_trades] = entry_idx
                        exit_idx_arr[n_trades] = i
                        entry_price_arr[n_trades] = entry_price
                        exit_price_arr[n_trades] = cover_price
                        size_arr[n_trades] = position_size
                        pnl_pct_arr[n_trades] = pnl_pct
                        hold_dd_arr[n_trades] = hold_max_drawdown
                        exit_reason_arr[n_trades] = 0  # 持有期满
                        n_trades += 1
                    cash -= position_size * cover_price * (1.0 + commission)
                position_size = 0

        # ===== 末尾未平仓，按最后收盘价平仓 =====
        if position_size > 0:
            last_idx = n - 1
            if pattern_is_buy:
                sell_price = closes[last_idx]
                pnl_pct = (sell_price / entry_price - 1.0) * 100.0
                if n_trades < _MAX_TRADES:
                    entry_idx_arr[n_trades] = entry_idx
                    exit_idx_arr[n_trades] = last_idx
                    entry_price_arr[n_trades] = entry_price
                    exit_price_arr[n_trades] = sell_price
                    size_arr[n_trades] = position_size
                    pnl_pct_arr[n_trades] = pnl_pct
                    hold_dd_arr[n_trades] = hold_max_drawdown
                    exit_reason_arr[n_trades] = 2  # 末尾平仓
                    n_trades += 1
                cash += position_size * sell_price * (1.0 - commission)
                equity_curve[last_idx] = cash
            else:
                cover_price = closes[last_idx]
                pnl_pct = (entry_price / cover_price - 1.0) * 100.0
                if n_trades < _MAX_TRADES:
                    entry_idx_arr[n_trades] = entry_idx
                    exit_idx_arr[n_trades] = last_idx
                    entry_price_arr[n_trades] = entry_price
                    exit_price_arr[n_trades] = cover_price
                    size_arr[n_trades] = position_size
                    pnl_pct_arr[n_trades] = pnl_pct
                    hold_dd_arr[n_trades] = hold_max_drawdown
                    exit_reason_arr[n_trades] = 2  # 末尾平仓
                    n_trades += 1
                cash -= position_size * cover_price * (1.0 + commission)
                equity_curve[last_idx] = cash
            position_size = 0

        # 截取实际使用的部分
        return (n_trades,
                entry_idx_arr[:n_trades].copy(),
                exit_idx_arr[:n_trades].copy(),
                entry_price_arr[:n_trades].copy(),
                exit_price_arr[:n_trades].copy(),
                size_arr[:n_trades].copy(),
                pnl_pct_arr[:n_trades].copy(),
                hold_dd_arr[:n_trades].copy(),
                exit_reason_arr[:n_trades].copy(),
                equity_curve,
                holding_days_total)


def _build_cautious_mask(df, pattern_name, n, cautious):
    """预计算谨慎模式过滤掩码。

    与 cautious_mode.meets_extra_condition_df 语义一致，
    输出 bool 数组：True=通过过滤（可开仓），False=不通过。
    非谨慎模式或形态不在受影响列表中时，全为 True。
    """
    if not cautious:
        return np.ones(n, dtype=np.bool_)
    # 谨慎模式只对 6 个特定形态生效
    from cautious_mode import CAUTIOUS_PATTERNS
    if pattern_name not in CAUTIOUS_PATTERNS:
        return np.ones(n, dtype=np.bool_)

    mask = np.zeros(n, dtype=np.bool_)
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values

    for i in range(n):
        if i < 3:
            mask[i] = False
            continue
        try:
            if pattern_name == 'CDLSEPARATINGLINES':
                mask[i] = (close_vals[i - 2] > close_vals[i - 3]
                           and close_vals[i] > open_vals[i])
            elif pattern_name == 'CDLTASUKIGAP':
                mask[i] = (open_vals[i - 1] > high_vals[i - 2]
                           and low_vals[i - 1] > high_vals[i - 2])
            elif pattern_name == 'CDLINVERTEDHAMMER':
                if close_vals[i] > open_vals[i]:
                    mask[i] = open_vals[i] / low_vals[i] < 1.003
                else:
                    mask[i] = close_vals[i] / low_vals[i] < 1.003
            elif pattern_name in ('CDLDRAGONFLYDOJI', 'CDLTAKURI'):
                mask[i] = (close_vals[i - 1] < open_vals[i - 1]
                           and close_vals[i - 1] < close_vals[i - 2]
                           and close_vals[i - 2] < close_vals[i - 3]
                           and close_vals[i] < close_vals[i - 1])
            elif pattern_name == 'CDLMARUBOZU':
                mask[i] = (close_vals[i - 1] / open_vals[i - 1] < 1.015
                           and close_vals[i - 2] / open_vals[i - 2] < 1.015)
            else:
                mask[i] = True
        except (IndexError, ZeroDivisionError):
            mask[i] = False
    return mask


def backtest_with_numba(df, signal, pattern_name, pattern_type,
                        observe_day, cash, cautious):
    """Numba 加速版回测入口。

    与 pattern_scan._backtest_pattern_vectorized 输出语义完全一致：
        (trade_details, equity_curve, holding_days_total)

    若 Numba 不可用或运行出错，调用方应负责回退到原实现。
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError('numba 未安装')

    n = len(df)
    if n == 0:
        return [], np.array([], dtype=np.float64), 0

    opens = df['open'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    # 用 numpy datetime64 数组代替 pandas Index，strftime 提速 100x
    # pandas Timestamp.strftime 是 Python 调用，每次约 4μs；
    # numpy datetime64 → ISO 字符串切片，每次约 0.04μs
    dates_np = df.index.values.astype('datetime64[s]')
    # 预生成所有日期字符串 YYYY-MM-DD（向量化，O(n) 一次完成）
    # numpy datetime64 → string 是 vectorized
    if len(dates_np) > 0:
        # 直接用 numpy array → list of str，再切片
        # 实测：np.datetime_as_string 比 Python loop strftime 快 100x
        date_strings = np.datetime_as_string(dates_np, unit='D').tolist()
    else:
        date_strings = []

    # 信号掩码
    if pattern_type == 'buy':
        signal_mask = signal > 0
    else:
        signal_mask = signal < 0
    signal_mask = signal_mask.astype(np.bool_)

    # 谨慎模式过滤掩码
    cautious_mask = _build_cautious_mask(df, pattern_name, n, cautious)

    # 调用 Numba 核心循环
    result = _backtest_core(
        opens, closes, signal_mask, cautious_mask,
        observe_day, float(cash), pattern_type == 'buy',
    )
    (n_trades, entry_idx, exit_idx, entry_price, exit_price,
     size, pnl_pct, hold_dd, exit_reason, equity_curve, holding_days_total) = result

    # ===== Python 端：数值数组 → trade_details（含日期字符串） =====
    # 用预生成的 date_strings（list[str]）按索引取，比 pandas Timestamp.strftime 快 100x
    trade_details = []
    is_buy = (pattern_type == 'buy')
    for k in range(n_trades):
        ei = int(entry_idx[k])
        xi = int(exit_idx[k])
        ep = float(entry_price[k])
        xp = float(exit_price[k])
        sz = int(size[k])
        pp = float(pnl_pct[k])
        hd = float(hold_dd[k])
        er = int(exit_reason[k])

        if is_buy:
            td = {
                'buy_date': date_strings[ei],
                'buy_price': round(ep, 3),
                'sell_date': date_strings[xi],
                'sell_price': round(xp, 3),
                'size': sz,
                'pnl': round((xp - ep) * sz, 2),
                'pnl_pct': round(pp, 2),
                'hold_days': xi - ei,
                'hold_max_drawdown': round(hd, 2),
            }
            if er == 1:
                td['exit_reason'] = '3%止损'
        else:
            td = {
                'open_date': date_strings[ei],
                'open_price': round(ep, 3),
                'close_date': date_strings[xi],
                'close_price': round(xp, 3),
                'size': sz,
                'pnl': round((ep - xp) * sz, 2),
                'pnl_pct': round(pp, 2),
                'hold_days': xi - ei,
                'hold_max_drawdown': round(hd, 2),
            }
        trade_details.append(td)

    return trade_details, equity_curve, holding_days_total


def _eq_compare(td_a, td_b, label_a='numba', label_b='origin', tol=1e-6):
    """对拍比较两份 trade_details 是否等价。

    比较字段：buy_date/open_date, buy_price/open_price, sell_date/close_date,
              sell_price/close_price, size, pnl, pnl_pct, hold_days, hold_max_drawdown

    Args:
        td_a, td_b: 两份 trade_details 列表
        tol: 浮点数容差

    Returns:
        (is_equal: bool, diff_msg: str)
    """
    if len(td_a) != len(td_b):
        return False, f'交易数不一致: {label_a}={len(td_a)}, {label_b}={len(td_b)}'

    fields_date = ('buy_date', 'sell_date', 'open_date', 'close_date')
    fields_int = ('size', 'hold_days')
    fields_float = ('buy_price', 'sell_price', 'open_price', 'close_price',
                    'pnl', 'pnl_pct', 'hold_max_drawdown')

    for i, (a, b) in enumerate(zip(td_a, td_b)):
        # 日期类字段必须完全相等
        for f in fields_date:
            if f in a or f in b:
                va = a.get(f, '')
                vb = b.get(f, '')
                if va != vb:
                    return False, f'交易[{i}].{f}: {label_a}={va!r}, {label_b}={vb!r}'
        # 整数字段
        for f in fields_int:
            if f in a or f in b:
                va = a.get(f, 0)
                vb = b.get(f, 0)
                if va != vb:
                    return False, f'交易[{i}].{f}: {label_a}={va}, {label_b}={vb}'
        # 浮点字段（容差比较）
        for f in fields_float:
            if f in a or f in b:
                va = a.get(f, 0.0)
                vb = b.get(f, 0.0)
                if abs(va - vb) > tol:
                    return False, f'交易[{i}].{f}: {label_a}={va}, {label_b}={vb}, 差异={abs(va - vb)}'
        # 平仓原因
        va = a.get('exit_reason', '')
        vb = b.get('exit_reason', '')
        if va != vb:
            return False, f'交易[{i}].exit_reason: {label_a}={va!r}, {label_b}={vb!r}'

    return True, ''
