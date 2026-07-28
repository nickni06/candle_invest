import backtrader as bt

# 形态数据字典从 pattern_data 重新导出（向后兼容），避免 web_app 等模块
# import signal_utils 时被迫加载 backtrader/talib（在 Flask 主进程首次加载会段错误）。
# 新代码应直接 from pattern_data import ... 以避免重依赖。
from pattern_data import (
    PATTERN_CN_NAMES,
    PATTERN_DESCRIPTIONS,
    BUY_PATTERNS,
    SELL_PATTERNS,
    ALL_PATTERNS,
    get_pattern_description,
)


def compute_all_signals(data):
    signals = {}
    for name in ALL_PATTERNS:
        func = getattr(bt.talib, name, None)
        if func is not None:
            signals[name] = func(data.open, data.high, data.low, data.close)
    return signals


def check_buy_signal(signals, name, data=None):
    val = signals[name][0]
    if val <= 0:
        return False
    extra = _BUY_EXTRA_CHECKS.get(name)
    if extra and data:
        return extra(data)
    return True


def check_sell_signal(signals, name, data=None):
    val = signals[name][0]
    if val >= 0:
        return False
    extra = _SELL_EXTRA_CHECKS.get(name)
    if extra and data:
        return extra(data)
    return True


def _downtrend(data, n=3):
    for i in range(1, n + 1):
        if data.close[-i] >= data.close[-(i + 1)]:
            return False
    return data.close[0] < data.close[-1]


_BUY_EXTRA_CHECKS = {}

_BUY_EXTRA_CHECKS['CDLDRAGONFLYDOJI'] = lambda d: _downtrend(d)
_BUY_EXTRA_CHECKS['CDLTAKURI'] = lambda d: _downtrend(d)
_BUY_EXTRA_CHECKS['CDLSEPARATINGLINES'] = lambda d: (
    d.close[-2] > d.close[-3] and d.close[0] > d.open[0]
)
_BUY_EXTRA_CHECKS['CDLTASUKIGAP'] = lambda d: (
    d.open[-1] > d.high[-2] and d.low[-1] > d.high[-2]
)
_BUY_EXTRA_CHECKS['CDLINVERTEDHAMMER'] = lambda d: (
    (d.close[0] > d.open[0] and d.open[0] / d.low[0] < 1.003)
    or (d.close[0] < d.open[0] and d.close[0] / d.low[0] < 1.003)
)
_BUY_EXTRA_CHECKS['CDLMARUBOZU'] = lambda d: (
    d.close[-1] / d.open[-1] < 1.015 and d.close[-2] / d.open[-2] < 1.015
)

_SELL_EXTRA_CHECKS = {}


def find_triggered_buy(signals, data=None):
    for name in ALL_PATTERNS:
        if check_buy_signal(signals, name, data):
            yield name


def find_triggered_sell(signals, data=None):
    for name in ALL_PATTERNS:
        if check_sell_signal(signals, name, data):
            yield name


def batch_init_signals(data, prefix=''):
    result = {}
    for name in ALL_PATTERNS:
        func = getattr(bt.talib, name, None)
        if func is not None:
            result[f'{prefix}{name}'] = func(data.open, data.high, data.low, data.close)
    return result
