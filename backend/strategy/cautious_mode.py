"""谨慎模式：6个特定K线形态的额外买入过滤条件

谨慎模式开启时，以下6个形态在原始TA-Lib信号触发后，还需满足额外条件才会实际买入。
其他形态不受谨慎模式影响，仅这6个形态受影响。

受影响的形态及额外条件：
    CDLSEPARATINGLINES 分手线：
        前2日处于上涨趋势（close[-2] > close[-3]）+ 当日收阳（close[0] > open[0]）
        原因：分手线本身方向不明，需在上涨趋势中且当日收阳才确认多方力量
    CDLTASUKIGAP 跳空缺口：
        昨日跳空高开（open[-1] > high[-2]）+ 昨日最低价高于前日最高价（low[-1] > high[-2]）
        原因：跳空缺口需是真跳空（不回补），否则信号无效
    CDLINVERTEDHAMMER 倒锤子：
        实体极小，接近最低价（阳线时 open/low<1.003 或阴线时 close/low<1.003）
        原因：倒锤子实体应极小，否则不是标准形态
    CDLDRAGONFLYDOJI 蜻蜓十字：
        前3日连续下跌（close[-1]<open[-1], close[-1]<close[-2], close[-2]<close[-3], close[0]<close[-1]）
        原因：蜻蜓十字需在下跌趋势中出现才有反转意义
    CDLTAKURI 探水竿：
        同蜻蜓十字，前3日连续下跌
        原因：探水竿是蜻蜓十字的变种，需在下跌趋势中
    CDLMARUBOZU 光头光脚：
        前2日实体幅度均小于1.5%（close[-1]/open[-1]<1.015, close[-2]/open[-2]<1.015）
        原因：光头光脚在前期波动小时更可靠，避免在已有大阳线后追高
"""
import backtrader as bt


# 受谨慎模式影响的6个形态
CAUTIOUS_PATTERNS = {
    'CDLSEPARATINGLINES', 'CDLTASUKIGAP', 'CDLINVERTEDHAMMER',
    'CDLDRAGONFLYDOJI', 'CDLTAKURI', 'CDLMARUBOZU',
}


def meets_extra_condition(pattern_name, data):
    """检查当前K线是否满足该形态的额外买入条件

    Args:
        pattern_name: 形态名（如 'CDLDRAGONFLYDOJI'）
        data: backtrader数据源，需能访问 data.open[0], data.close[-1] 等

    Returns:
        True = 满足额外条件（或形态不在6个受影响列表中，无额外条件）
        False = 不满足额外条件，应放弃买入
    """
    if pattern_name not in CAUTIOUS_PATTERNS:
        return True  # 不受谨慎模式影响的形态，直接通过

    try:
        if pattern_name == 'CDLSEPARATINGLINES':
            # 前2日上涨 + 当日收阳
            return (data.close[-2] > data.close[-3]
                    and data.close[0] > data.open[0])

        if pattern_name == 'CDLTASUKIGAP':
            # 昨日跳空高开 + 昨日最低价高于前日最高价
            return (data.open[-1] > data.high[-2]
                    and data.low[-1] > data.high[-2])

        if pattern_name == 'CDLINVERTEDHAMMER':
            # 实体极小，接近最低价
            if data.close[0] > data.open[0]:  # 阳线
                return data.open[0] / data.low[0] < 1.003
            else:  # 阴线
                return data.close[0] / data.low[0] < 1.003

        if pattern_name in ('CDLDRAGONFLYDOJI', 'CDLTAKURI'):
            # 前3日连续下跌
            return (data.close[-1] < data.open[-1]
                    and data.close[-1] < data.close[-2]
                    and data.close[-2] < data.close[-3]
                    and data.close[0] < data.close[-1])

        if pattern_name == 'CDLMARUBOZU':
            # 前2日实体幅度均小于1.5%
            return (data.close[-1] / data.open[-1] < 1.015
                    and data.close[-2] / data.open[-2] < 1.015)
    except (IndexError, ZeroDivisionError):
        # 数据不足（如历史开头几天），保守起见放弃买入
        return False

    return True


def meets_extra_condition_df(pattern_name, df, idx):
    """检查第 idx 根K线是否满足该形态的额外买入条件（pandas / numpy 版本）

    Args:
        pattern_name: 形态名
        df: 包含 open/high/low/close 列的 DataFrame，已按时间升序排列
        idx: 当前 bar 的整数位置（对应 backtrader 的 data.open[0]）

    Returns:
        True/False，语义与 meets_extra_condition 完全一致
    """
    if pattern_name not in CAUTIOUS_PATTERNS:
        return True

    # 需要至少往前看 3 根 bar
    if idx < 3:
        return False

    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values

    try:
        if pattern_name == 'CDLSEPARATINGLINES':
            return (close_vals[idx - 2] > close_vals[idx - 3]
                    and close_vals[idx] > open_vals[idx])

        if pattern_name == 'CDLTASUKIGAP':
            return (open_vals[idx - 1] > high_vals[idx - 2]
                    and low_vals[idx - 1] > high_vals[idx - 2])

        if pattern_name == 'CDLINVERTEDHAMMER':
            if close_vals[idx] > open_vals[idx]:
                return open_vals[idx] / low_vals[idx] < 1.003
            else:
                return close_vals[idx] / low_vals[idx] < 1.003

        if pattern_name in ('CDLDRAGONFLYDOJI', 'CDLTAKURI'):
            return (close_vals[idx - 1] < open_vals[idx - 1]
                    and close_vals[idx - 1] < close_vals[idx - 2]
                    and close_vals[idx - 2] < close_vals[idx - 3]
                    and close_vals[idx] < close_vals[idx - 1])

        if pattern_name == 'CDLMARUBOZU':
            return (close_vals[idx - 1] / open_vals[idx - 1] < 1.015
                    and close_vals[idx - 2] / open_vals[idx - 2] < 1.015)
    except (IndexError, ZeroDivisionError):
        return False

    return True
