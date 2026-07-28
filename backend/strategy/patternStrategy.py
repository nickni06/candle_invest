import numpy as np
import backtrader as bt
import datetime
import indicator as ind
import baseStrategy as bs
import pandas as pd

# 延续patternUp的策略，并增加买入条件：前几日上涨幅度限制
class patternUp_combine_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.buy_signal_CDLSEPARATINGLINES = getattr(bt.talib, 'CDLSEPARATINGLINES')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTASUKIGAP = getattr(bt.talib, 'CDLTASUKIGAP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLINVERTEDHAMMER = getattr(bt.talib, 'CDLINVERTEDHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDRAGONFLYDOJI = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTAKURI = getattr(bt.talib, 'CDLTAKURI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMARUBOZU = getattr(bt.talib, 'CDLMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)

        self.sell_signal_CDLADVANCEBLOCK = getattr(bt.talib, 'CDLADVANCEBLOCK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDARKCLOUDCOVER = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLEVENINGDOJISTAR = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLEVENINGSTAR = getattr(bt.talib, 'CDLEVENINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLINNECK = getattr(bt.talib, 'CDLINNECK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLONNECK = getattr(bt.talib, 'CDLONNECK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSTALLEDPATTERN = getattr(bt.talib, 'CDLSTALLEDPATTERN')(self.data.open, self.data.high, self.data.low, self.data.close)

        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

        self.stock_data_df = pd.read_csv('策略表现/策略字典.csv')


    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)
        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if buy_size > 0 and not self.have_position:
            self.buy_signal = None
            if (self.buy_signal_CDLSEPARATINGLINES[0] > 0 and self.data.close[-2] > self.data.close[-3]
                    and self.data.close[0] > self.data.open[0]):
                self.buy_signal = 'buy_signal_CDLSEPARATINGLINES'

            if (self.buy_signal_CDLTASUKIGAP[0] > 0 and self.data.open[-1] > self.data.high[-2]
                    and self.data.low[-1] > self.data.high[-2]):
                self.buy_signal = 'buy_signal_CDLTASUKIGAP'

            if (self.buy_signal_CDLINVERTEDHAMMER[0] > 0 and ((self.data.close[0] > self.data.open[0]
                                and self.data.open[0] / self.data.low[0] < 1.003)
                                or (self.data.close[0] < self.data.open[0] and
                                self.data.close[0] / self.data.low[0] < 1.003))):
                self.buy_signal = 'buy_signal_CDLINVERTEDHAMMER'

            if (self.buy_signal_CDLDRAGONFLYDOJI[0] > 0 and self.data.close[-1] < self.data.open[-1]  # 处于下降趋势
                    and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                    and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                    and self.data.close[0] < self.data.close[-1]):
                self.buy_signal = 'buy_signal_CDLDRAGONFLYDOJI'

            if (self.buy_signal_CDLTAKURI[0] > 0 and self.data.close[-1] < self.data.open[-1]  # 处于下降趋势
                    and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                    and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                    and self.data.close[0] < self.data.close[-1]):  # 处于下降趋势
                self.buy_signal = 'buy_signal_CDLTAKURI'

            if (self.buy_signal_CDLMARUBOZU[0] > 0 and self.data.close[-1] / self.data.open[-1] < 1.015
                    and self.data.close[-2] / self.data.open[-2] < 1.015):
                self.buy_signal = 'buy_signal_CDLTAKURI'

            if self.buy_signal != None:
                self.log('买入：' + self.buy_signal)
                #self.log(self.stock_data_df[self.stock_data_df['策略代码'] == self.buy_signal]['策略名称'].values[0])
                self.buy(size=buy_size)  # 满仓买入
                self.have_position = True
                self.buyday = 0
                self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损


        # 检查是否达到卖出天数，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
            self.log('卖出：达到卖出天数')

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        elif self.have_position:
            self.sell_signal = None

            if self.sell_signal_CDLADVANCEBLOCK[0] < 0:
                self.sell_signal = 'sell_signal_CDLADVANCEBLOCK'

            if self.sell_signal_CDLDARKCLOUDCOVER[0] < 0:
                self.sell_signal = 'sell_signal_CDLDARKCLOUDCOVER'

            if self.sell_signal_CDLEVENINGDOJISTAR[0] < 0:
                self.sell_signal = 'sell_signal_CDLEVENINGDOJISTAR'

            if self.sell_signal_CDLEVENINGSTAR[0] < 0:
                self.sell_signal = 'sell_signal_CDLEVENINGSTAR'

            if self.sell_signal_CDLINNECK[0] < 0:
                self.sell_signal = 'sell_signal_CDLINNECK'

            if self.sell_signal_CDLONNECK[0] < 0:
                self.sell_signal = 'sell_signal_CDLONNECK'

            if self.sell_signal_CDLEVENINGDOJISTAR[0] < 0:
                self.sell_signal = 'sell_signal_CDLEVENINGDOJISTAR'

            if self.sell_signal_CDLDARKCLOUDCOVER[0] < 0:
                self.sell_signal = 'sell_signal_CDLDARKCLOUDCOVER'

            if self.sell_signal_CDLSTALLEDPATTERN[0] < 0:
                self.sell_signal = 'sell_signal_CDLSTALLEDPATTERN'

            if self.sell_signal:
                self.sell(size=self.position.size)  # 卖出全部持仓
                self.buyday = 0  # 重新计算买入天数
                self.have_position = False
                self.log('卖出：' + self.sell_signal)

        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1


# 延续patternUp的策略，并增加买入条件：前几日上涨幅度限制
class patternUp_CDLMARUBOZU_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)
        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if (self.signal[0] > 0
                and self.data.close[-1] / self.data.open[-1] < 1.015
                and self.data.close[-2] / self.data.open[-2] < 1.015
                and buy_size > 0 and not self.have_position):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1


# 延续patternUp的策略，并增加买入条件：处于下降趋势
class patternUp_CDLTAKURI_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)
        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if (self.signal[0] > 0
                and self.data.close[-1] < self.data.open[-1]  # 处于下降趋势
                and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                and self.data.close[0] < self.data.close[-1]  # 处于下降趋势
                and buy_size > 0 and not self.have_position):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1




# 延续patternUp的策略，并增加买入条件：不做更改
class patternUp_CDLDRAGONFLYDOJI_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）


    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)
        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if (self.signal[0] > 0
                and self.data.close[-1] < self.data.open[-1] # 处于下降趋势
                and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                and self.data.close[0] < self.data.close[-1]  # 处于下降趋势
                and buy_size > 0 and not self.have_position):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓（次日开盘价）
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1



# 延续patternUp的策略，并增加买入条件：
class patternUp_CDLINVERTEDHAMMER_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)

        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if (self.signal[0] > 0
                and ((self.data.close[0] > self.data.open[0] and self.data.open[0] / self.data.low[0] < 1.003)
                         or (self.data.close[0] < self.data.open[0] and self.data.close[0] / self.data.low[0] < 1.003))
                and buy_size > 0 and not self.have_position):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1


# 延续patternUp的策略，并增加买入条件：股票pattern跳涨-开盘价高于昨日最高价
class patternUp_CDLTASUKIGAP_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)

        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if (self.signal[0] > 0 and self.data.open[-1] > self.data.high[-2] and self.data.low[-1] > self.data.high[-2]
                #and self.data.open[0] <= self.data.high[-1]
                #and self.data.close[0] >= self.data.low[-1]
                and buy_size > 0 and not self.have_position):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1


# 延续patternUp的策略，并增加买入条件：股票pattern前处于上涨趋势中
class patternUp_CDLSEPARATINGLINES_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
    )
    def __init__(self):
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)

        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if (self.signal[0] > 0 and self.data.close[-2] > self.data.close[-3] and self.data.close[0] > self.data.open[0]
                and buy_size > 0 and not self.have_position):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.close[0]  # 记录买入价，用于3%固定止损

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1

