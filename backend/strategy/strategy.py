import numpy as np
import backtrader as bt

import indicator as ind
import baseStrategy as bs

#根据pattern的卖出策略
class patternUp_Strategy(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
        ('cautious', False), # 谨慎模式：开启后6个特定形态需满足额外条件才买入
    )
    def __init__(self):
        if not self.p.name:
            raise ValueError("pattern_name不能为空，请选择一个K线形态")
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

#根据pattern的卖出策略
class patternDown_Strategy(bs.BaseDownCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
        ('cautious', False), # 谨慎模式：开启后6个特定形态需满足额外条件才买入
    )
    def __init__(self):
        if not self.p.name:
            raise ValueError("pattern_name不能为空，请选择一个K线形态")
        self.signal = getattr(bt.talib, self.p.name)(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False

# breakout + doubleMA
class ComboStrategy(bt.Strategy):
    params = {"window_length": 20,
              "short_period": 10,
              "long_period": 60}

    def __init__(self):
        self.count_1 = 0

        # breakout 策略
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.window_length)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.window_length)
        self.buy_signal_1 = bt.indicators.CrossUp(self.data.close, self.highest(-1)) # 向上突破信号
        self.sell_signal_1 = bt.indicators.CrossDown(self.data.close, self.lowest(-1))  # 向下突破信号

        # doubleMA 策略
        self.short_ma = bt.indicators.SMA(self.datas[0].close, period=self.p.short_period)
        self.long_ma = bt.indicators.SMA(self.datas[0].close, period=self.p.long_period)
        # 短期均线上穿长期均线时买入，反之卖出
        self.buy_signal_2 = bt.indicators.CrossUp(self.short_ma, self.long_ma)
        self.sell_signal_2 = bt.indicators.CrossDown(self.short_ma, self.long_ma)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        # 可购买的股票数量，确保是100的整数倍
        buy_size = int(cash // (stock_price*1.001) //100*100)
        # 若两种策略均同意买入，并且当前没有持仓，进行满仓买入
        if ((self.buy_signal_1 or self.buy_signal_2)
                and buy_size > 0 and self.position.size == 0):
            self.log('买入： {}' .format(self.data.close[0]))
            self.buy(size=buy_size)  # 满仓买入
        # 若两种策略均同意卖出，并且当前有持仓，卖出全部持仓
        elif ((self.sell_signal_1 or self.sell_signal_2)
              and self.position.size > 0):
            self.log('卖出： {}'.format(self.data.close[0]))
            self.sell(size=self.position.size) # 卖出全部持仓


# breakout + doubleMA
class indicatorStrategy(bt.Strategy):
    params = {("breakout_window_length", 20),
              ("doublema_short_period", 10),
              ("doublema_long_period", 60),
              ('turtle_long_period', 20),
              ('turtle_short_period', 10),
              }   #doublema

    def __init__(self):
        # breakout 策略
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.breakout_window_length)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.breakout_window_length)
        self.buy_signal_breakout = bt.indicators.CrossUp(self.data.close, self.highest(-1)) # 向上突破信号
        self.sell_signal_breakout = bt.indicators.CrossDown(self.data.close, self.lowest(-1))  # 向下突破信号

        # doubleMA 策略
        self.short_ma = bt.indicators.SMA(self.datas[0].close, period=self.p.doublema_short_period)
        self.long_ma = bt.indicators.SMA(self.datas[0].close, period=self.p.doublema_long_period)
        # 短期均线上穿长期均线时买入，反之卖出
        self.buy_signal_doubleMA = bt.indicators.CrossUp(self.short_ma, self.long_ma)
        self.sell_signal_doubleMA = bt.indicators.CrossDown(self.short_ma, self.long_ma)

        #turtle
        self.buy_count_turtle = 0
        self.buy_price = 0
        self.H_line = bt.indicators.Highest(self.data.high(-1), period=self.p.turtle_long_period)
        self.L_line = bt.indicators.Lowest(self.data.low(-1), period=self.p.turtle_short_period)
        self.TR = bt.indicators.Max((self.data.high(0) - self.data.low(0)),
                                    abs(self.data.close(-1) - self.data.high(0)),
                                    abs(self.data.close(-1) - self.data.low(0)))
        self.ATR = bt.indicators.SimpleMovingAverage(self.TR, period=14)
        # 价格与上下轨线的交叉
        self.buy_signal_turtle = bt.ind.CrossOver(self.data.close(0), self.H_line)
        self.sell_signal_turtle = bt.ind.CrossOver(self.data.close(0), self.L_line)


    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        # 可购买的股票数量，确保是100的整数倍
        #print(self.datas[0].datetime.date(0), cash, stock_price)
        try: # 可能会出现收盘价为空的情况
            buy_size = int(cash // (stock_price*1.001) //100*100)
        except:
            buy_size = 0
        # 若策略其中之一同意买入，并且当前没有持仓，进行满仓买入
        if buy_size > 0 and self.position.size == 0:
            self.buy_signal = None
            if self.buy_signal_breakout:
                self.buy_signal = 'buy_signal_breakout'
            if self.buy_signal_doubleMA:
                self.buy_signal = 'buy_signal_doubleMA'

            #turtle
            if self.buy_signal_turtle[0]:
                self.buy_signal = 'buy_signal_turtle'
                self.log('turtle即将第' + str(self.buy_count_turtle+1) + '次买入')
                self.buy_size = int(self.broker.getcash() * 0.01 / self.ATR / 100) * 100
                if self.buy_count_turtle == 0:
                    self.sizer.p.stake = self.buy_size
                    self.buy_count_turtle += 1
                    print(self.buy_size, self.broker.getcash(), stock_price)
                    self.buy(size=self.buy_size)

                # 加仓：价格上涨了买入价的0.5的ATR且加仓次数少于3次（含）
                elif self.data.close[0] > self.buy_price + 0.5 * self.ATR[0] and self.buy_count_turtle <= 4:
                    self.buy_size = int(self.broker.getcash() * 0.01 / self.ATR / 100) * 100
                    self.sizer.p.stake = self.buy_size
                    self.buy_count_turtle += 1
                    self.buy(size=self.buy_size)

            if self.buy_signal != None:
                self.log('触发策略： {}'.format(self.buy_signal))
                if self.buy_signal != 'buy_signal_turtle': #turtle在策略触发中买入
                    self.buy(size=buy_size)  # 满仓买入

        # 若策略其中之一同意卖出，并且当前有持仓，卖出全部持仓
        elif self.position.size > 0:
            self.sell_signal = None
            if self.sell_signal_breakout:
                self.sell_signal = 'sell_signal_breakout'
            if self.sell_signal_doubleMA:
                self.sell_signal = 'sell_signal_doubleMA'

            #turtle离场：价格跌破下轨线且持仓时
            if ((self.sell_signal_turtle < 0 or self.data.close[0] < (self.buy_price - 2*self.ATR[0]))
                    and self.buy_count_turtle > 0):
                self.sell_signal = 'sell_signal_turtle'
                self.buy_count_turtle = 0

            if self.sell_signal != None:
                self.log('卖出： {}'.format(self.data.close[0]))
                self.log('触发策略： {}'.format(self.sell_signal))
                self.sell(size=self.position.size)  # 卖出全部持仓


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.log('买入价格： %.2f' % (order.executed.price))
            elif order.issell():
                self.log('卖出价格： %.2f\n' % order.executed.price)


        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Margin:
                self.log('Order Canceled/Margin/Rejected: order.Margin!')
            elif order.status == order.Rejected:
                self.log('Order Canceled/Margin/Rejected: order.Rejected!')
            else:
                self.log('Order Canceled/Margin/Rejected: order.Canceled!')

        self.order = None


# 支撑和阻力线策略
class BreakoutStrategy(bt.Strategy):
    params =(
        ('window_length',200),  # 窗口期长度
    )
    def __init__(self):
        # 初始化高点和低点的滚动窗口
        self.highest = bt.indicators.Highest(self.data.high, period=round(self.p.window_length))
        self.lowest = bt.indicators.Lowest(self.data.low, period=round(self.p.window_length))
        self.buy_signal = bt.indicators.CrossUp(self.data.close, self.highest(-1)) # 向上突破信号
        self.sell_signal = bt.indicators.CrossDown(self.data.close, self.lowest(-1))  # 向下突破信号

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        # 可购买的股票数量，确保是100的整数倍
        buy_size = int(cash // (stock_price*1.001) //100*100)
        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if self.buy_signal and buy_size > 0 and self.position.size == 0:
            self.log('买入： {}' .format(self.data.close[0]))
            self.buy(size=buy_size)  # 满仓买入
        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        elif self.sell_signal and self.position.size >0:
            self.log('卖出： {}'.format(self.position.size))
            self.sell(size=self.position.size) # 卖出全部持仓


# 单均线策略
class MAStrategy(bt.Strategy):
    """
    主策略程序
    """
    params = (
        ("ma", 20),)  # 全局设定交易策略的参数

    def __init__(self):
        """
        初始化函数
        """
        self.data_close = self.datas[0].close  # 指定价格序列
        # 添加移动均线指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=round(self.params.ma)
        )
        self.buy_signal = bt.indicators.CrossUp(self.data.close, self.sma(-1))
        self.sell_signal = bt.indicators.CrossDown(self.data.close, self.sma(-1))  # 向下突破信号

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        buy_size = int(cash // (stock_price*1.001) //100*100)
        if self.buy_signal and buy_size > 0 and self.position.size == 0:
            self.buy(size=buy_size)
        # 执行卖出条件判断：收盘价格跌破20日均线
        elif self.sell_signal and self.position.size > 0:
            self.sell(size=self.position.size)


# 双均线策略
class DoubleMAStrategy(bt.Strategy):
    params = {"short_period": 10, "long_period": 60}

    def __init__(self):
        # 一般用于计算指标或者预先加载数据，定义变量使用
        self.short_ma = bt.indicators.SMA(self.datas[0].close, period=round(self.p.short_period))
        self.long_ma = bt.indicators.SMA(self.datas[0].close, period=round(self.p.long_period))
        # 短期均线上穿长期均线时买入，反之卖出
        self.buy_signal = bt.indicators.CrossUp(self.short_ma, self.long_ma)
        self.sell_signal = bt.indicators.CrossDown(self.short_ma, self.long_ma)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        try:
            buy_size = int(cash // (stock_price*1.005) //100*100)
        except:
            buy_size = 0
        # 若当前无仓位，且短期均线上穿长期均线时买入
        if (buy_size > 0 and self.position.size == 0
                and self.buy_signal):
            self.buy(size=buy_size)
        # 若当前有仓位，且短期均线下穿长期均线时卖出
        elif (self.position.size > 0
                and self.sell_signal):
            self.sell(size=self.position.size)


# 双均线策略
class HighestBreakoutStrategy(bt.Strategy):
    params = {"period": 10}

    def __init__(self):
        # 一般用于计算指标或者预先加载数据，定义变量使用
        self.H_line = bt.indicators.Highest(self.data.high, period=self.p.breakout_window_length)
        self.L_line = bt.indicators.Lowest(self.data.low(-1), period=self.p.turtle_short_period)
        # 短期均线上穿长期均线时买入，反之卖出
        self.buy_signal_turtle = bt.ind.CrossOver(self.data.close(0), self.H_line)
        self.sell_signal_turtle = bt.ind.CrossOver(self.data.close(0), self.L_line)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        try:
            buy_size = int(cash // (stock_price*1.005) //100*100)
        except:
            buy_size = 0
        # 若当前无仓位，且短期均线上穿长期均线时买入
        if (buy_size > 0 and self.position.size == 0
                and self.buy_signal):
            self.buy(size=buy_size)
        # 若当前有仓位，且短期均线下穿长期均线时卖出
        elif (self.position.size > 0
                and self.sell_signal):
            self.sell(size=self.position.size)


# MABuySell 均线附近买入 均线突破策略
class MABuySellStrategy(bt.Strategy):
    params = {("MABuySell_period_1", 10),
              ("MABuySell_period_2", 60),
              ('MABuySell_period_3', 90),
              ('MABuySell_period_4', 60), # 固定的
              ('MABuySell_period_5', 90), # 固定的
              ('MABuySell_diff_max', 0.01)
             }

    def __init__(self):
        # MABuySell均线附近买入，有均线支撑就买入，跌破卖出
        self.MABuySell_ma_1 = bt.indicators.SMA(self.datas[0].close, period=round(self.p.MABuySell_period_1))
        self.MABuySell_ma_2 = bt.indicators.SMA(self.datas[0].close, period=round(self.p.MABuySell_period_2))
        self.MABuySell_ma_3 = bt.indicators.SMA(self.datas[0].close, period=round(self.p.MABuySell_period_3))
        self.MABuySell_ma_4 = bt.indicators.SMA(self.datas[0].close, period=round(self.p.MABuySell_period_4))
        self.MABuySell_ma_5 = bt.indicators.SMA(self.datas[0].close, period=round(self.p.MABuySell_period_5))
        self.diff_MABuySell_1 = self.data.close - self.MABuySell_ma_1 # 向上突破信号
        self.diff_MABuySell_2 = self.data.close - self.MABuySell_ma_2
        self.diff_MABuySell_3 = self.data.close - self.MABuySell_ma_3
        self.diff_MABuySell_4 = self.data.close - self.MABuySell_ma_4
        self.diff_MABuySell_5 = self.data.close - self.MABuySell_ma_5
        self.sell_signal_MABuySell_1 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_1)  # 向下突破信号
        self.sell_signal_MABuySell_2 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_2)
        self.sell_signal_MABuySell_3 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_3)
        self.sell_signal_MABuySell_4 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_4)
        self.sell_signal_MABuySell_5 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_5)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        try: # 可能会出现收盘价为空的情况
            self.buy_size = int(cash // (stock_price*1.001) //100*100)
        except:
            self.buy_size = 0
        diff_max = self.data.close[0] * self.p.MABuySell_diff_max  # 触发MABuySell的最大差值
        if (0 < self.diff_MABuySell_1[0] < diff_max or 0 < self.diff_MABuySell_2[0] < diff_max or
                0 < self.diff_MABuySell_3[0] < diff_max or 0 < self.diff_MABuySell_4[0] < diff_max or
                0 < self.diff_MABuySell_5[0] < diff_max) :
            self.buy(size=self.buy_size)  # 满仓买入
            self.p_value = self.buy_size * stock_price

        # 若当前有仓位，且短期均线下穿长期均线时卖出
        if (self.position.size > 0 and (self.sell_signal_MABuySell_1[0] or self.sell_signal_MABuySell_2[0] or self.sell_signal_MABuySell_3[0] or
                self.sell_signal_MABuySell_4[0] or self.sell_signal_MABuySell_5[0])):
            self.sell(size=self.position.size)


# 临近window期内的highest卖出，临近lowest买入
class WindowBuySellStrategy(bt.Strategy):
    params =(
        ('window',30),  # 窗口期长度
        ('diff_max', 0.01), # 触发临近值的判断
    )
    def __init__(self):
        # 初始化高点和低点的滚动窗口
        self.highest = bt.indicators.Highest(self.data.high, period=round(self.p.window))
        self.lowest = bt.indicators.Lowest(self.data.low, period=round(self.p.window))


    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 当前的收盘价
        try: # 可能会出现收盘价为空的情况
            self.buy_size = int(cash // (stock_price*1.001) //100*100)
        except:
            self.buy_size = 0
        diff_max = self.data.close[0] * self.p.diff_max  # 触发MABuySell的最大差值
        if -diff_max < self.data.close[0] - self.lowest[0] < diff_max:
            self.buy(size=self.buy_size)  # 满仓买入
            self.p_value = self.buy_size * stock_price

        # 若当前有仓位，且收盘价触达近期最高价
        if self.position.size > 0 and -diff_max < self.data.close[0] - self.highest[0] < diff_max:
            self.sell(size=self.position.size)




