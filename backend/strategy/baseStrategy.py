import backtrader as bt
from cautious_mode import meets_extra_condition

# 买入信号：测试买入信号的准确性，即是否在买入信号5日内有上涨趋势
class BaseUpCandleStrategy(bt.Strategy):
    params = (
        ('log', True), # 是否打印日志
        ('observe_day', 2), # 买入信号2日内有上涨趋势
        ('cautious', False), # 谨慎模式：开启后6个特定形态需满足额外条件才买入
    )
    def __init__(self):
        # 获取每日十字星/锤子形状的判断结果
        self.signal = bt.talib.CDLDOJISTAR(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

    def log(self, txt, dt=None):
        if self.p.log:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)

        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        # 谨慎模式开启时，6个特定形态需满足额外条件（meets_extra_condition返回True）
        if (self.signal[0] > 0 and buy_size > 0 and not self.have_position
                and (not self.p.cautious or meets_extra_condition(self.p.name, self.data))):
            self.buy(size=buy_size)  # 满仓买入
            self.have_position = True
            self.buyday = 0
            self.buy_price = self.data.open[0]  # 次日开盘价成交（与 pattern_scan.py 一致）

        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            self.buy_price = 0.0  # 重置买入价

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False

        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受
            return

        if order.status in [order.Completed]:
            if order.isbuy():
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



# 逆买入信号：测试卖出信号的准确性，即是否在卖出信号5日内有下跌趋势
class BaseDownCandleStrategy(bt.Strategy):
    params = (
        ('log', True), # 是否打印日志
        ('observe_day', 2), # 卖出信号2日内有下跌趋势
        ('cautious', False), # 谨慎模式：开启后6个特定形态需满足额外条件才买入
    )

    def __init__(self):
        # 获取每日十字星/锤子形状的判断结果
        self.signal = bt.talib.CDLDOJISTAR(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buyday = 0  # 已买入天数
        self.have_position = False

    def log(self, txt, dt=None):
        if self.p.log:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0]  # 用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)

        # 卖出信号出现时做空（与 pattern_scan.py 的 SellPatternStrategy 逻辑一致）
        # 谨慎模式开启时，6个特定形态需满足额外条件
        if (self.signal[0] < 0 and buy_size > 0 and not self.have_position
                and (not self.p.cautious or meets_extra_condition(self.p.name, self.data))):
            self.sell(size=buy_size)  # 做空
            self.have_position = True
            self.buyday = 0
        # 持有 observe_day 天后买回平仓
        elif self.buyday == self.p.observe_day and self.have_position and self.position.size < 0:
            self.buy(size=abs(self.position.size))  # 买回平仓
            self.buyday = 0
            self.have_position = False

        # 跟踪持有天数
        if self.have_position:
            self.buyday += 1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受
            return

        if order.status in [order.Completed]:
            if order.isbuy():
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
