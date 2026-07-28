import numpy as np
import backtrader as bt
import main

import indicator as ind
import baseStrategy as bs

# breakout + doubleMA
class indicatorStrategy(bt.Strategy):
    params = {("breakout_window_length", 20),
              ("doublema_short_period", 10),
              ("doublema_long_period", 60),
              ('turtle_long_period', 20),
              ('turtle_short_period', 10),
              ("MABuySell_period_1", 10),
              ("MABuySell_period_2", 20),
              ('MABuySell_period_3', 60),
              ('MABuySell_period_4', 90),
              ('MABuySell_diff_max', 0.01),
             }

    def __init__(self):
        # breakout 策略
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.breakout_window_length)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.breakout_window_length)
        self.buy_signal_breakout = bt.indicators.CrossUp(self.data.close, self.highest(-1)) # 向上突破信号
        self.sell_signal_breakout = bt.indicators.CrossDown(self.data.close, self.lowest(-1))  # 向下突破信号

        # doubleMA 策略
        self.short_ma = bt.indicators.SMA(self.datas[0].close, period=self.p.doublema_short_period)
        self.long_ma = bt.indicators.SMA(self.datas[0].close, period=self.p.doublema_long_period)
        self.buy_signal_doubleMA = bt.indicators.CrossUp(self.short_ma, self.long_ma)
        self.sell_signal_doubleMA = bt.indicators.CrossDown(self.short_ma, self.long_ma)

        # turtle
        self.buy_count_turtle = 0
        self.buy_price = 0
        self.H_line = bt.indicators.Highest(self.data.high(-1), period=self.p.turtle_long_period)
        self.L_line = bt.indicators.Lowest(self.data.low(-1), period=self.p.turtle_short_period)
        self.TR = bt.indicators.Max((self.data.high(0) - self.data.low(0)),
                                    abs(self.data.close(-1) - self.data.high(0)),
                                    abs(self.data.close(-1) - self.data.low(0)))
        self.ATR = bt.indicators.SimpleMovingAverage(self.TR, period=14)
        self.buy_signal_turtle = bt.ind.CrossOver(self.data.close(0), self.H_line)
        self.sell_signal_turtle = bt.ind.CrossOver(self.data.close(0), self.L_line)

        # MABuySell均线附近买入，有均线支撑就买入，跌破卖出
        self.MABuySell_ma_1 = bt.indicators.SMA(self.datas[0].close, period=self.p.MABuySell_period_1)
        self.MABuySell_ma_2 = bt.indicators.SMA(self.datas[0].close, period=self.p.MABuySell_period_2)
        self.MABuySell_ma_3 = bt.indicators.SMA(self.datas[0].close, period=self.p.MABuySell_period_3)
        self.MABuySell_ma_4 = bt.indicators.SMA(self.datas[0].close, period=self.p.MABuySell_period_4)
        self.diff_MABuySell_1 = self.data.close - self.MABuySell_ma_1 # 向上突破信号
        self.diff_MABuySell_2 = self.data.close - self.MABuySell_ma_2
        self.diff_MABuySell_3 = self.data.close - self.MABuySell_ma_3
        self.diff_MABuySell_4 = self.data.close - self.MABuySell_ma_4
        self.sell_signal_MABuySell_1 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_1)  # 向下突破信号
        self.sell_signal_MABuySell_2 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_2)
        self.sell_signal_MABuySell_3 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_3)
        self.sell_signal_MABuySell_4 = bt.indicators.CrossDown(self.data.close, self.MABuySell_ma_4)

        self.initial_cash = self.broker.getcash()


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
            self.buy_size = int(cash // (stock_price*1.001) //100*100)
        except:
            self.buy_size = 0
        # 若策略其中之一同意买入，并且当前没有持仓，进行满仓买入
        if self.buy_size > 0:
            self.buy_signal = None
            if self.buy_signal_breakout[0]:
                self.buy_signal = 'buy_signal_breakout'
            if self.buy_signal_doubleMA[0]:
                self.buy_signal = 'buy_signal_doubleMA'

            diff_max = self.data.close[0] * self.p.MABuySell_diff_max #触发MABuySell的最大差值
            if 0<self.diff_MABuySell_1[0]<diff_max  or 0<self.diff_MABuySell_2[0]<diff_max or 0<self.diff_MABuySell_3[0]<diff_max or 0<self.diff_MABuySell_4[0]<diff_max:
                self.buy_signal = 'buy_signal_MABuySell'

            #turtle
            if self.buy_signal_turtle[0] and self.buy_count_turtle <= 4:
                self.buy_signal = 'buy_signal_turtle'
                self.log('turtle即将第' + str(self.buy_count_turtle+1) + '次买入')
                #self.buy_size = int(self.broker.getcash() * 0.01 / self.ATR[0] / 100) * 100
                # 暂时将每次的加仓仓位设置为cash的1/3
                try:  # 可能会出现收盘价为空的情况
                    self.single_cash = cash / 3
                    self.buy_size = int(self.single_cash // (stock_price * 1.001) // 100 * 100)
                except:
                    self.buy_size = 0
                    self.single_cash = None
                    self.log('Error：收盘价可能为空')
                if self.buy_count_turtle == 0:
                    self.sizer.p.stake = self.buy_size
                    self.buy_count_turtle += 1
                    print(self.buy_size, self.single_cash, stock_price, self.ATR[0])
                    self.buy(size=self.buy_size)
                    self.p_value = self.buy_size * stock_price
                    print('106', self.p_value)


                # 加仓：价格上涨了买入价的0.5的ATR且加仓次数少于3次（含）
                elif self.data.close[0] > self.buy_price + 0.5 * self.ATR[0] and self.buy_count_turtle <= 4:
                    try:  # 可能会出现收盘价为空的情况
                        self.buy_size = int(self.single_cash // (stock_price * 1.001) // 100 * 100)
                    except:
                        self.buy_size = 0
                        self.single_cash = None
                        self.log('Error：收盘价可能为空')
                    #print(self.buy_size, self.single_cash, stock_price, self.ATR[0])
                    if self.buy_size > 0:
                        self.sizer.p.stake = self.buy_size
                        self.buy_count_turtle += 1
                        self.buy(size=self.buy_size)
                        self.p_value = self.buy_size * stock_price

            # 若有策略触发则买入
            if self.buy_signal != None:
                self.log('触发策略： {}'.format(self.buy_signal))
                if self.buy_signal != 'buy_signal_turtle': #turtle在策略触发中买入
                    self.buy(size=self.buy_size)  # 满仓买入
                    self.p_value = self.buy_size * stock_price



        # 若策略其中之一同意卖出，并且当前有持仓，卖出全部持仓
        elif self.position.size > 0:
            self.sell_signal = None
            if self.sell_signal_breakout[0]:
                self.sell_signal = 'sell_signal_breakout'
            if self.sell_signal_doubleMA[0]:
                self.sell_signal = 'sell_signal_doubleMA'
            if self.sell_signal_MABuySell_1[0] or self.sell_signal_MABuySell_2[0] or self.sell_signal_MABuySell_3[0] or self.sell_signal_MABuySell_4[0]:
                self.sell_signal ='sell_signal_MABuySell'

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
            elif order.issell() or order.isclose():
                self.log('卖出价格： %.2f\n' % order.executed.price)


        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Margin:
                self.log('Order Canceled/Margin/Rejected: order.Margin!')
            elif order.status == order.Rejected:
                self.log('Order Canceled/Margin/Rejected: order.Rejected!')
            else:
                self.log('Order Canceled/Margin/Rejected: order.Canceled!')

        self.order = None

    # 记录交易收益情况
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'本次收益率 {trade.pnl/self.p_value * 100:.2f}%')
        self.log(f'当前总收益率：{(self.broker.getvalue() / self.initial_cash - 1) * 100:.2f}%\n')


# indicator联合策略
def run_indicatorStrategy(code, start_date, end_date,
                          window_length, doublema_short_period, doublema_long_period, turtle_long_period, turtle_short_period,
                          MABuySell_period_1, MABuySell_period_2, MABuySell_period_3, MABuySell_period_4, MABuySell_diff_max,
                          data_folder_dir='', plot=False, get_new_data=False, cash=100000000, commission=0.0001):
    cerebro = main.settingStrategy(code=code, start_date=start_date, end_date=end_date, data_folder_dir=data_folder_dir,
                              settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(indicatorStrategy,
                        breakout_window_length=window_length,
                        doublema_short_period=doublema_short_period,
                        doublema_long_period=doublema_long_period,
                        turtle_long_period=turtle_long_period,
                        turtle_short_period=turtle_short_period,
                        MABuySell_period_1=MABuySell_period_1,
                        MABuySell_period_2=MABuySell_period_2,
                        MABuySell_period_3=MABuySell_period_3,
                        MABuySell_period_4=MABuySell_period_4,
                        MABuySell_diff_max=MABuySell_diff_max)


    # 设置初始资金
    cerebro.broker.setcash(cash)
    # 设置手续费
    cerebro.broker.setcommission(commission=commission)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')

    # 获取收益率、夏普比率和最大回撤
    result = cerebro.run()
    returns = result[0].analyzers._Returns.get_analysis()['rtot'] * 100
    avg_return = result[0].analyzers._Returns.get_analysis()['rnorm'] * 100 # 年化收益率
    sharpe_ratio = result[0].analyzers.sharperatio.get_analysis()['sharperatio']
    max_drawdown = result[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyze = result[0].analyzers._TradeAnalyzer.get_analysis()

    if trade_analyze['total']['total'] != 0:
        try:
            print(f'\n交易次数：{trade_analyze['total']['total']:.0f}\n'
                  f'胜率：{trade_analyze['won']['total']/trade_analyze['total']['total']*100:.0f}%')
            print(f'简易收益率: {(np.exp(returns/100) - 1) * 100:.2f}%\n'
                  f'年化收益率: {avg_return:.2f}%\n'
                  f'夏普比率: {sharpe_ratio:.2f}\n'
                  f'最大回撤: {max_drawdown:.2f}%\n')
        except:
            print(f'\n交易次数：0\n'
                  f'胜率：0')
            print(f'简易收益率: 0\n'
                  f'年化收益率: 0\n'
                  f'夏普比率: 0\n'
                  f'最大回撤: 0\n')

    if plot:
        cerebro.plot(
             style='candle',  # 设置主图行情数据的样式为蜡烛图
             plotdist=0.1,    # 设置图形之间的间距
             barup = '#ff9896', bardown='#98df8a', # 设置蜡烛图上涨和下跌的颜色
             volup='#ff9896', voldown='#98df8a') # 设置成交量在行情上涨和下跌情况下的颜色)

    return returns


if __name__ == '__main__':
    start_date = '20100104'
    end_date = '20231229'

    code = 'DJI'  # 'DJI', 'FCHI', 'SPX', 'GDAXI', 'N225'

    breakout_window_length = 20 # 最低最高价突破
    doublema_short_period = 10
    doublema_long_period = 30
    turtle_short_period = 10 # 海龟
    turtle_long_period = 30
    MABuySell_period_1 = 12 # 均线突破
    MABuySell_period_2 = 12
    MABuySell_period_3 = 60
    MABuySell_period_4 = 100
    MABuySell_diff_max = 0.015

    index_dir = '数据/指数/训练测试库/训练/'
    stock_dir = '数据/A股/训练测试库/训练/'

    plot = False
    get_new_data = False

    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        data_folder_dir = index_dir
    else:
        data_folder_dir = stock_dir

    run_indicatorStrategy(code,
                          start_date, end_date,
                          breakout_window_length, doublema_short_period, doublema_long_period, turtle_short_period, turtle_long_period,
                          MABuySell_period_1, MABuySell_period_2, MABuySell_period_3, MABuySell_period_4, MABuySell_diff_max,
                          plot=plot, get_new_data=get_new_data, data_folder_dir=data_folder_dir)