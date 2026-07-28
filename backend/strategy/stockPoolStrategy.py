import csv
import warnings
from logging import exception

import backtrader as bt
import mplfinance as mpf
import numpy as np
import pandas as pd
import tushare as ts
from backtrader.writer import WriterFile

import strategy
import patternStrategy
import tracking
import tools
import main


def settingStrategy(code, start_date, end_date, settings, get_new_data, data_folder_dir, save_data=False):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(settings['cash'])  # 设置初始资金
    cerebro.broker.setcommission(commission=settings['commission'])  # 设置佣金
    if type(code) == list:
        for stock in code:
            data = read_data(stock, start_date, end_date, get_new_data=get_new_data, data_folder_dir=data_folder_dir, save_data=save_data)
            cerebro.adddata(data, name=stock)  # 添加数据源
    else:
        data = read_data(code, start_date, end_date, get_new_data=get_new_data, data_folder_dir=data_folder_dir, save_data=save_data)
        cerebro.adddata(data)  # 添加数据源
    return cerebro


# 从tushare获取股票日线数据
def read_data(code, start_date, end_date, get_new_data, data_folder_dir='', save_data=True):
    import os
    pq_path = os.path.join(data_folder_dir, str(code) + '_daily.parquet')
    csv_path = os.path.join(data_folder_dir, str(code) + '_daily.csv')
    if os.path.exists(pq_path) and os.path.getsize(pq_path) > 0:
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(csv_path)
    df = df.sort_values(by=['trade_date'], ascending=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  # 将日期转换为datetime格式
    df = df.set_index('trade_date', drop=True)  # 将日期设置为索引

    #按照日期范围截取df
    filtered_df = df.loc[start_date:end_date]

    if code in ['AU100g']: # 金价
        data = bt.feeds.PandasData(dataname=filtered_df, datetime=None,
                               open=1, high=2, low=3, close=4, volume=5, openinterest=-1)  # 创建数据源
    elif len(code) < 9: # 指数
        data = bt.feeds.PandasData(dataname=filtered_df, datetime=None,
                               open=1, high=3, low=4, close=2, volume=8, openinterest=-1)  # 创建数据源
    else: # a股
        data = bt.feeds.PandasData(dataname=filtered_df, datetime=None,
                               open=1, high=2, low=3, close=4, volume=8, openinterest=-1)  # 创建数据源
    # 如果df为空，警告
    if filtered_df.empty:
        raise ValueError("df为空")
    return data


def run_stock_pool_Strategy( code,
                             start_date,
                             end_date,
                             pattern_name='',
                             pattern_type='buy',
                             plot=False,
                             log=True, # 输出到控制台
                             get_new_data=False,
                             save_data=False,
                             print_performance=True,
                             data_folder_dir='数据/',
                             observe_day=2,
                             max_stock=5,
                             cash=100000000,
                             commission=0.0001,
                             track_date='2025-01-01',
                             to_log=True,
                             code_name=''): # 输出到log文件
    cerebro = settingStrategy(code, start_date, end_date, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data, data_folder_dir=data_folder_dir, save_data=save_data)


    #根据买入卖出信号的不同，选择不同的策略
    strategy_name = 'stockPoolStrategy'
    cerebro.addstrategy(globals().get(strategy_name),
                        name=pattern_name,
                        code=code,
                        code_name=code_name,
                        log=log,
                        to_log=to_log,
                        observe_day=observe_day,
                        track_date=track_date,
                        max_stock=max_stock,)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')

    # 获取收益率、夏普比率和最大回撤
    result = cerebro.run()
    #returns = result[0].analyzers._Returns.get_analysis()['rtot'] * 100
    #returns = int((np.exp(returns / 100) - 1) * 10000) / 100
    returns = int(cerebro.broker.getvalue() * 10000 / cash - 10000) / 100
    sharpe_ratio = result[0].analyzers.sharperatio.get_analysis()['sharperatio']
    max_drawdown = result[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyze = result[0].analyzers._TradeAnalyzer.get_analysis()
    if sharpe_ratio is not None:
        sharpe_ratio = round(sharpe_ratio, 2)

    # 如果策略有交易行为，就输出结果
    if trade_analyze['total']['total'] != 0:
        if print_performance:
            try:
                print(f'\n交易次数：{trade_analyze['total']['total']:.0f}\n'
                      f'胜率：{trade_analyze['won']['total']/trade_analyze['total']['total']*100:.0f}%')
                print(f'简易收益率: {returns:.2f}%\n'
                      f'夏普比率: {sharpe_ratio}\n'
                      f'最大回撤: {max_drawdown:.2f}%\n')
            except Exception as e:
                print(e)
                print(f'交易次数：0\n'
                      f'胜率：0')
                print(f'简易收益率: 0\n'
                      f'夏普比率: 0\n'
                      f'最大回撤: 0\n')
        # 记录策略表现
        try:
            strategy_performance = {'交易次数': trade_analyze['total']['total'],
                                    '胜率(%)': int(trade_analyze['won']['total']/trade_analyze['total']['total']*100),
                                    '简易收益率(%)': returns,
                                    '夏普比率': sharpe_ratio,
                                    '最大回撤(%)': max_drawdown}
        except Exception:
            strategy_performance = {'交易次数': 0,
                                    '胜率(%)': 0,
                                    '简易收益率(%)': 0,
                                    '夏普比率': 0,
                                    '最大回撤(%)': 0}

    else:
        if print_performance:
            print('该策略无交易行为\n')
        strategy_performance = {'交易次数': 0,
                                '胜率(%)': 0,
                                '简易收益率(%)': 0,
                                '夏普比率': 0,
                                '最大回撤(%)': 0}

    if plot:
        cerebro.plot(
             style='candle',  # 设置主图行情数据的样式为蜡烛图
             #plotdist=0.1,    # 设置图形之间的间距
             barup = '#ff9896', bardown='#98df8a', # 设置蜡烛图上涨和下跌的颜色
             volup='#ff9896', voldown='#98df8a', volume=False) # 设置成交量在行情上涨和下跌情况下的颜色)
    return strategy_performance


class stockPoolStrategy(bt.Strategy):
    params = (
        ('name', ''),
        ('log', True),
        ('to_log', True),
        ('track_date', '2025-01-01'),
        ('code', ''),
        ('code_name', ''),
        ('observe_day', 2),
        ('max_stock', 5),
        ('cautious', False), # 谨慎模式占位（股票池模块自带额外条件，此参数暂不生效）
    )

    def __init__(self):
        self.set_initials() #set indicators的空字典

        self.bought_stock_cnt = 0 #已买入股票数量
        self.buyday_dict = {} # 个股的买入天数统计
        self.have_position = False
        self.buyday_dict = {}

        self.strategy_data_df = pd.read_csv('策略表现/策略字典.csv')
        index_dir = '数据/指数/'
        a_market_dir = '数据/A股/'
        self.buy_strategy_performance_df_dict = {}
        self.sell_strategy_performance_df_dict = {}
        for code in self.p.code:
            if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
                performance_dir = index_dir
            else:
                performance_dir = a_market_dir
            self.buy_strategy_performance_df_dict[code] = pd.read_csv(
                performance_dir + '个股策略表现/' + code + '_buy_strategy_performance_test.csv')
            self.sell_strategy_performance_df_dict[code] = pd.read_csv(
                performance_dir + '个股策略表现/' + code + '_sell_strategy_performance_test.csv')
        self.stock_data_df = pd.read_csv(a_market_dir + 'stock_data.csv')
        self.trade_cnt = self.win_cnt = 0
        self.good_strategy_cnt = 0
        self.good_strategy = False
        self.max_stock = self.p.max_stock
        self.initial_cash = self.broker.getcash()

    def next(self):
        total_value = self.broker.getvalue()
        self.p_value = total_value * 0.99 / self.max_stock #该股的买入总额
        for data in self.datas:
            name_match = self.stock_data_df[self.stock_data_df['ts_code'] == data._name]['name']
            self.p.code_name = name_match.values[0] if len(name_match) > 0 else data._name
            # 获取仓位
            pos = self.getposition(data).size
            if not pos and self.bought_stock_cnt < self.max_stock: # 若当前无仓位
                self.buy_signal = None
                if self.buy_signal_CDL3INSIDE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDL3INSIDE'
                if self.buy_signal_CDL3OUTSIDE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDL3OUTSIDE'
                if self.buy_signal_CDLBELTHOLD[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLBELTHOLD'
                if (self.buy_signal_CDLMARUBOZU[data._name][0] > 0 and data.close[-1] / data.open[-1] < 1.015
                        and data.close[-2] / data.open[-2] < 1.015):
                    self.buy_signal = 'buy_signal_CDLMARUBOZU'
                if self.buy_signal_CDLCOUNTERATTACK[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLCOUNTERATTACK'
                if self.buy_signal_CDLDOJI[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLDOJI'
                if self.buy_signal_CDLDOJISTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLDOJISTAR'
                if (self.buy_signal_CDLDRAGONFLYDOJI[data._name][0] > 0 and data.close[-1] < data.open[-1]  # 处于下降趋势
                        and data.close[-1] < data.close[-2]  # 处于下降趋势
                        and data.close[-2] < data.close[-3]  # 处于下降趋势
                        and data.close[0] < data.close[-1]):
                    self.buy_signal = 'buy_signal_CDLDRAGONFLYDOJI'
                if self.buy_signal_CDLENGULFING[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLENGULFING'
                if self.buy_signal_CDLGAPSIDESIDEWHITE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLGAPSIDESIDEWHITE'
                if self.buy_signal_CDLGRAVESTONEDOJI[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLGRAVESTONEDOJI'
                if self.buy_signal_CDLHAMMER[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHAMMER'
                if self.buy_signal_CDLHARAMI[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHARAMI'
                if self.buy_signal_CDLHARAMICROSS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHARAMICROSS'
                if self.buy_signal_CDLHIGHWAVE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHIGHWAVE'
                if self.buy_signal_CDLHIKKAKE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHIKKAKE'
                if self.buy_signal_CDLHIKKAKEMOD[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHIKKAKEMOD'
                if self.buy_signal_CDLHOMINGPIGEON[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHOMINGPIGEON'
                if (self.buy_signal_CDLINVERTEDHAMMER[data._name][0] > 0 and ((data.close[0] > data.open[0]
                                                                   and data.open[0] / data.low[0] < 1.003)
                                                                  or (data.close[0] < data.open[0] and
                                                                      data.close[0] / data.low[0] < 1.003))):
                    self.buy_signal = 'buy_signal_CDLINVERTEDHAMMER'
                if self.buy_signal_CDLKICKING[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLKICKING'
                if self.buy_signal_CDLKICKINGBYLENGTH[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLKICKINGBYLENGTH'
                if self.buy_signal_CDLMATCHINGLOW[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLMATCHINGLOW'
                if self.buy_signal_CDLMORNINGDOJISTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLMORNINGDOJISTAR'
                if self.buy_signal_CDLMORNINGSTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLMORNINGSTAR'
                if self.buy_signal_CDLPIERCING[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLPIERCING'
                if self.buy_signal_CDLRICKSHAWMAN[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLRICKSHAWMAN'
                if self.buy_signal_CDLRISEFALL3METHODS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLRISEFALL3METHODS'
                if (self.buy_signal_CDLSEPARATINGLINES[data._name][0] > 0 and data.close[-2] > data.close[-3]
                        and data.close[0] > data.open[0]):
                    self.buy_signal = 'buy_signal_CDLSEPARATINGLINES'
                if self.buy_signal_CDLSHORTLINE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLSHORTLINE'
                if self.buy_signal_CDLSPINNINGTOP[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLSPINNINGTOP'
                if self.buy_signal_CDLSTICKSANDWICH[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLSTICKSANDWICH'
                if (self.buy_signal_CDLTAKURI[data._name][0] > 0 and data.close[-1] < data.open[-1]  # 处于下降趋势
                        and data.close[-1] < data.close[-2]  # 处于下降趋势
                        and data.close[-2] < data.close[-3]  # 处于下降趋势
                        and data.close[0] < data.close[-1]):  # 处于下降趋势
                    self.buy_signal = 'buy_signal_CDLTAKURI'
                if (self.buy_signal_CDLTASUKIGAP[data._name][0] > 0 and data.open[-1] > data.high[-2]
                        and data.low[-1] > data.high[-2]):
                    self.buy_signal = 'buy_signal_CDLTASUKIGAP'
                if self.buy_signal_CDLUNIQUE3RIVER[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLUNIQUE3RIVER'
                if self.buy_signal_CDLXSIDEGAP3METHODS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLXSIDEGAP3METHODS'
                # 可能的买入信号
                if self.buy_signal_CDL2CROWS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLADVANCEBLOCK'
                if self.buy_signal_CDL3BLACKCROWS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDL3BLACKCROWS'
                if self.buy_signal_CDL3LINESTRIKE[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDL3LINESTRIKE'
                if self.buy_signal_CDL3STARSINSOUTH[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDL3STARSINSOUTH'
                if self.buy_signal_CDL3WHITESOLDIERS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDL3WHITESOLDIERS'
                if self.buy_signal_CDLABANDONEDBABY[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLABANDONEDBABY'
                if self.buy_signal_CDLADVANCEBLOCK[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLADVANCEBLOCK'
                if self.buy_signal_CDLBREAKAWAY[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLBREAKAWAY'
                if self.buy_signal_CDLCONCEALBABYSWALL[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLCONCEALBABYSWALL'
                if self.buy_signal_CDLDARKCLOUDCOVER[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLDARKCLOUDCOVER'
                if self.buy_signal_CDLEVENINGDOJISTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLEVENINGDOJISTAR'
                if self.buy_signal_CDLEVENINGSTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLEVENINGSTAR'
                if self.buy_signal_CDLHANGINGMAN[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLHANGINGMAN'
                if self.buy_signal_CDLIDENTICAL3CROWS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLIDENTICAL3CROWS'
                if self.buy_signal_CDLINNECK[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLINNECK'
                if self.buy_signal_CDLSHOOTINGSTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLSHOOTINGSTAR'
                if self.buy_signal_CDLSTALLEDPATTERN[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLSTALLEDPATTERN'
                if self.buy_signal_CDLTHRUSTING[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLTHRUSTING'
                if self.buy_signal_CDLTRISTAR[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLTRISTAR'
                if self.buy_signal_CDLUPSIDEGAP2CROWS[data._name][0] > 0:
                    self.buy_signal = 'buy_signal_CDLUPSIDEGAP2CROWS'

                if self.buy_signal != None:
                    performace_list = self.buy_strategy_performance_df_dict[data._name][
                        self.buy_strategy_performance_df_dict[data._name]['策略名称'] == (
                            'buy_' + self.buy_signal[11:])][['交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']].values[0]

                    if performace_list[0] > 10 and performace_list[1] > 50 and performace_list[2] > 0 and performace_list[3] > 0.1:
                        self.good_strategy = True
                        self.good_strategy_cnt += 1
                        self.log(self.p.code_name + ' 符合优质策略，次数：' + str(self.good_strategy_cnt))

                    if self.good_strategy:
                        self.good_strategy = False
                        size = int(self.p_value / 100 / data.close[0]) * 100
                        self.buy(data=data, size=size)
                        self.buyday_dict[data._name] = 0
                        self.have_position = True
                        self.bought_stock_cnt += 1

                        self.log(self.p.code_name + ' 买入原因：' +
                                     self.strategy_data_df[self.strategy_data_df['策略代码'] == self.buy_signal[11:]][
                                         '策略名称'].values[0] + '(' + self.buy_signal + ')')
                        self.log(self.p.code_name + ' 触发形态：' +
                                     self.strategy_data_df[self.strategy_data_df['策略代码'] == self.buy_signal[11:]][
                                         '触发条件'].values[0] + '(' + self.buy_signal + ')')
                        self.log(self.p.code_name + ' 策略在该股的交易次数：' + str(round(performace_list[0], 0))
                                     + ', 胜率(%)：' + str(round(performace_list[1], 2))
                                     + ', 简易收益率(%)：' + str(round(performace_list[2], 2))
                                     + ', 夏普比率：' + str(round(performace_list[3], 3))
                                     + ', 最大回撤(%)：' + str(round(performace_list[4], 2)))
            if pos > 0: # 当前有仓位
                self.buyday_dict[data._name] += 1

                # 检查是否达到卖出天数，并且当前有持仓，卖出全部持仓
                if self.buyday_dict[data._name] == self.p.observe_day:
                    self.close(data=data)  # 卖出全部持仓
                    self.bought_stock_cnt -= 1
                    self.buyday_dict[data._name] = 0  # 重新计算买入天数
                    self.have_position = False
                    if self.p.to_log:
                        self.log(self.p.code_name + ' 卖出提示：次日达到买入天数_' + str(self.p.observe_day) + '天')
                # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
                self.sell_signal = None

                if self.sell_signal_CDL2CROWS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLADVANCEBLOCK'
                if self.sell_signal_CDL3BLACKCROWS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDL3BLACKCROWS'
                if self.sell_signal_CDL3LINESTRIKE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDL3LINESTRIKE'
                if self.sell_signal_CDL3STARSINSOUTH[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDL3STARSINSOUTH'
                if self.sell_signal_CDL3WHITESOLDIERS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDL3WHITESOLDIERS'
                if self.sell_signal_CDLABANDONEDBABY[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLABANDONEDBABY'
                if self.sell_signal_CDLADVANCEBLOCK[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLADVANCEBLOCK'
                if self.sell_signal_CDLBREAKAWAY[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLBREAKAWAY'
                if self.sell_signal_CDLCONCEALBABYSWALL[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLCONCEALBABYSWALL'
                if self.sell_signal_CDLDARKCLOUDCOVER[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLDARKCLOUDCOVER'
                if self.sell_signal_CDLEVENINGDOJISTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLEVENINGDOJISTAR'
                if self.sell_signal_CDLEVENINGSTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLEVENINGSTAR'
                if self.sell_signal_CDLHANGINGMAN[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHANGINGMAN'
                if self.sell_signal_CDLIDENTICAL3CROWS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLIDENTICAL3CROWS'
                if self.sell_signal_CDLINNECK[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLINNECK'
                if self.sell_signal_CDLSHOOTINGSTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLSHOOTINGSTAR'
                if self.sell_signal_CDLSTALLEDPATTERN[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLSTALLEDPATTERN'
                if self.sell_signal_CDLTHRUSTING[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLTHRUSTING'
                if self.sell_signal_CDLTRISTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLTRISTAR'
                if self.sell_signal_CDLUPSIDEGAP2CROWS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLUPSIDEGAP2CROWS'

                # 可能的卖出信号
                if self.sell_signal_CDL3INSIDE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDL3INSIDE'
                if self.sell_signal_CDL3OUTSIDE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDL3OUTSIDE'
                if self.sell_signal_CDLBELTHOLD[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLBELTHOLD'
                if self.sell_signal_CDLMARUBOZU[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLMARUBOZU'
                if self.sell_signal_CDLCOUNTERATTACK[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLCOUNTERATTACK'
                if self.sell_signal_CDLDOJI[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLDOJI'
                if self.sell_signal_CDLDOJISTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLDOJISTAR'
                if self.sell_signal_CDLDRAGONFLYDOJI[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLDRAGONFLYDOJI'
                if self.sell_signal_CDLENGULFING[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLENGULFING'
                if self.sell_signal_CDLGAPSIDESIDEWHITE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLGAPSIDESIDEWHITE'
                if self.sell_signal_CDLGRAVESTONEDOJI[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLGRAVESTONEDOJI'
                if self.sell_signal_CDLHAMMER[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHAMMER'
                if self.sell_signal_CDLHARAMI[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHARAMI'
                if self.sell_signal_CDLHARAMICROSS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHARAMICROSS'
                if self.sell_signal_CDLHIGHWAVE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHIGHWAVE'
                if self.sell_signal_CDLHIKKAKE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHIKKAKE'
                if self.sell_signal_CDLHIKKAKEMOD[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHIKKAKEMOD'
                if self.sell_signal_CDLHOMINGPIGEON[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLHOMINGPIGEON'
                if self.sell_signal_CDLINVERTEDHAMMER[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLINVERTEDHAMMER'
                if self.sell_signal_CDLKICKING[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLKICKING'
                if self.sell_signal_CDLKICKINGBYLENGTH[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLKICKINGBYLENGTH'
                if self.sell_signal_CDLMATCHINGLOW[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLMATCHINGLOW'
                if self.sell_signal_CDLMORNINGDOJISTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLMORNINGDOJISTAR'
                if self.sell_signal_CDLMORNINGSTAR[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLMORNINGSTAR'
                if self.sell_signal_CDLPIERCING[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLPIERCING'
                if self.sell_signal_CDLRICKSHAWMAN[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLRICKSHAWMAN'
                if self.sell_signal_CDLRISEFALL3METHODS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLRISEFALL3METHODS'
                if self.sell_signal_CDLSEPARATINGLINES[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLSEPARATINGLINES'
                if self.sell_signal_CDLSHORTLINE[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLSHORTLINE'
                if self.sell_signal_CDLSPINNINGTOP[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLSPINNINGTOP'
                if self.sell_signal_CDLSTICKSANDWICH[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLSTICKSANDWICH'
                if self.sell_signal_CDLTAKURI[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLTAKURI'
                if self.sell_signal_CDLTASUKIGAP[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLTASUKIGAP'
                if self.sell_signal_CDLUNIQUE3RIVER[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLUNIQUE3RIVER'
                if self.sell_signal_CDLXSIDEGAP3METHODS[data._name][0] < 0:
                    self.sell_signal = 'sell_signal_CDLXSIDEGAP3METHODS'

                if self.sell_signal != None and self.buy_signal is None and self.have_position:
                    if self.p.to_log:
                        performace_list = self.sell_strategy_performance_df_dict[data._name][
                            self.sell_strategy_performance_df_dict[data._name]['策略名称'] == 'sell_' + self.sell_signal[12:]][
                            ['交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']].values[0]
                        # 如果胜率小于55%，并且收益率小于0%，则输出策略名称和触发条件
                        #print(self.sell_signal[12:])
                        self.log(self.p.code_name + ' 卖出原因：' +
                                     self.strategy_data_df[self.strategy_data_df['策略代码'] == self.sell_signal[12:]][
                                         '策略名称'].values[0] + '(' + self.sell_signal + ')')
                        self.log(self.p.code_name + ' 触发形态：' +
                                     self.strategy_data_df[self.strategy_data_df['策略代码'] == self.sell_signal[12:]][
                                         '触发条件'].values[0] + '(' + self.sell_signal + ')')

                        self.log(self.p.code_name + ' 策略在该股的交易次数：' + str(round(performace_list[0], 0))
                                         + ', 胜率(%)：' + str(round(performace_list[1], 2))
                                         + ', 简易收益率(%)：' + str(round(performace_list[2], 2))
                                         + ', 夏普比率：' + str(round(performace_list[3], 3))
                                         + ', 最大回撤(%)：' + str(round(performace_list[4], 2)) + '\n')
                    else:
                        self.log('卖出提示：' + self.sell_signal)
                    self.buyday_dict[data._name] = 0
                    self.have_position = False
                    self.close(data=data)
                    self.bought_stock_cnt -= 1

    def log(self, txt, dt=None):
        if self.p.log:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('买入价格： %.2f' % (order.executed.price))
                self.log(f'当前持股数：{self.bought_stock_cnt:.0f}\n')
            elif order.issell():
                self.log('卖出价格： %.2f' % order.executed.price)


        if order.status in [order.Canceled, order.Margin, order.Rejected]:
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
        self.log(f'当前总收益率：{(self.broker.getvalue() / self.initial_cash - 1) * 100:.2f}%')
        self.log(f'当前持股数：{self.bought_stock_cnt:.0f}\n')

    def set_initials(self):
        self.buy_signal_CDL3INSIDE = {}
        self.buy_signal_CDL3OUTSIDE = {}
        self.buy_signal_CDLBELTHOLD = {}
        self.buy_signal_CDLCLOSINGMARUBOZU = {}
        self.buy_signal_CDLCOUNTERATTACK = {}
        self.buy_signal_CDLDOJI = {}
        self.buy_signal_CDLDOJISTAR = {}
        self.buy_signal_CDLDRAGONFLYDOJI = {}
        self.buy_signal_CDLENGULFING = {}
        self.buy_signal_CDLGAPSIDESIDEWHITE = {}
        self.buy_signal_CDLGRAVESTONEDOJI = {}
        self.buy_signal_CDLHAMMER = {}
        self.buy_signal_CDLHARAMI = {}
        self.buy_signal_CDLHARAMICROSS = {}
        self.buy_signal_CDLHIGHWAVE = {}
        self.buy_signal_CDLHIKKAKE = {}
        self.buy_signal_CDLHIKKAKEMOD = {}
        self.buy_signal_CDLHOMINGPIGEON = {}
        self.buy_signal_CDLINVERTEDHAMMER = {}
        self.buy_signal_CDLKICKING = {}
        self.buy_signal_CDLKICKINGBYLENGTH = {}
        self.buy_signal_CDLLADDERBOTTOM = {}
        self.buy_signal_CDLLONGLEGGEDDOJI = {}
        self.buy_signal_CDLLONGLINE = {}
        self.buy_signal_CDLMARUBOZU = {}
        self.buy_signal_CDLMATCHINGLOW = {}
        self.buy_signal_CDLMORNINGDOJISTAR = {}
        self.buy_signal_CDLMORNINGSTAR = {}
        self.buy_signal_CDLPIERCING = {}
        self.buy_signal_CDLRICKSHAWMAN = {}
        self.buy_signal_CDLRISEFALL3METHODS = {}
        self.buy_signal_CDLSEPARATINGLINES = {}
        self.buy_signal_CDLSHORTLINE = {}
        self.buy_signal_CDLSPINNINGTOP = {}
        self.buy_signal_CDLSTICKSANDWICH = {}
        self.buy_signal_CDLTAKURI = {}
        self.buy_signal_CDLTASUKIGAP = {}
        self.buy_signal_CDLUNIQUE3RIVER = {}
        self.buy_signal_CDLXSIDEGAP3METHODS = {}
        # 可能的买入信号
        self.buy_signal_CDL2CROWS = {}
        self.buy_signal_CDL3BLACKCROWS = {}
        self.buy_signal_CDL3LINESTRIKE = {}
        self.buy_signal_CDL3STARSINSOUTH = {}
        self.buy_signal_CDL3WHITESOLDIERS = {}
        self.buy_signal_CDLABANDONEDBABY = {}
        self.buy_signal_CDLADVANCEBLOCK = {}
        self.buy_signal_CDLBREAKAWAY = {}
        self.buy_signal_CDLCONCEALBABYSWALL = {}
        self.buy_signal_CDLDARKCLOUDCOVER = {}
        self.buy_signal_CDLEVENINGDOJISTAR = {}
        self.buy_signal_CDLEVENINGSTAR = {}
        self.buy_signal_CDLHANGINGMAN = {}
        self.buy_signal_CDLIDENTICAL3CROWS = {}
        self.buy_signal_CDLINNECK = {}
        self.buy_signal_CDLSHOOTINGSTAR = {}
        self.buy_signal_CDLSTALLEDPATTERN = {}
        self.buy_signal_CDLTHRUSTING = {}
        self.buy_signal_CDLTRISTAR = {}
        self.buy_signal_CDLUPSIDEGAP2CROWS = {}
        # 卖出信号
        self.sell_signal_CDL3INSIDE = {}
        self.sell_signal_CDL3OUTSIDE = {}
        self.sell_signal_CDLBELTHOLD = {}
        self.sell_signal_CDLCLOSINGMARUBOZU = {}
        self.sell_signal_CDLCOUNTERATTACK = {}
        self.sell_signal_CDLDOJI = {}
        self.sell_signal_CDLDOJISTAR = {}
        self.sell_signal_CDLDRAGONFLYDOJI = {}
        self.sell_signal_CDLENGULFING = {}
        self.sell_signal_CDLGAPSIDESIDEWHITE = {}
        self.sell_signal_CDLGRAVESTONEDOJI = {}
        self.sell_signal_CDLHAMMER = {}
        self.sell_signal_CDLHARAMI = {}
        self.sell_signal_CDLHARAMICROSS = {}
        self.sell_signal_CDLHIGHWAVE = {}
        self.sell_signal_CDLHIKKAKE = {}
        self.sell_signal_CDLHIKKAKEMOD = {}
        self.sell_signal_CDLHOMINGPIGEON = {}
        self.sell_signal_CDLINVERTEDHAMMER = {}
        self.sell_signal_CDLKICKING = {}
        self.sell_signal_CDLKICKINGBYLENGTH = {}
        self.sell_signal_CDLLADDERBOTTOM = {}
        self.sell_signal_CDLLONGLEGGEDDOJI = {}
        self.sell_signal_CDLLONGLINE = {}
        self.sell_signal_CDLMARUBOZU = {}
        self.sell_signal_CDLMATCHINGLOW = {}
        self.sell_signal_CDLMORNINGDOJISTAR = {}
        self.sell_signal_CDLMORNINGSTAR = {}
        self.sell_signal_CDLPIERCING = {}
        self.sell_signal_CDLRICKSHAWMAN = {}
        self.sell_signal_CDLRISEFALL3METHODS = {}
        self.sell_signal_CDLSEPARATINGLINES = {}
        self.sell_signal_CDLSHORTLINE = {}
        self.sell_signal_CDLSPINNINGTOP = {}
        self.sell_signal_CDLSTICKSANDWICH = {}
        self.sell_signal_CDLTAKURI = {}
        self.sell_signal_CDLTASUKIGAP = {}
        self.sell_signal_CDLUNIQUE3RIVER = {}
        self.sell_signal_CDLXSIDEGAP3METHODS = {}
        # 可能的卖出信号
        self.sell_signal_CDL2CROWS = {}
        self.sell_signal_CDL3BLACKCROWS = {}
        self.sell_signal_CDL3LINESTRIKE = {}
        self.sell_signal_CDL3STARSINSOUTH = {}
        self.sell_signal_CDL3WHITESOLDIERS = {}
        self.sell_signal_CDLABANDONEDBABY = {}
        self.sell_signal_CDLADVANCEBLOCK = {}
        self.sell_signal_CDLBREAKAWAY = {}
        self.sell_signal_CDLCONCEALBABYSWALL = {}
        self.sell_signal_CDLDARKCLOUDCOVER = {}
        self.sell_signal_CDLEVENINGDOJISTAR = {}
        self.sell_signal_CDLEVENINGSTAR = {}
        self.sell_signal_CDLHANGINGMAN = {}
        self.sell_signal_CDLIDENTICAL3CROWS = {}
        self.sell_signal_CDLINNECK = {}
        self.sell_signal_CDLSHOOTINGSTAR = {}
        self.sell_signal_CDLSTALLEDPATTERN = {}
        self.sell_signal_CDLTHRUSTING = {}
        self.sell_signal_CDLTRISTAR = {}
        self.sell_signal_CDLUPSIDEGAP2CROWS = {}
        # 遍历所有股票,计算20日均线
        for data in self.datas:
            #self.mas[data._name] = bt.ind.SMA(data.close, period=self.p.period)
            # 将buy pattern加入信号
            self.buy_signal_CDL3INSIDE[data._name] = getattr(bt.talib, 'CDL3INSIDE')(data.open, data.high,
                                                                                      data.low,
                                                                                      data.close)
            self.buy_signal_CDL3OUTSIDE[data._name] = getattr(bt.talib, 'CDL3OUTSIDE')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLBELTHOLD[data._name] = getattr(bt.talib, 'CDLBELTHOLD')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLCLOSINGMARUBOZU[data._name] = getattr(bt.talib, 'CDLCLOSINGMARUBOZU')(data.open,
                                                                                                      data.high,
                                                                                                      data.low,
                                                                                                      data.close)
            self.buy_signal_CDLCOUNTERATTACK[data._name] = getattr(bt.talib, 'CDLCOUNTERATTACK')(data.open,
                                                                                                  data.high,
                                                                                                  data.low,
                                                                                                  data.close)
            self.buy_signal_CDLDOJI[data._name] = getattr(bt.talib, 'CDLDOJI')(data.open, data.high,
                                                                                data.low,
                                                                                data.close)
            self.buy_signal_CDLDOJISTAR[data._name] = getattr(bt.talib, 'CDLDOJISTAR')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLDRAGONFLYDOJI[data._name] = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(data.open,
                                                                                                  data.high,
                                                                                                  data.low,
                                                                                                  data.close)
            self.buy_signal_CDLENGULFING[data._name] = getattr(bt.talib, 'CDLENGULFING')(data.open, data.high,
                                                                                          data.low, data.close)
            self.buy_signal_CDLGAPSIDESIDEWHITE[data._name] = getattr(bt.talib, 'CDLGAPSIDESIDEWHITE')(data.open,
                                                                                                        data.high,
                                                                                                        data.low,
                                                                                                        data.close)
            self.buy_signal_CDLGRAVESTONEDOJI[data._name] = getattr(bt.talib, 'CDLGRAVESTONEDOJI')(data.open,
                                                                                                    data.high,
                                                                                                    data.low,
                                                                                                    data.close)
            self.buy_signal_CDLHAMMER[data._name] = getattr(bt.talib, 'CDLHAMMER')(data.open, data.high,
                                                                                    data.low,
                                                                                    data.close)
            self.buy_signal_CDLHARAMI[data._name] = getattr(bt.talib, 'CDLHARAMI')(data.open, data.high,
                                                                                    data.low,
                                                                                    data.close)
            self.buy_signal_CDLHARAMICROSS[data._name] = getattr(bt.talib, 'CDLHARAMICROSS')(data.open,
                                                                                              data.high,
                                                                                              data.low,
                                                                                              data.close)
            self.buy_signal_CDLHIGHWAVE[data._name] = getattr(bt.talib, 'CDLHIGHWAVE')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLHIKKAKE[data._name] = getattr(bt.talib, 'CDLHIKKAKE')(data.open, data.high,
                                                                                      data.low,
                                                                                      data.close)
            self.buy_signal_CDLHIKKAKEMOD[data._name] = getattr(bt.talib, 'CDLHIKKAKEMOD')(data.open, data.high,
                                                                                            data.low, data.close)
            self.buy_signal_CDLHOMINGPIGEON[data._name] = getattr(bt.talib, 'CDLHOMINGPIGEON')(data.open,
                                                                                                data.high,
                                                                                                data.low,
                                                                                                data.close)
            self.buy_signal_CDLINVERTEDHAMMER[data._name] = getattr(bt.talib, 'CDLINVERTEDHAMMER')(data.open,
                                                                                                    data.high,
                                                                                                    data.low,
                                                                                                    data.close)
            self.buy_signal_CDLKICKING[data._name] = getattr(bt.talib, 'CDLKICKING')(data.open, data.high,
                                                                                      data.low,
                                                                                      data.close)
            self.buy_signal_CDLKICKINGBYLENGTH[data._name] = getattr(bt.talib, 'CDLKICKINGBYLENGTH')(data.open,
                                                                                                      data.high,
                                                                                                      data.low,
                                                                                                      data.close)
            self.buy_signal_CDLLADDERBOTTOM[data._name] = getattr(bt.talib, 'CDLLADDERBOTTOM')(data.open,
                                                                                                data.high,
                                                                                                data.low,
                                                                                                data.close)
            self.buy_signal_CDLLONGLEGGEDDOJI[data._name] = getattr(bt.talib, 'CDLLONGLEGGEDDOJI')(data.open,
                                                                                                    data.high,
                                                                                                    data.low,
                                                                                                    data.close)
            self.buy_signal_CDLLONGLINE[data._name] = getattr(bt.talib, 'CDLLONGLINE')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLMARUBOZU[data._name] = getattr(bt.talib, 'CDLMARUBOZU')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLMATCHINGLOW[data._name] = getattr(bt.talib, 'CDLMATCHINGLOW')(data.open,
                                                                                              data.high,
                                                                                              data.low,
                                                                                              data.close)
            self.buy_signal_CDLMORNINGDOJISTAR[data._name] = getattr(bt.talib, 'CDLMORNINGDOJISTAR')(data.open,
                                                                                                      data.high,
                                                                                                      data.low,
                                                                                                      data.close)
            self.buy_signal_CDLMORNINGSTAR[data._name] = getattr(bt.talib, 'CDLMORNINGSTAR')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLPIERCING[data._name] = getattr(bt.talib, 'CDLPIERCING')(data.open, data.high,
                                                                                        data.low, data.close)
            self.buy_signal_CDLRICKSHAWMAN[data._name] = getattr(bt.talib, 'CDLRICKSHAWMAN')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLRISEFALL3METHODS[data._name] = getattr(bt.talib, 'CDLRISEFALL3METHODS')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLSEPARATINGLINES[data._name] = getattr(bt.talib, 'CDLSEPARATINGLINES')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLSHORTLINE[data._name] = getattr(bt.talib, 'CDLSHORTLINE')(data.open, data.high,
                                                                                          data.low, data.close)
            self.buy_signal_CDLSPINNINGTOP[data._name] = getattr(bt.talib, 'CDLSPINNINGTOP')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLSTICKSANDWICH[data._name] = getattr(bt.talib, 'CDLSTICKSANDWICH')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLTAKURI[data._name] = getattr(bt.talib, 'CDLTAKURI')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLTASUKIGAP[data._name] = getattr(bt.talib, 'CDLTASUKIGAP')(data.open, data.high,
                                                                                          data.low, data.close)
            self.buy_signal_CDLUNIQUE3RIVER[data._name] = getattr(bt.talib, 'CDLUNIQUE3RIVER')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLXSIDEGAP3METHODS[data._name] = getattr(bt.talib, 'CDLXSIDEGAP3METHODS')(data.open, data.high,
                                                                                           data.low, data.close)
            # 可能的买入信号
            self.buy_signal_CDL2CROWS[data._name] = getattr(bt.talib, 'CDL2CROWS')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDL3BLACKCROWS[data._name] = getattr(bt.talib, 'CDL3BLACKCROWS')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDL3LINESTRIKE[data._name] = getattr(bt.talib, 'CDL3LINESTRIKE')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDL3STARSINSOUTH[data._name] = getattr(bt.talib, 'CDL3STARSINSOUTH')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDL3WHITESOLDIERS[data._name] = getattr(bt.talib, 'CDL3WHITESOLDIERS')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLABANDONEDBABY[data._name] = getattr(bt.talib, 'CDLABANDONEDBABY')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLADVANCEBLOCK[data._name] = getattr(bt.talib, 'CDLADVANCEBLOCK')(data.open, data.high,
                                                                                           data.low, data.close)
            self.buy_signal_CDLBREAKAWAY[data._name] = getattr(bt.talib, 'CDLBREAKAWAY')(data.open, data.high,
                                                                                          data.low, data.close)
            self.buy_signal_CDLCONCEALBABYSWALL[data._name] = getattr(bt.talib, 'CDLCONCEALBABYSWALL')(data.open,
                                                                                                        data.high,
                                                                                                        data.low,
                                                                                                        data.close)
            self.buy_signal_CDLDARKCLOUDCOVER[data._name] = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(data.open,
                                                                                                    data.high,
                                                                                                    data.low,
                                                                                                    data.close)
            self.buy_signal_CDLEVENINGDOJISTAR[data._name] = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(data.open,
                                                                                                      data.high,
                                                                                                      data.low,
                                                                                                      data.close)
            self.buy_signal_CDLEVENINGSTAR[data._name] = getattr(bt.talib, 'CDLEVENINGSTAR')(data.open,
                                                                                              data.high,
                                                                                              data.low,
                                                                                              data.close)
            self.buy_signal_CDLHANGINGMAN[data._name] = getattr(bt.talib, 'CDLHANGINGMAN')(data.open, data.high,
                                                                                            data.low, data.close)
            self.buy_signal_CDLIDENTICAL3CROWS[data._name] = getattr(bt.talib, 'CDLIDENTICAL3CROWS')(data.open,
                                                                                                      data.high,
                                                                                                      data.low,
                                                                                                      data.close)
            self.buy_signal_CDLINNECK[data._name] = getattr(bt.talib, 'CDLINNECK')(data.open, data.high,
                                                                                    data.low,
                                                                                    data.close)
            self.buy_signal_CDLSHOOTINGSTAR[data._name] = getattr(bt.talib, 'CDLSHOOTINGSTAR')(data.open,
                                                                                                data.high,
                                                                                                data.low,
                                                                                                data.close)
            self.buy_signal_CDLSTALLEDPATTERN[data._name] = getattr(bt.talib, 'CDLSTALLEDPATTERN')(data.open,
                                                                                                    data.high,
                                                                                                    data.low,
                                                                                                    data.close)
            self.buy_signal_CDLTHRUSTING[data._name] = getattr(bt.talib, 'CDLTHRUSTING')(data.open, data.high,
                                                                                          data.low, data.close)
            self.buy_signal_CDLTRISTAR[data._name] = getattr(bt.talib, 'CDLTRISTAR')(data.open, data.high,
                                                                                      data.low,
                                                                                      data.close)
            self.buy_signal_CDLUPSIDEGAP2CROWS[data._name] = getattr(bt.talib, 'CDLUPSIDEGAP2CROWS')(data.open,
                                                                                                      data.high,
                                                                                                      data.low,
                                                                                                      data.close)

            # 将sell pattern加入信号
            self.sell_signal_CDL2CROWS[data._name] = getattr(bt.talib, 'CDL2CROWS')(data.open, data.high,
                                                                                     data.low,
                                                                                     data.close)
            self.sell_signal_CDL3BLACKCROWS[data._name] = getattr(bt.talib, 'CDL3BLACKCROWS')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDL3LINESTRIKE[data._name] = getattr(bt.talib, 'CDL3LINESTRIKE')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDL3STARSINSOUTH[data._name] = getattr(bt.talib, 'CDL3STARSINSOUTH')(data.open,
                                                                                                   data.high,
                                                                                                   data.low,
                                                                                                   data.close)
            self.sell_signal_CDL3WHITESOLDIERS[data._name] = getattr(bt.talib, 'CDL3WHITESOLDIERS')(data.open,
                                                                                                     data.high,
                                                                                                     data.low,
                                                                                                     data.close)
            self.sell_signal_CDLABANDONEDBABY[data._name] = getattr(bt.talib, 'CDLABANDONEDBABY')(data.open,
                                                                                                   data.high,
                                                                                                   data.low,
                                                                                                   data.close)
            self.sell_signal_CDLADVANCEBLOCK[data._name] = getattr(bt.talib, 'CDLADVANCEBLOCK')(data.open,
                                                                                                 data.high,
                                                                                                 data.low,
                                                                                                 data.close)
            self.sell_signal_CDLBREAKAWAY[data._name] = getattr(bt.talib, 'CDLBREAKAWAY')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLCONCEALBABYSWALL[data._name] = getattr(bt.talib, 'CDLCONCEALBABYSWALL')(data.open,
                                                                                                         data.high,
                                                                                                         data.low,
                                                                                                         data.close)
            self.sell_signal_CDLDARKCLOUDCOVER[data._name] = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(data.open,
                                                                                                     data.high,
                                                                                                     data.low,
                                                                                                     data.close)
            self.sell_signal_CDLEVENINGDOJISTAR[data._name] = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(data.open,
                                                                                                       data.high,
                                                                                                       data.low,
                                                                                                       data.close)
            self.sell_signal_CDLEVENINGSTAR[data._name] = getattr(bt.talib, 'CDLEVENINGSTAR')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDLHANGINGMAN[data._name] = getattr(bt.talib, 'CDLHANGINGMAN')(data.open, data.high,
                                                                                             data.low, data.close)
            self.sell_signal_CDLIDENTICAL3CROWS[data._name] = getattr(bt.talib, 'CDLIDENTICAL3CROWS')(data.open,
                                                                                                       data.high,
                                                                                                       data.low,
                                                                                                       data.close)
            self.sell_signal_CDLINNECK[data._name] = getattr(bt.talib, 'CDLINNECK')(data.open, data.high,
                                                                                     data.low,
                                                                                     data.close)
            self.sell_signal_CDLSHOOTINGSTAR[data._name] = getattr(bt.talib, 'CDLSHOOTINGSTAR')(data.open,
                                                                                                 data.high,
                                                                                                 data.low,
                                                                                                 data.close)
            self.sell_signal_CDLSTALLEDPATTERN[data._name] = getattr(bt.talib, 'CDLSTALLEDPATTERN')(data.open,
                                                                                                     data.high,
                                                                                                     data.low,
                                                                                                     data.close)
            self.sell_signal_CDLTHRUSTING[data._name] = getattr(bt.talib, 'CDLTHRUSTING')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLTRISTAR[data._name] = getattr(bt.talib, 'CDLTRISTAR')(data.open, data.high,
                                                                                       data.low, data.close)
            self.sell_signal_CDLUPSIDEGAP2CROWS[data._name] = getattr(bt.talib, 'CDLUPSIDEGAP2CROWS')(data.open,
                                                                                                       data.high,
                                                                                                       data.low,
                                                                                                       data.close)

            # 可能的卖出信号
            self.sell_signal_CDL3INSIDE[data._name] = getattr(bt.talib, 'CDL3INSIDE')(data.open, data.high,
                                                                                       data.low, data.close)
            self.sell_signal_CDL3OUTSIDE[data._name] = getattr(bt.talib, 'CDL3OUTSIDE')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLBELTHOLD[data._name] = getattr(bt.talib, 'CDLBELTHOLD')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLCLOSINGMARUBOZU[data._name] = getattr(bt.talib, 'CDLCLOSINGMARUBOZU')(data.open,
                                                                                                       data.high,
                                                                                                       data.low,
                                                                                                       data.close)
            self.sell_signal_CDLCOUNTERATTACK[data._name] = getattr(bt.talib, 'CDLCOUNTERATTACK')(data.open,
                                                                                                   data.high,
                                                                                                   data.low,
                                                                                                   data.close)
            self.sell_signal_CDLDOJI[data._name] = getattr(bt.talib, 'CDLDOJI')(data.open, data.high,
                                                                                 data.low,
                                                                                 data.close)
            self.sell_signal_CDLDOJISTAR[data._name] = getattr(bt.talib, 'CDLDOJISTAR')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLDRAGONFLYDOJI[data._name] = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(data.open,
                                                                                                   data.high,
                                                                                                   data.low,
                                                                                                   data.close)
            self.sell_signal_CDLENGULFING[data._name] = getattr(bt.talib, 'CDLENGULFING')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLGAPSIDESIDEWHITE[data._name] = getattr(bt.talib, 'CDLGAPSIDESIDEWHITE')(data.open,
                                                                                                         data.high,
                                                                                                         data.low,
                                                                                                         data.close)
            self.sell_signal_CDLGRAVESTONEDOJI[data._name] = getattr(bt.talib, 'CDLGRAVESTONEDOJI')(data.open,
                                                                                                     data.high,
                                                                                                     data.low,
                                                                                                     data.close)
            self.sell_signal_CDLHAMMER[data._name] = getattr(bt.talib, 'CDLHAMMER')(data.open, data.high,
                                                                                     data.low,
                                                                                     data.close)
            self.sell_signal_CDLHARAMI[data._name] = getattr(bt.talib, 'CDLHARAMI')(data.open, data.high,
                                                                                     data.low,
                                                                                     data.close)
            self.sell_signal_CDLHARAMICROSS[data._name] = getattr(bt.talib, 'CDLHARAMICROSS')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDLHIGHWAVE[data._name] = getattr(bt.talib, 'CDLHIGHWAVE')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLHIKKAKE[data._name] = getattr(bt.talib, 'CDLHIKKAKE')(data.open, data.high,
                                                                                       data.low, data.close)
            self.sell_signal_CDLHIKKAKEMOD[data._name] = getattr(bt.talib, 'CDLHIKKAKEMOD')(data.open, data.high,
                                                                                             data.low, data.close)
            self.sell_signal_CDLHOMINGPIGEON[data._name] = getattr(bt.talib, 'CDLHOMINGPIGEON')(data.open,
                                                                                                 data.high,
                                                                                                 data.low,
                                                                                                 data.close)
            self.sell_signal_CDLINVERTEDHAMMER[data._name] = getattr(bt.talib, 'CDLINVERTEDHAMMER')(data.open,
                                                                                                     data.high,
                                                                                                     data.low,
                                                                                                     data.close)
            self.sell_signal_CDLKICKING[data._name] = getattr(bt.talib, 'CDLKICKING')(data.open, data.high,
                                                                                       data.low, data.close)
            self.sell_signal_CDLKICKINGBYLENGTH[data._name] = getattr(bt.talib, 'CDLKICKINGBYLENGTH')(data.open,
                                                                                                       data.high,
                                                                                                       data.low,
                                                                                                       data.close)
            self.sell_signal_CDLLADDERBOTTOM[data._name] = getattr(bt.talib, 'CDLLADDERBOTTOM')(data.open,
                                                                                                 data.high,
                                                                                                 data.low,
                                                                                                 data.close)
            self.sell_signal_CDLLONGLEGGEDDOJI[data._name] = getattr(bt.talib, 'CDLLONGLEGGEDDOJI')(data.open,
                                                                                                     data.high,
                                                                                                     data.low,
                                                                                                     data.close)
            self.sell_signal_CDLLONGLINE[data._name] = getattr(bt.talib, 'CDLLONGLINE')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLMARUBOZU[data._name] = getattr(bt.talib, 'CDLMARUBOZU')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLMATCHINGLOW[data._name] = getattr(bt.talib, 'CDLMATCHINGLOW')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDLMORNINGDOJISTAR[data._name] = getattr(bt.talib, 'CDLMORNINGDOJISTAR')(data.open,
                                                                                                       data.high,
                                                                                                       data.low,
                                                                                                       data.close)
            self.sell_signal_CDLMORNINGSTAR[data._name] = getattr(bt.talib, 'CDLMORNINGSTAR')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDLPIERCING[data._name] = getattr(bt.talib, 'CDLPIERCING')(data.open, data.high,
                                                                                         data.low, data.close)
            self.sell_signal_CDLRICKSHAWMAN[data._name] = getattr(bt.talib, 'CDLRICKSHAWMAN')(data.open,
                                                                                               data.high,
                                                                                               data.low,
                                                                                               data.close)
            self.sell_signal_CDLRISEFALL3METHODS[data._name] = getattr(bt.talib, 'CDLRISEFALL3METHODS')(data.open,
                                                                                                         data.high,
                                                                                                         data.low,
                                                                                                         data.close)
            self.sell_signal_CDLSEPARATINGLINES[data._name] = getattr(bt.talib, 'CDLSEPARATINGLINES')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLSHORTLINE[data._name] = getattr(bt.talib, 'CDLSHORTLINE')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLSPINNINGTOP[data._name] = getattr(bt.talib, 'CDLSPINNINGTOP')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLSTICKSANDWICH[data._name] = getattr(bt.talib, 'CDLSTICKSANDWICH')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLTAKURI[data._name] = getattr(bt.talib, 'CDLTAKURI')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLTASUKIGAP[data._name] = getattr(bt.talib, 'CDLTASUKIGAP')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLUNIQUE3RIVER[data._name] = getattr(bt.talib, 'CDLUNIQUE3RIVER')(data.open, data.high,
                                                                                           data.low, data.close)
            self.sell_signal_CDLXSIDEGAP3METHODS[data._name] = getattr(bt.talib, 'CDLXSIDEGAP3METHODS')(data.open, data.high,
                                                                                           data.low, data.close)


if __name__ == '__main__':
    observe_day = 2
    max_stock = 10

    start_date = '20240101'
    end_date = '20241231'

    get_new_data = False
    save_data = False
    to_log = True

    index_dir = '数据/指数/'
    a_market_dir = '数据/A股/'
    stock_data_df = pd.read_csv(a_market_dir + 'stock_data.csv')

    stock_index = 'index'
    index_pool = ['DJI', 'FCHI', 'SPX', 'N225', 'GDAXI', '000300.SH', '399006.SZ'] #, 'AU100g'
    stock_pool = stock_data_df[(stock_data_df['total_mv'] > 5000000)]['ts_code'].tolist()
    #stock_list = stock_data_df[(stock_data_df['total_mv'] > 10000000) & (stock_data_df['industry'] == '银行')]['ts_code'].tolist()

    if stock_index == 'index':
        pool_list = index_pool
        test_data_dir = index_dir + '训练测试库/测试/'
    else:
        pool_list = stock_pool
        test_data_dir = a_market_dir + '训练测试库/测试/'

    pattern_category = 'combine'
    pattern_type = 'stockPool'

    pattern_performance = run_stock_pool_Strategy( pool_list,
                                                   start_date,
                                                   end_date,
                                                   pattern_name='',
                                                   pattern_type=pattern_type,
                                                   observe_day=observe_day,
                                                   max_stock=max_stock,
                                                   plot=False,
                                                   log=to_log,
                                                   get_new_data=get_new_data,
                                                   data_folder_dir=test_data_dir)