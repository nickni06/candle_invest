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
import stockPoolStrategy
import optunity
import optunity.metrics

# 过滤掉FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
#pro = ts.pro_api('20241217202523-6dc513df-e2f2-4ab8-8dfd-038be46b739c')
#pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'

from config import config
try:
    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()
except Exception as _ts_init_err:
    # 多进程子进程环境下 token 文件可能不可用，或免费 token 权限不足
    # 此时 pro=None，后续代码应通过 data_source（akshare）获取数据
    import logging as _logging
    _logging.getLogger('trader_system').warning(f'[main] tushare 初始化失败（将使用 akshare 兜底）: {_ts_init_err}')
    pro = None


def _resolve_strategy_class(strategy_name):
    """从 'module.ClassName' 字符串解析策略类，替代eval()提升安全性。

    Args:
        strategy_name: 如 'strategy.patternUp_Strategy' / 'patternStrategy.patternUp_CDL3INSIDE_Strategy'
    Returns:
        策略类对象
    """
    import importlib
    module_path, class_name = strategy_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def test_strategy():
    code = 'FCHI' #'000001.SZ'
    #code = '000001.SZ'
    #df = ts.pro_bar(ts_code=code, start_date='20240101', end_date='20241128', ma=[5, 20, 50])
    df = pro.index_global(ts_code=code, start_date='20240101', end_date='20241202')
    print(df)
    df = df.sort_values(by=['trade_date'], ascending=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    #ma5 = df['ma5']
    #print(df.loc[0,'ma5'] > df.loc[0,'ma20'])

    window_length = 20  # 窗口长度设为20
    ratio = 0.005  # 保守画线的浮动比例

    # 计算过去20个交易日最低价的最小值，作为支撑线
    df['支撑线'] = df['low'].rolling(window=window_length).min()
    df['保守支撑线'] = df['支撑线'] * (1+ratio)
    # 计算过去20个交易日最高价的最大值，作为阻力线
    df['阻力线'] = df['high'].rolling(window=window_length).max()
    df['保守阻力线'] = df['阻力线'] * (1-ratio)
    # 识别看涨突破（收盘价高于阻力线）
    df['看涨突破'] = df['close']> df['保守阻力线'].shift(1)
    # 识别看跌突破（收盘价低于支撑线）
    df['看跌突破'] = df['close']< df['保守支撑线'].shift(1)
    # 为看涨和看跌突破点创建新列，并用NaN值填充
    df1=df.loc['20240101':]
    df1['看涨突破点']= np.nan
    df1['看跌突破点']= np.nan
    # 在出现看涨和看跌突破的位置，用收盘价填充新列
    df1.loc[df1['看涨突破'],'看涨突破点']= df1['close']
    df1.loc[df1['看跌突破'],'看跌突破点']= df1['close']
    # 为支撑线、阻力线、看涨突破点和看跌突破点创建附加图
    ap1 = mpf.make_addplot(df1['支撑线'], color='green')
    ap2 = mpf.make_addplot(df1['保守支撑线'], color='green', linestyle='--')
    ap3 = mpf.make_addplot(df1['阻力线'], color='red')
    ap4 = mpf.make_addplot(df1['保守阻力线'], color='red', linestyle='--')
    ap5 = mpf.make_addplot(df1['看涨突破点'], scatter=True, markersize=100, color='blue')
    ap6 = mpf.make_addplot(df1['看跌突破点'], scatter=True, markersize=100, color='orange')
    # 创建带有附加图的K线图
    mpf.plot(df1, type='candle', style='charles',  addplot=[ap1, ap2, ap3, ap4, ap5, ap6], figsize=(10,6))


def settingStrategy(code, start_date, end_date, settings, get_new_data, data_folder_dir, save_data=False):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(settings['cash'])  # 设置初始资金
    cerebro.broker.setcommission(commission=settings['commission'])  # 设置佣金
    if type(code) == list:
        for stock in code:
            data = tools.get_data(stock, start_date, end_date, get_new_data=get_new_data, daily_folder_dir=data_folder_dir, save_data=save_data)
            cerebro.adddata(data, name=stock)  # 添加数据源
    else:
        data = tools.get_data(code, start_date, end_date, get_new_data=get_new_data, daily_folder_dir=data_folder_dir, save_data=save_data)
        cerebro.adddata(data)  # 添加数据源
    return cerebro


def run_BreakoutStrategy(code, settings, window_length, plot=False, get_new_data=False, cash=10000000, commission=0.0001):
    cerebro = settingStrategy(code, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(strategy.BreakoutStrategy, window_length=window_length)  # 添加策略
    cerebro.run()
    profit_rate = int(cerebro.broker.getvalue() * 10000 / settings['cash'] - 10000) / 100
    print('window_length: %.i 策略收益率: %.2f%%' % (window_length, profit_rate))
    if plot:
        cerebro.plot()
    return profit_rate


def run_MAStrategy(ma, plot=False, get_new_data=False, cash=10000000, commission=0.0001):
    cerebro = settingStrategy(code, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(strategy.MAStrategy, ma = ma)  # 添加策略
    cerebro.run()
    profit_rate = int(cerebro.broker.getvalue() * 10000 / cash - 10000) / 100
    print('ma: %.i 策略收益率: %.2f%%' % (ma, profit_rate))
    if plot:
        cerebro.plot()
    return profit_rate


def run_DoubleMAStrategy(short_period, long_period, data_folder_dir='数据/指数/训练测试库/训练/', plot=False, get_new_data=False, cash=10000000, commission=0.0001):
    code = 'DJI'
    cerebro = settingStrategy(code, start_date, end_date, data_folder_dir=data_folder_dir, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(strategy.DoubleMAStrategy, short_period=short_period, long_period=long_period)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 获取收益率、夏普比率和最大回撤
    result = cerebro.run()
    returns = result[0].analyzers._Returns.get_analysis()['rtot'] * 100
    avg_return = result[0].analyzers._Returns.get_analysis()['rnorm'] * 100 # 年化收益率
    sharpe_ratio = result[0].analyzers.sharperatio.get_analysis()['sharperatio']
    max_drawdown = result[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    try:
        sharpe_ratio = round(sharpe_ratio, 2)
    except Exception:
        sharpe_ratio = 0
    print(f'参数: {short_period}, {long_period}\n'
          f'简易收益率: {(np.exp(returns/100) - 1) * 100:.2f}%\n'
          f'年化收益率: {avg_return:.2f}%\n'
          f'夏普比率: {sharpe_ratio} \n'
          f'最大回撤: {max_drawdown:.2f}%')

    if plot:
        cerebro.plot(
             style='candle',  # 设置主图行情数据的样式为蜡烛图
             plotdist=0.1,    # 设置图形之间的间距
             barup = '#ff9896', bardown='#98df8a', # 设置蜡烛图上涨和下跌的颜色
             volup='#ff9896', voldown='#98df8a') # 设置成交量在行情上涨和下跌情况下的颜色)

    return returns

# 双策略投票
def run_ComboStrategy(window_length, short_period, long_period, plot=False, get_new_data=False, cash=10000000, commission=0.0001):
    cerebro = settingStrategy(code, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(strategy.ComboStrategy, window_length=window_length, short_period=short_period, long_period=long_period)
    # 获取收益率、夏普比率和最大回撤
    result = cerebro.run()
    returns = result[0].analyzers._Returns.get_analysis()['rtot'] * 100
    sharpe_ratio = result[0].analyzers.sharperatio.get_analysis()['sharperatio']
    max_drawdown = result[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    print(f'参数: {window_length}, {short_period}, {long_period}\n'
          f'简易收益率: {(np.exp(returns/100) - 1) * 100:.2f}%\n'
          f'对数收益率: {returns:.2f}%\n'
          f'夏普比率: {sharpe_ratio:.2f} \n'
          f'最大回撤: {max_drawdown:.2f}%')
    if plot:
        cerebro.plot(
             style='candle',  # 设置主图行情数据的样式为蜡烛图
             plotdist=0.1,    # 设置图形之间的间距
             barup = '#ff9896', bardown='#98df8a', # 设置蜡烛图上涨和下跌的颜色
             volup='#ff9896', voldown='#98df8a') # 设置成交量在行情上涨和下跌情况下的颜色)

    return returns


def run_MABuySellStrategy(period_1, period_2, period_3, MABuySell_diff_max, data_folder_dir='数据/指数/训练测试库/训练/', plot=False, get_new_data=False, cash=10000000, commission=0.0001):
    code = global_code
    data_folder_dir = global_data_folder_dir
    cerebro = settingStrategy(code, start_date, end_date, data_folder_dir=data_folder_dir, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(strategy.MABuySellStrategy, MABuySell_period_1=period_1, MABuySell_period_2=period_2, MABuySell_period_3=period_3, MABuySell_diff_max=MABuySell_diff_max)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 获取收益率、夏普比率和最大回撤
    result = cerebro.run()[0]
    returns = result.analyzers._Returns.get_analysis()['rtot'] * 100
    avg_return = result.analyzers._Returns.get_analysis()['rnorm'] * 100 # 年化收益率
    sharpe_ratio = result.analyzers.sharperatio.get_analysis()['sharperatio']
    max_drawdown = result.analyzers.drawdown.get_analysis()['max']['drawdown']
    try:
        sharpe_ratio = round(sharpe_ratio, 2)
    except Exception:
        sharpe_ratio = 0
    print(f'参数: {period_1}, {period_2}, {period_3}\n'
          f'简易收益率: {(np.exp(returns/100) - 1) * 100:.2f}%\n'
          f'年化收益率: {avg_return:.2f}%\n'
          f'夏普比率: {sharpe_ratio} \n'
          f'最大回撤: {max_drawdown:.2f}%')

    return returns


def run_WindowBuySellStrategy(window, diff_max, data_folder_dir='数据/指数/训练测试库/训练/', plot=False, get_new_data=False, cash=10000000, commission=0.0001):
    code = global_code
    data_folder_dir = global_data_folder_dir
    cerebro = settingStrategy(code, start_date, end_date, data_folder_dir=data_folder_dir, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data)
    cerebro.addstrategy(strategy.WindowBuySellStrategy, window=window, diff_max=diff_max)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 获取收益率、夏普比率和最大回撤
    result = cerebro.run()[0]
    returns = result.analyzers._Returns.get_analysis()['rtot'] * 100
    avg_return = result.analyzers._Returns.get_analysis()['rnorm'] * 100 # 年化收益率
    sharpe_ratio = result.analyzers.sharperatio.get_analysis()['sharperatio']
    max_drawdown = result.analyzers.drawdown.get_analysis()['max']['drawdown']
    try:
        sharpe_ratio = round(sharpe_ratio, 2)
    except Exception:
        sharpe_ratio = 0
    print(f'参数: {round(window)}, {round(diff_max,3)}\n'
          f'简易收益率: {(np.exp(returns/100) - 1) * 100:.2f}%\n'
          f'年化收益率: {avg_return:.2f}%\n'
          f'夏普比率: {sharpe_ratio} \n'
          f'最大回撤: {max_drawdown:.2f}%')

    return returns


'''
蜡烛线图形指标
'''

# pattern_category: '' / '_sellIfDown'
def run_pattern_recognition_Strategy(code,
                                     start_date,
                                     end_date,
                                     pattern_category='',
                                     pattern_name='',
                                     pattern_type='buy',
                                     plot=False,
                                     log=True, # 输出到控制台
                                     get_new_data=False,
                                     save_data=False,
                                     print_performance=True,
                                     data_folder_dir='数据/',
                                     observe_day=2,
                                     cash=100000000,
                                     commission=0.0001,
                                     track_date='2025-01-01',
                                     to_log=True,
                                     code_name='',
                                     cautious=False): # 谨慎模式：6个特定形态需满足额外条件
    cerebro = settingStrategy(code, start_date, end_date, settings={'cash': cash, 'commission': commission}, get_new_data=get_new_data, data_folder_dir=data_folder_dir, save_data=save_data)


    #根据买入卖出信号的不同，选择不同的策略
    if pattern_type == 'buy':
        if pattern_category == '':
            strategy_name = 'strategy.patternUp_Strategy'
        else:
            strategy_name ='patternStrategy.patternUp_' + pattern_name + '_Strategy'

        cerebro.addstrategy(_resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            log=log,
                            observe_day=observe_day,
                            cautious=cautious)
    elif pattern_type == 'sell':
        if pattern_category == '':
            strategy_name = 'strategy.patternDown_Strategy'
        else:
            strategy_name ='patternStrategy.patternDown_' + pattern_name + '_Strategy'

        cerebro.addstrategy(_resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            log=log,
                            observe_day=observe_day,
                            cautious=cautious)

    elif pattern_type == 'tracking':
        if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
            strategy_name = 'tracking.patternUp_' + pattern_name + '_Strategy_index'
        else:
            strategy_name = 'tracking.patternUp_' + pattern_name + '_Strategy_stock'

        cerebro.addstrategy(_resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            code=code,
                            code_name=code_name,
                            log=log,
                            to_log=to_log,
                            observe_day=observe_day,
                            track_date=track_date,
                            cautious=cautious)

    elif pattern_type == 'stockPool':
        strategy_name = 'stockPoolStrategy.stockPoolStrategy'
        cerebro.addstrategy(_resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            code=code,
                            code_name=code_name,
                            log=log,
                            to_log=to_log,
                            observe_day=observe_day,
                            track_date=track_date,
                            cautious=cautious)

    elif pattern_type == 'else':
        if pattern_category == '':
            strategy_name = 'strategy.patternDown_Strategy'
        else:
            strategy_name ='patternStrategy.patternDown_' + pattern_name + '_Strategy'

        cerebro.addstrategy(_resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            log=log,
                            observe_day=observe_day,
                            cautious=cautious)
    else:
        print('pattern_type 未识别成功，已退出！')
        return

    if print_performance:
        print('策略名：', strategy_name, ':', pattern_name, '\n')

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
                print(f'交易次数：{trade_analyze['total']['total']:.0f}\n'
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
                                    '胜率(%)': round(trade_analyze['won']['total']/trade_analyze['total']['total']*100),
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



def performance_test(code, start_date, end_date, file_name, plot, get_new_data):
    results = []  # 存放策略的性能指标

    # for pattern in pn.patternList()[:5]:
    buy_pattern = ['CDL3INSIDE', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLCOUNTERATTACK', 'CDLDOJI',
                   'CDLDOJISTAR',
                   'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
                   'CDLHARAMI',
                   'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
                   'CDLINVERTEDHAMMER',
                   'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
                   'CDLMARUBOZU',
                   'CDLMATCHINGLOW', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLRICKSHAWMAN',
                   'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTICKSANDWICH',
                   'CDLTAKURI',
                   'CDLTASUKIGAP', 'CDLUNIQUE3RIVER', 'CDLXSIDEGAP3METHODS']
    sell_pattern = ['CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
                    'CDLABANDONEDBABY',
                    'CDLADVANCEBLOCK', 'CDLBREAKAWAY', 'CDLCONCEALBABYSWALL', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR',
                    'CDLEVENINGSTAR', 'CDLHANGINGMAN', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLMATHOLD', 'CDLONNECK',
                    'CDLSHOOTINGSTAR', 'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUPSIDEGAP2CROWS']

    buy_pattern = ['MABuySell', 'WindowBuySell']

    observe_days = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # 观察天数
    observe_days = [3]  # 观察天数

    for observe_day in observe_days:
        # 遍历所有的买入信号
        # buy_pattern/sell_pattern
        for pattern in buy_pattern:
            print(pattern)
            pattern_performance = run_pattern_recognition_Strategy(code,
                                                                   start_date,
                                                                   end_date,
                                                                   pattern_category='', #''/'self'
                                                                   pattern_name=pattern,
                                                                   pattern_type='else', #buy/sell
                                                                   observe_day=observe_day,
                                                                   plot=False,
                                                                   log=False,
                                                                   get_new_data=get_new_data)
            if pattern_performance['交易次数'] > 0:
                # 将策略的性能指标添加到 results 列表中
                results.append({
                    '股票代码': code,
                    '买入天数': observe_day,
                    '策略名称': pattern,
                    '交易次数': pattern_performance['交易次数'],
                    '胜率(%)': pattern_performance['胜率(%)'],
                    '简易收益率(%)': pattern_performance['简易收益率(%)'],
                    '夏普比率': pattern_performance['夏普比率'],
                    '最大回撤(%)': pattern_performance['最大回撤(%)']
                })

    # 将策略表现写入文件，默认继续累加写入上一个文件，若不存在该文件则创建
    try:
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            #existing_data = list(reader)
    except FileNotFoundError:
        #existing_data = []
        # 如果文件不存在，则创建一个新的，并写入标题行
        with open(file_name, 'w', newline='') as csvfile:
            fieldnames = ['股票代码', '买入天数', '策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率',
                          '最大回撤(%)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    with open(file_name, 'a', newline='') as csvfile:
        fieldnames = ['股票代码', '买入天数', '策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率',
                      '最大回撤(%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        for result in results:
            writer.writerow(result)


def codes_test(pattern_name, observe_day, start_date, end_date, file_name, plot, get_new_data):
    print('pattern_name, observe_day: ', pattern_name, observe_day)

    win_pctg_results = []
    revenue_results = []
    sharpe_ratio_results = []
    max_drawdown_results = []

    frequent_codes = []

    for code in ['DJI', 'FCHI', 'SPX', 'GDAXI', 'N225',
                 'IBOVESPA', 'RTS', 'TWII', 'CKLSE', 'SPTSX', 'CSX5P', 'RUT']:
        print(code)
        pattern_performance = run_pattern_recognition_Strategy(code,
                                                               start_date,
                                                               end_date,
                                                               pattern_category='',  # _inUp
                                                               pattern_name=pattern_name,
                                                               pattern_type='buy',
                                                               observe_day=observe_day,
                                                               plot=False,
                                                               log=False,
                                                               get_new_data=get_new_data)
        if pattern_performance['交易次数'] >= 10:
            win_pctg_results.append(round(pattern_performance['胜率(%)'], 2))
            revenue_results.append(round(pattern_performance['简易收益率(%)'], 2))
            sharpe_ratio_results.append(round(pattern_performance['夏普比率'], 2))
            max_drawdown_results.append(round(pattern_performance['最大回撤(%)'], 2))

            frequent_codes.append(code)

    # 运行对比策略
    print("\n优化后的策略表现：")

    win_pctg_results_2 = []
    revenue_results_2 = []
    sharpe_ratio_results_2 = []
    max_drawdown_results_2 = []
    trade_cnt_2 = []

    unfrequent_code = 0 # 记录调整后交易次数为0的策略，计算平均值时剔除

    for code in frequent_codes:
        print(code)
        pattern_performance = run_pattern_recognition_Strategy(code,
                                                               start_date,
                                                               end_date,
                                                               pattern_category='_inUp',  # _sellIfDown
                                                               pattern_name=pattern_name,
                                                               pattern_type='buy',
                                                               observe_day=observe_day,
                                                               plot=False,
                                                               log=False)

        win_pctg_results_2.append(round(pattern_performance['胜率(%)'], 2))
        revenue_results_2.append(round(pattern_performance['简易收益率(%)'], 2))
        sharpe_ratio_results_2.append(round(pattern_performance['夏普比率'], 2))
        max_drawdown_results_2.append(round(pattern_performance['最大回撤(%)'], 2))
        trade_cnt_2.append(pattern_performance['交易次数'])

        if pattern_performance['交易次数'] == 0:
            unfrequent_code += 1

    diff_win_pctg = tools.list_subtraction(win_pctg_results_2, win_pctg_results)
    diff_revenue = tools.list_subtraction(revenue_results_2, revenue_results)
    diff_sharpe_ratio = tools.list_subtraction(sharpe_ratio_results_2, sharpe_ratio_results)
    diff_max_drawdown = tools.list_subtraction(max_drawdown_results_2, max_drawdown_results)

    print('-胜率(%):')
    print(diff_win_pctg)
    print('优化策略值：\n', win_pctg_results_2)
    print('优化策略平均值：\n', round(sum(win_pctg_results_2) / (len(revenue_results_2) - unfrequent_code), 2))
    print('平均差值：\n', round(sum(win_pctg_results_2) / (len(revenue_results_2) - unfrequent_code) - sum(win_pctg_results) / len(revenue_results), 2), '\n')

    print('-简易收益率(%)')
    print(diff_revenue)
    print('优化策略值：\n', revenue_results_2)
    print('优化策略平均值：\n', round(sum(revenue_results_2) / (len(revenue_results_2) - unfrequent_code), 2), '\n')

    print('-夏普比率')
    print(diff_sharpe_ratio)
    print('平均差值：\n', round(sum(diff_sharpe_ratio) / len(diff_sharpe_ratio), 2), '\n')

    print('-最大回撤(%)')
    print(diff_max_drawdown)
    print('平均差值：\n', round(sum(diff_max_drawdown) / len(diff_max_drawdown), 2), '\n')

    print('-调整后策略交易次数')
    print(trade_cnt_2)
    print(frequent_codes)

    return (round(sum(win_pctg_results_2) / (len(revenue_results_2) - unfrequent_code), 2),
            round(sum(win_pctg_results) / len(win_pctg_results), 2))


def opt(opt_func, space, num_evals):

    opt = optunity.maximize(
        f=opt_func,
        num_evals=num_evals,  # 回测x次 获取最优参数
        **space
    )

    optimal_pars, details, _ = opt  # optimal_pars 最优参数组合
    print(optimal_pars)
    return optimal_pars

'''
步骤：
先test看调整方向，
调整后再codes_test看调整效果，
最后find_observe_day看调整后的策略的通用性并选择最佳参数
'''
if __name__ == '__main__':
    start_date = '20240101'
    end_date = '20241231'

    code = 'GDAXI'
    #code = '000001.SZ' # '601111.SH'
    pattern_name = 'combine'

    plot = False
    get_new_data = False

    file_name = 'sell_strategy_performance_combined.csv'

    mode = 'performance_test'  # 'test'/'codes_test'/ 对比策略调整前后表现 codes_test_compare / 最后一步找到最佳买入天数参数 find_observe_day
                                 # indicator_test 指标类策略测试 /
                                # performance_test 对个股遍历策略并记录到csv

    # 寻找最优参数
    # 批量测试策略参数，输出策略表现到csv
    if mode == 'opt':
        #code_list = ['000166.SZ', 'FCHI']

        opt_func = run_WindowBuySellStrategy

        period_file_name = 'WindowBuySellStrategy_window.csv'

        # 寻找最优参数
        start_date = '20100104'
        end_date = '20231229'

        # 参数取值范围
        param_space = {
            #'short_period': [5, 30],
            #'long_period': [15, 80]
            'window': [10, 100],
            'diff_max': [0, 0.03]
        }

        num_evals = 5

        a_market_dir = '数据/A股/'


        try:
            stock_data_df = pd.read_csv(a_market_dir + 'stock_data.csv')
            filtered_stocks = stock_data_df[stock_data_df['total_mv'] > 5000000]['ts_code'].tolist()
        except FileNotFoundError:
            print("股票数据文件未找到")

        index_dir_train = '数据/指数/训练测试库/训练/'
        stock_dir_train = '数据/A股/训练测试库/训练/'

        index_dir_test = '数据/指数/训练测试库/测试/'
        stock_dir_test = '数据/A股/训练测试库/测试/'

        # 全局变量
        global global_code
        global global_data_folder_dir

        results = []
        for code in filtered_stocks[:2]:
            global_code = code # 刷新全局code变量

            if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
                train_data_folder_dir = index_dir_train
            else:
                train_data_folder_dir = stock_dir_train

            global_data_folder_dir = train_data_folder_dir

            result = opt(opt_func, param_space, num_evals)

            results.append({'股票代码': code,
                            'window': round(result['window']),
                            'diff_max': round(result['diff_max'], 3) })

        # 将策略表现写入文件，默认继续累加写入上一个文件，若不存在该文件则创建
        try:
            with open(period_file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
        except FileNotFoundError:
            # 如果文件不存在，则创建一个新的，并写入标题行
            with open(period_file_name, 'w', newline='') as csvfile:
                fieldnames = ['股票代码', 'window', 'diff_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        with open(period_file_name, 'a', newline='') as csvfile:
            fieldnames = ['股票代码', 'window', 'diff_max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for result in results:
                writer.writerow(result)


    # 批量测试策略参数，输出策略表现到csv - 指数版
    elif mode == 'performance_test':
        for code in ['DJI', 'FCHI', 'SPX', 'GDAXI', 'N225']:
            print('\n', code,'\n')
            file_name = code + '_buy_strategy_performance_test.csv'
            performance_test(code, start_date, end_date, file_name, plot, get_new_data)

    # 测试indicator strategy中单个策略表现
    elif mode == 'indicator_test':
        code = 'DJI'

        start_date = '20100104'
        end_date = '20231229'

        #short_period = 25
        #long_period = 38
        period_1 = 12
        period_2 = 12
        period_3 = 60
        period_4 = 100

        plot = False
        get_new_data = False

        index_dir_train = '数据/指数/训练测试库/训练/'
        stock_dir_train = '数据/A股/训练测试库/训练/'

        index_dir_test = '数据/指数/训练测试库/测试/'
        stock_dir_test = '数据/A股/训练测试库/测试/'

        if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
            train_data_folder_dir = index_dir_train
            test_data_folder_dir = index_dir_test
        else:
            train_data_folder_dir = stock_dir_train
            test_data_folder_dir = stock_dir_test
        print('\n训练数据表现\n')
        #run_DoubleMAStrategy(short_period, long_period, data_folder_dir=train_data_folder_dir, plot=plot, get_new_data=get_new_data, cash=100000000, commission=0.0001)
        run_MABuySellStrategy(period_1, period_2, period_3, period_4, data_folder_dir=train_data_folder_dir, plot=plot, get_new_data=get_new_data, cash=100000000, commission=0.0001)

        start_date = '20240102'
        end_date = '20241231'

        print('\n测试数据表现')
        #run_DoubleMAStrategy(short_period, long_period, data_folder_dir=test_data_folder_dir, plot=plot, get_new_data=get_new_data, cash=100000000, commission=0.0001)
        run_MABuySellStrategy(period_1, period_2, period_3, period_4, data_folder_dir=test_data_folder_dir, plot=plot, get_new_data=get_new_data, cash=100000000, commission=0.0001)

    # 测试调整后的策略
    elif mode == 'codes_test_compare':
        observe_day = 2
        codes_test(pattern_name, observe_day, start_date, end_date, file_name, plot, get_new_data)

    elif mode == 'codes_test':
        observe_day = 4
        for code in ['DJI', 'FCHI', 'SPX', 'GDAXI', 'N225',
                     'IBOVESPA', 'RTS', 'TWII', 'CKLSE', 'SPTSX', 'CSX5P', 'RUT']:
            print('\n', code)
            pattern_performance = run_pattern_recognition_Strategy(code,
                                                                   start_date,
                                                                   end_date,
                                                                   pattern_category='united',  # _inUp
                                                                   pattern_name=pattern_name,
                                                                   pattern_type='buy',
                                                                   observe_day=observe_day,
                                                                   plot=False,
                                                                   log=False,
                                                                   get_new_data=False)

    elif mode == 'pattern_test':
        observe_day = 2
        code = "DJI" #'DJI', 'FCHI', 'SPX', 'GDAXI', 'N225'
        code = '002600.SZ'
        pattern_name = 'CDLENGULFING'
        pattern_category = ''
        get_new_data = True
        pattern_performance = run_pattern_recognition_Strategy(code,
                                                               start_date,
                                                               end_date,
                                                               pattern_category=pattern_category,  # _inUp
                                                               pattern_name=pattern_name,
                                                               pattern_type='buy',
                                                               observe_day=observe_day,
                                                               plot=True,
                                                               log=True,
                                                               get_new_data=get_new_data)

    elif mode == 'find_observe_day':
        observe_days = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        avg_win_ratios = []
        for observe_day in observe_days:
            avg_win_ratios.append(codes_test(pattern_name, observe_day, start_date, end_date, file_name, plot, get_new_data))

        for i, win_ratio in enumerate(avg_win_ratios):
            print(f"observe_day {observe_days[i]}: {list(win_ratio)} diff: {round(list(win_ratio)[0] - list(win_ratio)[1], 2)}")
