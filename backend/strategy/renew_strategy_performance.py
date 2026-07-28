import csv
import warnings
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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
import main

# 过滤掉FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import config

try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False
try:
    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()
except Exception as _ts_init_err:
    # 多进程子进程环境下 token 文件可能不可用，pro=None 时由 data_source 兜底
    import logging as _logging
    _logging.getLogger('trader_system').warning(f'[renew] tushare 初始化失败（将使用 akshare 兜底）: {_ts_init_err}')
    pro = None

# 模块级logger，子进程中输出到stderr，被web_app的subprocess捕获到log_streams
logger = logging.getLogger('trader_system')

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


# 个股日线DataFrame缓存：批量回测61个形态读同一code文件时避免重复IO
# key: (code, data_folder_dir) -> {'df': DataFrame, 'mtime': float}
_daily_df_cache = {}


def _effective_data_path(code, data_folder_dir):
    """返回指定目录中 code 的有效数据文件：优先 Parquet，回退 CSV。"""
    import os
    pq_path = os.path.join(data_folder_dir, f'{code}_daily.parquet')
    if _PARQUET_AVAILABLE and os.path.exists(pq_path) and os.path.getsize(pq_path) > 0:
        return pq_path, True
    csv_path = os.path.join(data_folder_dir, f'{code}_daily.csv')
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path, False
    return None, False


def _load_daily_df(code, data_folder_dir):
    """读取并预处理个股日线 Parquet/CSV，带mtime缓存。

    返回sort+set_index后的DataFrame；文件不存在或读取失败返回None。
    """
    import os
    data_path, is_parquet = _effective_data_path(code, data_folder_dir)
    if not data_path:
        return None
    try:
        mtime = os.path.getmtime(data_path)
    except OSError:
        return None
    key = (code, data_folder_dir)
    cached = _daily_df_cache.get(key)
    if cached and cached['mtime'] == mtime:
        return cached['df']
    try:
        if is_parquet:
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        df = df.sort_values(by=['trade_date'], ascending=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date', drop=True)
        _daily_df_cache[key] = {'df': df, 'mtime': mtime}
        return df
    except Exception:
        return None


# 从tushare获取股票日线数据
def read_data(code, start_date, end_date, get_new_data, data_folder_dir='', save_data=True):
    import os
    # 优先使用缓存（批量回测61形态读同一code时显著减少IO）
    df = _load_daily_df(code, data_folder_dir)
    if df is None:
        # 缓存未命中（文件不存在等），回退到直接读取以保留原异常行为
        data_path, is_parquet = _effective_data_path(code, data_folder_dir)
        if not data_path:
            raise FileNotFoundError(f'{code} 在 {data_folder_dir} 无 Parquet/CSV 数据')
        if is_parquet:
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        df = df.sort_values(by=['trade_date'], ascending=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  # 将日期转换为datetime格式
        df = df.set_index('trade_date', drop=True)  # 将日期设置为索引

    #按照日期范围截取df（用 .copy() 避免修改缓存中的原始 DataFrame）
    filtered_df = df.loc[start_date:end_date].copy()

    # 填充 NaN 值：OHLC 等关键列向前填充，避免回测时因 NaN 崩溃
    for col in filtered_df.columns:
        if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            filtered_df[col] = filtered_df[col].ffill().fillna(0)

    # 使用列名定位，兼容指数/A股/ETF等不同数据源 CSV 的列顺序差异
    cols = filtered_df.columns.tolist()
    vol_col = 'vol' if 'vol' in cols else ('volume' if 'volume' in cols else -1)
    data = bt.feeds.PandasData(
        dataname=filtered_df, datetime=None,
        open='open', high='high', low='low', close='close',
        volume=vol_col, openinterest=-1,
    )
    # 如果df为空，警告
    if filtered_df.empty:
        raise ValueError("df为空")
    return data


def run_performance_testing( code,
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

        cerebro.addstrategy(main._resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            log=log,
                            observe_day=observe_day,
                            cautious=cautious)
    elif pattern_type == 'sell':
        if pattern_category == '':
            strategy_name = 'strategy.patternDown_Strategy'
        else:
            strategy_name ='patternStrategy.patternDown_' + pattern_name + '_Strategy'

        cerebro.addstrategy(main._resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            log=log,
                            observe_day=observe_day,
                            cautious=cautious)

    elif pattern_type == 'tracking':
        if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
            strategy_name = 'tracking.patternUp_' + pattern_name + '_Strategy_index'
        else:
            strategy_name = 'tracking.patternUp_' + pattern_name + '_Strategy_stock'

        cerebro.addstrategy(main._resolve_strategy_class(strategy_name),
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
        cerebro.addstrategy(main._resolve_strategy_class(strategy_name),
                            name=pattern_name,
                            code=code,
                            code_name=code_name,
                            log=log,
                            to_log=to_log,
                            observe_day=observe_day,
                            track_date=track_date,
                            cautious=cautious)

    else:
        print('pattern_type 只能是 buy 或者 sell，已退出！')
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
            # 修复：胜率用 round() 四舍五入（与 main.py 和 pattern_scan.py 一致，原 int() 向下取整会导致1%差异）
            strategy_performance = {'交易次数': trade_analyze['total']['total'],
                                    '胜率(%)': round(trade_analyze['won']['total']/trade_analyze['total']['total']*100),
                                    '简易收益率(%)': returns,
                                    '夏普比率': sharpe_ratio,
                                    '最大回撤(%)': max_drawdown}
        except Exception as e:
            print(e)
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


# 每只股票的策略表现分开存放
# 形态列表（模块级常量，避免每次调用renew_performances重建）
# 与 signal_update.py / strategy_signals.py 保持一致（59个形态）
PATTERN_LIST = ['CDL3INSIDE', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLCOUNTERATTACK', 'CDLDOJI',
               'CDLDOJISTAR', 'CDLHARAMI', 'CDLINVERTEDHAMMER', 'CDLMARUBOZU', 'CDLTAKURI',
               'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
               'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
               'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
               'CDLMATCHINGLOW', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLRICKSHAWMAN',
               'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTICKSANDWICH',
               'CDLTASUKIGAP', 'CDLUNIQUE3RIVER', 'CDLXSIDEGAP3METHODS', 'CDL2CROWS', 'CDL3BLACKCROWS',
               'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY','CDL3LINESTRIKE',
                'CDLADVANCEBLOCK', 'CDLBREAKAWAY', 'CDLCONCEALBABYSWALL', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLHANGINGMAN', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
                'CDLSHOOTINGSTAR', 'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUPSIDEGAP2CROWS']


def _test_one_pattern(task_args):
    """进程池worker：回测单个形态。模块级函数，可被pickle。

    修复：buy/sell 类型直接调用 pattern_scan.run_single_pattern，
    统一交易明细、胜率、收益率、夏普、最大回撤的计算逻辑，
    避免维护两套独立回测引擎（原 run_performance_testing 用账户价值复利计算收益率，
    pattern_scan 用各笔交易 pnl_pct 算术和，导致胜率/收益/夏普/回撤都有差异）。
    """
    (code, pattern, buy_sell, start_date, end_date, observe_day,
     get_new_data, data_folder_dir, cautious) = task_args
    try:
        # buy/sell 类型直接调用 pattern_scan.run_single_pattern，统一计算口径
        import pattern_scan
        r = pattern_scan.run_single_pattern(
            code=code,
            pattern_name=pattern,
            pattern_type=buy_sell,  # buy/sell
            start_date=start_date,
            end_date=end_date,
            data_folder_dir=data_folder_dir,
            observe_day=observe_day,
            cautious=cautious,
        )
        # pattern_scan 返回字段名与 CSV 字段名映射
        return {
            '股票代码': code,
            '买入天数': observe_day,
            '策略名称': buy_sell + '_' + pattern,
            '交易次数': r.get('trades', 0),
            '胜率(%)': r.get('win_rate', 0),
            '简易收益率(%)': r.get('return_pct', 0),
            '夏普比率': r.get('sharpe', 0),
            '最大回撤(%)': r.get('hold_max_drawdown', 0),
        }
    except Exception as e:
        # 单个形态失败不影响其他形态
        return {
            '股票代码': code,
            '买入天数': observe_day,
            '策略名称': buy_sell + '_' + pattern,
            '交易次数': 0,
            '胜率(%)': 0,
            '简易收益率(%)': 0,
            '夏普比率': 0,
            '最大回撤(%)': 0,
            'error': str(e),
        }


def renew_performances(code, buy_sell, start_date, end_date, performance_file_name, data_folder_dir, get_new_data=True, cautious=False):
    if buy_sell not in ['buy', 'sell']:
        logger.error('[批量回测] buy_sell 只能是 buy 或 sell，已退出')
        print('buy_sell只能是buy或者sell')
        return

    # 父进程预拉取数据：检查本地csv是否存在且覆盖所需日期范围，缺失则自动拉取
    # 多进程子进程禁止网络访问，由父进程单进程串行拉取（akshare 优先，规避 tushare 限流）
    try:
        import data_source as _ds
        import os as _os
        csv_path = _os.path.join(data_folder_dir, f'{code}_daily.csv')
        need_fetch = True
        if _os.path.exists(csv_path) and _os.path.getsize(csv_path) > 0:
            # 检查日期范围是否覆盖
            try:
                _chk = pd.read_csv(csv_path, usecols=['trade_date'])
                _dates = _chk['trade_date'].astype(str)
                if _dates.min() <= start_date and _dates.max() >= end_date:
                    need_fetch = False
            except Exception:
                need_fetch = True
        if need_fetch:
            print(f'[批量回测] 本地无 {code} 数据或日期范围不足，父进程预拉取中...')
            logger.info(f'[批量回测] 预拉取 {code}（start={start_date}, end={end_date}）')
            df = _ds.get_kline_df(code, start_date, end_date, prefer_local=True, allow_network=True)
            if df is not None and not df.empty:
                _ds.save_kline_to_local(code, df, data_folder_dir)
                print(f'[批量回测] 预拉取 {code} 成功（{len(df)}行）')
            else:
                logger.warning(f'[批量回测] 预拉取 {code} 失败：数据为空')
                print(f'[批量回测] 警告: 预拉取 {code} 失败，可能影响回测结果')
    except Exception as _fetch_err:
        logger.warning(f'[批量回测] 预拉取 {code} 异常: {_fetch_err}')
        print(f'[批量回测] 预拉取 {code} 异常: {_fetch_err}')

    results = []  # 存放策略的性能指标

    # 遍历所有的买入信号
    # buy_pattern/sell_pattern
    observe_day = 2  # 观察天数默认2天

    # 构建任务列表
    tasks = [(code, pattern, buy_sell, start_date, end_date, observe_day,
              get_new_data, data_folder_dir, cautious) for pattern in PATTERN_LIST]
    total = len(tasks)

    # 决定进程数：0=自动（CPU核心数），1=串行
    workers = config.SCAN_WORKERS
    if workers <= 0:
        workers = multiprocessing.cpu_count()
    workers = min(workers, total)

    import time
    _start_ts = time.time()
    logger.info(f'[批量回测] 开始: code={code}, buy_sell={buy_sell}, 形态数={total}, 进程数={workers}, cautious={cautious}')

    if workers <= 1:
        # 串行模式
        for task_args in tasks:
            r = _test_one_pattern(task_args)
            results.append(r)
    else:
        # 并行模式：多进程同时回测多个形态
        results = [None] * total
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(_test_one_pattern, t): i for i, t in enumerate(tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

    # 过滤掉None（理论上不会出现，保留兜底）
    results = [r for r in results if r is not None]

    fieldnames = ['股票代码', '买入天数', '策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']

    # 读取已有记录，按 (股票代码, 策略名称) 去重：新结果覆盖旧记录，避免重复运行累积重复行
    dedup_map = {}
    import os
    if os.path.exists(performance_file_name):
        try:
            with open(performance_file_name, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    key = (row.get('股票代码', ''), row.get('策略名称', ''))
                    dedup_map[key] = row
        except Exception:
            pass  # 文件损坏时忽略已有记录，直接用新结果覆盖

    # 新结果覆盖同key的旧记录
    for result in results:
        key = (result['股票代码'], result['策略名称'])
        dedup_map[key] = {k: v for k, v in result.items() if k in fieldnames}

    # 全量覆盖写入（不再追加，避免重复行）
    with open(performance_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dedup_map.values():
            # 补全缺失字段，避免DictWriter报错
            for fn in fieldnames:
                row.setdefault(fn, '')
            writer.writerow({fn: row.get(fn, '') for fn in fieldnames})

    # 统计失败数（带error字段的result）
    failed = sum(1 for r in results if 'error' in r)
    logger.info(f'[批量回测] 完成: code={code}, 成功={len(results) - failed}, 失败={failed}, '
                f'耗时={time.time() - _start_ts:.1f}秒, 文件={performance_file_name}')

if __name__ == '__main__':
    # 批量测试策略参数，输出策略表现到csv - A股版，每只股票分开存放
    start_date = '20100104'
    end_date = '20231229'
    buy_sell = 'sell'
    index_stock = 'index'  # 填入index/stock

    get_new_data = False
    a_market_dir = '数据/A股/'
    index_dir = '数据/指数/'
    stock_data_df = pd.read_csv(a_market_dir + 'stock_data.csv')
    # 筛选大市值的股票
    # filtered_stocks = stock_data_df[(stock_data_df['total_mv'] > 5000000)]['ts_code'].tolist()
    filtered_stocks = ['DJI', 'FCHI', 'SPX', 'N225', 'GDAXI', '000300.SH', '399006.SZ']
    # filtered_stocks = ['000300.SH', '399006.SZ', 'gold']
    for code in filtered_stocks[:3]:
        if index_stock == 'index':  # 回测指数
            data_dir = index_dir + '训练测试库/训练/'
            performance_file_name = index_dir + '个股策略表现/' + code + '_' + buy_sell + '_strategy_performance_test.csv'
        else:  # 回测个股
            data_dir = a_market_dir + '训练测试库/训练/'
            performance_file_name = a_market_dir + '个股策略表现/' + code + '_' + buy_sell + '_strategy_performance_test.csv'
        renew_performances(code,
                           buy_sell,
                           start_date,
                           end_date,
                           performance_file_name,
                           data_dir,
                           get_new_data=get_new_data)

