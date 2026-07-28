import baseStrategy as bs
import main
import tools
import pandas as pd
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import backtrader as bt
from config import config

# stock_data.csv 模块级缓存：避免tracking()和run_tracking()重复读取
# 文件mtime变化时自动失效
_stock_data_cache = {'data': None, 'mtime': 0, 'path': ''}


def _load_stock_data():
    """加载stock_data.csv，带mtime缓存。

    多次调用同一进程内复用DataFrame；文件被更新（mtime变化）时自动重读。
    返回DataFrame，文件不存在时返回空DataFrame。
    """
    path = str(config.STOCK_DATA_FILE)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return pd.DataFrame()
    if (_stock_data_cache['data'] is not None
            and _stock_data_cache['mtime'] == mtime
            and _stock_data_cache['path'] == path):
        return _stock_data_cache['data']
    try:
        df = pd.read_csv(path)
        _stock_data_cache['data'] = df
        _stock_data_cache['mtime'] = mtime
        _stock_data_cache['path'] = path
        return df
    except Exception:
        return pd.DataFrame()

class patternUp_combine_Strategy_index(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
        ('to_log', True),
        ('track_date', '2025-01-01'),
        ('code', ''),
        ('code_name', ''),
        ('observe_day', 2),
        ('cautious', False), # 谨慎模式：开启后6个特定形态需满足额外条件才买入

    )
    def __init__(self):
        # 将buy pattern加入信号
        self.buy_signal_CDL3INSIDE = getattr(bt.talib, 'CDL3INSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3OUTSIDE = getattr(bt.talib, 'CDL3OUTSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLBELTHOLD = getattr(bt.talib, 'CDLBELTHOLD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLCLOSINGMARUBOZU = getattr(bt.talib, 'CDLCLOSINGMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLCOUNTERATTACK = getattr(bt.talib, 'CDLCOUNTERATTACK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDOJI = getattr(bt.talib, 'CDLDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDOJISTAR = getattr(bt.talib, 'CDLDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDRAGONFLYDOJI = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLENGULFING = getattr(bt.talib, 'CDLENGULFING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLGAPSIDESIDEWHITE = getattr(bt.talib, 'CDLGAPSIDESIDEWHITE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLGRAVESTONEDOJI = getattr(bt.talib, 'CDLGRAVESTONEDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHAMMER = getattr(bt.talib, 'CDLHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHARAMI = getattr(bt.talib, 'CDLHARAMI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHARAMICROSS = getattr(bt.talib, 'CDLHARAMICROSS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHIGHWAVE = getattr(bt.talib, 'CDLHIGHWAVE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHIKKAKE = getattr(bt.talib, 'CDLHIKKAKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHIKKAKEMOD = getattr(bt.talib, 'CDLHIKKAKEMOD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHOMINGPIGEON = getattr(bt.talib, 'CDLHOMINGPIGEON')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLINVERTEDHAMMER = getattr(bt.talib, 'CDLINVERTEDHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLKICKING = getattr(bt.talib, 'CDLKICKING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLKICKINGBYLENGTH = getattr(bt.talib, 'CDLKICKINGBYLENGTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLLADDERBOTTOM = getattr(bt.talib, 'CDLLADDERBOTTOM')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLLONGLEGGEDDOJI = getattr(bt.talib, 'CDLLONGLEGGEDDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLLONGLINE = getattr(bt.talib, 'CDLLONGLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMARUBOZU = getattr(bt.talib, 'CDLMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMATCHINGLOW = getattr(bt.talib, 'CDLMATCHINGLOW')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMORNINGDOJISTAR = getattr(bt.talib, 'CDLMORNINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMORNINGSTAR = getattr(bt.talib, 'CDLMORNINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLPIERCING = getattr(bt.talib, 'CDLPIERCING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLRICKSHAWMAN = getattr(bt.talib, 'CDLRICKSHAWMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLRISEFALL3METHODS = getattr(bt.talib, 'CDLRISEFALL3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSEPARATINGLINES = getattr(bt.talib, 'CDLSEPARATINGLINES')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSHORTLINE = getattr(bt.talib, 'CDLSHORTLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSPINNINGTOP = getattr(bt.talib, 'CDLSPINNINGTOP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSTICKSANDWICH = getattr(bt.talib, 'CDLSTICKSANDWICH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTAKURI = getattr(bt.talib, 'CDLTAKURI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTASUKIGAP = getattr(bt.talib, 'CDLTASUKIGAP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLUNIQUE3RIVER = getattr(bt.talib, 'CDLUNIQUE3RIVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLXSIDEGAP3METHODS = getattr(bt.talib, 'CDLXSIDEGAP3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)
        #可能的买入信号
        self.buy_signal_CDL2CROWS = getattr(bt.talib, 'CDL2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3BLACKCROWS = getattr(bt.talib, 'CDL3BLACKCROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3LINESTRIKE = getattr(bt.talib, 'CDL3LINESTRIKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3STARSINSOUTH = getattr(bt.talib, 'CDL3STARSINSOUTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3WHITESOLDIERS = getattr(bt.talib, 'CDL3WHITESOLDIERS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLABANDONEDBABY = getattr(bt.talib, 'CDLABANDONEDBABY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLADVANCEBLOCK = getattr(bt.talib, 'CDLADVANCEBLOCK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLBREAKAWAY = getattr(bt.talib, 'CDLBREAKAWAY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLCONCEALBABYSWALL = getattr(bt.talib, 'CDLCONCEALBABYSWALL')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDARKCLOUDCOVER = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLEVENINGDOJISTAR = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLEVENINGSTAR = getattr(bt.talib, 'CDLEVENINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHANGINGMAN = getattr(bt.talib, 'CDLHANGINGMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLIDENTICAL3CROWS = getattr(bt.talib, 'CDLIDENTICAL3CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLINNECK = getattr(bt.talib, 'CDLINNECK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSHOOTINGSTAR = getattr(bt.talib, 'CDLSHOOTINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSTALLEDPATTERN = getattr(bt.talib, 'CDLSTALLEDPATTERN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTHRUSTING = getattr(bt.talib, 'CDLTHRUSTING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTRISTAR = getattr(bt.talib, 'CDLTRISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLUPSIDEGAP2CROWS = getattr(bt.talib, 'CDLUPSIDEGAP2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)

        # 将sell pattern加入信号
        self.sell_signal_CDL2CROWS = getattr(bt.talib, 'CDL2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3BLACKCROWS = getattr(bt.talib, 'CDL3BLACKCROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3LINESTRIKE = getattr(bt.talib, 'CDL3LINESTRIKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3STARSINSOUTH = getattr(bt.talib, 'CDL3STARSINSOUTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3WHITESOLDIERS = getattr(bt.talib, 'CDL3WHITESOLDIERS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLABANDONEDBABY = getattr(bt.talib, 'CDLABANDONEDBABY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLADVANCEBLOCK = getattr(bt.talib, 'CDLADVANCEBLOCK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLBREAKAWAY = getattr(bt.talib, 'CDLBREAKAWAY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLCONCEALBABYSWALL = getattr(bt.talib, 'CDLCONCEALBABYSWALL')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDARKCLOUDCOVER = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLEVENINGDOJISTAR = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLEVENINGSTAR = getattr(bt.talib, 'CDLEVENINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHANGINGMAN = getattr(bt.talib, 'CDLHANGINGMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLIDENTICAL3CROWS = getattr(bt.talib, 'CDLIDENTICAL3CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLINNECK = getattr(bt.talib, 'CDLINNECK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSHOOTINGSTAR = getattr(bt.talib, 'CDLSHOOTINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSTALLEDPATTERN = getattr(bt.talib, 'CDLSTALLEDPATTERN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTHRUSTING = getattr(bt.talib, 'CDLTHRUSTING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTRISTAR = getattr(bt.talib, 'CDLTRISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLUPSIDEGAP2CROWS = getattr(bt.talib, 'CDLUPSIDEGAP2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        #可能的卖出信号
        self.sell_signal_CDL3INSIDE = getattr(bt.talib, 'CDL3INSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3OUTSIDE = getattr(bt.talib, 'CDL3OUTSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLBELTHOLD = getattr(bt.talib, 'CDLBELTHOLD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLCLOSINGMARUBOZU = getattr(bt.talib, 'CDLCLOSINGMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLCOUNTERATTACK = getattr(bt.talib, 'CDLCOUNTERATTACK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDOJI = getattr(bt.talib, 'CDLDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDOJISTAR = getattr(bt.talib, 'CDLDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDRAGONFLYDOJI = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLENGULFING = getattr(bt.talib, 'CDLENGULFING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLGAPSIDESIDEWHITE = getattr(bt.talib, 'CDLGAPSIDESIDEWHITE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLGRAVESTONEDOJI = getattr(bt.talib, 'CDLGRAVESTONEDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHAMMER = getattr(bt.talib, 'CDLHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHARAMI = getattr(bt.talib, 'CDLHARAMI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHARAMICROSS = getattr(bt.talib, 'CDLHARAMICROSS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHIGHWAVE = getattr(bt.talib, 'CDLHIGHWAVE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHIKKAKE = getattr(bt.talib, 'CDLHIKKAKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHIKKAKEMOD = getattr(bt.talib, 'CDLHIKKAKEMOD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHOMINGPIGEON = getattr(bt.talib, 'CDLHOMINGPIGEON')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLINVERTEDHAMMER = getattr(bt.talib, 'CDLINVERTEDHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLKICKING = getattr(bt.talib, 'CDLKICKING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLKICKINGBYLENGTH = getattr(bt.talib, 'CDLKICKINGBYLENGTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLLADDERBOTTOM = getattr(bt.talib, 'CDLLADDERBOTTOM')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLLONGLEGGEDDOJI = getattr(bt.talib, 'CDLLONGLEGGEDDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLLONGLINE = getattr(bt.talib, 'CDLLONGLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMARUBOZU = getattr(bt.talib, 'CDLMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMATCHINGLOW = getattr(bt.talib, 'CDLMATCHINGLOW')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMORNINGDOJISTAR = getattr(bt.talib, 'CDLMORNINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMORNINGSTAR = getattr(bt.talib, 'CDLMORNINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLPIERCING = getattr(bt.talib, 'CDLPIERCING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLRICKSHAWMAN = getattr(bt.talib, 'CDLRICKSHAWMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLRISEFALL3METHODS = getattr(bt.talib, 'CDLRISEFALL3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSEPARATINGLINES = getattr(bt.talib, 'CDLSEPARATINGLINES')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSHORTLINE = getattr(bt.talib, 'CDLSHORTLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSPINNINGTOP = getattr(bt.talib, 'CDLSPINNINGTOP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSTICKSANDWICH = getattr(bt.talib, 'CDLSTICKSANDWICH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTAKURI = getattr(bt.talib, 'CDLTAKURI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTASUKIGAP = getattr(bt.talib, 'CDLTASUKIGAP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLUNIQUE3RIVER = getattr(bt.talib, 'CDLUNIQUE3RIVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLXSIDEGAP3METHODS = getattr(bt.talib, 'CDLXSIDEGAP3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)

        self.buyday = 0 # 已买入天数
        self.have_position = False

        self.strategy_data_df = pd.read_csv('策略表现/策略字典.csv')
        index_dir = '数据/指数/'
        # 策略表现CSV可能不存在（未跑批量回测的新标的），缺失时用空DataFrame避免子进程崩溃
        _perf_cols = ['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']
        try:
            self.buy_strategy_performance_df = pd.read_csv(index_dir + '个股策略表现/' + self.p.code + '_buy_strategy_performance_test.csv')
        except FileNotFoundError:
            self.buy_strategy_performance_df = pd.DataFrame(columns=_perf_cols)
            print(f'[跟踪] 警告: {self.p.code} 无 buy 策略表现CSV，相关信号将无法输出')
        try:
            self.sell_strategy_performance_df = pd.read_csv(index_dir + '个股策略表现/' + self.p.code + '_sell_strategy_performance_test.csv')
        except FileNotFoundError:
            self.sell_strategy_performance_df = pd.DataFrame(columns=_perf_cols)

    def next(self): #TODO;若无持仓，仍输出卖出建议，若有持仓，仍输出买入建议
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)

        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        self.buy_signal = None
        if self.buy_signal_CDL3INSIDE[0] > 0:
            self.buy_signal = 'buy_signal_CDL3INSIDE'
        if self.buy_signal_CDL3OUTSIDE[0] > 0:
            self.buy_signal = 'buy_signal_CDL3OUTSIDE'
        if self.buy_signal_CDLBELTHOLD[0] > 0:
            self.buy_signal = 'buy_signal_CDLBELTHOLD'
        if (self.buy_signal_CDLMARUBOZU[0] > 0 and (not self.p.cautious
                     or (self.data.close[-1] / self.data.open[-1] < 1.015
                     and self.data.close[-2] / self.data.open[-2] < 1.015))):
            self.buy_signal = 'buy_signal_CDLMARUBOZU'
        if self.buy_signal_CDLCOUNTERATTACK[0] > 0:
            self.buy_signal = 'buy_signal_CDLCOUNTERATTACK'
        if self.buy_signal_CDLDOJI[0] > 0:
            self.buy_signal = 'buy_signal_CDLDOJI'
        if self.buy_signal_CDLDOJISTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLDOJISTAR'
        if (self.buy_signal_CDLDRAGONFLYDOJI[0] > 0 and (not self.p.cautious
                     or (self.data.close[-1] < self.data.open[-1] # 处于下降趋势
                     and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                     and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                     and self.data.close[0] < self.data.close[-1]))):
            self.buy_signal = 'buy_signal_CDLDRAGONFLYDOJI'
        if self.buy_signal_CDLENGULFING[0] > 0:
            self.buy_signal = 'buy_signal_CDLENGULFING'
        if self.buy_signal_CDLGAPSIDESIDEWHITE[0] > 0:
            self.buy_signal = 'buy_signal_CDLGAPSIDESIDEWHITE'
        if self.buy_signal_CDLGRAVESTONEDOJI[0] > 0:
            self.buy_signal = 'buy_signal_CDLGRAVESTONEDOJI'
        if self.buy_signal_CDLHAMMER[0] > 0:
            self.buy_signal = 'buy_signal_CDLHAMMER'
        if self.buy_signal_CDLHARAMI[0] > 0:
            self.buy_signal = 'buy_signal_CDLHARAMI'
        if self.buy_signal_CDLHARAMICROSS[0] > 0:
            self.buy_signal = 'buy_signal_CDLHARAMICROSS'
        if self.buy_signal_CDLHIGHWAVE[0] > 0:
            self.buy_signal = 'buy_signal_CDLHIGHWAVE'
        if self.buy_signal_CDLHIKKAKE[0] > 0:
            self.buy_signal = 'buy_signal_CDLHIKKAKE'
        if self.buy_signal_CDLHIKKAKEMOD[0] > 0:
            self.buy_signal = 'buy_signal_CDLHIKKAKEMOD'
        if self.buy_signal_CDLHOMINGPIGEON[0] > 0:
            self.buy_signal = 'buy_signal_CDLHOMINGPIGEON'
        if (self.buy_signal_CDLINVERTEDHAMMER[0] > 0 and (not self.p.cautious
                     or ((self.data.close[0] > self.data.open[0]
                     and self.data.open[0] / self.data.low[0] < 1.003)
                     or (self.data.close[0] < self.data.open[0] and self.data.close[0] / self.data.low[0] < 1.003)))):
            self.buy_signal = 'buy_signal_CDLINVERTEDHAMMER'
        if self.buy_signal_CDLKICKING[0] > 0:
            self.buy_signal = 'buy_signal_CDLKICKING'
        if self.buy_signal_CDLKICKINGBYLENGTH[0] > 0:
            self.buy_signal = 'buy_signal_CDLKICKINGBYLENGTH'
        if self.buy_signal_CDLMATCHINGLOW[0] > 0:
            self.buy_signal = 'buy_signal_CDLMATCHINGLOW'
        if self.buy_signal_CDLMORNINGDOJISTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLMORNINGDOJISTAR'
        if self.buy_signal_CDLMORNINGSTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLMORNINGSTAR'
        if self.buy_signal_CDLPIERCING[0] > 0:
            self.buy_signal = 'buy_signal_CDLPIERCING'
        if self.buy_signal_CDLRICKSHAWMAN[0] > 0:
            self.buy_signal = 'buy_signal_CDLRICKSHAWMAN'
        if self.buy_signal_CDLRISEFALL3METHODS[0] > 0:
            self.buy_signal = 'buy_signal_CDLRISEFALL3METHODS'
        if (self.buy_signal_CDLSEPARATINGLINES[0] > 0 and (not self.p.cautious
                or (self.data.close[-2] > self.data.close[-3]
                and self.data.close[0] > self.data.open[0]))):
            self.buy_signal = 'buy_signal_CDLSEPARATINGLINES'
        if self.buy_signal_CDLSHORTLINE[0] > 0:
            self.buy_signal = 'buy_signal_CDLSHORTLINE'
        if self.buy_signal_CDLSPINNINGTOP[0] > 0:
            self.buy_signal = 'buy_signal_CDLSPINNINGTOP'
        if self.buy_signal_CDLSTICKSANDWICH[0] > 0:
            self.buy_signal = 'buy_signal_CDLSTICKSANDWICH'
        if (self.buy_signal_CDLTAKURI[0] > 0 and (not self.p.cautious
                     or (self.data.close[-1] < self.data.open[-1]  # 处于下降趋势
                     and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                     and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                     and self.data.close[0] < self.data.close[-1]))):  # 处于下降趋势
            self.buy_signal = 'buy_signal_CDLTAKURI'
        if (self.buy_signal_CDLTASUKIGAP[0] > 0 and (not self.p.cautious
                or (self.data.open[-1] > self.data.high[-2]
                and self.data.low[-1] > self.data.high[-2]))):
            self.buy_signal ='buy_signal_CDLTASUKIGAP'
        if self.buy_signal_CDLUNIQUE3RIVER[0] > 0:
            self.buy_signal = 'buy_signal_CDLUNIQUE3RIVER'
        if self.buy_signal_CDLXSIDEGAP3METHODS[0] > 0:
            self.buy_signal = 'buy_signal_CDLXSIDEGAP3METHODS'
        #可能的买入信号
        if self.buy_signal_CDL2CROWS[0] > 0:
            self.buy_signal = 'buy_signal_CDLADVANCEBLOCK'
        if self.buy_signal_CDL3BLACKCROWS[0] > 0:
            self.buy_signal = 'buy_signal_CDL3BLACKCROWS'
        if self.buy_signal_CDL3LINESTRIKE[0] > 0:
            self.buy_signal = 'buy_signal_CDL3LINESTRIKE'
        if self.buy_signal_CDL3STARSINSOUTH[0] > 0:
            self.buy_signal = 'buy_signal_CDL3STARSINSOUTH'
        if self.buy_signal_CDL3WHITESOLDIERS[0] > 0:
            self.buy_signal = 'buy_signal_CDL3WHITESOLDIERS'
        if self.buy_signal_CDLABANDONEDBABY[0] > 0:
            self.buy_signal = 'buy_signal_CDLABANDONEDBABY'
        if self.buy_signal_CDLADVANCEBLOCK[0] > 0:
            self.buy_signal = 'buy_signal_CDLADVANCEBLOCK'
        if self.buy_signal_CDLBREAKAWAY[0] > 0:
            self.buy_signal = 'buy_signal_CDLBREAKAWAY'
        if self.buy_signal_CDLCONCEALBABYSWALL[0] > 0:
            self.buy_signal = 'buy_signal_CDLCONCEALBABYSWALL'
        if self.buy_signal_CDLDARKCLOUDCOVER[0] > 0:
            self.buy_signal = 'buy_signal_CDLDARKCLOUDCOVER'
        if self.buy_signal_CDLEVENINGDOJISTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLEVENINGDOJISTAR'
        if self.buy_signal_CDLEVENINGSTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLEVENINGSTAR'
        if self.buy_signal_CDLHANGINGMAN[0] > 0:
            self.buy_signal = 'buy_signal_CDLHANGINGMAN'
        if self.buy_signal_CDLIDENTICAL3CROWS[0] > 0:
            self.buy_signal = 'buy_signal_CDLIDENTICAL3CROWS'
        if self.buy_signal_CDLINNECK[0] > 0:
            self.buy_signal = 'buy_signal_CDLINNECK'
        if self.buy_signal_CDLSHOOTINGSTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLSHOOTINGSTAR'
        if self.buy_signal_CDLSTALLEDPATTERN[0] > 0:
            self.buy_signal = 'buy_signal_CDLSTALLEDPATTERN'
        if self.buy_signal_CDLTHRUSTING[0] > 0:
            self.buy_signal = 'buy_signal_CDLTHRUSTING'
        if self.buy_signal_CDLTRISTAR[0] > 0:
            self.buy_signal = 'buy_signal_CDLTRISTAR'
        if self.buy_signal_CDLUPSIDEGAP2CROWS[0] > 0:
            self.buy_signal = 'buy_signal_CDLUPSIDEGAP2CROWS'
        if self.buy_signal != None:
            self.log(self.buy_signal)
            if str(self.datas[0].datetime.date(0)) == self.p.track_date:
                try:
                    performace_list = self.buy_strategy_performance_df[self.buy_strategy_performance_df['策略名称'] == ('buy_' + self.buy_signal[11:])][['交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']].values[0]
                except IndexError:
                    performace_list = None  # CSV中无该策略记录，跳过
                # 如果胜率大于40%，并且收益率大于0%，则输出策略名称和触发条件
                if performace_list is not None and performace_list[1] > 50 and performace_list[2] > 0:
                    logging.info('买入原因：' + self.strategy_data_df[self.strategy_data_df['策略代码'] == self.buy_signal[11:]]['策略名称'].values[0] + '(' + self.buy_signal + ')',
                                            extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
                    logging.info('触发形态：' + self.strategy_data_df[self.strategy_data_df['策略代码'] == self.buy_signal[11:]]['触发条件'].values[0] + '(' + self.buy_signal + ')',
                                            extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
                    logging.info('策略在该股的交易次数：' + str(round(performace_list[0], 0))
                                 + ', 胜率(%)：' + str(round(performace_list[1],2) )
                                 + ', 简易收益率(%)：' + str(round(performace_list[2],2))
                                 + ', 夏普比率：' + str(round(performace_list[3],3))
                                 + ', 最大回撤(%)：' + str(round(performace_list[4], 2)) + '\n',
                                extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        self.sell_signal = None

        if self.sell_signal_CDL2CROWS[0] < 0:
            self.sell_signal = 'sell_signal_CDLADVANCEBLOCK'
        if self.sell_signal_CDL3BLACKCROWS[0] < 0:
            self.sell_signal ='sell_signal_CDL3BLACKCROWS'
        if self.sell_signal_CDL3LINESTRIKE[0] < 0:
            self.sell_signal ='sell_signal_CDL3LINESTRIKE'
        if self.sell_signal_CDL3STARSINSOUTH[0] < 0:
            self.sell_signal ='sell_signal_CDL3STARSINSOUTH'
        if self.sell_signal_CDL3WHITESOLDIERS[0] < 0:
            self.sell_signal ='sell_signal_CDL3WHITESOLDIERS'
        if self.sell_signal_CDLABANDONEDBABY[0] < 0:
            self.sell_signal ='sell_signal_CDLABANDONEDBABY'
        if self.sell_signal_CDLADVANCEBLOCK[0] < 0:
            self.sell_signal ='sell_signal_CDLADVANCEBLOCK'
        if self.sell_signal_CDLBREAKAWAY[0] < 0:
            self.sell_signal ='sell_signal_CDLBREAKAWAY'
        if self.sell_signal_CDLCONCEALBABYSWALL[0] < 0:
            self.sell_signal ='sell_signal_CDLCONCEALBABYSWALL'
        if self.sell_signal_CDLDARKCLOUDCOVER[0] < 0:
            self.sell_signal ='sell_signal_CDLDARKCLOUDCOVER'
        if self.sell_signal_CDLEVENINGDOJISTAR[0] < 0:
            self.sell_signal ='sell_signal_CDLEVENINGDOJISTAR'
        if self.sell_signal_CDLEVENINGSTAR[0] < 0:
            self.sell_signal ='sell_signal_CDLEVENINGSTAR'
        if self.sell_signal_CDLHANGINGMAN[0] < 0:
            self.sell_signal ='sell_signal_CDLHANGINGMAN'
        if self.sell_signal_CDLIDENTICAL3CROWS[0] < 0:
            self.sell_signal ='sell_signal_CDLIDENTICAL3CROWS'
        if self.sell_signal_CDLINNECK[0] < 0:
            self.sell_signal ='sell_signal_CDLINNECK'
        if self.sell_signal_CDLSHOOTINGSTAR[0] < 0:
            self.sell_signal ='sell_signal_CDLSHOOTINGSTAR'
        if self.sell_signal_CDLSTALLEDPATTERN[0] < 0:
            self.sell_signal ='sell_signal_CDLSTALLEDPATTERN'
        if self.sell_signal_CDLTHRUSTING[0] < 0:
            self.sell_signal ='sell_signal_CDLTHRUSTING'
        if self.sell_signal_CDLTRISTAR[0] < 0:
            self.sell_signal ='sell_signal_CDLTRISTAR'
        if self.sell_signal_CDLUPSIDEGAP2CROWS[0] < 0:
            self.sell_signal ='sell_signal_CDLUPSIDEGAP2CROWS'

        #可能的卖出信号
        if self.sell_signal_CDL3INSIDE[0] < 0:
            self.sell_signal = 'sell_signal_CDL3INSIDE'
        if self.sell_signal_CDL3OUTSIDE[0] < 0:
            self.sell_signal = 'sell_signal_CDL3OUTSIDE'
        if self.sell_signal_CDLBELTHOLD[0] < 0:
            self.sell_signal = 'sell_signal_CDLBELTHOLD'
        if self.sell_signal_CDLMARUBOZU[0] < 0:
            self.sell_signal = 'sell_signal_CDLMARUBOZU'
        if self.sell_signal_CDLCOUNTERATTACK[0] < 0:
            self.sell_signal = 'sell_signal_CDLCOUNTERATTACK'
        if self.sell_signal_CDLDOJI[0] < 0:
            self.sell_signal = 'sell_signal_CDLDOJI'
        if self.sell_signal_CDLDOJISTAR[0] < 0:
            self.sell_signal = 'sell_signal_CDLDOJISTAR'
        if self.sell_signal_CDLDRAGONFLYDOJI[0] < 0:
            self.sell_signal = 'sell_signal_CDLDRAGONFLYDOJI'
        if self.sell_signal_CDLENGULFING[0] < 0:
            self.sell_signal = 'sell_signal_CDLENGULFING'
        if self.sell_signal_CDLGAPSIDESIDEWHITE[0] < 0:
            self.sell_signal = 'sell_signal_CDLGAPSIDESIDEWHITE'
        if self.sell_signal_CDLGRAVESTONEDOJI[0] < 0:
            self.sell_signal = 'sell_signal_CDLGRAVESTONEDOJI'
        if self.sell_signal_CDLHAMMER[0] < 0:
            self.sell_signal = 'sell_signal_CDLHAMMER'
        if self.sell_signal_CDLHARAMI[0] < 0:
            self.sell_signal = 'sell_signal_CDLHARAMI'
        if self.sell_signal_CDLHARAMICROSS[0] < 0:
            self.sell_signal = 'sell_signal_CDLHARAMICROSS'
        if self.sell_signal_CDLHIGHWAVE[0] < 0:
            self.sell_signal = 'sell_signal_CDLHIGHWAVE'
        if self.sell_signal_CDLHIKKAKE[0] < 0:
            self.sell_signal = 'sell_signal_CDLHIKKAKE'
        if self.sell_signal_CDLHIKKAKEMOD[0] < 0:
            self.sell_signal = 'sell_signal_CDLHIKKAKEMOD'
        if self.sell_signal_CDLHOMINGPIGEON[0] < 0:
            self.sell_signal = 'sell_signal_CDLHOMINGPIGEON'
        if self.sell_signal_CDLINVERTEDHAMMER[0] < 0:
            self.sell_signal = 'sell_signal_CDLINVERTEDHAMMER'
        if self.sell_signal_CDLKICKING[0] < 0:
            self.sell_signal = 'sell_signal_CDLKICKING'
        if self.sell_signal_CDLKICKINGBYLENGTH[0] < 0:
            self.sell_signal = 'sell_signal_CDLKICKINGBYLENGTH'
        if self.sell_signal_CDLMATCHINGLOW[0] < 0:
            self.sell_signal = 'sell_signal_CDLMATCHINGLOW'
        if self.sell_signal_CDLMORNINGDOJISTAR[0] < 0:
            self.sell_signal = 'sell_signal_CDLMORNINGDOJISTAR'
        if self.sell_signal_CDLMORNINGSTAR[0] < 0:
            self.sell_signal = 'sell_signal_CDLMORNINGSTAR'
        if self.sell_signal_CDLPIERCING[0] < 0:
            self.sell_signal = 'sell_signal_CDLPIERCING'
        if self.sell_signal_CDLRICKSHAWMAN[0] < 0:
            self.sell_signal = 'sell_signal_CDLRICKSHAWMAN'
        if self.sell_signal_CDLRISEFALL3METHODS[0] < 0:
            self.sell_signal = 'sell_signal_CDLRISEFALL3METHODS'
        if self.sell_signal_CDLSEPARATINGLINES[0] < 0:
            self.sell_signal = 'sell_signal_CDLSEPARATINGLINES'
        if self.sell_signal_CDLSHORTLINE[0] < 0:
            self.sell_signal = 'sell_signal_CDLSHORTLINE'
        if self.sell_signal_CDLSPINNINGTOP[0] < 0:
            self.sell_signal = 'sell_signal_CDLSPINNINGTOP'
        if self.sell_signal_CDLSTICKSANDWICH[0] < 0:
            self.sell_signal = 'sell_signal_CDLSTICKSANDWICH'
        if self.sell_signal_CDLTAKURI[0] < 0:  # 处于下降趋势
            self.sell_signal = 'sell_signal_CDLTAKURI'
        if self.sell_signal_CDLTASUKIGAP[0] < 0:
            self.sell_signal ='sell_signal_CDLTASUKIGAP'
        if self.sell_signal_CDLUNIQUE3RIVER[0] < 0:
            self.sell_signal = 'sell_signal_CDLUNIQUE3RIVER'
        if self.sell_signal_CDLXSIDEGAP3METHODS[0] < 0:
            self.sell_signal = 'sell_signal_CDLXSIDEGAP3METHODS'

        if self.sell_signal != None:
            if str(self.datas[0].datetime.date(0)) == self.p.track_date:
                if self.p.to_log:
                    try:
                        # 修复：CSV中策略名称格式为 'sell_CDLXXX'，需加 'sell_' 前缀（与 signal_update.py 输出一致）
                        performace_list = self.sell_strategy_performance_df[self.sell_strategy_performance_df['策略名称'] == ('sell_' + self.sell_signal[12:])][
                            ['交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']].values[0]
                    except IndexError:
                        performace_list = None  # CSV中无该策略记录，跳过
                    # 卖出信号筛选：胜率>50% 且 收益率>0%（与 strategy_signals.py 一致，原40%阈值已统一为50%）
                    if performace_list is not None and performace_list[1] > 50 and performace_list[2] > 0:
                        logging.info('卖出原因：' + self.strategy_data_df[self.strategy_data_df['策略代码'] == ('sell_' + self.sell_signal[12:])]['策略名称'].values[0] + '(' + self.sell_signal + ')',
                                     extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
                        logging.info('策略在该股的交易次数：' + str(round(performace_list[0], 0))
                                     + ', 胜率(%)：' + str(round(performace_list[1], 2))
                                     + ', 简易收益率(%)：' + str(round(performace_list[2], 2))
                                     + ', 夏普比率：' + str(round(performace_list[3], 3))
                                     + ', 最大回撤(%)：' + str(round(performace_list[4], 2)) + '\n',
                                     extra={'track_date': self.p.track_date, 'code': self.p.code,
                                            'code_name': self.p.code_name})
                else:
                    self.log('卖出提示：' + self.sell_signal)


class patternUp_combine_Strategy_stock(bs.BaseUpCandleStrategy):
    params = (
        ('name', ''),
        ('log', True),
        ('to_log', True),
        ('track_date', '2025-01-01'),
        ('code', ''),
        ('code_name', ''),
        ('cautious', False), # 谨慎模式：开启后6个特定形态需满足额外条件才买入

    )
    def __init__(self):
        # 将buy pattern加入信号
        self.buy_signal_CDL3INSIDE = getattr(bt.talib, 'CDL3INSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3OUTSIDE = getattr(bt.talib, 'CDL3OUTSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLBELTHOLD = getattr(bt.talib, 'CDLBELTHOLD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLCLOSINGMARUBOZU = getattr(bt.talib, 'CDLCLOSINGMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLCOUNTERATTACK = getattr(bt.talib, 'CDLCOUNTERATTACK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDOJI = getattr(bt.talib, 'CDLDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDOJISTAR = getattr(bt.talib, 'CDLDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDRAGONFLYDOJI = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLENGULFING = getattr(bt.talib, 'CDLENGULFING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLGAPSIDESIDEWHITE = getattr(bt.talib, 'CDLGAPSIDESIDEWHITE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLGRAVESTONEDOJI = getattr(bt.talib, 'CDLGRAVESTONEDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHAMMER = getattr(bt.talib, 'CDLHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHARAMI = getattr(bt.talib, 'CDLHARAMI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHARAMICROSS = getattr(bt.talib, 'CDLHARAMICROSS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHIGHWAVE = getattr(bt.talib, 'CDLHIGHWAVE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHIKKAKE = getattr(bt.talib, 'CDLHIKKAKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHIKKAKEMOD = getattr(bt.talib, 'CDLHIKKAKEMOD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHOMINGPIGEON = getattr(bt.talib, 'CDLHOMINGPIGEON')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLINVERTEDHAMMER = getattr(bt.talib, 'CDLINVERTEDHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLKICKING = getattr(bt.talib, 'CDLKICKING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLKICKINGBYLENGTH = getattr(bt.talib, 'CDLKICKINGBYLENGTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLLADDERBOTTOM = getattr(bt.talib, 'CDLLADDERBOTTOM')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLLONGLEGGEDDOJI = getattr(bt.talib, 'CDLLONGLEGGEDDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLLONGLINE = getattr(bt.talib, 'CDLLONGLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMARUBOZU = getattr(bt.talib, 'CDLMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMATCHINGLOW = getattr(bt.talib, 'CDLMATCHINGLOW')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMORNINGDOJISTAR = getattr(bt.talib, 'CDLMORNINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLMORNINGSTAR = getattr(bt.talib, 'CDLMORNINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLPIERCING = getattr(bt.talib, 'CDLPIERCING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLRICKSHAWMAN = getattr(bt.talib, 'CDLRICKSHAWMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLRISEFALL3METHODS = getattr(bt.talib, 'CDLRISEFALL3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSEPARATINGLINES = getattr(bt.talib, 'CDLSEPARATINGLINES')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSHORTLINE = getattr(bt.talib, 'CDLSHORTLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSPINNINGTOP = getattr(bt.talib, 'CDLSPINNINGTOP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSTICKSANDWICH = getattr(bt.talib, 'CDLSTICKSANDWICH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTAKURI = getattr(bt.talib, 'CDLTAKURI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTASUKIGAP = getattr(bt.talib, 'CDLTASUKIGAP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLUNIQUE3RIVER = getattr(bt.talib, 'CDLUNIQUE3RIVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLXSIDEGAP3METHODS = getattr(bt.talib, 'CDLXSIDEGAP3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)
        #可能的买入信号
        self.buy_signal_CDL2CROWS = getattr(bt.talib, 'CDL2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3BLACKCROWS = getattr(bt.talib, 'CDL3BLACKCROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3LINESTRIKE = getattr(bt.talib, 'CDL3LINESTRIKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3STARSINSOUTH = getattr(bt.talib, 'CDL3STARSINSOUTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDL3WHITESOLDIERS = getattr(bt.talib, 'CDL3WHITESOLDIERS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLABANDONEDBABY = getattr(bt.talib, 'CDLABANDONEDBABY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLADVANCEBLOCK = getattr(bt.talib, 'CDLADVANCEBLOCK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLBREAKAWAY = getattr(bt.talib, 'CDLBREAKAWAY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLCONCEALBABYSWALL = getattr(bt.talib, 'CDLCONCEALBABYSWALL')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLDARKCLOUDCOVER = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLEVENINGDOJISTAR = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLEVENINGSTAR = getattr(bt.talib, 'CDLEVENINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLHANGINGMAN = getattr(bt.talib, 'CDLHANGINGMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLIDENTICAL3CROWS = getattr(bt.talib, 'CDLIDENTICAL3CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLINNECK = getattr(bt.talib, 'CDLINNECK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSHOOTINGSTAR = getattr(bt.talib, 'CDLSHOOTINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLSTALLEDPATTERN = getattr(bt.talib, 'CDLSTALLEDPATTERN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTHRUSTING = getattr(bt.talib, 'CDLTHRUSTING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLTRISTAR = getattr(bt.talib, 'CDLTRISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.buy_signal_CDLUPSIDEGAP2CROWS = getattr(bt.talib, 'CDLUPSIDEGAP2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)

        # 将sell pattern加入信号
        self.sell_signal_CDL2CROWS = getattr(bt.talib, 'CDL2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3BLACKCROWS = getattr(bt.talib, 'CDL3BLACKCROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3LINESTRIKE = getattr(bt.talib, 'CDL3LINESTRIKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3STARSINSOUTH = getattr(bt.talib, 'CDL3STARSINSOUTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3WHITESOLDIERS = getattr(bt.talib, 'CDL3WHITESOLDIERS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLABANDONEDBABY = getattr(bt.talib, 'CDLABANDONEDBABY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLADVANCEBLOCK = getattr(bt.talib, 'CDLADVANCEBLOCK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLBREAKAWAY = getattr(bt.talib, 'CDLBREAKAWAY')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLCONCEALBABYSWALL = getattr(bt.talib, 'CDLCONCEALBABYSWALL')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDARKCLOUDCOVER = getattr(bt.talib, 'CDLDARKCLOUDCOVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLEVENINGDOJISTAR = getattr(bt.talib, 'CDLEVENINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLEVENINGSTAR = getattr(bt.talib, 'CDLEVENINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHANGINGMAN = getattr(bt.talib, 'CDLHANGINGMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLIDENTICAL3CROWS = getattr(bt.talib, 'CDLIDENTICAL3CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLINNECK = getattr(bt.talib, 'CDLINNECK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSHOOTINGSTAR = getattr(bt.talib, 'CDLSHOOTINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSTALLEDPATTERN = getattr(bt.talib, 'CDLSTALLEDPATTERN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTHRUSTING = getattr(bt.talib, 'CDLTHRUSTING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTRISTAR = getattr(bt.talib, 'CDLTRISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLUPSIDEGAP2CROWS = getattr(bt.talib, 'CDLUPSIDEGAP2CROWS')(self.data.open, self.data.high, self.data.low, self.data.close)
        #可能的卖出信号
        self.sell_signal_CDL3INSIDE = getattr(bt.talib, 'CDL3INSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDL3OUTSIDE = getattr(bt.talib, 'CDL3OUTSIDE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLBELTHOLD = getattr(bt.talib, 'CDLBELTHOLD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLCLOSINGMARUBOZU = getattr(bt.talib, 'CDLCLOSINGMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLCOUNTERATTACK = getattr(bt.talib, 'CDLCOUNTERATTACK')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDOJI = getattr(bt.talib, 'CDLDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDOJISTAR = getattr(bt.talib, 'CDLDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLDRAGONFLYDOJI = getattr(bt.talib, 'CDLDRAGONFLYDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLENGULFING = getattr(bt.talib, 'CDLENGULFING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLGAPSIDESIDEWHITE = getattr(bt.talib, 'CDLGAPSIDESIDEWHITE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLGRAVESTONEDOJI = getattr(bt.talib, 'CDLGRAVESTONEDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHAMMER = getattr(bt.talib, 'CDLHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHARAMI = getattr(bt.talib, 'CDLHARAMI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHARAMICROSS = getattr(bt.talib, 'CDLHARAMICROSS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHIGHWAVE = getattr(bt.talib, 'CDLHIGHWAVE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHIKKAKE = getattr(bt.talib, 'CDLHIKKAKE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHIKKAKEMOD = getattr(bt.talib, 'CDLHIKKAKEMOD')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLHOMINGPIGEON = getattr(bt.talib, 'CDLHOMINGPIGEON')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLINVERTEDHAMMER = getattr(bt.talib, 'CDLINVERTEDHAMMER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLKICKING = getattr(bt.talib, 'CDLKICKING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLKICKINGBYLENGTH = getattr(bt.talib, 'CDLKICKINGBYLENGTH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLLADDERBOTTOM = getattr(bt.talib, 'CDLLADDERBOTTOM')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLLONGLEGGEDDOJI = getattr(bt.talib, 'CDLLONGLEGGEDDOJI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLLONGLINE = getattr(bt.talib, 'CDLLONGLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMARUBOZU = getattr(bt.talib, 'CDLMARUBOZU')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMATCHINGLOW = getattr(bt.talib, 'CDLMATCHINGLOW')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMORNINGDOJISTAR = getattr(bt.talib, 'CDLMORNINGDOJISTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLMORNINGSTAR = getattr(bt.talib, 'CDLMORNINGSTAR')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLPIERCING = getattr(bt.talib, 'CDLPIERCING')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLRICKSHAWMAN = getattr(bt.talib, 'CDLRICKSHAWMAN')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLRISEFALL3METHODS = getattr(bt.talib, 'CDLRISEFALL3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSEPARATINGLINES = getattr(bt.talib, 'CDLSEPARATINGLINES')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSHORTLINE = getattr(bt.talib, 'CDLSHORTLINE')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSPINNINGTOP = getattr(bt.talib, 'CDLSPINNINGTOP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLSTICKSANDWICH = getattr(bt.talib, 'CDLSTICKSANDWICH')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTAKURI = getattr(bt.talib, 'CDLTAKURI')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLTASUKIGAP = getattr(bt.talib, 'CDLTASUKIGAP')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLUNIQUE3RIVER = getattr(bt.talib, 'CDLUNIQUE3RIVER')(self.data.open, self.data.high, self.data.low, self.data.close)
        self.sell_signal_CDLXSIDEGAP3METHODS = getattr(bt.talib, 'CDLXSIDEGAP3METHODS')(self.data.open, self.data.high, self.data.low, self.data.close)

        self.buyday = 0 # 已买入天数
        self.have_position = False
        self.buy_price = 0.0  # 买入价（用于3%固定止损）

        self.stock_data_df = pd.read_csv('策略表现/策略字典.csv')
        a_market_dir = '数据/A股/'
        # 策略表现CSV可能不存在（未跑批量回测的新股），缺失时用空DataFrame避免子进程崩溃
        try:
            self.buy_strategy_performance_df = pd.read_csv(a_market_dir + '个股策略表现/' + self.p.code + '_buy_strategy_performance_test.csv')
        except FileNotFoundError:
            self.buy_strategy_performance_df = pd.DataFrame(columns=['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)'])
            print(f'[跟踪] 警告: {self.p.code} 无 buy 策略表现CSV，相关信号将无法输出')
        try:
            self.sell_strategy_performance_df = pd.read_csv(a_market_dir + '个股策略表现/' + self.p.code + '_sell_strategy_performance_test.csv')
        except FileNotFoundError:
            self.sell_strategy_performance_df = pd.DataFrame(columns=['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)'])

    def next(self):
        # 计算可购买的股票数量，并确保是100的整数倍
        cash = self.broker.getcash()  # 获取当前的现金
        stock_price = self.data.close[0] #用收盘价估算买入数量（接近次日开盘价）
        # 可购买的股票数量，确保是100的整数倍；稍微少买一点，避免买不进去
        buy_size = int(cash * 0.995 // (stock_price) // 100 * 100)
        # 检查是否有买入信号，并且当前没有持仓，进行满仓买入
        if buy_size > 0 and not self.have_position:
            self.buy_signal = None

            if self.buy_signal_CDL3INSIDE[0] > 0:
                self.buy_signal = 'buy_signal_CDL3INSIDE'
            if self.buy_signal_CDL3OUTSIDE[0] > 0:
                self.buy_signal = 'buy_signal_CDL3OUTSIDE'
            if self.buy_signal_CDLBELTHOLD[0] > 0:
                self.buy_signal = 'buy_signal_CDLBELTHOLD'
            if (self.buy_signal_CDLMARUBOZU[0] > 0 and (not self.p.cautious
                         or (self.data.close[-1] / self.data.open[-1] < 1.015
                         and self.data.close[-2] / self.data.open[-2] < 1.015))):
                self.buy_signal = 'buy_signal_CDLMARUBOZU'
            if self.buy_signal_CDLCOUNTERATTACK[0] > 0:
                self.buy_signal = 'buy_signal_CDLCOUNTERATTACK'
            if self.buy_signal_CDLDOJI[0] > 0:
                self.buy_signal = 'buy_signal_CDLDOJI'
            if self.buy_signal_CDLDOJISTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLDOJISTAR'
            if (self.buy_signal_CDLDRAGONFLYDOJI[0] > 0 and (not self.p.cautious
                         or (self.data.close[-1] < self.data.open[-1] # 处于下降趋势
                         and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                         and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                         and self.data.close[0] < self.data.close[-1]))):
                self.buy_signal = 'buy_signal_CDLDRAGONFLYDOJI'
            if self.buy_signal_CDLENGULFING[0] > 0:
                self.buy_signal = 'buy_signal_CDLENGULFING'
            if self.buy_signal_CDLGAPSIDESIDEWHITE[0] > 0:
                self.buy_signal = 'buy_signal_CDLGAPSIDESIDEWHITE'
            if self.buy_signal_CDLGRAVESTONEDOJI[0] > 0:
                self.buy_signal = 'buy_signal_CDLGRAVESTONEDOJI'
            if self.buy_signal_CDLHAMMER[0] > 0:
                self.buy_signal = 'buy_signal_CDLHAMMER'
            if self.buy_signal_CDLHARAMI[0] > 0:
                self.buy_signal = 'buy_signal_CDLHARAMI'
            if self.buy_signal_CDLHARAMICROSS[0] > 0:
                self.buy_signal = 'buy_signal_CDLHARAMICROSS'
            if self.buy_signal_CDLHIGHWAVE[0] > 0:
                self.buy_signal = 'buy_signal_CDLHIGHWAVE'
            if self.buy_signal_CDLHIKKAKE[0] > 0:
                self.buy_signal = 'buy_signal_CDLHIKKAKE'
            if self.buy_signal_CDLHIKKAKEMOD[0] > 0:
                self.buy_signal = 'buy_signal_CDLHIKKAKEMOD'
            if self.buy_signal_CDLHOMINGPIGEON[0] > 0:
                self.buy_signal = 'buy_signal_CDLHOMINGPIGEON'
            if (self.buy_signal_CDLINVERTEDHAMMER[0] > 0 and (not self.p.cautious
                         or ((self.data.close[0] > self.data.open[0]
                         and self.data.open[0] / self.data.low[0] < 1.003)
                         or (self.data.close[0] < self.data.open[0] and self.data.close[0] / self.data.low[0] < 1.003)))):
                self.buy_signal = 'buy_signal_CDLINVERTEDHAMMER'
            if self.buy_signal_CDLKICKING[0] > 0:
                self.buy_signal = 'buy_signal_CDLKICKING'
            if self.buy_signal_CDLKICKINGBYLENGTH[0] > 0:
                self.buy_signal = 'buy_signal_CDLKICKINGBYLENGTH'
            if self.buy_signal_CDLMATCHINGLOW[0] > 0:
                self.buy_signal = 'buy_signal_CDLMATCHINGLOW'
            if self.buy_signal_CDLMORNINGDOJISTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLMORNINGDOJISTAR'
            if self.buy_signal_CDLMORNINGSTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLMORNINGSTAR'
            if self.buy_signal_CDLPIERCING[0] > 0:
                self.buy_signal = 'buy_signal_CDLPIERCING'
            if self.buy_signal_CDLRICKSHAWMAN[0] > 0:
                self.buy_signal = 'buy_signal_CDLRICKSHAWMAN'
            if self.buy_signal_CDLRISEFALL3METHODS[0] > 0:
                self.buy_signal = 'buy_signal_CDLRISEFALL3METHODS'
            if (self.buy_signal_CDLSEPARATINGLINES[0] > 0 and (not self.p.cautious
                    or (self.data.close[-2] > self.data.close[-3]
                    and self.data.close[0] > self.data.open[0]))):
                self.buy_signal = 'buy_signal_CDLSEPARATINGLINES'
            if self.buy_signal_CDLSHORTLINE[0] > 0:
                self.buy_signal = 'buy_signal_CDLSHORTLINE'
            if self.buy_signal_CDLSPINNINGTOP[0] > 0:
                self.buy_signal = 'buy_signal_CDLSPINNINGTOP'
            if self.buy_signal_CDLSTICKSANDWICH[0] > 0:
                self.buy_signal = 'buy_signal_CDLSTICKSANDWICH'
            if (self.buy_signal_CDLTAKURI[0] > 0 and (not self.p.cautious
                         or (self.data.close[-1] < self.data.open[-1]  # 处于下降趋势
                         and self.data.close[-1] < self.data.close[-2]  # 处于下降趋势
                         and self.data.close[-2] < self.data.close[-3]  # 处于下降趋势
                         and self.data.close[0] < self.data.close[-1]))):  # 处于下降趋势
                self.buy_signal = 'buy_signal_CDLTAKURI'
            if (self.buy_signal_CDLTASUKIGAP[0] > 0 and (not self.p.cautious
                    or (self.data.open[-1] > self.data.high[-2]
                    and self.data.low[-1] > self.data.high[-2]))):
                self.buy_signal ='buy_signal_CDLTASUKIGAP'
            if self.buy_signal_CDLUNIQUE3RIVER[0] > 0:
                self.buy_signal = 'buy_signal_CDLUNIQUE3RIVER'
            if self.buy_signal_CDLXSIDEGAP3METHODS[0] > 0:
                self.buy_signal = 'buy_signal_CDLXSIDEGAP3METHODS'
            #可能的买入信号
            if self.buy_signal_CDL2CROWS[0] > 0:
                self.buy_signal = 'buy_signal_CDLADVANCEBLOCK'
            if self.buy_signal_CDL3BLACKCROWS[0] > 0:
                self.buy_signal = 'buy_signal_CDL3BLACKCROWS'
            if self.buy_signal_CDL3LINESTRIKE[0] > 0:
                self.buy_signal = 'buy_signal_CDL3LINESTRIKE'
            if self.buy_signal_CDL3STARSINSOUTH[0] > 0:
                self.buy_signal = 'buy_signal_CDL3STARSINSOUTH'
            if self.buy_signal_CDL3WHITESOLDIERS[0] > 0:
                self.buy_signal = 'buy_signal_CDL3WHITESOLDIERS'
            if self.buy_signal_CDLABANDONEDBABY[0] > 0:
                self.buy_signal = 'buy_signal_CDLABANDONEDBABY'
            if self.buy_signal_CDLADVANCEBLOCK[0] > 0:
                self.buy_signal = 'buy_signal_CDLADVANCEBLOCK'
            if self.buy_signal_CDLBREAKAWAY[0] > 0:
                self.buy_signal = 'buy_signal_CDLBREAKAWAY'
            if self.buy_signal_CDLCONCEALBABYSWALL[0] > 0:
                self.buy_signal = 'buy_signal_CDLCONCEALBABYSWALL'
            if self.buy_signal_CDLDARKCLOUDCOVER[0] > 0:
                self.buy_signal = 'buy_signal_CDLDARKCLOUDCOVER'
            if self.buy_signal_CDLEVENINGDOJISTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLEVENINGDOJISTAR'
            if self.buy_signal_CDLEVENINGSTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLEVENINGSTAR'
            if self.buy_signal_CDLHANGINGMAN[0] > 0:
                self.buy_signal = 'buy_signal_CDLHANGINGMAN'
            if self.buy_signal_CDLIDENTICAL3CROWS[0] > 0:
                self.buy_signal = 'buy_signal_CDLIDENTICAL3CROWS'
            if self.buy_signal_CDLINNECK[0] > 0:
                self.buy_signal = 'buy_signal_CDLINNECK'
            if self.buy_signal_CDLSHOOTINGSTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLSHOOTINGSTAR'
            if self.buy_signal_CDLSTALLEDPATTERN[0] > 0:
                self.buy_signal = 'buy_signal_CDLSTALLEDPATTERN'
            if self.buy_signal_CDLTHRUSTING[0] > 0:
                self.buy_signal = 'buy_signal_CDLTHRUSTING'
            if self.buy_signal_CDLTRISTAR[0] > 0:
                self.buy_signal = 'buy_signal_CDLTRISTAR'
            if self.buy_signal_CDLUPSIDEGAP2CROWS[0] > 0:
                self.buy_signal = 'buy_signal_CDLUPSIDEGAP2CROWS'

            if self.buy_signal != None:
                self.buy(size=buy_size)  # 满仓买入
                self.have_position = True
                self.buyday = 0
                # 修复：buy_price 用次日开盘价（与 pattern_scan.py 和 baseStrategy.py 一致，原 close[0] 会导致3%止损计算基准不一致）
                self.buy_price = self.data.open[0]  # 记录买入价，用于3%固定止损
                if str(self.datas[0].datetime.date(0)) == self.p.track_date:
                    try:
                        performace_list = self.buy_strategy_performance_df[self.buy_strategy_performance_df['策略名称'] == ('buy_' + self.buy_signal[11:])][['交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']].values[0]
                    except IndexError:
                        performace_list = None  # CSV中无该策略记录，跳过
                    # 买入信号筛选：胜率>50% 且 收益率>0%（与 strategy_signals.py 一致，移除夏普>0.1的额外阈值）
                    if performace_list is not None and performace_list[1] > 50 and performace_list[2] > 0:
                        logging.info('买入原因：' + self.stock_data_df[self.stock_data_df['策略代码'] == self.buy_signal[11:]]['策略名称'].values[0] + '(' + self.buy_signal + ')',
                                                extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
                        logging.info('触发形态：' + self.stock_data_df[self.stock_data_df['策略代码'] == self.buy_signal[11:]]['触发条件'].values[0] + '(' + self.buy_signal + ')',
                                                extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
                        logging.info('策略在该股的交易次数：' + str(round(performace_list[0], 0))
                                     + ', 胜率(%)：' + str(round(performace_list[1],2) )
                                     + ', 简易收益率(%)：' + str(round(performace_list[2],2))
                                     + ', 夏普比率：' + str(round(performace_list[3],3))
                                     + ', 最大回撤(%)：' + str(round(performace_list[4], 2)) + '\n',
                                    extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})

                    # 检查是否达到卖出天数，并且当前有持仓，卖出全部持仓
        # 3%固定止损：买入后亏损达到3%立即卖出（仅对买入信号触发的持仓，卖出信号不加）
        elif self.have_position and self.buy_price > 0 and (self.data.close[0] / self.buy_price - 1) <= -0.03:
            self.sell(size=self.position.size)
            self.buyday = 0
            self.have_position = False
            if str(self.datas[0].datetime.date(0)) == self.p.track_date:
                if self.p.to_log:
                    logging.info('卖出提示：触发3%%固定止损（买入价%.2f，当前价%.2f，跌幅%.2f%%）\n' % (
                        self.buy_price, self.data.close[0],
                        (self.data.close[0] / self.buy_price - 1) * 100),
                        extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
            self.buy_price = 0.0  # 重置买入价
        elif self.buyday == self.p.observe_day and self.have_position:
            self.sell(size=self.position.size) # 卖出全部持仓
            self.buyday = 0 # 重新计算买入天数
            self.have_position = False
            if str(self.datas[0].datetime.date(0)) == self.p.track_date:
                if self.p.to_log:
                    logging.info('卖出提示：达到买入天数：' + str(self.p.observe_day) + '天\n',
                                 extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})

        # 检查是否有卖出信号，并且当前有持仓，卖出全部持仓
        elif self.have_position:
            self.sell_signal = None

            if self.sell_signal_CDL2CROWS[0] < 0:
                self.sell_signal = 'sell_signal_CDLADVANCEBLOCK'
            if self.sell_signal_CDL3BLACKCROWS[0] < 0:
                self.sell_signal ='sell_signal_CDL3BLACKCROWS'
            if self.sell_signal_CDL3LINESTRIKE[0] < 0:
                self.sell_signal ='sell_signal_CDL3LINESTRIKE'
            if self.sell_signal_CDL3STARSINSOUTH[0] < 0:
                self.sell_signal ='sell_signal_CDL3STARSINSOUTH'
            if self.sell_signal_CDL3WHITESOLDIERS[0] < 0:
                self.sell_signal ='sell_signal_CDL3WHITESOLDIERS'
            if self.sell_signal_CDLABANDONEDBABY[0] < 0:
                self.sell_signal ='sell_signal_CDLABANDONEDBABY'
            if self.sell_signal_CDLADVANCEBLOCK[0] < 0:
                self.sell_signal ='sell_signal_CDLADVANCEBLOCK'
            if self.sell_signal_CDLBREAKAWAY[0] < 0:
                self.sell_signal ='sell_signal_CDLBREAKAWAY'
            if self.sell_signal_CDLCONCEALBABYSWALL[0] < 0:
                self.sell_signal ='sell_signal_CDLCONCEALBABYSWALL'
            if self.sell_signal_CDLDARKCLOUDCOVER[0] < 0:
                self.sell_signal ='sell_signal_CDLDARKCLOUDCOVER'
            if self.sell_signal_CDLEVENINGDOJISTAR[0] < 0:
                self.sell_signal ='sell_signal_CDLEVENINGDOJISTAR'
            if self.sell_signal_CDLEVENINGSTAR[0] < 0:
                self.sell_signal ='sell_signal_CDLEVENINGSTAR'
            if self.sell_signal_CDLHANGINGMAN[0] < 0:
                self.sell_signal ='sell_signal_CDLHANGINGMAN'
            if self.sell_signal_CDLIDENTICAL3CROWS[0] < 0:
                self.sell_signal ='sell_signal_CDLIDENTICAL3CROWS'
            if self.sell_signal_CDLINNECK[0] < 0:
                self.sell_signal ='sell_signal_CDLINNECK'
            if self.sell_signal_CDLSHOOTINGSTAR[0] < 0:
                self.sell_signal ='sell_signal_CDLSHOOTINGSTAR'
            if self.sell_signal_CDLSTALLEDPATTERN[0] < 0:
                self.sell_signal ='sell_signal_CDLSTALLEDPATTERN'
            if self.sell_signal_CDLTHRUSTING[0] < 0:
                self.sell_signal ='sell_signal_CDLTHRUSTING'
            if self.sell_signal_CDLTRISTAR[0] < 0:
                self.sell_signal ='sell_signal_CDLTRISTAR'
            if self.sell_signal_CDLUPSIDEGAP2CROWS[0] < 0:
                self.sell_signal ='sell_signal_CDLUPSIDEGAP2CROWS'

            #可能的卖出信号
            if self.sell_signal_CDL3INSIDE[0] < 0:
                self.sell_signal = 'sell_signal_CDL3INSIDE'
            if self.sell_signal_CDL3OUTSIDE[0] < 0:
                self.sell_signal = 'sell_signal_CDL3OUTSIDE'
            if self.sell_signal_CDLBELTHOLD[0] < 0:
                self.sell_signal = 'sell_signal_CDLBELTHOLD'
            if self.sell_signal_CDLMARUBOZU[0] < 0:
                self.sell_signal = 'sell_signal_CDLMARUBOZU'
            if self.sell_signal_CDLCOUNTERATTACK[0] < 0:
                self.sell_signal = 'sell_signal_CDLCOUNTERATTACK'
            if self.sell_signal_CDLDOJI[0] < 0:
                self.sell_signal = 'sell_signal_CDLDOJI'
            if self.sell_signal_CDLDOJISTAR[0] < 0:
                self.sell_signal = 'sell_signal_CDLDOJISTAR'
            if self.sell_signal_CDLDRAGONFLYDOJI[0] < 0:
                self.sell_signal = 'sell_signal_CDLDRAGONFLYDOJI'
            if self.sell_signal_CDLENGULFING[0] < 0:
                self.sell_signal = 'sell_signal_CDLENGULFING'
            if self.sell_signal_CDLGAPSIDESIDEWHITE[0] < 0:
                self.sell_signal = 'sell_signal_CDLGAPSIDESIDEWHITE'
            if self.sell_signal_CDLGRAVESTONEDOJI[0] < 0:
                self.sell_signal = 'sell_signal_CDLGRAVESTONEDOJI'
            if self.sell_signal_CDLHAMMER[0] < 0:
                self.sell_signal = 'sell_signal_CDLHAMMER'
            if self.sell_signal_CDLHARAMI[0] < 0:
                self.sell_signal = 'sell_signal_CDLHARAMI'
            if self.sell_signal_CDLHARAMICROSS[0] < 0:
                self.sell_signal = 'sell_signal_CDLHARAMICROSS'
            if self.sell_signal_CDLHIGHWAVE[0] < 0:
                self.sell_signal = 'sell_signal_CDLHIGHWAVE'
            if self.sell_signal_CDLHIKKAKE[0] < 0:
                self.sell_signal = 'sell_signal_CDLHIKKAKE'
            if self.sell_signal_CDLHIKKAKEMOD[0] < 0:
                self.sell_signal = 'sell_signal_CDLHIKKAKEMOD'
            if self.sell_signal_CDLHOMINGPIGEON[0] < 0:
                self.sell_signal = 'sell_signal_CDLHOMINGPIGEON'
            if self.sell_signal_CDLINVERTEDHAMMER[0] < 0:
                self.sell_signal = 'sell_signal_CDLINVERTEDHAMMER'
            if self.sell_signal_CDLKICKING[0] < 0:
                self.sell_signal = 'sell_signal_CDLKICKING'
            if self.sell_signal_CDLKICKINGBYLENGTH[0] < 0:
                self.sell_signal = 'sell_signal_CDLKICKINGBYLENGTH'
            if self.sell_signal_CDLMATCHINGLOW[0] < 0:
                self.sell_signal = 'sell_signal_CDLMATCHINGLOW'
            if self.sell_signal_CDLMORNINGDOJISTAR[0] < 0:
                self.sell_signal = 'sell_signal_CDLMORNINGDOJISTAR'
            if self.sell_signal_CDLMORNINGSTAR[0] < 0:
                self.sell_signal = 'sell_signal_CDLMORNINGSTAR'
            if self.sell_signal_CDLPIERCING[0] < 0:
                self.sell_signal = 'sell_signal_CDLPIERCING'
            if self.sell_signal_CDLRICKSHAWMAN[0] < 0:
                self.sell_signal = 'sell_signal_CDLRICKSHAWMAN'
            if self.sell_signal_CDLRISEFALL3METHODS[0] < 0:
                self.sell_signal = 'sell_signal_CDLRISEFALL3METHODS'
            if self.sell_signal_CDLSEPARATINGLINES[0] < 0:
                self.sell_signal = 'sell_signal_CDLSEPARATINGLINES'
            if self.sell_signal_CDLSHORTLINE[0] < 0:
                self.sell_signal = 'sell_signal_CDLSHORTLINE'
            if self.sell_signal_CDLSPINNINGTOP[0] < 0:
                self.sell_signal = 'sell_signal_CDLSPINNINGTOP'
            if self.sell_signal_CDLSTICKSANDWICH[0] < 0:
                self.sell_signal = 'sell_signal_CDLSTICKSANDWICH'
            if self.sell_signal_CDLTAKURI[0] < 0:  # 处于下降趋势
                self.sell_signal = 'sell_signal_CDLTAKURI'
            if self.sell_signal_CDLTASUKIGAP[0] < 0:
                self.sell_signal ='sell_signal_CDLTASUKIGAP'
            if self.sell_signal_CDLUNIQUE3RIVER[0] < 0:
                self.sell_signal = 'sell_signal_CDLUNIQUE3RIVER'
            if self.sell_signal_CDLXSIDEGAP3METHODS[0] < 0:
                self.sell_signal = 'sell_signal_CDLXSIDEGAP3METHODS'

            if self.sell_signal != None:
                self.sell(size=self.position.size)  # 卖出全部持仓
                self.buyday = 0  # 重新计算买入天数
                self.have_position = False
                if str(self.datas[0].datetime.date(0)) == self.p.track_date:
                    if self.p.to_log:
                        try:
                            # 修复：CSV中策略名称格式为 'sell_CDLXXX'，需加 'sell_' 前缀（与 signal_update.py 输出一致）
                            performace_list = self.sell_strategy_performance_df[self.sell_strategy_performance_df['策略名称'] == ('sell_' + self.sell_signal[12:])][
                                ['交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']].values[0]
                        except IndexError:
                            performace_list = None  # CSV中无该策略记录，跳过
                        # 卖出信号筛选：胜率>50% 且 收益率>0%（与 strategy_signals.py 一致，原40%阈值已统一为50%）
                        if performace_list is not None and performace_list[1] > 50 and performace_list[2] > 0:
                            logging.info('卖出原因：' + self.stock_data_df[self.stock_data_df['策略代码'] == ('sell_' + self.sell_signal[12:])]['策略名称'].values[0] + '(' + self.sell_signal + ')',
                                         extra={'track_date': self.p.track_date, 'code': self.p.code, 'code_name': self.p.code_name})
                            logging.info('策略在该股的交易次数：' + str(round(performace_list[0], 0))
                                         + ', 胜率(%)：' + str(round(performace_list[1], 2))
                                         + ', 简易收益率(%)：' + str(round(performace_list[2], 2))
                                         + ', 夏普比率：' + str(round(performace_list[3], 3))
                                         + ', 最大回撤(%)：' + str(round(performace_list[4], 2)) + '\n',
                                         extra={'track_date': self.p.track_date, 'code': self.p.code,
                                                'code_name': self.p.code_name})
                    else:
                        self.log('卖出提示：' + self.sell_signal)

        # 跟踪买入天数
        if self.have_position:
            self.buyday += 1


def _track_one_code(task_args):
    """进程池worker：跟踪单个标的。模块级函数，可被pickle。

    子进程内重新设置logger，确保日志写入正确的跟踪日志文件。
    多进程同时写同一日志文件时，POSIX对小于PIPE_BUF(4KB)的write是原子的，
    单条日志记录通常小于4KB，因此不会交错。每条记录带时间戳和code，可追溯。
    """
    (code, code_name, get_new_data, save_data, start_date, end_date,
     observe_day, folder, to_log, track_date, cautious) = task_args
    try:
        # 清空root logger的handler，避免fork模式下继承父进程handler导致重复输出
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        # 子进程内重新设置logger，确保FileHandler指向正确的跟踪日志文件
        tools.set_logger(track_date)
        main.run_pattern_recognition_Strategy(code,
                                               start_date,
                                               end_date,
                                               pattern_category='united',  # _inUp
                                               pattern_name='combine',
                                               pattern_type='tracking',
                                               observe_day=observe_day,
                                               plot=False,
                                               log=False,
                                               get_new_data=get_new_data,
                                               save_data=save_data,
                                               print_performance=False,
                                               data_folder_dir=folder,
                                               track_date=track_date,
                                               to_log=to_log,
                                               code_name=code_name,
                                               cautious=cautious)
        return {'code': code, 'success': True, 'error': ''}
    except Exception as e:
        import traceback
        # 子进程内用 print 输出到 stderr（被web_app的subprocess捕获到log_streams）
        # 不用 logging.exception，因为子进程的 logger handler 状态可能不正确
        tb = traceback.format_exc()
        print(f'跟踪失败: {code} - {e}\n{tb}', flush=True)
        return {'code': code, 'success': False, 'error': str(e), 'traceback': tb}


def _prefetch_kline_data(code_list, start_date, end_date, folder):
    """父进程并行预拉取日K数据到本地目录。

    使用 ThreadPoolExecutor 并行调用 akshare（IO 密集型，线程并行安全），
    并发数由 config.TRACKING_PREFETCH_WORKERS 控制（默认 4）。
    多进程子进程中调用 akshare 会因网络 IO 阻塞导致进程卡死，
    因此子进程禁止网络访问，由父进程预拉取到本地 CSV。

    Args:
        code_list: 标的代码列表
        start_date/end_date: YYYYMMDD
        folder: 保存目录（tracking 用的 folder，如 '数据/A股/每日跟踪/'）
    Returns:
        dict: {code: True/False} 表示预拉取是否成功
    """
    import data_source
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    total = len(code_list)
    if total == 0:
        return results

    workers = max(1, min(getattr(config, 'TRACKING_PREFETCH_WORKERS', 4), 16))
    print(f'[预拉取] 启动并行预拉取：共 {total} 个标的，{workers} 线程')

    def _fetch_one(code):
        try:
            df = data_source.get_kline_df(code, start_date, end_date,
                                           prefer_local=True, allow_network=True)
            if df is not None and not df.empty:
                data_source.save_kline_to_local(code, df, folder)
                return code, True
            return code, False
        except Exception as e:
            print(f'[预拉取] {code} 失败: {e}')
            return code, False

    ok = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, c): c for c in code_list}
        for future in as_completed(futures):
            code, success = future.result()
            results[code] = success
            if success:
                ok += 1
            done += 1
            if done % 50 == 0 or done == total:
                print(f'[预拉取] 进度：{done}/{total}（成功 {ok}）')
    print(f'[预拉取] 完成：共 {total}，成功 {ok}，失败 {total - ok}')
    return results


def tracking(code_list, get_new_data, save_data, start_date, track_date, observe_day, folder, to_log, cautious=False):
    stock_data_df = _load_stock_data()

    end_date = track_date.replace('-', '')
    total = len(code_list)
    print(f'跟踪日：{track_date}，共 {total} 个标的，数据目录：{folder}')

    # 父进程预拉取数据：多进程子进程禁止网络访问
    # - get_new_data=True：全量预拉取
    # - get_new_data=False：仅对本地无 csv 的标的预拉取（补全式），避免子进程回退 tushare 被限流
    worker_get_new_data = get_new_data
    if total > 0:
        if get_new_data:
            print('[tracking] 父进程预拉取数据中（全量）...')
            _prefetch_kline_data(code_list, start_date, end_date, folder)
            worker_get_new_data = False
        else:
            # 检查哪些标的本地无 csv，仅补全这些标的
            import data_source as _ds
            missing = [c for c in code_list if _ds._find_local_data(str(c)) is None]
            if missing:
                print(f'[tracking] 父进程补全预拉取 {len(missing)} 个本地无数据的标的...')
                _prefetch_kline_data(missing, start_date, end_date, folder)
            # 子进程仍保持 get_new_data=False，只读本地
            worker_get_new_data = False

    # 构建任务列表：父进程一次性读取stock_data.csv，避免每个worker重复读取
    tasks = []
    for code in code_list:
        name_match = stock_data_df[stock_data_df['ts_code'] == code]['name']
        code_name = name_match.values[0] if len(name_match) > 0 else code
        tasks.append((code, code_name, worker_get_new_data, save_data, start_date, end_date,
                      observe_day, folder, to_log, track_date, cautious))

    if total == 0:
        print('无标的需跟踪')
        return

    # 决定进程数：优先 TRACKING_POOL_WORKERS，回退 SCAN_WORKERS；0=自动（CPU核心数）；1=串行
    workers = getattr(config, 'TRACKING_POOL_WORKERS', None) or config.SCAN_WORKERS
    if workers <= 0:
        workers = multiprocessing.cpu_count()
    workers = min(workers, total, getattr(config, 'MAX_WORKERS', 4))
    workers = max(1, workers)
    worker_timeout = getattr(config, 'WORKER_TIMEOUT_SECONDS', 30)
    print(f'[tracking] 启动跟踪：{total} 个标的，{workers} 进程并行')

    # 进度打印频率：小批量每条打，大批量每 5% 打一次
    progress_step = max(1, total // 20)

    done = 0
    failed = 0
    if workers <= 1:
        # 串行模式（与原逻辑一致，便于对比验证）
        for task_args in tasks:
            r = _track_one_code(task_args)
            done += 1
            if not r['success']:
                failed += 1
                print(f'Failed: {r["code"]} - {r["error"]}')
            if done % progress_step == 0 or done == total:
                print(f'进度：{done}/{total}（{done*100//total}%，失败 {failed}）')
    else:
        # 并行模式：多进程同时跟踪多个标的
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_code = {executor.submit(_track_one_code, t): t[0] for t in tasks}
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    r = future.result(timeout=worker_timeout)
                except TimeoutError:
                    r = {'code': code, 'success': False, 'error': f'任务超时（{worker_timeout}秒）'}
                except Exception as e:
                    r = {'code': code, 'success': False, 'error': str(e)}
                done += 1
                if not r['success']:
                    failed += 1
                    print(f'Failed: {r["code"]} - {r["error"]}')
                if done % progress_step == 0 or done == total:
                    print(f'进度：{done}/{total}（{done*100//total}%，失败 {failed}）', flush=True)

    print(f'跟踪完成：共 {total} 个标的，成功 {total - failed}，失败 {failed}')
    # 输出结构化信号汇总
    _summarize_tracking_signals(track_date, code_list, total, failed)


def _reset_summary_file(track_date):
    """删除指定跟踪日的 summary.json，避免与上次结果合并。

    在 run_tracking 入口调用，确保 mode='all' 多次调用 tracking() 时累积的是本次结果，
    而不是上一次运行的残留数据。
    """
    summary_file = 'log/' + track_date.replace('-', '') + '_summary.json'
    try:
        if os.path.exists(summary_file):
            os.remove(summary_file)
            print(f'[tracking] 已清空旧 summary: {summary_file}')
    except Exception as e:
        print(f'[tracking] 清空 summary 失败: {e}')


def _summarize_tracking_signals(track_date, code_list, total, failed, append=True):
    """解析当日跟踪日志，输出结构化信号汇总到 JSON 文件。

    在 tracking() 末尾调用，仅汇总本次 code_list 内的标的信号。
    JSON 文件路径：log/{YYYYMMDD}_summary.json，供 web_app 读取后传给前端结构化展示。
    同时输出简短的 print 状态行到 output 流。

    Args:
        append: True 时与已有 summary.json 合并（mode='all' 多次调用 tracking 场景）；
                False 时覆盖。合并时标的列表和信号列表累加，总数/成败数累加。
    """
    import re
    import json

    # 加载 stock_data 用于查 code_name
    _stock_data_df = _load_stock_data()
    # 全球指数代码不在 stock_data.csv 中，单独维护名称映射
    _index_name_map = {
        'DJI': '道琼斯', 'FCHI': '法国CAC40', 'SPX': '标普500',
        'N225': '日经225', 'GDAXI': '德国DAX',
        '000300.SH': '沪深300', '399006.SZ': '创业板指',
    }

    def _get_code_name(code):
        """从 stock_data.csv 查 code 对应的名称，指数用内置映射，找不到返回空字符串。"""
        if code in _index_name_map:
            return _index_name_map[code]
        if _stock_data_df.empty:
            return ''
        try:
            name_match = _stock_data_df[_stock_data_df['ts_code'] == code]['name']
            return str(name_match.values[0]) if len(name_match) > 0 else ''
        except Exception:
            return ''

    log_file = 'log/' + track_date.replace('-', '') + '_tracking.log'
    summary_file = 'log/' + track_date.replace('-', '') + '_summary.json'

    # 标的分类：指数 vs 个股
    index_codes = [c for c in code_list if len(str(c)) < 9 or c in ['000300.SH', '399006.SZ']]
    stock_codes = [c for c in code_list if c not in index_codes]

    # 解析日志，按 code 聚合信号
    # 日志行格式：asctime 跟踪日：track_date - code - code_name: message
    signals = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    m = re.match(r'^\S+\s+\S+\s+跟踪日：\S+\s+-\s+(\S+)\s+-\s+(.+?):\s+(.+)$', line)
                    if not m:
                        continue
                    code, code_name, msg = m.groups()
                    if code not in code_list:
                        continue
                    if code not in signals:
                        signals[code] = {'name': code_name, 'type': 'index' if code in index_codes else 'stock', 'buy': [], 'sell': []}

                    if msg.startswith('买入原因：'):
                        pat_m = re.search(r'\(buy_signal_(\w+)\)', msg)
                        pattern = pat_m.group(1) if pat_m else 'unknown'
                        signals[code]['buy'].append({'pattern': pattern, 'win_rate': None, 'return': None})
                    elif msg.startswith('策略在该股的交易次数'):
                        if signals[code]['buy']:
                            last = signals[code]['buy'][-1]
                            win_m = re.search(r'胜率\(%\)：([\d.-]+)', msg)
                            ret_m = re.search(r'简易收益率\(%\)：([\d.-]+)', msg)
                            if win_m:
                                last['win_rate'] = float(win_m.group(1))
                            if ret_m:
                                last['return'] = float(ret_m.group(1))
                    elif msg.startswith('卖出提示：'):
                        if '达到买入天数' in msg:
                            signals[code]['sell'].append({'reason': '达到持有天数', 'pattern': None})
                        else:
                            pat_m = re.search(r'sell_signal_(\w+)', msg)
                            pattern = pat_m.group(1) if pat_m else 'unknown'
                            signals[code]['sell'].append({'reason': '形态卖出', 'pattern': pattern})
        except Exception as e:
            print(f'[汇总] 解析日志失败: {e}')

    # 收集买入/卖出信号（扁平化列表）
    buy_list = []
    sell_list = []
    codes_with_signals = []
    for code, sig in signals.items():
        if sig['buy'] or sig['sell']:
            codes_with_signals.append(code)
        for s in sig['buy']:
            buy_list.append({'code': code, 'name': sig['name'], 'type': sig['type'],
                             'pattern': s['pattern'], 'win_rate': s['win_rate'], 'return': s['return']})
        for s in sig['sell']:
            sell_list.append({'code': code, 'name': sig['name'], 'type': sig['type'],
                              'reason': s['reason'], 'pattern': s['pattern']})

    # 计算平均胜率
    wins = [s['win_rate'] for s in buy_list if s['win_rate'] is not None]
    avg_win_rate = round(sum(wins) / len(wins), 2) if wins else None

    summary = {
        'track_date': track_date,
        'total': total,
        'success': total - failed,
        'failed': failed,
        'index_count': len(index_codes),
        'stock_count': len(stock_codes),
        'codes_with_signals': len(codes_with_signals),
        'buy_count': len(buy_list),
        'sell_count': len(sell_list),
        'avg_win_rate': avg_win_rate,
        'buy_signals': buy_list,
        'sell_signals': sell_list,
        'has_signals': len(buy_list) + len(sell_list) > 0,
        # 标的列表（含名称），供前端点击统计数字查看明细
        'index_codes': [{'code': c, 'name': _get_code_name(c)} for c in index_codes],
        'stock_codes': [{'code': c, 'name': _get_code_name(c)} for c in stock_codes],
    }

    # 写入 JSON 文件（append 模式下与已有 summary 合并）
    try:
        if append and os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                # 累加统计数字
                summary['total'] = old.get('total', 0) + total
                summary['success'] = old.get('success', 0) + (total - failed)
                summary['failed'] = old.get('failed', 0) + failed
                summary['index_count'] = old.get('index_count', 0) + len(index_codes)
                summary['stock_count'] = old.get('stock_count', 0) + len(stock_codes)
                # 合并标的列表（按 code 去重）
                old_idx_codes = {item['code']: item for item in old.get('index_codes', [])}
                old_stk_codes = {item['code']: item for item in old.get('stock_codes', [])}
                for c in index_codes:
                    if c not in old_idx_codes:
                        old_idx_codes[c] = {'code': c, 'name': _get_code_name(c)}
                for c in stock_codes:
                    if c not in old_stk_codes:
                        old_stk_codes[c] = {'code': c, 'name': _get_code_name(c)}
                summary['index_codes'] = list(old_idx_codes.values())
                summary['stock_codes'] = list(old_stk_codes.values())
                # 合并信号列表
                summary['buy_signals'] = old.get('buy_signals', []) + buy_list
                summary['sell_signals'] = old.get('sell_signals', []) + sell_list
                summary['buy_count'] = len(summary['buy_signals'])
                summary['sell_count'] = len(summary['sell_signals'])
                summary['codes_with_signals'] = len(set(
                    [s['code'] for s in summary['buy_signals']] +
                    [s['code'] for s in summary['sell_signals']]
                ))
                summary['has_signals'] = summary['buy_count'] + summary['sell_count'] > 0
                # 重新计算平均胜率
                all_wins = [s['win_rate'] for s in summary['buy_signals']
                            if s.get('win_rate') is not None]
                summary['avg_win_rate'] = round(sum(all_wins) / len(all_wins), 2) if all_wins else None
                print(f'[汇总] 合并已有 summary.json：本次 +{total}，累计 {summary["total"]} 个标的')
            except Exception as merge_err:
                print(f'[汇总] 合并旧 summary 失败，覆盖写入: {merge_err}')

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[汇总] 写入 summary.json 失败: {e}')

    # 简短状态行输出到 output 流（详细展示由前端从 summary.json 渲染）
    if not summary['has_signals']:
        print(f'[汇总] 本次跟踪 {total} 个标的（指数 {len(index_codes)}，个股 {len(stock_codes)}），无信号输出。')
    else:
        win_str = f'，平均胜率 {avg_win_rate}%' if avg_win_rate is not None else ''
        print(f'[汇总] 本次跟踪 {total} 个标的，共 {len(buy_list) + len(sell_list)} 个信号'
              f'（买入 {len(buy_list)}，卖出 {len(sell_list)}{win_str}）。详见上方信号卡片。')


def run_tracking(min_mv=None, max_mv=None):
    """运行跟踪任务。

    Args:
        min_mv: 总市值下限（单位：万元），None 表示不限制
        max_mv: 总市值上限（单位：万元），None 表示不限制
    """
    global track_date, start_date, observe_day, get_new_data, save_data, to_log, mode, cautious
    tools.set_logger(track_date)
    index_dir = str(config.INDEX_DIR) + '/'
    a_market_dir = str(config.A_MARKET_DIR) + '/'
    stock_data_df = _load_stock_data()
    index_list = ['DJI', 'FCHI', 'SPX', 'N225', 'GDAXI', '000300.SH', '399006.SZ']

    # 按市值范围筛选 A 股个股；min_mv/max_mv 为 None 时该侧不限制
    if stock_data_df.empty:
        stock_list = []
        print(f'[tracking] 警告: stock_data.csv 为空，A 股个股跟踪范围 0 只')
    else:
        mask = pd.Series([True] * len(stock_data_df), index=stock_data_df.index)
        if min_mv is not None:
            mask &= stock_data_df['total_mv'] >= min_mv
        if max_mv is not None:
            mask &= stock_data_df['total_mv'] <= max_mv
        stock_list = stock_data_df[mask]['ts_code'].tolist()
        mv_desc = f'市值范围 '
        if min_mv is not None and max_mv is not None:
            mv_desc += f'[{min_mv/10000:.0f}亿, {max_mv/10000:.0f}亿]'
        elif min_mv is not None:
            mv_desc += f'>= {min_mv/10000:.0f}亿'
        elif max_mv is not None:
            mv_desc += f'<= {max_mv/10000:.0f}亿'
        else:
            mv_desc = '全市场（无市值限制）'
        print(f'[tracking] A 股个股筛选条件：{mv_desc}，共 {len(stock_list)} 只')

    if mode == 'all':
        # 清空旧 summary，避免与上次跟踪结果合并
        _reset_summary_file(track_date)
        tracking(index_list, get_new_data, save_data, start_date, track_date, observe_day, index_dir+'每日跟踪/', to_log, cautious=cautious)
        tracking(stock_list, get_new_data, save_data, start_date, track_date, observe_day, a_market_dir+'每日跟踪/', to_log, cautious=cautious)
    elif mode == 'index':
        _reset_summary_file(track_date)
        tracking(index_list, get_new_data, save_data, start_date, track_date, observe_day, index_dir+'每日跟踪/', to_log, cautious=cautious)
    elif mode == 'stock':
        _reset_summary_file(track_date)
        tracking(stock_list, get_new_data, save_data, start_date, track_date, observe_day, a_market_dir+'每日跟踪/', to_log, cautious=cautious)


if __name__ == '__main__':
    observe_day = 2
    start_date = '20250101'
    track_date = '2025-03-31'
    get_new_data = True
    save_data = True
    to_log = False
    cautious = False
    from config import config
    mode = 'stock'
    run_tracking()




