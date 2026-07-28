import numpy as np
import pandas as pd
import tushare as ts
import time
import logging
import optunity
import optunity.metrics
import main
import backtrader as bt

from config import config

# 模块级logger，便于追溯API调用问题
logger = logging.getLogger('trader_system')

try:
    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()
except Exception as _ts_init_err:
    # 多进程子进程环境下 token 文件可能不可用，或免费 token 权限不足
    # pro=None 时由 data_source（akshare）兜底
    logger.warning(f'[tools] tushare 初始化失败（将使用 akshare 兜底）: {_ts_init_err}')
    pro = None


def _call_tushare_with_retry(api_fn, *args, max_retries=3, base_sleep=60, **kwargs):
    """包装Tushare API调用，遇到频率限制自动重试退避

    Tushare对部分接口限制每分钟1次，触发限制时抛出异常消息包含"每分钟"或"频率"。
    遇到限制时等待 base_sleep 秒后重试，最多 max_retries 次。

    Args:
        api_fn: tushare API函数（如 pro.index_daily / pro.index_global / ts.pro_bar）
        max_retries: 最大重试次数
        base_sleep: 基础等待秒数（指数退避：base_sleep * (retry+1)）
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return api_fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_msg = str(e)
            # 识别频率限制类错误
            is_rate_limit = any(k in err_msg for k in ['每分钟', '频率', 'limit', '次数', '抱歉'])
            if not is_rate_limit or attempt == max_retries:
                raise
            sleep_secs = base_sleep * (attempt + 1)
            logger.warning(f'[Tushare] API频率限制，{sleep_secs}秒后重试（第 {attempt+1}/{max_retries} 次）: {err_msg[:80]}')
            time.sleep(sleep_secs)
    raise last_err

def list_subtraction(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length for subtraction.")

    result = []
    for i in range(len(list1)):
        result.append(round(list1[i] - list2[i],2))

    return result

def get_stock_list():
    # 获取A股股票列表
    stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    #stock_list.to_csv(folder_name + 'stock_list.csv', index=False)

    # 打印股票列表的前几行，以查看数据结构
    print(stock_list.head())

    return stock_list

def get_stock_data(code, trade_date):
    df = pro.daily_basic(ts_code=code, trade_date=trade_date,
                         fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,circ_mv')
    return df


# 更新指定交易日的股票数据。
def renew_stock_data():
    trade_date = '20250113'
    folder_name = '数据/A股/'
    stock_list_df = get_stock_list()
    stock_list = stock_list_df['ts_code'].tolist()  # ['000001.SZ']
    '''
    # 为stock_list_df新建列
    stock_list_df['turnover_rate'] = np.nan
    stock_list_df['volume_ratio'] = np.nan
    stock_list_df['pe'] = np.nan
    stock_list_df['pb'] = np.nan
    stock_list_df['total_mv'] = np.nan
'''
    for stock_code in stock_list:
        try:
            stock_data_df = get_stock_data(stock_code, trade_date)
            # 将stock_data_df获取的数据按照ts_code加到stock_list_df对应的列中
            stock_list_df.loc[stock_list_df['ts_code'] == stock_code, 'turnover_rate'] = \
            stock_data_df['turnover_rate'].values[0]
            stock_list_df.loc[stock_list_df['ts_code'] == stock_code, 'volume_ratio'] = \
            stock_data_df['volume_ratio'].values[0]
            stock_list_df.loc[stock_list_df['ts_code'] == stock_code, 'total_mv'] = stock_data_df['circ_mv'].values[0]
            stock_list_df.loc[stock_list_df['ts_code'] == stock_code, 'pb'] = stock_data_df['pb'].values[0]
            stock_list_df.loc[stock_list_df['ts_code'] == stock_code, 'pe'] = stock_data_df['pe'].values[0]
            time.sleep(0.3)

        except Exception as e:
            print(stock_code, e)
    stock_list_df.to_csv(folder_name + 'stock_data.csv', index=False)

def set_logger(track_date='auto'):
    """配置跟踪日志。

    使用 Filter 为每条 record 注入 track_date/code/code_name 默认值，
    避免未传 extra 的日志（如 tools.py 的 logger.info）触发 KeyError。
    """
    import logging
    track_date_raw = track_date
    track_date = track_date.replace('-', '')    # 去除日期中的分隔符

    class _TrackDefaultsFilter(logging.Filter):
        """为未传 extra 的 record 注入默认字段，兼容带 track_date 的 format。"""
        def filter(self, record):
            for attr, default in (('track_date', track_date_raw),
                                  ('code', '-'),
                                  ('code_name', '-')):
                if not hasattr(record, attr):
                    setattr(record, attr, default)
            return True

    _fmt = '%(asctime)s 跟踪日：%(track_date)s - %(code)s - %(code_name)s: %(message)s'
    logging.basicConfig(filename='log/' + track_date + '_tracking.log', level=logging.INFO,
                        format=_fmt, datefmt='%Y-%m-%d %H:%M')
    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置控制台处理器的日志级别
    console_handler.setFormatter(logging.Formatter(_fmt))

    # 获取根日志记录器，添加 Filter + 控制台处理器
    root_logger = logging.getLogger()
    _flt = _TrackDefaultsFilter()
    # 给已有的 FileHandler 也加 Filter（basicConfig 已加了一个 FileHandler）
    for h in list(root_logger.handlers):
        h.addFilter(_flt)
    console_handler.addFilter(_flt)
    root_logger.addHandler(console_handler)


# 自动寻找最优参数
# TODO: 添加夏普比率和最大回撤作为参数优化指标
def opt_params(num_evals, short_period, long_period):
    # 定义参数空间
    code = 'DJI'
    space = {
        'short_period': [short_period[0], short_period[1]],
        'long_period': [long_period[0], long_period[1]]
    }
    opt = optunity.maximize(
        f=main.run_DoubleMAStrategy,
        num_evals=num_evals,  # 回测x次 获取最优参数
        solver_name='particle swarm',  # 使用粒子群优化算法
        **space
    )
    optimal_pars, details, _ = opt  # optimal_pars 最优参数组合
    print(optimal_pars)


# 为train data和test data建立数据库，获取数据并分片保存
def save_data(code, start_date, end_date, get_new_data, data_folder_dir='数据/', save_data=True):
    print('获取日线数据：', code)
    # 优先使用 data_source（akshare 免费数据源）
    try:
        import data_source
        df = data_source.get_kline_df(code, start_date, end_date, prefer_local=not get_new_data)
        if df is not None and not df.empty:
            if save_data:
                try:
                    data_source.save_kline_to_local(code, df, data_folder_dir)
                except Exception as _save_err:
                    logger.warning(f'[tools.save_data] 保存 {code} 到本地失败: {_save_err}')
            # 转换为 backtrader 格式并返回（与原 save_data 行为一致）
            df = df.sort_values(by=['trade_date'], ascending=True)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date', drop=True)
            filtered_df = df.loc[start_date:end_date]
            if not filtered_df.empty:
                return
            logger.warning(f'[tools.save_data] data_source 返回 {code} 数据为空，回退到 tushare')
        else:
            logger.info(f'[tools.save_data] data_source 无 {code} 数据，回退到 tushare')
    except Exception as e:
        logger.warning(f'[tools.save_data] data_source 获取 {code} 失败，回退到 tushare: {e}')

    # ===== 以下为原有 tushare 逻辑（fallback） =====
    # pro=None 时（token 失败）直接抛错，避免子进程崩溃污染进程池
    if pro is None:
        raise FileNotFoundError(f'tushare 未初始化（pro=None），且 data_source 无 {code} 数据')
    if code == 'AU100g':
            df = pro.sge_daily(ts_code=code, start_date=start_date, end_date=end_date,
                               fields='ts_code, trade_date, open, high, low, close, vol')
            # 保存数据供后使用
            df.to_csv(data_folder_dir + str(code) + '_daily.csv', index=False)

    elif len(code) < 9:
        #如果重新获取数据
            df = pro.index_global(ts_code=code, start_date=start_date, end_date=end_date)
            # 保存数据供后使用
            df.to_csv(data_folder_dir + str(code) + '_daily.csv', index=False)

    elif code in ['000300.SH', '399006.SZ']:
        df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
        # 保存数据供后使用
        df.to_csv(data_folder_dir + str(code) + '_daily.csv', index=False)
    # 如果代码为国内股票
    else:
        df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start_date, end_date=end_date)
        # 保存数据供后使用
        df.to_csv(data_folder_dir + str(code) + '_daily.csv', index=False)

    df = df.sort_values(by=['trade_date'], ascending=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  # 将日期转换为datetime格式
    df = df.set_index('trade_date', drop=True)  # 将日期设置为索引

    #按照日期范围截取df
    filtered_df = df.loc[start_date:end_date]

    #如果df为空，警告
    if filtered_df.empty:
        raise ValueError("df为空")


# 从tushare获取股票日线数据
def get_data(code, start_date, end_date, get_new_data, daily_folder_dir='数据/', save_data=True):
    '''获取日K数据并构造 backtrader 数据源

    优先使用 data_source（akshare 免费数据源），失败时回退到 tushare。
    '''
    # 优先使用 data_source（akshare 免费数据源，无需 token）
    try:
        import data_source
        # 多进程子进程环境下（get_new_data=False）禁止网络访问，避免子进程卡死
        df = data_source.get_kline_df(code, start_date, end_date,
                                       prefer_local=not get_new_data,
                                       allow_network=get_new_data)
        if df is not None and not df.empty:
            # 保存到本地（如果需要）
            if save_data and get_new_data:
                try:
                    data_source.save_kline_to_local(code, df, daily_folder_dir)
                except Exception as _save_err:
                    logger.warning(f'[tools.get_data] 保存 {code} 到本地失败: {_save_err}')
            # 转换为 backtrader 格式
            df = df.sort_values(by=['trade_date'], ascending=True)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date', drop=True)
            filtered_df = df.loc[start_date:end_date]
            if not filtered_df.empty:
                # 修复：NaN 填充，避免回测时崩溃（与 pattern_scan.py / renew_strategy_performance.py 一致）
                for col in filtered_df.columns:
                    if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        filtered_df[col] = filtered_df[col].ffill().fillna(0)
                # 修复：用列名定位，与 pattern_scan.py / renew_strategy_performance.py 保持一致
                cols = filtered_df.columns.tolist()
                vol_col = 'vol' if 'vol' in cols else ('volume' if 'volume' in cols else -1)
                data = bt.feeds.PandasData(dataname=filtered_df, datetime=None,
                                           open='open', high='high', low='low', close='close',
                                           volume=vol_col, openinterest=-1)
                return data
            logger.warning(f'[tools.get_data] data_source 返回 {code} 数据为空（日期范围 {start_date}-{end_date}），回退到 tushare')
        else:
            logger.info(f'[tools.get_data] data_source 无 {code} 数据，回退到 tushare')
    except Exception as e:
        logger.warning(f'[tools.get_data] data_source 获取 {code} 失败，回退到 tushare: {e}')

    # ===== 以下为原有 tushare 逻辑（fallback） =====
    # pro=None 时（token 失败）直接抛错，避免子进程崩溃污染进程池
    if pro is None:
        raise FileNotFoundError(f'tushare 未初始化（pro=None），且 data_source 无 {code} 数据')
    '''
    # 如果代码为国外指数
    if not get_new_data:
        df = pd.read_csv(daily_folder_dir + str(code) + '_daily.csv')
        if int(end_date) not in df['trade_date'].values.tolist() or int(start_date) not in df['trade_date'].values.tolist():
            raise ValueError("end_date 或 start_date 不在数据中")'''
    if code == 'AU100g':
        if get_new_data:
            df = _call_tushare_with_retry(pro.sge_daily, ts_code=code, start_date=start_date, end_date=end_date,
                                          fields='ts_code, trade_date, open, high, low, close, vol')
            # 保存数据供后使用（统一走 data_source，生成 Parquet + CSV）
            if save_data:
                data_source.save_kline_to_local(code, df, daily_folder_dir)
        else:
            try: # 如果有数据则直接使用
                df = data_source._read_local_data(code, start_date, end_date) if hasattr(data_source, '_read_local_data') else None
                if df is None:
                    df = pd.read_csv(daily_folder_dir + str(code) + '_daily.csv')
            except Exception: # 如果没有数据则重新获取
                df = _call_tushare_with_retry(pro.index_global, ts_code=code, start_date=start_date, end_date=end_date)
                # 保存数据供后使用
                if save_data:
                    data_source.save_kline_to_local(code, df, daily_folder_dir)
    elif len(code) < 9:
        #如果重新获取数据
        if get_new_data:
            df = _call_tushare_with_retry(pro.index_global, ts_code=code, start_date=start_date, end_date=end_date)
            # 保存数据供后使用
            if save_data:
                data_source.save_kline_to_local(code, df, daily_folder_dir)
        else:
            #try: # 如果有数据则直接使用
                df = pd.read_csv(daily_folder_dir + str(code) + '_daily.csv')
            #except: # 如果没有数据则重新获取
            #    df = pro.index_global(ts_code=code, start_date=start_date, end_date=end_date)
                # 保存数据供后使用
            #    df.to_csv(data_folder_dir + str(code) + '_daily.csv', index=False)
                #print("没有数据，已重新获取并保存数据")

    elif code in ['000300.SH', '399006.SZ']:
        if get_new_data:
            df = _call_tushare_with_retry(pro.index_daily, ts_code=code, start_date=start_date, end_date=end_date)
            # 保存数据供后使用
            if save_data:
                data_source.save_kline_to_local(code, df, daily_folder_dir)
        else:
            try: # 优先读本地文件，避免Tushare API频率限制
                df = pd.read_csv(daily_folder_dir + str(code) + '_daily.csv')
            except Exception: # 本地无数据时才调用API
                df = _call_tushare_with_retry(pro.index_daily, ts_code=code, start_date=start_date, end_date=end_date)
                if save_data:
                    data_source.save_kline_to_local(code, df, daily_folder_dir)
    # 如果代码为国内股票
    else:
        if get_new_data:
            df = _call_tushare_with_retry(ts.pro_bar, ts_code=code, adj='qfq', start_date=start_date, end_date=end_date)
            # 保存数据供后使用
            if save_data:
                data_source.save_kline_to_local(code, df, daily_folder_dir)
        else:
            try:
                df = pd.read_csv(daily_folder_dir + str(code) + '_daily.csv')
            except Exception:
                # 多进程子进程禁网模式下不回退 tushare（避免限流拖垮进程池）
                # 父进程预拉取应已补全数据；若仍无数据则直接抛错让该标的失败
                raise FileNotFoundError(f'本地无 {code} 日K数据，且子进程禁网模式不回退 tushare')
    df = df.sort_values(by=['trade_date'], ascending=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  # 将日期转换为datetime格式
    df = df.set_index('trade_date', drop=True)  # 将日期设置为索引

    #按照日期范围截取df
    filtered_df = df.loc[start_date:end_date]

    # 修复：NaN 填充，避免回测时崩溃（与 pattern_scan.py / renew_strategy_performance.py 一致）
    if not filtered_df.empty:
        for col in filtered_df.columns:
            if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                filtered_df[col] = filtered_df[col].ffill().fillna(0)

    # 修复：用列名定位 OHLCV，避免不同数据源列顺序差异导致错位
    # （原代码用位置索引定位，但 tushare/akshare 返回的列顺序不一致，会导致 high/low/close 错位）
    cols = filtered_df.columns.tolist()
    vol_col = 'vol' if 'vol' in cols else ('volume' if 'volume' in cols else -1)
    data = bt.feeds.PandasData(dataname=filtered_df, datetime=None,
                               open='open', high='high', low='low', close='close',
                               volume=vol_col, openinterest=-1)

    #如果df为空，尝试从另一个目录（训练/测试）读取
    if filtered_df.empty and not get_new_data:
        alt_dir = _get_alt_data_dir(daily_folder_dir, code)
        if alt_dir:
            try:
                alt_pq = alt_dir + str(code) + '_daily.parquet'
                alt_csv = alt_dir + str(code) + '_daily.csv'
                if os.path.exists(alt_pq) and os.path.getsize(alt_pq) > 0:
                    alt_df = pd.read_parquet(alt_pq)
                else:
                    alt_df = pd.read_csv(alt_csv)
                alt_df = alt_df.sort_values(by=['trade_date'], ascending=True)
                alt_df['trade_date'] = pd.to_datetime(alt_df['trade_date'], format='%Y%m%d')
                alt_df = alt_df.set_index('trade_date', drop=True)
                alt_filtered = alt_df.loc[start_date:end_date]
                if not alt_filtered.empty:
                    filtered_df = alt_filtered
                    # 修复：列名定位，避免错位
                    cols = filtered_df.columns.tolist()
                    vol_col = 'vol' if 'vol' in cols else ('volume' if 'volume' in cols else -1)
                    data = bt.feeds.PandasData(dataname=filtered_df, datetime=None,
                                               open='open', high='high', low='low', close='close',
                                               volume=vol_col, openinterest=-1)
            except Exception:
                pass

    if filtered_df.empty:
        raise ValueError("df为空")
    return data


def _get_alt_data_dir(current_dir, code):
    """在训练和测试目录之间切换"""
    import os
    current_dir = os.path.abspath(str(current_dir))
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        train = os.path.abspath(str(config.TRAIN_DATA_INDEX_DIR))
        test = os.path.abspath(str(config.TEST_DATA_INDEX_DIR))
    else:
        train = os.path.abspath(str(config.TRAIN_DATA_A_DIR))
        test = os.path.abspath(str(config.TEST_DATA_A_DIR))
    if current_dir == train:
        return test + '/'
    elif current_dir == test:
        return train + '/'
    return None


if __name__ == '__main__':
    start_date = '20240101'
    end_date = '20241231'
    get_new_data = True

    a_market_dir = '数据/A股/'
    data_folder_dir = '数据/A股/训练测试库/测试/'

    stock_data_df = pd.read_csv(a_market_dir + 'stock_data.csv')

    code_list = stock_data_df[(stock_data_df['total_mv'] > 5000000)]['ts_code'].tolist()
    #code_list = ['DJI', 'FCHI', 'SPX', 'N225', 'GDAXI', '000300.SH', '399006.SZ']  # , 'AU100g']

    for code in code_list:
        save_data(code, start_date, end_date, get_new_data, data_folder_dir=data_folder_dir)
