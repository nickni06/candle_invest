"""统一数据源模块：优先使用免费数据源（akshare），减少对 tushare 的依赖

数据获取优先级：
1. 本地 Parquet/CSV 缓存（优先 Parquet，避免任何 API 调用）
2. akshare（免费，无需 token）
3. tushare（备选，需要 token 且有权限限制）
4. 腾讯财经（akshare/tushare 均不可用时的兜底免费源）

支持的标的类型：
- A股个股（如 000533.SZ）：akshare stock_zh_a_hist
- 国内指数（如 000300.SH）：akshare stock_zh_index_daily
- 海外指数 DJI/SPX/IXIC：akshare index_us_stock_sina
- 海外指数 FCHI/GDAXI/N225：akshare index_global_hist_sina（新浪环球市场）
- 黄金 AU100g：仅本地 csv

返回统一的 tushare 兼容 DataFrame 格式：
    ts_code, trade_date, open, high, low, close, vol, pct_chg（可选）
    trade_date 格式：YYYYMMDD 字符串
    按 trade_date 升序
"""
import os
import logging
import socket
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from config import config

logger = logging.getLogger('trader_system')

# Parquet 支持状态：未安装 pyarrow 时自动回退到 CSV
try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False

# akshare 单次调用超时（秒）：akshare 内部 requests 无超时，需外部强制
AKSHARE_CALL_TIMEOUT = 25

# 腾讯财经单次调用超时（秒）
TENCENT_CALL_TIMEOUT = 15


# ============================================================================
# 本地 K 线元数据索引（SQLite），用于加速 "本地数据是否覆盖日期范围" 的检查
# ============================================================================
def _init_meta_db():
    """初始化 SQLite 元数据库（WAL 模式，支持多进程并发读写）。"""
    db_path = Path(config.DATA_META_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute('''CREATE TABLE IF NOT EXISTS kline_meta (
            path TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            min_date TEXT,
            max_date TEXT,
            mtime REAL,
            updated_at TEXT
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_kline_meta_code ON kline_meta(code)')
        conn.commit()
    finally:
        conn.close()


_init_meta_db()


def _remove_meta_paths(paths):
    """删除已不存在的文件对应的 meta 记录。"""
    if not paths:
        return
    try:
        conn = sqlite3.connect(str(config.DATA_META_DB), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executemany('DELETE FROM kline_meta WHERE path=?', [(p,) for p in paths])
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.debug(f'[数据源] 清理 kline_meta 失败: {e}')


def _update_kline_meta(code, data_path):
    """读取单个本地 K 线文件并更新其日期范围到元数据库。"""
    data_path = Path(data_path)
    if not data_path.exists() or data_path.stat().st_size == 0:
        return
    try:
        mtime = data_path.stat().st_mtime
        if data_path.suffix == '.parquet' and _PARQUET_AVAILABLE:
            df = pd.read_parquet(data_path, columns=['trade_date'])
        else:
            df = pd.read_csv(data_path, dtype={'ts_code': str}, usecols=['trade_date'])
        if df.empty:
            return
        dates = df['trade_date'].astype(str).tolist()
        min_d, max_d = min(dates), max(dates)
        conn = sqlite3.connect(str(config.DATA_META_DB), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute('''INSERT INTO kline_meta (path, code, min_date, max_date, mtime, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                code=excluded.code, min_date=excluded.min_date, max_date=excluded.max_date,
                mtime=excluded.mtime, updated_at=excluded.updated_at''',
                (str(data_path), code, min_d, max_d, mtime, datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.debug(f'[数据源] 更新 kline_meta 失败 {data_path}: {e}')


def _get_meta_date_range(code, dirs):
    """从元数据库查询某 code 在所有候选目录下的最小/最大日期。

    Returns:
        (min_date_str, max_date_str) 或 None（无有效记录）
    """
    if not config.DATA_META_DB.exists():
        return None
    try:
        conn = sqlite3.connect(str(config.DATA_META_DB), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            rows = conn.execute(
                'SELECT path, min_date, max_date, mtime FROM kline_meta WHERE code=?',
                (code,)
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return None

    all_dates = []
    stale_paths = []
    for path_str, min_d, max_d, mtime in rows:
        p = Path(path_str)
        if not p.exists():
            stale_paths.append(path_str)
            continue
        if p.stat().st_mtime != mtime:
            # 文件已被修改：重新读取并更新 meta，本次不使用旧记录
            _update_kline_meta(code, p)
            continue
        all_dates.extend([min_d, max_d])

    if stale_paths:
        _remove_meta_paths(stale_paths)

    if not all_dates:
        return None
    return min(all_dates), max(all_dates)


def _is_index_code(code):
    """判断是否为指数代码（海外指数 len<9，或国内指数 000300.SH/399006.SZ）"""
    code = str(code)
    return len(code) < 9 or code in ['000300.SH', '399006.SZ']


def _to_akshare_symbol(code):
    """将 tushare 代码转换为 akshare 对应的 symbol

    A股: 000533.SZ → 000533
    国内指数: 000300.SH → sh000300, 399006.SZ → sz399006
    海外指数: DJI → .DJI, SPX → .INX, IXIC → .IXIC
    """
    code = str(code)
    # 国内指数
    if code == '000300.SH':
        return 'sh000300'
    if code == '399006.SZ':
        return 'sz399006'
    # A股个股：去掉后缀
    if len(code) == 9 and (code.endswith('.SH') or code.endswith('.SZ')):
        return code[:6]
    # 海外指数（akshare index_us_stock_sina 支持）
    ak_overseas_map = {'DJI': '.DJI', 'SPX': '.INX', 'IXIC': '.IXIC'}
    if code in ak_overseas_map:
        return ak_overseas_map[code]
    # 海外指数（akshare index_global_hist_sina 支持，需中文名）
    # 在 _fetch_from_akshare 中单独处理，这里返回 None
    return None


# 海外指数 FCHI/GDAXI/N225 通过 akshare index_global_hist_sina 拉取，
# 该接口需要传中文名作为 symbol（如 "法CAC40指数"），见 ak.index_global_name_table()
_SINA_GLOBAL_INDEX_MAP = {
    'FCHI': '法CAC40指数',
    'GDAXI': '德国DAX 30种股价指数',
    'N225': '日经225指数',
}


def _find_local_data(code):
    """在所有数据目录中查找本地数据文件，优先 Parquet，回退 CSV。

    返回 (path, is_parquet) 或 None。
    """
    if _is_index_code(code):
        dirs = [config.DAILY_TRACKING_INDEX_DIR, config.TRAIN_DATA_INDEX_DIR, config.TEST_DATA_INDEX_DIR]
    else:
        dirs = [config.DAILY_TRACKING_A_DIR, config.TRAIN_DATA_A_DIR, config.TEST_DATA_A_DIR]
    for d in dirs:
        pq_path = d / f'{code}_daily.parquet'
        if _PARQUET_AVAILABLE and pq_path.exists() and pq_path.stat().st_size > 0:
            return pq_path, True
        csv_path = d / f'{code}_daily.csv'
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return csv_path, False
    return None


def _read_local_data(code, start_date, end_date):
    """读取本地数据（优先 Parquet）并按日期范围过滤。

    返回 tushare 兼容格式的 DataFrame，或 None（文件不存在/为空）
    """
    found = _find_local_data(code)
    if found is None:
        return None
    data_path, is_parquet = found
    try:
        if is_parquet:
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, dtype={'ts_code': str})
        if df.empty:
            return None
        # 过滤日期范围（trade_date 格式为 YYYYMMDD）
        mask = (df['trade_date'].astype(str) >= start_date) & (df['trade_date'].astype(str) <= end_date)
        filtered = df[mask].copy()
        if filtered.empty:
            return None
        # 按 trade_date 升序
        filtered = filtered.sort_values(by='trade_date', ascending=True).reset_index(drop=True)
        return filtered
    except Exception as e:
        logger.warning(f'[数据源] 读取本地数据失败 {data_path}: {e}')
        return None


def _local_data_has_date_range(code, start_date, end_date):
    """检查本地数据（所有候选目录合并后）是否覆盖所需的日期范围。

    使用整数日期比较，并允许合理偏差：
    - start_date 允许向后偏差最多 10 天（处理周末/节假日导致的首个交易日滞后）
    - end_date 允许向前偏差最多 5 天（处理当天未收盘或数据源延迟 1~2 天）
    """
    if _is_index_code(code):
        dirs = [config.DAILY_TRACKING_INDEX_DIR, config.TRAIN_DATA_INDEX_DIR, config.TEST_DATA_INDEX_DIR]
    else:
        dirs = [config.DAILY_TRACKING_A_DIR, config.TRAIN_DATA_A_DIR, config.TEST_DATA_A_DIR]

    # 统一转换为整数日期（兼容 YYYY-MM-DD 与 YYYYMMDD）
    start_int = int(str(start_date).replace('-', ''))
    end_int = int(str(end_date).replace('-', ''))

    # 优先使用 SQLite 元数据索引，避免每次全市场逐文件读 CSV
    meta_range = _get_meta_date_range(code, dirs)
    if meta_range is not None:
        min_int = int(str(meta_range[0]).replace('-', ''))
        max_int = int(str(meta_range[1]).replace('-', ''))
        return min_int <= start_int + 10 and max_int >= end_int - 5

    # 元数据未命中时回退到直接读取，并同步更新索引
    all_dates = []
    for d in dirs:
        pq_path = d / f'{code}_daily.parquet'
        csv_path = d / f'{code}_daily.csv'
        data_path = None
        is_parquet = False
        if _PARQUET_AVAILABLE and pq_path.exists() and pq_path.stat().st_size > 0:
            data_path = pq_path
            is_parquet = True
        elif csv_path.exists() and csv_path.stat().st_size > 0:
            data_path = csv_path
        if not data_path:
            continue
        try:
            if is_parquet:
                df = pd.read_parquet(data_path, columns=['trade_date'])
            else:
                df = pd.read_csv(data_path, dtype={'ts_code': str}, usecols=['trade_date'])
            if not df.empty:
                all_dates.extend(df['trade_date'].astype(str).tolist())
                _update_kline_meta(code, data_path)
        except Exception:
            continue
    if not all_dates:
        return False

    all_ints = [int(str(d).replace('-', '')) for d in all_dates]
    min_d, max_d = min(all_ints), max(all_ints)
    # 起始端允许最多 10 天偏差；结束端允许最多 5 天偏差
    return min_d <= start_int + 10 and max_d >= end_int - 5


def _normalize_akshare_a_stock(df, code):
    """将 akshare A股个股 DataFrame 转换为 tushare 兼容格式

    akshare 列: 日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    tushare 列: ts_code, trade_date, open, high, low, close, vol, amount, pct_chg
    """
    result = pd.DataFrame(index=df.index)
    result['ts_code'] = code
    result['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
    result['open'] = df['开盘'].values
    result['high'] = df['最高'].values
    result['low'] = df['最低'].values
    result['close'] = df['收盘'].values
    result['vol'] = df['成交量'].values
    if '成交额' in df.columns:
        result['amount'] = df['成交额']
    if '涨跌幅' in df.columns:
        result['pct_chg'] = df['涨跌幅']
    if '换手率' in df.columns:
        result['turnover_rate'] = df['换手率']
    return result


def _normalize_akshare_index_zh(df, code):
    """将 akshare 国内指数 DataFrame 转换为 tushare 兼容格式

    akshare 列: date, open, high, low, close, volume
    tushare 列: ts_code, trade_date, open, high, low, close, vol
    """
    result = pd.DataFrame(index=df.index)
    result['ts_code'] = code
    result['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    result['open'] = df['open'].values
    result['high'] = df['high'].values
    result['low'] = df['low'].values
    result['close'] = df['close'].values
    result['vol'] = df['volume'].values
    return result


def _normalize_akshare_index_us(df, code):
    """将 akshare 海外指数 DataFrame 转换为 tushare 兼容格式

    akshare 列: date, open, high, low, close, volume, amount
    tushare 列: ts_code, trade_date, open, high, low, close, vol
    """
    result = pd.DataFrame(index=df.index)
    result['ts_code'] = code
    result['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    result['open'] = df['open'].values
    result['high'] = df['high'].values
    result['low'] = df['low'].values
    result['close'] = df['close'].values
    result['vol'] = df['volume'].values
    return result


def _normalize_akshare_index_global(df, code):
    """将 akshare index_global_hist_sina 的 DataFrame 转换为 tushare 兼容格式

    akshare 列: date, open, high, low, close, volume（date 为 datetime.date）
    tushare 列: ts_code, trade_date, open, high, low, close, vol
    """
    result = pd.DataFrame(index=df.index)
    result['ts_code'] = code  # 标量赋值，自动广播到所有行
    result['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    result['open'] = df['open'].values
    result['high'] = df['high'].values
    result['low'] = df['low'].values
    result['close'] = df['close'].values
    result['vol'] = df['volume'].values
    return result


def _call_akshare_with_timeout(func, *args, **kwargs):
    """在线程池中调用 akshare 接口，强制超时（akshare 内部 requests 无超时参数）。

    超时后线程仍在后台运行（无法强制杀掉线程），但主流程会立即返回 None，
    避免预拉取串行流程被单个卡死的 API 调用永久阻塞。
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=AKSHARE_CALL_TIMEOUT)
        except FuturesTimeoutError:
            logger.warning(f'[数据源] akshare 调用超时（{AKSHARE_CALL_TIMEOUT}秒），跳过')
            return None
        except Exception as e:
            logger.warning(f'[数据源] akshare 调用异常: {e}')
            return None


def _fetch_from_akshare(code, start_date, end_date, max_retries=2):
    """从 akshare 获取日K数据，返回 tushare 兼容格式的 DataFrame，或 None

    包含简单重试机制，应对 akshare 偶发的网络抖动（RemoteDisconnected）。
    每次 akshare 调用都有 AKSHARE_CALL_TIMEOUT 秒超时保护，避免永久挂起。
    """
    import akshare as ak
    import time
    code = str(code)
    ak_symbol = _to_akshare_symbol(code)

    # FCHI/GDAXI/N225：通过 akshare index_global_hist_sina 拉取（需中文名）
    if ak_symbol is None and code in _SINA_GLOBAL_INDEX_MAP:
        cn_name = _SINA_GLOBAL_INDEX_MAP[code]
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                df = _call_akshare_with_timeout(ak.index_global_hist_sina, symbol=cn_name)
                if df is None or df.empty:
                    return None
                # 过滤日期范围
                dates = pd.to_datetime(df['date'])
                mask = (dates >= start_dt) & (dates <= end_dt)
                filtered = df[mask].copy()
                if filtered.empty:
                    return None
                return _normalize_akshare_index_global(filtered, code)
            except Exception as e:
                last_err = e
                err_msg = str(e)
                is_network_err = any(k in err_msg for k in ['RemoteDisconnected', 'Connection aborted', 'timeout', 'TimeoutError'])
                if not is_network_err or attempt == max_retries:
                    logger.warning(f'[数据源] akshare index_global_hist_sina 获取 {code}({cn_name}) 失败: {e}')
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    # akshare 不支持的标的
    if ak_symbol is None:
        logger.info(f'[数据源] akshare 不支持 {code}，仅本地可用')
        return None

    # 统一转换为 Timestamp 进行比较，避免 datetime.date 与 Timestamp 比较报错
    start_dt = pd.to_datetime(start_date, format='%Y%m%d')
    end_dt = pd.to_datetime(end_date, format='%Y%m%d')

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            # A股个股：优先东财接口，失败则降级到新浪接口
            if len(code) == 9 and (code.endswith('.SH') or code.endswith('.SZ')):
                # 优先使用东财接口（数据更全，含涨跌幅/换手率）
                df = _call_akshare_with_timeout(
                    ak.stock_zh_a_hist,
                    symbol=ak_symbol, period='daily',
                    start_date=start_date, end_date=end_date, adjust='qfq'
                )
                if df is not None and not df.empty:
                    return _normalize_akshare_a_stock(df, code)

                # 东财接口失败/无数据，降级到新浪接口
                logger.info(f'[数据源] {code} 东财接口无数据，降级到新浪接口')
                # 新浪接口 symbol 格式：sz000533 / sh600000
                sina_symbol = ('sh' if code.endswith('.SH') else 'sz') + code[:6]
                df = _call_akshare_with_timeout(ak.stock_zh_a_daily, symbol=sina_symbol, start_date=start_date, end_date=end_date, adjust='qfq')
                if df is not None and not df.empty:
                    return _normalize_akshare_a_stock_sina(df, code)
                return None

            # 国内指数
            if code in ['000300.SH', '399006.SZ']:
                df = _call_akshare_with_timeout(ak.stock_zh_index_daily, symbol=ak_symbol)
                if df is None or df.empty:
                    return None
                # 统一转 Timestamp 后再比较，避免 datetime.date 类型不匹配
                dates = pd.to_datetime(df['date'])
                mask = (dates >= start_dt) & (dates <= end_dt)
                filtered = df[mask].copy()
                if filtered.empty:
                    return None
                return _normalize_akshare_index_zh(filtered, code)

            # 海外指数（DJI/SPX/IXIC）
            if code in ['DJI', 'SPX', 'IXIC']:
                df = _call_akshare_with_timeout(ak.index_us_stock_sina, symbol=ak_symbol)
                if df is None or df.empty:
                    return None
                # 统一转 Timestamp 后再比较
                dates = pd.to_datetime(df['date'])
                mask = (dates >= start_dt) & (dates <= end_dt)
                filtered = df[mask].copy()
                if filtered.empty:
                    return None
                return _normalize_akshare_index_us(filtered, code)

            # 黄金 AU100g 等特殊代码：akshare 不支持
            logger.info(f'[数据源] akshare 不支持 {code}，仅本地可用')
            return None

        except Exception as e:
            last_err = e
            err_msg = str(e)
            # 网络抖动类错误才重试
            is_network_err = any(k in err_msg for k in ['RemoteDisconnected', 'Connection aborted', 'timeout', 'TimeoutError'])
            if not is_network_err or attempt == max_retries:
                logger.warning(f'[数据源] akshare 获取 {code} 失败: {e}')
                return None
            # 短暂等待后重试
            time.sleep(1.5 * (attempt + 1))

    if last_err:
        logger.warning(f'[数据源] akshare 获取 {code} 重试 {max_retries} 次后仍失败: {last_err}')
    return None


def _normalize_akshare_a_stock_sina(df, code):
    """将 akshare 新浪接口 A股 DataFrame 转换为 tushare 兼容格式

    akshare 新浪接口列: date, open, high, low, close, volume, amount, outstanding_share, turnover
    tushare 列: ts_code, trade_date, open, high, low, close, vol, amount, turnover_rate
    """
    result = pd.DataFrame(index=df.index)
    result['ts_code'] = code
    result['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    result['open'] = df['open'].values
    result['high'] = df['high'].values
    result['low'] = df['low'].values
    result['close'] = df['close'].values
    result['vol'] = df['volume'].values
    if 'amount' in df.columns:
        result['amount'] = df['amount'].values
    if 'turnover' in df.columns:
        result['turnover_rate'] = (df['turnover'].values) * 100  # 新浪返回小数，转为百分比
    return result


def _fetch_from_tushare(code, start_date, end_date):
    """从 tushare 获取 A 股个股日K数据，返回 tushare 兼容格式的 DataFrame，或 None

    tushare 的数据更新及时，通常当日收盘后即可获取。
    作为 akshare 数据延迟时的备选数据源。
    """
    try:
        import tushare as ts
        token = getattr(config, 'TUSHARE_TOKEN', '')
        if not token:
            logger.info('[数据源] tushare token 未配置，跳过')
            return None
        pro = ts.pro_api(token)
        # A股个股
        if len(code) == 9 and (code.endswith('.SH') or code.endswith('.SZ')):
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return None
            # tushare 返回的字段：ts_code, trade_date, open, high, low, close, vol, amount, pct_chg, change
            # vol 单位是手，转成股
            df['vol'] = df['vol'] * 100
            # 按日期升序
            df = df.sort_values('trade_date').reset_index(drop=True)
            logger.info(f'[数据源] tushare 获取 {code} 成功（{len(df)}行，最后日期 {df["trade_date"].iloc[-1]}）')
            return df
        # 国内指数
        if code in ['000300.SH', '399006.SZ']:
            df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return None
            df = df.sort_values('trade_date').reset_index(drop=True)
            logger.info(f'[数据源] tushare 获取指数 {code} 成功（{len(df)}行，最后日期 {df["trade_date"].iloc[-1]}）')
            return df
        return None
    except Exception as e:
        logger.warning(f'[数据源] tushare 获取 {code} 失败: {e}')
        return None


def _normalize_tencent_a_stock(df, code):
    """将腾讯财经 A股 DataFrame 转换为 tushare 兼容格式

    腾讯财经列: date, open, close, high, low, volume
    tushare 列: ts_code, trade_date, open, high, low, close, vol
    """
    result = pd.DataFrame(index=df.index)
    result['ts_code'] = code
    result['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    result['open'] = df['open'].values
    result['high'] = df['high'].values
    result['low'] = df['low'].values
    result['close'] = df['close'].values
    result['vol'] = df['volume'].values
    return result


def _fetch_from_tencent(code, start_date, end_date):
    """从腾讯财经获取 A 股个股日K数据，返回 tushare 兼容格式的 DataFrame，或 None

    作为 akshare/tushare 均不可用时的兜底数据源。
    仅支持沪深 A 股个股（sh6xxxxx / sz0xxxxx / sz3xxxxx）。
    """
    code = str(code)
    if not (len(code) == 9 and (code.endswith('.SH') or code.endswith('.SZ'))):
        return None

    import requests
    try:
        market = 'sh' if code.endswith('.SH') else 'sz'
        tencent_symbol = f'{market}{code[:6]}'
        start_fmt = pd.to_datetime(start_date, format='%Y%m%d').strftime('%Y-%m-%d')
        end_fmt = pd.to_datetime(end_date, format='%Y%m%d').strftime('%Y-%m-%d')
        # datalen 取日期跨度交易日数上限（按自然日估算并留余量）
        days_span = (pd.to_datetime(end_date, format='%Y%m%d') - pd.to_datetime(start_date, format='%Y%m%d')).days + 1
        datalen = max(days_span, 20)
        url = (
            f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
            f'?param={tencent_symbol},day,{start_fmt},{end_fmt},{datalen},qfq'
        )
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=TENCENT_CALL_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or data.get('code') != 0:
            return None
        raw_data = data.get('data', {})
        if not isinstance(raw_data, dict):
            return None
        symbol_data = raw_data.get(tencent_symbol, {})
        klines = symbol_data.get('qfqday') or symbol_data.get('day')
        if not klines:
            return None
        # 腾讯接口偶发会在末尾附加除权除息信息对象，导致列数不一致；仅取前 6 列
        trimmed = [row[:6] for row in klines]
        df = pd.DataFrame(trimmed, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna(subset=['open', 'close', 'high', 'low', 'volume'])
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        if df.empty:
            return None
        logger.info(f'[数据源] 腾讯财经获取 {code} 成功（{len(df)}行，最后日期 {df["date"].iloc[-1].strftime("%Y%m%d")}）')
        return _normalize_tencent_a_stock(df, code)
    except Exception as e:
        logger.warning(f'[数据源] 腾讯财经获取 {code} 失败: {e}')
        return None


def _to_sina_realtime_symbol(code):
    """将 tushare 代码转换为新浪财经实时行情 symbol"""
    code = str(code)
    if len(code) == 9 and code.endswith('.SH'):
        return 'sh' + code[:6]
    if len(code) == 9 and code.endswith('.SZ'):
        return 'sz' + code[:6]
    if code == 'DJI':
        return 'gb_dji'
    if code == 'SPX':
        return 'gb_inx'
    if code == 'IXIC':
        return 'gb_ixic'
    return None


def _parse_sina_realtime_quote(text, sina_symbol):
    """解析 hq.sinajs.cn 返回的 JS 变量，返回最新一个交易日的 OHLCV 字典。

    返回: {'date': 'YYYYMMDD', 'open', 'high', 'low', 'close', 'volume'}
    如果无法解析或日期不匹配则返回 None。
    """
    import re
    try:
        m = re.search(r'var hq_str_' + re.escape(sina_symbol) + r'=["\']([^"\']*)["\']', text)
        if not m:
            return None
        parts = m.group(1).split(',')
        if not parts or not parts[0]:
            return None

        # A 股：name, open, prev_close, close, high, low, ..., volume(股), amount(元), ..., date, time
        if sina_symbol.startswith(('sh', 'sz')):
            if len(parts) < 33:
                return None
            return {
                'date': pd.to_datetime(parts[30]).strftime('%Y%m%d'),
                'open': float(parts[1]),
                'high': float(parts[4]),
                'low': float(parts[5]),
                'close': float(parts[3]),
                'volume': float(parts[8]),
            }

        # 美股指数（gb_dji/gb_inx/gb_ixic）：name, close, change_pct, datetime, change, open, high, low, ...
        if sina_symbol.startswith('gb_'):
            if len(parts) < 11:
                return None
            dt = pd.to_datetime(parts[3])
            # 新浪返回的是中国时间；美股交易日对应中国时间 04:00~次日 04:00 左右
            # 简单处理：凌晨 0~4 点仍属于前一天美股收盘时段，其余按当天算
            if dt.hour < 4:
                dt = dt - pd.Timedelta(days=1)
            return {
                'date': dt.strftime('%Y%m%d'),
                'open': float(parts[5]),
                'high': float(parts[6]),
                'low': float(parts[7]),
                'close': float(parts[1]),
                'volume': float(parts[10]),
            }
        return None
    except Exception:
        return None


def _fetch_from_sina_realtime(code, end_date):
    """从新浪财经实时行情获取最新一个交易日的 OHLCV，返回单行的 tushare 兼容 DataFrame，或 None。

    仅用于其他历史数据源均失败、本地数据又缺少最后一日时的兜底补齐。
    """
    sina_symbol = _to_sina_realtime_symbol(code)
    if sina_symbol is None:
        return None
    import requests
    try:
        url = f'https://hq.sinajs.cn/list={sina_symbol}'
        resp = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'},
            timeout=10,
        )
        resp.encoding = 'GB18030'
        resp.raise_for_status()
        quote = _parse_sina_realtime_quote(resp.text, sina_symbol)
        if quote is None or quote['date'] != end_date:
            return None
        df = pd.DataFrame([{
            'ts_code': code,
            'trade_date': quote['date'],
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'vol': quote['volume'],
        }])
        logger.info(f'[数据源] 新浪财经实时行情获取 {code} 最新日 {quote["date"]} 成功')
        return df
    except Exception as e:
        logger.warning(f'[数据源] 新浪财经实时行情获取 {code} 失败: {e}')
        return None


def _merge_sina_latest(df, code, end_date):
    """如果 df 缺少 end_date 数据，尝试用新浪财经实时行情补齐最后一根 K 线。"""
    if df is None or df.empty:
        return df
    try:
        dates = df['trade_date'].astype(str)
        if dates.max() >= end_date:
            return df
        latest = _fetch_from_sina_realtime(code, end_date)
        if latest is None or latest.empty:
            return df
        # 统一 trade_date 为字符串，避免 int/str 混排导致 sort_values 失败
        df_copy = df.copy()
        df_copy['trade_date'] = df_copy['trade_date'].astype(str)
        merged = pd.concat([df_copy, latest], ignore_index=True)
        merged = merged.drop_duplicates(subset=['trade_date'], keep='last')
        return merged.sort_values('trade_date').reset_index(drop=True)
    except Exception:
        return df


def _df_has_end_date(df, end_date):
    """检查 DataFrame 是否包含 end_date 这一天的数据。"""
    if df is None or df.empty:
        return False
    try:
        return df['trade_date'].astype(str).max() >= end_date
    except Exception:
        return False


def get_kline_df(code, start_date, end_date, prefer_local=True, allow_network=True):
    """获取日K线数据（统一入口）

    优先级：
    1. 盈湖（Yinghu DB）：全市场统一存储，覆盖范围走 SQLite 索引
    2. 本地 Parquet/CSV（兼容旧数据，如果 prefer_local=True）
    3. akshare（免费数据源，仅当 allow_network=True 时）→ 拉取成功后入盈湖
    4. tushare（备选数据源，akshare 数据延迟时使用）→ 拉取成功后入盈湖
    5. 腾讯财经（akshare/tushare 均不可用时的兜底免费源）→ 拉取成功后入盈湖
    6. 新浪财经实时行情（历史源全部失败时，用最新行情补齐最后一日）
    7. 本地 Parquet/CSV 部分数据（即使不覆盖整个日期范围）

    Args:
        code: 标的代码（tushare 格式，如 000533.SZ / DJI / 000300.SH）
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        prefer_local: 是否优先读本地（盈湖优先级最高，此参数控制旧目录的回退）
        allow_network: 是否允许调用网络数据源（akshare/tushare/腾讯财经/新浪财经）。
            多进程子进程环境下建议设为 False，避免网络阻塞导致子进程卡死。

    Returns:
        DataFrame（tushare 兼容格式，按 trade_date 升序）或 None
    """
    code = str(code)

    # best_df / best_source 初始化为 None，后续分支可能赋值
    # （盈湖 partial / 本地 partial / 网络拉取 等场景）
    best_df = None
    best_source = None

    # 0. 优先读盈湖（Yinghu DB）：全市场统一存储，SQLite 索引快速判断覆盖范围
    # 注意：check_coverage 允许 end_date 向前偏差 5 天（处理周末/节假日），
    # 但返回的数据可能不包含 end_date 当天。此时不直接返回，继续走网络拉取
    # 以确保跟踪日/回测结束日的数据被补齐。
    try:
        import yinghu_db
        if yinghu_db.check_coverage(code, start_date, end_date):
            df = yinghu_db.get_kline(code, start_date, end_date)
            if df is not None and not df.empty:
                if _df_has_end_date(df, end_date):
                    return df
                # 盈湖数据不包含 end_date，保留作为 fallback，继续尝试网络拉取
                best_df = df
                best_source = 'yinghu_partial'
    except Exception as e:
        logger.debug(f'[数据源] 盈湖查询 {code} 失败，回退旧逻辑: {e}')

    # 1. 优先读本地旧目录（如果本地数据覆盖所需日期范围）
    # 同样：_local_data_has_date_range 允许 5 天偏差，需验证返回数据是否真包含 end_date
    if prefer_local and _local_data_has_date_range(code, start_date, end_date):
        df = _read_local_data(code, start_date, end_date)
        if df is not None:
            if _df_has_end_date(df, end_date):
                # 异步入库：旧目录有数据但盈湖缺失，迁入盈湖（不阻塞返回）
                try:
                    import yinghu_db
                    yinghu_db.save_kline(code, df, data_source_name='local_legacy')
                except Exception:
                    pass
                return df
            # 本地数据不包含 end_date，保留作为 fallback 合并源
            if best_df is None:
                best_df = df
                best_source = 'local_partial'

    # 2. 从 akshare 获取（仅当允许网络访问时）
    # best_df / best_source 可能已在盈湖/本地分支中赋值（部分覆盖场景），
    # 网络拉取成功且包含 end_date 时优先使用网络数据
    if allow_network:
        df = _fetch_from_akshare(code, start_date, end_date)
        if df is not None and not df.empty:
            # 网络数据比已有的 partial 数据更优，优先使用
            best_df = df
            best_source = 'akshare'
            if _df_has_end_date(best_df, end_date):
                _save_to_yinghu_db_async(code, best_df, best_source)
                return best_df

        # 3. akshare 失败或数据延迟，尝试 tushare 备选
        df = _fetch_from_tushare(code, start_date, end_date)
        if df is not None and not df.empty:
            best_df = df
            best_source = 'tushare'
            if _df_has_end_date(best_df, end_date):
                _save_to_yinghu_db_async(code, best_df, best_source)
                return best_df

        # 4. akshare/tushare 均失败，尝试腾讯财经兜底（仅 A 股个股）
        df = _fetch_from_tencent(code, start_date, end_date)
        if df is not None and not df.empty:
            best_df = df
            best_source = 'tencent'
            if _df_has_end_date(best_df, end_date):
                _save_to_yinghu_db_async(code, best_df, best_source)
                return best_df

    # 5. fallback / merge：读本地部分数据；若本地比网络更新则合并，避免网络数据
    #    缺少最新交易日时被旧本地数据覆盖。
    local_df = _read_local_data(code, start_date, end_date)
    if local_df is not None and not local_df.empty:
        if best_df is None:
            best_df = local_df
            best_source = 'local_legacy'
        else:
            local_max = local_df['trade_date'].astype(str).max()
            best_max = best_df['trade_date'].astype(str).max()
            if local_max > best_max:
                best_df = pd.concat([best_df, local_df], ignore_index=True)
                best_df = best_df.drop_duplicates(subset=['trade_date'], keep='last')
                best_df = best_df.sort_values('trade_date').reset_index(drop=True)
                best_source = f'{best_source}+local'

    # 6. 若历史源均缺少 end_date，尝试用新浪财经实时行情补齐最后一日
    if allow_network and best_df is not None:
        merged = _merge_sina_latest(best_df, code, end_date)
        if merged is not None and not merged.empty:
            best_df = merged
            best_source = f'{best_source}+sina'
            if _df_has_end_date(best_df, end_date):
                _save_to_yinghu_db_async(code, best_df, best_source)
                return best_df

    # 7. 本地完全没有数据时，尝试新浪财经实时行情兜底
    if best_df is None and allow_network:
        df = _fetch_from_sina_realtime(code, end_date)
        if df is not None:
            best_df = df
            best_source = 'sina_realtime'

    # 入库：网络拉取到的数据写入盈湖
    if best_df is not None and best_source:
        _save_to_yinghu_db_async(code, best_df, best_source)

    return best_df


def _save_to_yinghu_db_async(code, df, source_name):
    """将拉取到的数据写入盈湖（同步，但失败不阻塞业务流程）。

    Args:
        code: 标的代码
        df: 已通过质量校验的数据
        source_name: 数据来源
    """
    try:
        import yinghu_db
        rows, err = yinghu_db.save_kline(code, df, data_source_name=source_name)
        if err:
            logger.debug(f'[数据源] {code} 入盈湖跳过: {err}')
        elif rows > 0:
            logger.debug(f'[数据源] {code} 入盈湖 {rows} 行 (来源: {source_name})')
    except Exception as e:
        logger.debug(f'[数据源] {code} 入盈湖失败: {e}')


def save_kline_to_local(code, df, data_folder_dir=None):
    """将 DataFrame 保存到本地（优先 Parquet，保留 CSV 作为备份）

    Args:
        code: 标的代码
        df: tushare 兼容格式的 DataFrame
        data_folder_dir: 保存目录，默认根据 code 类型自动选择
    """
    if df is None or df.empty:
        return
    code = str(code)
    if data_folder_dir is None:
        if _is_index_code(code):
            data_folder_dir = config.DAILY_TRACKING_INDEX_DIR
        else:
            data_folder_dir = config.DAILY_TRACKING_A_DIR
    else:
        data_folder_dir = type(config.DAILY_TRACKING_INDEX_DIR)(data_folder_dir)

    csv_path = data_folder_dir / f'{code}_daily.csv'

    # 合并已有数据，防止旧数据/不完整数据覆盖较新的本地数据
    if csv_path.exists():
        try:
            old_df = pd.read_csv(str(csv_path))
            if not old_df.empty and 'trade_date' in old_df.columns:
                merged = pd.concat([old_df, df], ignore_index=True)
                merged = merged.drop_duplicates(subset=['trade_date'], keep='last')
                df = merged.sort_values('trade_date').reset_index(drop=True)
        except Exception:
            pass  # 合并失败则直接覆盖

    # 1. 保存 Parquet 格式（如果支持）：体积小、读取快
    if _PARQUET_AVAILABLE:
        pq_path = data_folder_dir / f'{code}_daily.parquet'
        try:
            # 确保 trade_date 为字符串，避免 Parquet 类型不一致
            df_to_save = df.copy()
            if 'trade_date' in df_to_save.columns:
                df_to_save['trade_date'] = df_to_save['trade_date'].astype(str)
            df_to_save.to_parquet(pq_path, index=False)
            logger.info(f'[数据源] 保存 Parquet {code} 至 {pq_path}（{len(df)}行）')
        except Exception as e:
            logger.warning(f'[数据源] Parquet 保存失败 {code}: {e}')

    # 2. 保留 CSV 作为备份/兼容格式
    df.to_csv(str(csv_path), index=False)
    logger.info(f'[数据源] 保存 CSV {code} 至 {csv_path}（{len(df)}行）')

    # 3. 同步更新元数据索引（CSV 必须更新；Parquet 也更新，若存在）
    _update_kline_meta(code, csv_path)
    if _PARQUET_AVAILABLE:
        pq_path = data_folder_dir / f'{code}_daily.parquet'
        if pq_path.exists():
            _update_kline_meta(code, pq_path)
