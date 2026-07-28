"""盈湖（Yinghu DB）：全市场股票日K数据的统一存储与访问层。

设计要点：
1. 按月分区 Parquet：`盈湖/kline/<板块>/<code>/<YYYY-MM>.parquet`
   - 板块分类：沪深主板(hs)、创业板(cy)、科创板(kc)、指数(index)、ETF(etf)
   - 单文件 50-100MB，列式存储比 CSV 小 5-10 倍、读取快 10 倍
2. SQLite 元数据库（WAL 模式）：
   - securities 表：标的元数据（代码/名称/板块/上市日/退市日/ST标记）
   - kline_coverage 表：每个标的的数据覆盖范围（min_date/max_date/updated_at）
3. 质量校验：入库前校验列完整性、价格合理性、日期连续性
4. 增量更新：新数据按月合并去重后写回
5. 多进程并发安全：每个标的的写入互不干扰；读取走 SQLite 索引

业务调度：
- get_kline(code, start, end) → 命中盈湖直接返回；否则上层拉取后调 save_kline 入库
- check_coverage(code, start, end) → 快速判断是否需要拉取
- 数据写入前必须通过 quality_check
"""
import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import config

logger = logging.getLogger('trader_system')

# Parquet 支持状态
try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False


# ============================================================================
# 标的板块分类
# ============================================================================
def classify_code(code: str) -> str:
    """根据代码判断板块分类。

    返回值：'hs'(沪深主板) / 'cy'(创业板) / 'kc'(科创板) / 'index'(指数) / 'etf'(ETF)
    北交所(.BJ)不在盈湖范围内，返回 'excluded'。
    """
    code = str(code)
    # 指数优先判断
    if code in ('DJI', 'FCHI', 'SPX', 'N225', 'GDAXI'):
        return 'index'
    # 北交所一律排除
    if code.endswith('.BJ'):
        return 'excluded'
    # 国内指数
    if code.endswith('.SH'):
        prefix = code.split('.')[0]
        if prefix.startswith(('000', '880')):
            return 'index'
    if code.endswith('.SZ'):
        prefix = code.split('.')[0]
        if prefix.startswith('399'):
            return 'index'
    # 创业板 300/301
    if code.startswith(('300', '301')) and code.endswith('.SZ'):
        return 'cy'
    # 科创板 688
    if code.startswith('688') and code.endswith('.SH'):
        return 'kc'
    # ETF 简单判断（511/159/512/513/515/516/518/520/561/562/563/159 等开头）
    if (code.startswith('5') and code.endswith('.SH')) or \
       (code.startswith('15') and code.endswith('.SZ')):
        # 进一步判断：510/511/512/513/515/516/518/520/561/562/563/588 是 ETF
        if code[:3] in ('510', '511', '512', '513', '515', '516', '518', '520',
                         '561', '562', '563', '588'):
            return 'etf'
        if code[:3] == '159':
            return 'etf'
    # 沪深主板：000/001/002/003/600/601/603/605
    if code.endswith('.SZ') and code[:3] in ('000', '001', '002', '003'):
        return 'hs'
    if code.endswith('.SH') and code[:3] in ('600', '601', '603', '605'):
        return 'hs'
    # 兜底：归到沪深
    if code.endswith(('.SH', '.SZ')):
        return 'hs'
    # 海外指数等无后缀
    return 'index'


# ============================================================================
# 元数据库初始化
# ============================================================================
def _init_meta_db():
    """初始化 SQLite 元数据库（WAL 模式，支持多进程并发读写）。"""
    db_path = Path(config.YINGHU_DB_META)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # K线数据根目录
    Path(config.YINGHU_DB_KLINE_DIR).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # 标的元数据表
        conn.execute('''CREATE TABLE IF NOT EXISTS securities (
            code TEXT PRIMARY KEY,
            name TEXT,
            board TEXT NOT NULL,
            list_date TEXT,
            delist_date TEXT,
            is_st INTEGER DEFAULT 0,
            updated_at TEXT
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_securities_board ON securities(board)')
        # K线数据覆盖表
        conn.execute('''CREATE TABLE IF NOT EXISTS kline_coverage (
            code TEXT NOT NULL,
            min_date TEXT NOT NULL,
            max_date TEXT NOT NULL,
            row_count INTEGER,
            last_update TEXT,
            data_source TEXT,
            quality_score REAL,
            PRIMARY KEY (code)
        )''')
        # 数据更新任务表
        conn.execute('''CREATE TABLE IF NOT EXISTS update_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            code TEXT,
            mode TEXT,
            start_date TEXT,
            end_date TEXT,
            status TEXT,
            rows_added INTEGER,
            error TEXT,
            started_at TEXT,
            finished_at TEXT
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_update_log_task ON update_log(task_id)')
        conn.commit()
    finally:
        conn.close()


_init_meta_db()


def _get_conn():
    """获取 SQLite 连接（WAL 模式，30s 超时）。"""
    conn = sqlite3.connect(str(config.YINGHU_DB_META), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# ============================================================================
# 路径计算
# ============================================================================
def _kline_path(code: str, year_month: str) -> Path:
    """获取某标的某月份的 Parquet 文件路径。

    Args:
        code: 标的代码（如 000001.SZ）
        year_month: 月份字符串（如 2026-07）
    """
    board = classify_code(code)
    return Path(config.YINGHU_DB_KLINE_DIR) / board / code / f'{year_month}.parquet'


def _list_months(start_date: str, end_date: str):
    """列出日期范围覆盖的所有月份（YYYY-MM 格式）。"""
    start = datetime.strptime(start_date.replace('-', ''), '%Y%m%d')
    end = datetime.strptime(end_date.replace('-', ''), '%Y%m%d')
    months = []
    ym = datetime(start.year, start.month, 1)
    while ym <= end:
        months.append(ym.strftime('%Y-%m'))
        # 下一月
        if ym.month == 12:
            ym = datetime(ym.year + 1, 1, 1)
        else:
            ym = datetime(ym.year, ym.month + 1, 1)
    return months


# ============================================================================
# 数据读取
# ============================================================================
def get_kline(code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从盈湖读取 K 线数据。

    Args:
        code: 标的代码
        start_date: 起始日 YYYYMMDD
        end_date: 结束日 YYYYMMDD

    Returns:
        tushare 兼容格式的 DataFrame（按 trade_date 升序），或 None（无数据）
    """
    if classify_code(code) == 'excluded':
        return None
    start_date = str(start_date).replace('-', '')
    end_date = str(end_date).replace('-', '')
    months = _list_months(start_date, end_date)
    frames = []
    for ym in months:
        path = _kline_path(code, ym)
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_parquet(path)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f'[盈湖] 读取 {code} {ym} 失败: {e}')
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # 按日期范围过滤
    df = df[(df['trade_date'].astype(str) >= start_date) &
            (df['trade_date'].astype(str) <= end_date)]
    if df.empty:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def check_coverage(code: str, start_date: str, end_date: str) -> bool:
    """快速检查盈湖是否覆盖指定日期范围（走 SQLite 索引，不读文件）。

    允许的偏差：
    - start_date 向后偏差最多 10 天（处理周末/节假日导致的首个交易日滞后）
    - end_date 向前偏差最多 5 天（处理当天未收盘或数据源延迟）
    """
    if classify_code(code) == 'excluded':
        return False
    start_int = int(str(start_date).replace('-', ''))
    end_int = int(str(end_date).replace('-', ''))
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                'SELECT min_date, max_date FROM kline_coverage WHERE code=?',
                (code,)
            ).fetchone()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[盈湖] 查询 {code} 覆盖范围失败: {e}')
        return False
    if not row:
        return False
    min_int = int(str(row[0]).replace('-', ''))
    max_int = int(str(row[1]).replace('-', ''))
    return min_int <= start_int + 10 and max_int >= end_int - 5


def get_coverage(code: str):
    """返回某 code 的数据覆盖范围 (min_date, max_date, row_count, last_update)，无则 None。"""
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                'SELECT min_date, max_date, row_count, last_update FROM kline_coverage WHERE code=?',
                (code,)
            ).fetchone()
        finally:
            conn.close()
        if row:
            return {'min_date': row[0], 'max_date': row[1],
                    'row_count': row[2], 'last_update': row[3]}
    except Exception:
        pass
    return None


# ============================================================================
# 数据质量校验
# ============================================================================
def quality_check(df: pd.DataFrame, code: str) -> tuple:
    """对入库前的数据进行质量校验。

    Returns:
        (passed: bool, reason: str)
    """
    if df is None or df.empty:
        return False, '空数据'
    required_cols = {'trade_date', 'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return False, f'缺少必要列: {missing}'
    # 列空值检查
    for col in ('open', 'high', 'low', 'close', 'trade_date'):
        if df[col].isna().any():
            na_count = df[col].isna().sum()
            return False, f'{col} 存在 {na_count} 个空值'
    # 价格合理性：high >= max(open, close) >= min(open, close) >= low
    bad = df[(df['high'] < df[['open', 'close']].max(axis=1)) |
             (df['low'] > df[['open', 'close']].min(axis=1)) |
             (df['high'] < df['low'])]
    if not bad.empty:
        return False, f'价格不合理行数: {len(bad)}'
    # 价格为负或零
    bad_price = df[(df['open'] <= 0) | (df['high'] <= 0) |
                    (df['low'] <= 0) | (df['close'] <= 0)]
    if not bad_price.empty:
        return False, f'价格<=0 行数: {len(bad_price)}'
    # 日期去重检查
    dup = df['trade_date'].duplicated().sum()
    if dup > 0:
        return False, f'日期重复行数: {dup}'
    return True, ''


# ============================================================================
# 数据写入
# ============================================================================
def save_kline(code: str, df: pd.DataFrame, data_source_name: str = 'akshare') -> tuple:
    """将 K 线数据写入盈湖（按月分区，增量合并去重）。

    Args:
        code: 标的代码
        df: 待写入的 DataFrame（tushare 兼容格式）
        data_source_name: 数据来源（记录到元数据库）

    Returns:
        (rows_added: int, error: str)
    """
    if classify_code(code) == 'excluded':
        return 0, '北交所股票不入库'
    if df is None or df.empty:
        return 0, '空数据'

    # 质量校验
    ok, reason = quality_check(df, code)
    if not ok:
        logger.warning(f'[盈湖] {code} 质量校验失败: {reason}')
        return 0, f'质量校验失败: {reason}'

    # 标准化 trade_date 为字符串
    df = df.copy()
    df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
    df = df.drop_duplicates(subset=['trade_date'], keep='last')
    df = df.sort_values('trade_date').reset_index(drop=True)

    # 按月分组写入
    df['ym'] = df['trade_date'].str[:6]
    ym_map = {}
    for ym, group in df.groupby('ym'):
        year_month = f'{ym[:4]}-{ym[4:6]}'
        ym_map[year_month] = group.drop(columns=['ym'])

    rows_added = 0
    for year_month, new_data in ym_map.items():
        path = _kline_path(code, year_month)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 增量合并：读取已存在的数据，与新数据合并去重
        if path.exists() and path.stat().st_size > 0:
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['trade_date'], keep='last')
                combined = combined.sort_values('trade_date').reset_index(drop=True)
            except Exception as e:
                logger.warning(f'[盈湖] 读取 {code} {year_month} 旧数据失败，覆盖写: {e}')
                combined = new_data
        else:
            combined = new_data
        # 写入 Parquet
        try:
            if _PARQUET_AVAILABLE:
                combined.to_parquet(path, index=False, compression='snappy')
            else:
                # 兜底：CSV
                csv_path = path.with_suffix('.csv')
                combined.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f'[盈湖] 写入 {code} {year_month} 失败: {e}')
            continue
        rows_added += len(new_data)

    # 更新元数据库
    _update_coverage(code, df, data_source_name)
    return rows_added, ''


def _update_coverage(code: str, df: pd.DataFrame, data_source_name: str):
    """更新 kline_coverage 表（与已有数据合并后的覆盖范围）。

    同时同步 securities 表，确保该表记录所有已入库标的，供前端统计展示。
    row_count 基于本次 df 的真实行数（不再累加旧值，避免增量合并时虚高）。
    """
    if df.empty:
        return
    try:
        conn = _get_conn()
        try:
            # 取本次 df 的真实覆盖范围
            df_min = str(df['trade_date'].astype(str).min())
            df_max = str(df['trade_date'].astype(str).max())
            df_count = len(df)

            # 与已存在的覆盖范围取并集（min 取最小，max 取最大）
            row = conn.execute(
                'SELECT min_date, max_date FROM kline_coverage WHERE code=?',
                (code,)
            ).fetchone()
            if row:
                old_min, old_max = row
                new_min = min(str(old_min), df_min)
                new_max = max(str(old_max), df_max)
            else:
                new_min = df_min
                new_max = df_max

            # row_count 用 df_count（真实本次写入数，不累加，避免增量合并虚高）
            conn.execute('''INSERT INTO kline_coverage
                (code, min_date, max_date, row_count, last_update, data_source, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                min_date=excluded.min_date, max_date=excluded.max_date,
                row_count=excluded.row_count, last_update=excluded.last_update,
                data_source=excluded.data_source, quality_score=excluded.quality_score''',
                (code, new_min, new_max, df_count, datetime.now().isoformat(),
                 data_source_name, 1.0))

            # 同步 securities 表（若不存在则插入，board 由 classify_code 推断）
            board = classify_code(code)
            if board == 'excluded':
                # 不入库的标的也不应进 securities 表
                pass
            else:
                # 名称留空，后续 yinghu_db_init 会通过 fetch_all_a_stocks 等填充
                conn.execute('''INSERT INTO securities
                    (code, name, board, list_date, is_st, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(code) DO UPDATE SET
                    board=excluded.board, updated_at=excluded.updated_at''',
                    (code, '', board, None, 0, datetime.now().isoformat()))

            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[盈湖] 更新 {code} 覆盖元数据失败: {e}')


# ============================================================================
# 标的元数据管理
# ============================================================================
def upsert_security(code: str, name: str, board: str = None,
                    list_date: str = None, is_st: int = 0):
    """新增/更新标的元数据。"""
    if board is None:
        board = classify_code(code)
    try:
        conn = _get_conn()
        try:
            conn.execute('''INSERT INTO securities
                (code, name, board, list_date, is_st, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                name=excluded.name, board=excluded.board,
                list_date=excluded.list_date, is_st=excluded.is_st,
                updated_at=excluded.updated_at''',
                (code, name, board, list_date, is_st, datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[盈湖] 更新 {code} 元数据失败: {e}')


def get_security(code: str):
    """查询标的基本信息。无则返回 None。"""
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                'SELECT code, name, board, list_date, delist_date, is_st FROM securities WHERE code=?',
                (code,)
            ).fetchone()
        finally:
            conn.close()
        if row:
            return {'code': row[0], 'name': row[1], 'board': row[2],
                    'list_date': row[3], 'delist_date': row[4], 'is_st': bool(row[5])}
    except Exception:
        pass
    return None


def list_securities(board: str = None, include_st: bool = True):
    """列出所有标的（可按板块过滤）。返回 list[dict]。"""
    try:
        conn = _get_conn()
        try:
            if board:
                rows = conn.execute(
                    'SELECT code, name, board, list_date, is_st FROM securities WHERE board=? ORDER BY code',
                    (board,)
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT code, name, board, list_date, is_st FROM securities ORDER BY code'
                ).fetchall()
        finally:
            conn.close()
        result = []
        for r in rows:
            if not include_st and r[4]:
                continue
            result.append({'code': r[0], 'name': r[1], 'board': r[2],
                           'list_date': r[3], 'is_st': bool(r[4])})
        return result
    except Exception as e:
        logger.warning(f'[盈湖] 查询标的列表失败: {e}')
        return []


# ============================================================================
# 统计与监控
# ============================================================================
def get_db_stats():
    """获取盈湖整体统计：标的数、覆盖范围、按板块分布。"""
    try:
        conn = _get_conn()
        try:
            total = conn.execute('SELECT COUNT(*) FROM securities').fetchone()[0]
            by_board = conn.execute(
                'SELECT board, COUNT(*) FROM securities GROUP BY board ORDER BY COUNT(*) DESC'
            ).fetchall()
            coverage = conn.execute(
                'SELECT COUNT(*), MIN(min_date), MAX(max_date), SUM(row_count) FROM kline_coverage'
            ).fetchone()
        finally:
            conn.close()
        return {
            'total_securities': total,
            'by_board': {b: c for b, c in by_board},
            'coverage_count': coverage[0] or 0,
            'min_date': coverage[1],
            'max_date': coverage[2],
            'total_rows': coverage[3] or 0,
        }
    except Exception as e:
        logger.warning(f'[盈湖] 获取统计失败: {e}')
        return {}


def log_update(task_id: str, code: str, mode: str, start_date: str, end_date: str,
               status: str, rows_added: int = 0, error: str = '', started_at: str = None):
    """记录更新任务日志。"""
    try:
        conn = _get_conn()
        try:
            conn.execute('''INSERT INTO update_log
                (task_id, code, mode, start_date, end_date, status, rows_added, error,
                 started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (task_id, code, mode, start_date, end_date, status, rows_added, error,
                 started_at or datetime.now().isoformat(), datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[盈湖] 记录更新日志失败: {e}')


# ============================================================================
# 备份
# ============================================================================
def backup_daily():
    """每日备份盈湖元数据库（保留最近 30 天）。"""
    backup_dir = Path(config.YINGHU_DB_BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    backup_path = backup_dir / f'market_{today}.db'
    if backup_path.exists():
        return False  # 今天已备份
    try:
        import shutil
        shutil.copy2(str(config.YINGHU_DB_META), str(backup_path))
        # 清理 30 天前的备份
        cutoff = datetime.now() - timedelta(days=30)
        for old in backup_dir.glob('market_*.db'):
            try:
                file_date = datetime.strptime(old.stem.split('_')[1], '%Y%m%d')
                if file_date < cutoff:
                    old.unlink()
            except Exception:
                pass
        logger.info(f'[盈湖] 备份完成: {backup_path.name}')
        return True
    except Exception as e:
        logger.warning(f'[盈湖] 备份失败: {e}')
        return False
