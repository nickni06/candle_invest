"""结果库（Result DB）：策略回测/信号跟踪结果的缓存层。

设计要点：
1. 缓存键：标的 + 策略 + 日期范围 + observe_day + cautious 模式 → SHA1 哈希
2. 失效策略：
   - 源数据 mtime 变化自动失效（盈湖 K 线文件被更新）
   - 策略逻辑变更主动清空（通过 PATTERN_VERSION 标记）
   - 超过保留期(365天)自动清理
3. 存储：Parquet 数据文件 + SQLite 索引
4. 用途：用户再跑同样日期同样标的，直接读库，秒级返回

接口设计：
- get_result(cache_key) → dict / None
- save_result(cache_key, result_dict) → None
- compute_cache_key(code, pattern_name, pattern_type, start_date, end_date,
                    observe_day, cautious, extra=None) → str
- check_fresh(cache_key) → bool  # 检查缓存是否仍然有效（源数据未变更）
"""
import os
import json
import hashlib
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import config

logger = logging.getLogger('trader_system')

# 结果库版本号：策略逻辑有重大变更时手动 +1，自动失效所有旧缓存
RESULT_VERSION = 'v1'

# Parquet 支持
try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False


# ============================================================================
# 元数据库初始化
# ============================================================================
def _init_meta_db():
    """初始化结果库的 SQLite 索引数据库。"""
    db_path = Path(config.RESULT_DB_META)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.RESULT_DB_DATA_DIR).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # 结果缓存索引表
        conn.execute('''CREATE TABLE IF NOT EXISTS result_cache (
            cache_key TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            pattern_name TEXT,
            pattern_type TEXT,
            start_date TEXT,
            end_date TEXT,
            observe_day INTEGER,
            cautious INTEGER,
            result_version TEXT,
            data_signature TEXT,
            data_file TEXT,
            created_at TEXT,
            last_access TEXT,
            access_count INTEGER DEFAULT 0
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_code ON result_cache(code)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_pattern ON result_cache(pattern_name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_created ON result_cache(created_at)')
        conn.commit()
    finally:
        conn.close()


_init_meta_db()


def _get_conn():
    conn = sqlite3.connect(str(config.RESULT_DB_META), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# ============================================================================
# 缓存键计算
# ============================================================================
def compute_cache_key(code: str, pattern_name: str, pattern_type: str,
                      start_date: str, end_date: str,
                      observe_day: int, cautious: bool, extra: dict = None) -> str:
    """计算结果缓存的唯一键。

    缓存键要素：
    - 标的代码
    - 策略名 + 策略类型（buy/sell）
    - 起止日期
    - observe_day
    - cautious 模式
    - 结果库版本号（策略逻辑变更时整体失效）
    - extra: 额外参数（如 cautious_mode 的具体过滤条件等）

    注意：源数据 mtime 不参与缓存键，而是作为 data_signature 单独存储，
    用于判断缓存是否仍然有效。
    """
    parts = [
        str(code),
        str(pattern_name),
        str(pattern_type),
        str(start_date).replace('-', ''),
        str(end_date).replace('-', ''),
        str(observe_day),
        '1' if cautious else '0',
        RESULT_VERSION,
    ]
    if extra:
        # extra 按 key 排序后拼接，保证幂等
        extra_str = json.dumps(extra, sort_keys=True, ensure_ascii=False)
        parts.append(extra_str)
    raw = '|'.join(parts)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:32]


def _compute_data_signature(code: str, start_date: str, end_date: str) -> str:
    """计算源数据签名：基于盈湖中相关月份文件的 mtime+size。

    如果盈湖中的源数据被更新（如新增了数据），签名会变化，旧缓存自动失效。
    """
    try:
        import yinghu_db
        from datetime import datetime as dt
        # 列出覆盖日期范围内的所有月份文件
        months = yinghu_db._list_months(start_date, end_date)
        parts = []
        for ym in months:
            path = yinghu_db._kline_path(code, ym)
            if path.exists():
                st = path.stat()
                parts.append(f'{path.name}:{st.st_mtime:.6f}:{st.st_size}')
        if not parts:
            # 盈湖无数据，使用旧目录的文件 mtime
            try:
                import data_source
                found = data_source._find_local_data(code)
                if found:
                    p = found[0]
                    st = p.stat()
                    parts.append(f'{p.name}:{st.st_mtime:.6f}:{st.st_size}')
            except Exception:
                pass
        return hashlib.md5('|'.join(parts).encode('utf-8')).hexdigest()[:16]
    except Exception as e:
        logger.debug(f'[结果库] 计算 {code} 数据签名失败: {e}')
        return ''


# ============================================================================
# 缓存读取
# ============================================================================
def get_result(cache_key: str) -> Optional[dict]:
    """读取缓存结果。

    Returns:
        dict: 缓存的结果数据；None 表示无缓存或缓存已失效
    """
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                '''SELECT code, start_date, end_date, result_version, data_signature,
                          data_file, created_at
                   FROM result_cache WHERE cache_key=?''',
                (cache_key,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        code, start_date, end_date, ver, old_sig, data_file, created = row
        # 版本号不匹配 → 失效
        if ver != RESULT_VERSION:
            return None
        # 数据签名变化 → 失效
        new_sig = _compute_data_signature(code, start_date, end_date)
        if new_sig and old_sig and new_sig != old_sig:
            logger.debug(f'[结果库] {code} 源数据已变更，缓存失效')
            return None
        # 检查保留期：超过 365 天的清理
        try:
            created_dt = datetime.fromisoformat(created)
            if datetime.now() - created_dt > timedelta(days=config.RESULT_DB_RETENTION_DAYS):
                _delete_cache(cache_key, data_file)
                return None
        except Exception:
            pass
        # 读取数据文件
        if not data_file:
            return None
        data_path = Path(config.RESULT_DB_DATA_DIR) / data_file
        if not data_path.exists():
            return None
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            # 更新访问时间和计数
            _touch_access(cache_key)
            return result
        except Exception as e:
            logger.warning(f'[结果库] 读取缓存 {cache_key} 失败: {e}')
            return None
    except Exception as e:
        logger.warning(f'[结果库] get_result 异常: {e}')
        return None


def _touch_access(cache_key: str):
    """更新缓存的最后访问时间和访问计数。"""
    try:
        conn = _get_conn()
        try:
            conn.execute(
                '''UPDATE result_cache
                   SET last_access=?, access_count=access_count+1
                   WHERE cache_key=?''',
                (datetime.now().isoformat(), cache_key)
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


# ============================================================================
# 缓存写入
# ============================================================================
def save_result(cache_key: str, code: str, pattern_name: str, pattern_type: str,
                start_date: str, end_date: str, observe_day: int, cautious: bool,
                result: dict):
    """保存结果到缓存。

    Args:
        cache_key: 缓存键
        code: 标的代码
        pattern_name: 策略名
        pattern_type: 策略类型（buy/sell）
        start_date / end_date: 日期范围
        observe_day: 观察日数
        cautious: 谨慎模式
        result: 结果字典（必须可 JSON 序列化）
    """
    if not result:
        return
    # 计算数据签名
    data_sig = _compute_data_signature(code, start_date, end_date)
    # 数据文件路径：按 cache_key 前两位分桶
    bucket = cache_key[:2]
    bucket_dir = Path(config.RESULT_DB_DATA_DIR) / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    data_file = f'{bucket}/{cache_key}.json'
    data_path = Path(config.RESULT_DB_DATA_DIR) / data_file
    try:
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
    except Exception as e:
        logger.warning(f'[结果库] 写入缓存文件 {cache_key} 失败: {e}')
        return
    # 更新索引
    try:
        conn = _get_conn()
        try:
            conn.execute('''INSERT INTO result_cache
                (cache_key, code, pattern_name, pattern_type, start_date, end_date,
                 observe_day, cautious, result_version, data_signature, data_file,
                 created_at, last_access, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                code=excluded.code, pattern_name=excluded.pattern_name,
                pattern_type=excluded.pattern_type, start_date=excluded.start_date,
                end_date=excluded.end_date, observe_day=excluded.observe_day,
                cautious=excluded.cautious, result_version=excluded.result_version,
                data_signature=excluded.data_signature, data_file=excluded.data_file,
                created_at=excluded.created_at, last_access=excluded.last_access''',
                (cache_key, code, pattern_name, pattern_type,
                 str(start_date).replace('-', ''), str(end_date).replace('-', ''),
                 observe_day, 1 if cautious else 0, RESULT_VERSION, data_sig, data_file,
                 datetime.now().isoformat(), datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[结果库] 写入索引 {cache_key} 失败: {e}')


def _delete_cache(cache_key: str, data_file: str = None):
    """删除缓存（索引 + 数据文件）。"""
    try:
        if not data_file:
            conn = _get_conn()
            try:
                row = conn.execute(
                    'SELECT data_file FROM result_cache WHERE cache_key=?',
                    (cache_key,)
                ).fetchone()
                if row:
                    data_file = row[0]
            finally:
                conn.close()
        # 删除数据文件
        if data_file:
            p = Path(config.RESULT_DB_DATA_DIR) / data_file
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        # 删除索引
        conn = _get_conn()
        try:
            conn.execute('DELETE FROM result_cache WHERE cache_key=?', (cache_key,))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[结果库] 删除缓存 {cache_key} 失败: {e}')


# ============================================================================
# 自动清理过期缓存
# ============================================================================
def cleanup_expired() -> int:
    """清理超过保留期的缓存。返回清理的条数。"""
    cutoff = datetime.now() - timedelta(days=config.RESULT_DB_RETENTION_DAYS)
    cutoff_str = cutoff.isoformat()
    deleted = 0
    try:
        conn = _get_conn()
        try:
            rows = conn.execute(
                'SELECT cache_key, data_file FROM result_cache WHERE created_at < ?',
                (cutoff_str,)
            ).fetchall()
        finally:
            conn.close()
        for cache_key, data_file in rows:
            _delete_cache(cache_key, data_file)
            deleted += 1
        if deleted > 0:
            logger.info(f'[结果库] 清理过期缓存 {deleted} 条')
    except Exception as e:
        logger.warning(f'[结果库] 清理过期缓存失败: {e}')
    return deleted


def get_stats():
    """获取结果库统计信息。"""
    try:
        conn = _get_conn()
        try:
            total = conn.execute('SELECT COUNT(*) FROM result_cache').fetchone()[0]
            by_pattern_type = conn.execute(
                'SELECT pattern_type, COUNT(*) FROM result_cache GROUP BY pattern_type'
            ).fetchall()
            recent = conn.execute(
                'SELECT COUNT(*) FROM result_cache WHERE created_at > ?',
                ((datetime.now() - timedelta(days=7)).isoformat(),)
            ).fetchone()[0]
            oldest = conn.execute(
                'SELECT MIN(created_at) FROM result_cache'
            ).fetchone()[0]
            newest = conn.execute(
                'SELECT MAX(created_at) FROM result_cache'
            ).fetchone()[0]
        finally:
            conn.close()
        # 计算数据目录大小
        data_size = 0
        for p in Path(config.RESULT_DB_DATA_DIR).rglob('*.json'):
            try:
                data_size += p.stat().st_size
            except Exception:
                pass
        return {
            'total_entries': total,
            'by_pattern_type': dict(by_pattern_type),
            'recent_7d': recent,
            'oldest_created': oldest,
            'newest_created': newest,
            'data_size_mb': round(data_size / 1024 / 1024, 2),
            'retention_days': config.RESULT_DB_RETENTION_DAYS,
        }
    except Exception as e:
        logger.warning(f'[结果库] 获取统计失败: {e}')
        return {}
