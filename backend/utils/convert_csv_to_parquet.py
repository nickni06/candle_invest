"""批量将现有的日线 CSV 文件转换为 Parquet 格式（保留 CSV 作为备份）。

用法：
    python3 convert_csv_to_parquet.py [数据目录]

默认扫描 config 中定义的所有日线数据目录：
- A股：每日跟踪 / 训练 / 测试
- 指数：每日跟踪 / 训练 / 测试
- ETF：每日跟踪 / 训练 / 测试

转换逻辑：
1. 仅当存在 CSV 且不存在 Parquet（或 Parquet 为空/损坏）时才转换
2. 转换后校验行数、关键列是否一致
3. 保留原 CSV，不删除
4. 使用进程池并行处理，充分利用 CPU
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import pandas as pd

from config import config

logger = logging.getLogger('trader_system')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False


def _data_dirs():
    """返回所有需要扫描的日线数据目录。"""
    dirs = [
        config.DAILY_TRACKING_A_DIR,
        config.DAILY_TRACKING_INDEX_DIR,
        config.DAILY_TRACKING_ETF_DIR,
        config.TRAIN_DATA_A_DIR,
        config.TRAIN_DATA_INDEX_DIR,
        config.TRAIN_DATA_ETF_DIR,
        config.TEST_DATA_A_DIR,
        config.TEST_DATA_INDEX_DIR,
        config.TEST_DATA_ETF_DIR,
    ]
    return [str(d) for d in dirs if d.exists()]


def _convert_one(csv_path: str, verify: bool = True) -> dict:
    """转换单个 CSV 文件为 Parquet。

    Returns:
        {'csv': str, 'parquet': str, 'status': 'converted'|'skipped'|'error',
         'rows': int, 'error': str|None}
    """
    csv_path = Path(csv_path)
    pq_path = csv_path.with_suffix('.parquet')
    result = {'csv': str(csv_path), 'parquet': str(pq_path),
              'status': 'error', 'rows': 0, 'error': None}

    if not _PARQUET_AVAILABLE:
        result['error'] = 'pyarrow 未安装'
        return result

    try:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            result['status'] = 'skipped'
            result['error'] = 'CSV 不存在或为空'
            return result

        # 如果 Parquet 已存在且有效，跳过
        if pq_path.exists() and pq_path.stat().st_size > 0:
            try:
                pq_df = pd.read_parquet(pq_path)
                if not pq_df.empty:
                    result['status'] = 'skipped'
                    result['rows'] = len(pq_df)
                    result['error'] = 'Parquet 已存在且非空'
                    return result
            except Exception:
                pass  # Parquet 损坏，重新生成

        df = pd.read_csv(str(csv_path), dtype={'ts_code': str})
        if df.empty:
            result['status'] = 'skipped'
            result['error'] = 'CSV 为空表'
            return result

        # 统一 trade_date 为字符串，避免 Parquet 类型推断不一致
        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)

        df.to_parquet(pq_path, index=False)

        if verify:
            pq_df = pd.read_parquet(pq_path)
            if len(df) != len(pq_df):
                raise ValueError(f'行数不一致: csv={len(df)}, parquet={len(pq_df)}')
            csv_cols = set(df.columns)
            pq_cols = set(pq_df.columns)
            if csv_cols != pq_cols:
                raise ValueError(f'列不一致: csv_only={csv_cols - pq_cols}, pq_only={pq_cols - csv_cols}')

        result['status'] = 'converted'
        result['rows'] = len(df)
        return result

    except Exception as e:
        result['error'] = str(e)
        # 转换失败时清理可能损坏的 Parquet
        try:
            if pq_path.exists():
                pq_path.unlink()
        except Exception:
            pass
        return result


def convert_all(data_dirs=None, max_workers=None, verify=True):
    """批量转换指定目录下的所有日线 CSV 文件。

    Args:
        data_dirs: 要扫描的目录列表，默认使用 config 中所有日线目录
        max_workers: 并行进程数，默认 CPU 核心数
        verify: 是否校验转换后的 Parquet

    Returns:
        dict: {'converted': int, 'skipped': int, 'failed': int, 'details': [dict]}
    """
    if not _PARQUET_AVAILABLE:
        logger.error('pyarrow 未安装，无法转换。请先执行: pip install pyarrow')
        return {'converted': 0, 'skipped': 0, 'failed': 0, 'details': []}

    dirs = data_dirs or _data_dirs()
    csv_files = []
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith('_daily.csv'):
                csv_files.append(os.path.join(d, f))

    if not csv_files:
        logger.info('未找到需要转换的 CSV 文件')
        return {'converted': 0, 'skipped': 0, 'failed': 0, 'details': []}

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    logger.info(f'发现 {len(csv_files)} 个 CSV 文件，启动 {max_workers} 个进程并行转换...')

    converted = 0
    skipped = 0
    failed = 0
    details = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_convert_one, path, verify): path for path in csv_files}
        for future in as_completed(futures):
            res = future.result()
            details.append(res)
            if res['status'] == 'converted':
                converted += 1
                if converted % 100 == 0:
                    logger.info(f'已转换 {converted} 个，跳过 {skipped} 个，失败 {failed} 个')
            elif res['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1
                logger.warning(f'转换失败 {res["csv"]}: {res["error"]}')

    logger.info(f'转换完成：成功 {converted}，跳过 {skipped}，失败 {failed}')
    return {'converted': converted, 'skipped': skipped, 'failed': failed, 'details': details}


def main():
    parser = argparse.ArgumentParser(description='批量转换日线 CSV 到 Parquet')
    parser.add_argument('dirs', nargs='*', help='可选：指定要扫描的目录')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数')
    parser.add_argument('--no-verify', action='store_true', help='跳过转换后校验')
    args = parser.parse_args()

    dirs = args.dirs if args.dirs else None
    convert_all(data_dirs=dirs, max_workers=args.workers, verify=not args.no_verify)


if __name__ == '__main__':
    main()
