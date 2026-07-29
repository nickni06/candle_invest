"""批量样本采集脚本：从盈湖全量 A 股生成训练样本。

用法:
    # 采集默认 50 只（从 stock_data.csv 随机）
    python backend/ai/batch_collect.py

    # 采集全部 A 股
    python backend/ai/batch_collect.py --n_codes 0

    # 采集指定标的
    python backend/ai/batch_collect.py --code_list 600519.SH,000001.SZ

    # 指定时间范围
    python backend/ai/batch_collect.py --start_date 20150101 --end_date 20260727

    # 同时构建 XGBoost 数据集
    python backend/ai/batch_collect.py --build_xgb
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.sample_collector import (  # noqa: E402
    collect_and_save,
    _load_default_codes_from_stock_data,
)


def main():
    parser = argparse.ArgumentParser(description='批量采集训练样本')
    parser.add_argument('--code_list', help='标的代码列表（逗号分隔），覆盖随机抽取')
    parser.add_argument('--n_codes', type=int, default=None,
                        help='随机抽取标的数（0=全部，默认读 config.AI_SAMPLE_DEFAULT_CODES）')
    parser.add_argument('--start_date', default=None, help='起始日 YYYYMMDD')
    parser.add_argument('--end_date', default=None, help='结束日 YYYYMMDD')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    parser.add_argument('--build_xgb', action='store_true',
                        help='采集后同时构建 XGBoost 数据集')
    args = parser.parse_args()

    # 读取默认配置
    try:
        from config import config
        default_n = getattr(config, 'AI_SAMPLE_DEFAULT_CODES', 50)
        default_start = getattr(config, 'AI_SAMPLE_START_DATE', '20100101')
        default_end = getattr(config, 'AI_SAMPLE_END_DATE', '20260727')
        default_output = str(getattr(config, 'AI_SAMPLE_DIR',
                                     _THIS_DIR / 'data' / 'train'))
    except Exception:
        default_n = 50
        default_start = '20100101'
        default_end = '20260727'
        default_output = str(_THIS_DIR / 'data' / 'train')

    # 决定标的列表
    if args.code_list:
        code_list = [c.strip() for c in args.code_list.split(',') if c.strip()]
    else:
        n = args.n_codes if args.n_codes is not None else default_n
        code_list = _load_default_codes_from_stock_data(n)

    start_date = args.start_date or default_start
    end_date = args.end_date or default_end
    output_dir = args.output_dir or default_output

    print(f'=== 批量样本采集 ===', flush=True)
    print(f'标的数: {len(code_list)}', flush=True)
    print(f'时间范围: {start_date} ~ {end_date}', flush=True)
    print(f'输出目录: {output_dir}', flush=True)
    print(f'前 5 只: {code_list[:5]}', flush=True)

    t0 = time.time()
    stats = collect_and_save(code_list, start_date, end_date, output_dir)
    elapsed = time.time() - t0

    print(f'\n=== 采集完成 ===', flush=True)
    print(f'耗时: {elapsed:.1f}s', flush=True)
    print(f'统计: {stats}', flush=True)

    if args.build_xgb:
        print(f'\n=== 构建 XGBoost 数据集 ===', flush=True)
        from ai.dataset_xgb import build_xgb_dataset_from_npy
        xgb_output = os.path.join(os.path.dirname(output_dir), 'xgb')
        xgb_stats = build_xgb_dataset_from_npy(output_dir, xgb_output)
        print(f'XGBoost 数据集: {xgb_stats}', flush=True)


if __name__ == '__main__':
    main()
