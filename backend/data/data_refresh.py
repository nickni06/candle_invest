"""集中数据补全模块：从环境变量读取参数，并行拉取指定标的的本地日K数据。

由 web_app.py 通过 subprocess 调用，输出实时进度到 stdout 供 SSE 捕获。
"""
import os
import sys
import json
import time

# 必须在导入 data_source 之前设置 fork 启动方式，避免 macOS 子进程问题
import platform
import multiprocessing

if platform.system() == 'Darwin':
    try:
        multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
import data_source
from signal_tracker import prefetch_data, is_index_code


def main():
    """主入口：从环境变量读取参数，执行集中数据补全。"""
    codes_str = os.environ.get('REFRESH_CODES', '')
    start_date = os.environ.get('REFRESH_START', '')
    end_date = os.environ.get('REFRESH_END', '')

    if not codes_str or not start_date or not end_date:
        print('[数据补全] 错误：缺少必要参数 REFRESH_CODES/REFRESH_START/REFRESH_END', flush=True)
        sys.exit(1)

    codes = [c.strip() for c in codes_str.split(',') if c.strip()]
    print(f'[数据补全] 共 {len(codes)} 个标的，日期范围 {start_date}~{end_date}', flush=True)

    # 按数据目录分组
    a_codes = [c for c in codes if not is_index_code(c)]
    idx_codes = [c for c in codes if is_index_code(c)]
    print(f'[数据补全] 个股 {len(a_codes)}，指数 {len(idx_codes)}', flush=True)

    ok_count = 0
    fail_count = 0
    failed_codes = []

    # 个股数据补全
    if a_codes:
        a_dir = str(config.DAILY_TRACKING_A_DIR)
        print(f'[数据补全] 开始个股数据补全（{len(a_codes)} 个）...', flush=True)
        results = prefetch_data(a_codes, start_date, end_date, a_dir)
        for code, ok in results.items():
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                failed_codes.append({'code': code, 'error': '无数据返回'})

    # 指数数据补全
    if idx_codes:
        idx_dir = str(config.DAILY_TRACKING_INDEX_DIR)
        print(f'[数据补全] 开始指数数据补全（{len(idx_codes)} 个）...', flush=True)
        results = prefetch_data(idx_codes, start_date, end_date, idx_dir)
        for code, ok in results.items():
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                failed_codes.append({'code': code, 'error': '无数据返回'})

    # 输出最终汇总（最后一行为 JSON，便于 web_app 解析）
    summary = {
        'total': len(codes),
        'success': ok_count,
        'failed': fail_count,
        'failed_codes': failed_codes,
    }
    print(f'[数据补全] 完成：共 {len(codes)}，成功 {ok_count}，失败 {fail_count}', flush=True)
    print(f'[数据补全] RESULT_JSON={json.dumps(summary, ensure_ascii=False)}', flush=True)


if __name__ == '__main__':
    main()
