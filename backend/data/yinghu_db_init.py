"""盈湖初始化脚本：拉取全市场股票（北交所除外）2010 年至今的日K数据。

使用方式：
    python3 yinghu_db_init.py                  # 默认全市场，2010-01-01 至今
    python3 yinghu_db_init.py --board hs       # 仅沪深主板
    python3 yinghu_db_init.py --code 000001.SZ # 单标的
    python3 yinghu_db_init.py --start 20200101  # 自定义起始日

执行约束：
- 4-8 进程并行（受 config.TRACKING_PREFETCH_WORKERS 控制）
- 单标超时 60 秒
- 每个 API 失败最多重试 3 次
- 支持断点续跑：已成功的标的跳过

输出：
- 实时进度到 stdout（被 web_app subprocess 捕获）
- 末尾输出 RESULT_JSON=... 便于解析
"""
import os
import sys
import json
import time
import argparse
import platform
import multiprocessing
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# macOS 使用 forkserver，避免 akshare JS 引擎 fork 不安全
if platform.system() == 'Darwin':
    try:
        multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
import yinghu_db
import data_source


# 指数代码（含海外）
INDEX_CODES = ['DJI', 'FCHI', 'SPX', 'N225', 'GDAXI', '000300.SH', '399006.SZ']
INDEX_NAME_MAP = {
    'DJI': '道琼斯', 'FCHI': '法国CAC40', 'SPX': '标普500',
    'N225': '日经225', 'GDAXI': '德国DAX',
    '000300.SH': '沪深300', '399006.SZ': '创业板指',
}


def fetch_all_a_stocks():
    """获取全 A 股标的列表（沪深主板 + 创业板 + 科创板，排除北交所）。"""
    import akshare as ak
    print('[初始化] 拉取全A股列表...', flush=True)
    try:
        df = ak.stock_info_a_code_name()
        codes = []
        for _, row in df.iterrows():
            code = str(row['code'])
            name = str(row['name'])
            # 拼接为 tushare 格式
            if code.startswith('6'):  # 沪市
                ts_code = f'{code}.SH'
            else:  # 深市主板/创业板
                ts_code = f'{code}.SZ'
            # 过滤北交所（akshare stock_info_a_code_name 不含北交所，但兜底过滤）
            if ts_code.endswith('.BJ'):
                continue
            # 板块分类
            board = yinghu_db.classify_code(ts_code)
            if board == 'excluded':
                continue
            codes.append({'code': ts_code, 'name': name, 'board': board})
        # 同步写入 securities 表
        for c in codes:
            yinghu_db.upsert_security(c['code'], c['name'], c['board'])
        print(f'[初始化] 全A股列表拉取完成：{len(codes)} 个标的', flush=True)
        return codes
    except Exception as e:
        print(f'[初始化] 拉取全A股列表失败: {e}', flush=True)
        return []


def fetch_etf_list():
    """获取 ETF 列表。"""
    import akshare as ak
    print('[初始化] 拉取 ETF 列表...', flush=True)
    try:
        df = ak.fund_etf_category_sina(symbol='ETF基金')
        codes = []
        for _, row in df.iterrows():
            code = str(row['代码'])
            name = str(row['名称'])
            if code.startswith('5'):
                ts_code = f'{code}.SH'
            elif code.startswith('1'):
                ts_code = f'{code}.SZ'
            else:
                continue
            board = yinghu_db.classify_code(ts_code)
            if board == 'etf':
                codes.append({'code': ts_code, 'name': name, 'board': 'etf'})
                yinghu_db.upsert_security(ts_code, name, 'etf')
        print(f'[初始化] ETF 列表拉取完成：{len(codes)} 个', flush=True)
        return codes
    except Exception as e:
        print(f'[初始化] 拉取 ETF 列表失败: {e}', flush=True)
        return []


def init_index_list():
    """初始化指数列表。"""
    print('[初始化] 写入指数列表...', flush=True)
    for code, name in INDEX_NAME_MAP.items():
        yinghu_db.upsert_security(code, name, 'index')


def _fetch_one_code(task):
    """子进程 worker：拉取单个标的的全历史数据并写入盈湖。"""
    code, name, board, start_date, end_date = task
    try:
        # 盈湖已覆盖则跳过
        if yinghu_db.check_coverage(code, start_date, end_date):
            return code, True, 'already_covered', 0
        # 拉取数据（get_kline_df 内部会自动入盈湖）
        df = data_source.get_kline_df(code, start_date, end_date,
                                       prefer_local=False, allow_network=True)
        if df is None or df.empty:
            return code, False, 'no_data', 0
        # 盈湖写入由 get_kline_df 内部完成，这里只统计行数
        rows = len(df)
        return code, True, '', rows
    except Exception as e:
        return code, False, str(e), 0


def run_init(start_date, end_date, board_filter=None, code_filter=None,
              include_etf=True, include_index=True, workers=None):
    """执行盈湖初始化。

    Args:
        start_date: 起始日 YYYYMMDD
        end_date: 结束日 YYYYMMDD
        board_filter: 仅初始化指定板块（hs/cy/kc/index/etf），None=全部
        code_filter: 仅初始化指定代码列表，None=全市场
        include_etf: 是否包含 ETF
        include_index: 是否包含指数
        workers: 并行进程数，None=自动
    """
    print(f'[初始化] 盈湖初始化开始：{start_date} ~ {end_date}', flush=True)
    print(f'[初始化] 板块过滤: {board_filter or "全部"}，标的过滤: {"指定" if code_filter else "全市场"}', flush=True)

    # 1. 获取标的列表
    if code_filter:
        # 自定义标的列表
        targets = []
        for code in code_filter:
            code = str(code)
            if code.endswith('.BJ'):
                continue
            board = yinghu_db.classify_code(code)
            if board == 'excluded':
                continue
            if board_filter and board != board_filter:
                continue
            sec = yinghu_db.get_security(code)
            name = sec['name'] if sec else ''
            targets.append({'code': code, 'name': name, 'board': board})
    else:
        # 拉取全市场列表
        targets = []
        if not board_filter or board_filter in ('hs', 'cy', 'kc'):
            targets.extend(fetch_all_a_stocks())
        if include_index and (not board_filter or board_filter == 'index'):
            init_index_list()
            for code, name in INDEX_NAME_MAP.items():
                targets.append({'code': code, 'name': name, 'board': 'index'})
        if include_etf and (not board_filter or board_filter == 'etf'):
            targets.extend(fetch_etf_list())
        # 应用板块过滤
        if board_filter:
            targets = [t for t in targets if t['board'] == board_filter]

    total = len(targets)
    print(f'[初始化] 待初始化标的: {total} 个', flush=True)
    if total == 0:
        print('[初始化] 无待初始化标的', flush=True)
        return

    # 2. 构造任务列表
    tasks = [(t['code'], t['name'], t['board'], start_date, end_date) for t in targets]

    # 3. 并行拉取
    if workers is None:
        workers = getattr(config, 'TRACKING_PREFETCH_WORKERS', 4)
    workers = max(1, min(workers, getattr(config, 'MAX_WORKERS', 8), total))
    timeout = getattr(config, 'PREFETCH_TIMEOUT_SECONDS', 60)

    print(f'[初始化] 启动并行拉取：{workers} 进程，单标超时 {timeout}s', flush=True)

    ok_count = 0
    fail_count = 0
    skip_count = 0
    rows_total = 0
    failed_codes = []
    start_time = time.time()
    last_report = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_code = {
            executor.submit(_fetch_one_code, task): task[0]
            for task in tasks
        }
        for i, future in enumerate(as_completed(future_to_code), 1):
            code = future_to_code[future]
            try:
                _, success, msg, rows = future.result(timeout=timeout)
            except Exception as e:
                success, msg, rows = False, str(e), 0
            if success:
                ok_count += 1
                rows_total += rows
                if msg == 'already_covered':
                    skip_count += 1
            else:
                fail_count += 1
                failed_codes.append({'code': code, 'error': msg[:100]})
            # 进度汇报（每 50 个或最后）
            if i - last_report >= 50 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_sec = (total - i) / rate if rate > 0 else 0
                eta_min = eta_sec / 60
                print(f'[初始化] 进度: {i}/{total} ({i*100//total}%) '
                       f'- 成功 {ok_count} (跳过 {skip_count}), 失败 {fail_count}, '
                       f'速率 {rate:.1f}/s, 预计剩余 {eta_min:.1f} 分钟',
                       flush=True)
                last_report = i

    elapsed = time.time() - start_time
    summary = {
        'total': total,
        'success': ok_count,
        'skipped': skip_count,
        'failed': fail_count,
        'rows_added': rows_total,
        'elapsed_seconds': round(elapsed, 1),
        'failed_codes': failed_codes[:50],  # 只保留前 50 个失败样例
    }
    print(f'[初始化] 完成：共 {total}，成功 {ok_count} (跳过 {skip_count})，'
           f'失败 {fail_count}，耗时 {elapsed:.1f}s', flush=True)
    print(f'[初始化] RESULT_JSON={json.dumps(summary, ensure_ascii=False)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='盈湖初始化')
    parser.add_argument('--start', default=config.YINGHU_DB_START_DATE,
                        help='起始日期 YYYYMMDD（默认 20100101）')
    parser.add_argument('--end', default=datetime.now().strftime('%Y%m%d'),
                        help='结束日期 YYYYMMDD（默认今天）')
    parser.add_argument('--board', choices=['hs', 'cy', 'kc', 'index', 'etf'],
                        help='仅初始化指定板块')
    parser.add_argument('--code', help='仅初始化指定代码（逗号分隔）')
    parser.add_argument('--no-etf', action='store_true', help='不初始化 ETF')
    parser.add_argument('--no-index', action='store_true', help='不初始化指数')
    parser.add_argument('--workers', type=int, help='并行进程数')
    args = parser.parse_args()

    code_filter = args.code.split(',') if args.code else None
    run_init(
        start_date=args.start.replace('-', ''),
        end_date=args.end.replace('-', ''),
        board_filter=args.board,
        code_filter=code_filter,
        include_etf=not args.no_etf,
        include_index=not args.no_index,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
