"""信号跟踪调度器（多进程并行 + 父进程预拉取 + 结构化结果）。

替代旧 tracking.py 的 tracking() 函数，核心改进：
1. worker 直接返回结构化信号 dict，不再依赖日志正则解析
2. 单次 run 只写一次 summary.json，mode='all' 在 run 内合并，不依赖文件级 append
3. 进度通过 print 实时输出（被 web_app subprocess 捕获）+ 结构化 summary.json
4. 停止通过 SIGTERM 整个进程组实现（web_app 负责）

业务逻辑保留点：
- ProcessPoolExecutor 并行（worker 数取 config.TRACKING_POOL_WORKERS 或 SCAN_WORKERS）
- 父进程串行预拉取（akshare 内部 JS 引擎不线程安全，并行会崩溃）
- 单个标的失败不影响进程池（try/except 兜底）
- 指数/个股按数据目录分组，目标个股单独跟踪
"""
import os
import json
import logging
import multiprocessing
import platform

# 多进程启动模式：
# - Linux 默认 fork，速度快且对纯计算任务稳定
# - macOS 使用 forkserver，避免 Objective-C / V8 / OpenBLAS 等库因 fork 不安全而崩溃，
#   同时比 spawn 大幅节省子进程初始化时间
# - Windows 仅支持 spawn
if platform.system() == 'Darwin':
    multiprocessing.set_start_method('forkserver', force=True)
elif platform.system() == 'Linux':
    multiprocessing.set_start_method('fork', force=True)
# Windows 保持默认 spawn

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from config import config
import data_source

logger = logging.getLogger('trader_system')

# 指数代码固定列表
INDEX_CODES = ['DJI', 'FCHI', 'SPX', 'N225', 'GDAXI', '000300.SH', '399006.SZ']
INDEX_NAME_MAP = {
    'DJI': '道琼斯', 'FCHI': '法国CAC40', 'SPX': '标普500',
    'N225': '日经225', 'GDAXI': '德国DAX',
    '000300.SH': '沪深300', '399006.SZ': '创业板指',
}

# stock_data.csv 缓存
_stock_data_cache = {'df': None, 'mtime': 0}


def _load_stock_data():
    """加载 stock_data.csv，带 mtime 缓存。"""
    path = str(config.STOCK_DATA_FILE)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        mtime = os.path.getmtime(path)
        if _stock_data_cache['df'] is not None and _stock_data_cache['mtime'] == mtime:
            return _stock_data_cache['df']
        df = pd.read_csv(path)
        _stock_data_cache['df'] = df
        _stock_data_cache['mtime'] = mtime
        return df
    except Exception as e:
        logger.warning(f'[tracker] 加载 stock_data.csv 失败: {e}')
        return pd.DataFrame()


def get_code_name(code):
    """查询 code 对应的名称，指数用内置映射，个股查 stock_data.csv。"""
    if code in INDEX_NAME_MAP:
        return INDEX_NAME_MAP[code]
    df = _load_stock_data()
    if df.empty:
        return ''
    try:
        m = df[df['ts_code'] == code]['name']
        return str(m.values[0]) if len(m) > 0 else ''
    except Exception:
        return ''


def is_index_code(code):
    """判断是否为指数代码。"""
    code = str(code)
    return len(code) < 9 or code in ['000300.SH', '399006.SZ']


# ============================================================================
# worker：跟踪单个标的（模块级函数，可被 pickle）
# ============================================================================
def _track_one_code(task_args):
    """进程池 worker：计算单个标的的信号。

    Args:
        task_args: (code, code_name, start_date, end_date, track_date,
                    cautious, is_index, data_folder_dir, perf_dir, allow_network,
                    track_patterns, held_codes)

    Returns:
        {
            'code': str, 'name': str, 'is_index': bool,
            'signals': [...],  # 结构化信号列表
            'error': str,      # 异常时填充
        }
    """
    (code, code_name, start_date, end_date, track_date,
     cautious, is_index, data_folder_dir, perf_dir, allow_network,
     track_patterns, held_codes) = task_args

    try:
        # 第一次 allow_network=False（子进程禁网），重试时 allow_network=True
        df = data_source.get_kline_df(code, start_date, end_date,
                                       prefer_local=True, allow_network=allow_network)
        if df is None or df.empty:
            # fallback：尝试直接读 data_folder_dir 下的 Parquet/CSV
            pq_path = os.path.join(data_folder_dir, f'{code}_daily.parquet')
            if os.path.exists(pq_path) and os.path.getsize(pq_path) > 0:
                try:
                    df = pd.read_parquet(pq_path)
                except Exception:
                    df = None
            if df is None or df.empty:
                csv_path = os.path.join(data_folder_dir, f'{code}_daily.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)

        if df is None or df.empty:
            return {'code': code, 'name': code_name, 'is_index': is_index,
                    'signals': [], 'error': '本地无数据' + ('' if not allow_network else '且网络拉取失败')}

        # 调用纯 TA-Lib 信号计算（卖出信号仅对持仓标的计算）
        import strategy_signals
        result = strategy_signals.compute_signals_for_code(
            df=df,
            code=code,
            code_name=code_name,
            track_date=track_date,
            cautious=cautious,
            is_index=is_index,
            perf_dir=perf_dir,
            track_patterns=track_patterns,
            held_codes=held_codes,
            data_folder_dir=data_folder_dir,
        )
        # 标记是否为定向跟踪
        if track_patterns:
            for sig in result.get('signals', []):
                sig['is_configured'] = True
        return result
    except Exception as e:
        import traceback
        return {'code': code, 'name': code_name, 'is_index': is_index,
                'signals': [], 'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}


# ============================================================================
# 父进程预拉取（进程级并行，akshare JS 引擎在每个子进程内独立）
# ============================================================================
def _prefetch_one_code(args):
    """预拉取 worker：每个子进程拥有独立的 akshare JS 引擎，可安全并行。

    优先走盈湖（Yinghu DB）：覆盖命中即跳过网络拉取；
    未命中才调 get_kline_df 触发网络拉取，并由其内部完成盈湖入库。
    """
    code, start_date, end_date, folder = args
    try:
        # 0. 盈湖优先：覆盖范围命中即跳过网络拉取
        try:
            import yinghu_db
            if yinghu_db.check_coverage(code, start_date, end_date):
                return code, True, 'yinghu_db_hit'
        except Exception:
            pass
        # 1. 盈湖未命中，调 get_kline_df 触发网络拉取（内部会自动入盈湖）
        df = data_source.get_kline_df(code, start_date, end_date,
                                       prefer_local=True, allow_network=True)
        if df is not None and not df.empty:
            # 兼容旧目录写入（保留一段时间，直到完全切换到盈湖）
            if folder:
                data_source.save_kline_to_local(code, df, folder)
            return code, True, ''
        else:
            return code, False, '无数据返回'
    except Exception as e:
        return code, False, str(e)


def prefetch_data(code_list, start_date, end_date, folder):
    """父进程并行预拉取日 K 数据到本地。

    说明：akshare 的 libmini_racer 是线程不安全，但进程之间隔离；
    使用 ProcessPoolExecutor 在每个子进程内独立调用 akshare，实现并行预拉取。

    Returns:
        dict: {code: True/False} 表示预拉取是否成功
    """
    results = {}
    total = len(code_list)
    if total == 0:
        return results

    workers = getattr(config, 'TRACKING_PREFETCH_WORKERS', 4)
    if workers <= 0:
        workers = min(multiprocessing.cpu_count(), 8)
    workers = min(workers, total)
    timeout = getattr(config, 'PREFETCH_TIMEOUT_SECONDS', 60)

    print(f'[预拉取] 启动并行预拉取：共 {total} 个标的，{workers} 进程', flush=True)

    ok = 0
    failed = 0
    if workers <= 1:
        # 兜底串行模式
        for code in code_list:
            _, success, err = _prefetch_one_code((code, start_date, end_date, folder))
            results[code] = success
            if success:
                ok += 1
            else:
                failed += 1
                print(f'[预拉取] {code} 失败: {err}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_code = {
                executor.submit(_prefetch_one_code, (code, start_date, end_date, folder)): code
                for code in code_list
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    _, success, err = future.result(timeout=timeout)
                except Exception as e:
                    success = False
                    err = str(e)
                results[code] = success
                if success:
                    ok += 1
                else:
                    failed += 1
                    print(f'[预拉取] {code} 失败: {err}', flush=True)
                done = ok + failed
                if done % 50 == 0 or done == total:
                    print(f'[预拉取] 进度：{done}/{total}（成功 {ok}，失败 {failed}）', flush=True)

    print(f'[预拉取] 完成：共 {total}，成功 {ok}，失败 {failed}', flush=True)
    return results


# ============================================================================
# 主入口：跟踪一组 code_list
# ============================================================================
def _build_track_patterns(code):
    """查询某 code 的定向跟踪形态列表。

    Returns:
        list[dict] | None: 形态列表 [{'pattern','pattern_type'}]；无配置返回 None（跑全量）
    """
    try:
        import strategy_config
        names, types, _observe_days = strategy_config.get_tracking_patterns(code)
        if not names:
            return None
        return [{'pattern': p, 'pattern_type': t} for p, t in zip(names, types)]
    except Exception as e:
        logger.warning(f'[tracker] 查询 {code} 定向配置失败: {e}')
        return None


def track_code_list(code_list, start_date, track_date, cautious,
                     data_folder_dir, perf_dir, label='', track_mode='full',
                     held_codes=None, global_progress=None):
    """跟踪一组标的，返回结构化结果。

    Args:
        code_list: 标的代码列表
        start_date: 数据起始日 YYYYMMDD
        track_date: 跟踪日 YYYY-MM-DD
        cautious: 谨慎模式
        data_folder_dir: 数据目录（如 '数据/A股/每日跟踪/'）
        perf_dir: 策略表现 CSV 目录
        label: 日志标签（如 '指数'/'个股'/'目标个股'）
        track_mode: 跟踪模式
            'full' - 全量跟踪，所有 code 跑全量形态
            'directional' - 定向跟踪，已配置的 code 只跑配置形态，未配置的跑全量
            'configured_only' - 仅跟踪已配置的 code，未配置的跳过
        held_codes: 持仓代码列表 set；非空时，不在列表中的标的不计算卖出信号
        global_progress: 可选 dict {'total': int, 'done': int, 'failed': int}，
            用于在 run_tracking 合并多组跟踪时输出统一的全局进度；本函数会原子更新 done/failed。

    Returns:
        {
            'total': int, 'success': int, 'failed': int,
            'failed_codes': [{'code', 'name', 'error'}],
            'signals': [...],  # 所有标的的信号汇总
            'codes': [{'code', 'name', 'is_index'}],  # 本次跟踪的标的列表
        }
    """
    end_date = track_date.replace('-', '')

    # 标的列表（含名称、是否指数）；定向/仅配置模式下过滤
    codes_info = []
    skipped_no_config = 0
    directional_count = 0
    skipped_bj = 0
    for code in code_list:
        code = str(code)
        # 跳过北交所股票：akshare/腾讯/新浪 均不支持 .BJ 后缀的代码格式
        if code.endswith('.BJ'):
            skipped_bj += 1
            continue
        name = get_code_name(code)
        is_index = is_index_code(code)
        # 指数不做定向跟踪（配置仅对个股生效）
        if track_mode in ('directional', 'configured_only') and not is_index:
            track_patterns = _build_track_patterns(code)
            if track_patterns is None:
                # 未配置
                if track_mode == 'configured_only':
                    skipped_no_config += 1
                    continue
                # directional：未配置走全量
                tp_for_task = None
            else:
                tp_for_task = track_patterns
                directional_count += 1
        else:
            tp_for_task = None
        codes_info.append({'code': code, 'name': name, 'is_index': is_index,
                            'track_patterns': tp_for_task})

    if skipped_bj > 0:
        print(f'[{label}] 跳过 {skipped_bj} 个北交所(.BJ)股票（数据源不支持）', flush=True)
    if skipped_no_config > 0:
        print(f'[{label}] 仅配置模式：跳过 {skipped_no_config} 个未配置个股', flush=True)
    if directional_count > 0:
        print(f'[{label}] 定向模式：{directional_count} 个个股按配置跟踪', flush=True)

    total = len(codes_info)
    # 修正全局总数：跳过 .BJ / 未配置 等过滤掉的标的，避免进度条永远到不了 100%
    if global_progress is not None:
        skipped_in_this_call = len(code_list) - total
        if skipped_in_this_call > 0:
            global_progress['total'] = max(0, global_progress['total'] - skipped_in_this_call)
    result = {
        'total': total, 'success': 0, 'failed': 0,
        'failed_codes': [], 'signals': [], 'codes': [
            {'code': c['code'], 'name': c['name'], 'is_index': c['is_index']}
            for c in codes_info
        ],
    }
    if total == 0:
        return result

    print(f'[{label}] 启动跟踪：{total} 个标的（track_mode={track_mode}）', flush=True)

    # 父进程预拉取：检查本地数据是否覆盖 [start_date, end_date] 范围
    # 不只看文件是否存在，还要看日期范围（避免用旧数据算形态导致每次信号相同）
    # 大数据量时此检查较慢（5000+标的每标的读一次CSV），输出进度避免用户等待焦虑
    print(f'[{label}] 检查 {total} 个标的数据覆盖范围...', flush=True)
    check_step = max(1, total // 20)
    missing = []
    for ci, info in enumerate(codes_info):
        if not data_source._local_data_has_date_range(info['code'], start_date, end_date):
            missing.append(info['code'])
        if (ci + 1) % check_step == 0 or ci + 1 == total:
            print(f'[{label}] 数据检查进度：{ci+1}/{total} {((ci+1)*100)//total}%', flush=True)
    if missing:
        print(f'[{label}] 父进程补全预拉取 {len(missing)} 个本地数据缺失/过期的标的...', flush=True)
        prefetch_data(missing, start_date, end_date, data_folder_dir)

    # 构造任务列表（第一次：子进程禁网，只读本地 CSV）
    tasks = []
    for info in codes_info:
        tasks.append((
            info['code'], info['name'], start_date, end_date, track_date,
            cautious, info['is_index'], data_folder_dir, perf_dir, False,
            info['track_patterns'], held_codes,
        ))

    # 决定 worker 数（受全局 MAX_WORKERS 上限约束）
    workers = getattr(config, 'TRACKING_POOL_WORKERS', None) or config.SCAN_WORKERS
    if workers <= 0:
        workers = multiprocessing.cpu_count()
    workers = min(workers, total, getattr(config, 'MAX_WORKERS', 4))
    workers = max(1, workers)
    worker_timeout = getattr(config, 'WORKER_TIMEOUT_SECONDS', 30)

    # 进度汇报：同时输出本组进度和全局进度（如有）
    progress_step = max(1, total // 100)
    done = 0
    failed = 0
    last_reported_done = 0
    last_reported_failed = 0

    def _emit_progress(force=False):
        nonlocal last_reported_done, last_reported_failed
        # 已经报到过 total，不再重复输出
        if last_reported_done == total:
            return
        if not force and done % progress_step != 0 and done != total:
            return
        delta_done = done - last_reported_done
        delta_failed = failed - last_reported_failed
        if global_progress is not None:
            global_progress['done'] += delta_done
            global_progress['failed'] += delta_failed
        local_pct = done * 100 // total if total > 0 else 0
        print(f'[{label}] 进度：{done}/{total}（{local_pct}%，失败 {failed}）', flush=True)
        if global_progress is not None:
            g_total = global_progress['total']
            g_done = global_progress['done']
            g_failed = global_progress['failed']
            g_pct = g_done * 100 // g_total if g_total > 0 else 0
            print(f'[全局] 进度：{g_done}/{g_total}（已完成 {g_done}，剩余 {g_total - g_done}，失败 {g_failed}，{g_pct}%）', flush=True)
        last_reported_done = done
        last_reported_failed = failed

    if workers <= 1:
        # 串行模式
        for task_args in tasks:
            r = _track_one_code(task_args)
            done += 1
            if r['error']:
                failed += 1
                result['failed_codes'].append({'code': r['code'], 'name': r['name'], 'error': r['error']})
                print(f'[{label}] Failed: {r["code"]} - {r["error"]}', flush=True)
            else:
                result['signals'].extend(r['signals'])
            _emit_progress()
    else:
        # 并行模式
        print(f'[{label}] {workers} 进程并行', flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_code = {executor.submit(_track_one_code, t): t[0] for t in tasks}
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    r = future.result(timeout=worker_timeout)
                except TimeoutError:
                    r = {'code': str(code), 'name': get_code_name(str(code)),
                         'is_index': is_index_code(str(code)),
                         'signals': [], 'error': f'任务超时（{worker_timeout}秒）'}
                except Exception as e:
                    r = {'code': str(code), 'name': get_code_name(str(code)),
                         'is_index': is_index_code(str(code)),
                         'signals': [], 'error': str(e)}
                done += 1
                if r['error']:
                    failed += 1
                    result['failed_codes'].append({'code': r['code'], 'name': r['name'], 'error': r['error']})
                    print(f'[{label}] Failed: {r["code"]} - {r["error"]}', flush=True)
                else:
                    result['signals'].extend(r['signals'])
                _emit_progress()

    # 确保最终进度一定输出（即使进度步长未对齐）
    _emit_progress(force=True)
    result['success'] = total - failed
    result['failed'] = failed
    print(f'[{label}] 完成：共 {total}，成功 {total - failed}，失败 {failed}', flush=True)

    # 自动重试失败的标的（允许网络拉取）
    if result['failed_codes']:
        retry_codes = [fc['code'] for fc in result['failed_codes']]
        print(f'[{label}] 自动重试 {len(retry_codes)} 个失败标的（允许网络拉取）: {retry_codes}', flush=True)
        # 重建 code -> track_patterns 映射，重试时保持定向配置
        retry_tp_map = {c['code']: c.get('track_patterns') for c in codes_info}
        retry_tasks = []
        for fc in result['failed_codes']:
            retry_tasks.append((
                fc['code'], fc['name'], start_date, end_date, track_date,
                cautious, is_index_code(fc['code']), data_folder_dir, perf_dir, True,
                retry_tp_map.get(fc['code']), held_codes,
            ))
        retry_workers = min(max(1, workers), len(retry_tasks))
        retry_failed = []
        retry_recovered = 0
        if retry_workers <= 1:
            for task_args in retry_tasks:
                r = _track_one_code(task_args)
                if r['error']:
                    retry_failed.append({'code': r['code'], 'name': r['name'], 'error': r['error']})
                    print(f'[{label}] 重试仍失败: {r["code"]} - {r["error"]}', flush=True)
                else:
                    result['signals'].extend(r['signals'])
                    retry_recovered += 1
                    print(f'[{label}] 重试成功: {r["code"]}（{len(r["signals"])} 个信号）', flush=True)
        else:
            print(f'[{label}] {retry_workers} 进程并行重试', flush=True)
            with ProcessPoolExecutor(max_workers=retry_workers) as executor:
                future_to_code = {executor.submit(_track_one_code, t): t[0] for t in retry_tasks}
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        r = future.result(timeout=worker_timeout)
                    except TimeoutError:
                        r = {'code': str(code), 'name': get_code_name(str(code)),
                             'is_index': is_index_code(str(code)),
                             'signals': [], 'error': f'重试超时（{worker_timeout}秒）'}
                    except Exception as e:
                        r = {'code': str(code), 'name': get_code_name(str(code)),
                             'is_index': is_index_code(str(code)),
                             'signals': [], 'error': str(e)}
                    if r['error']:
                        retry_failed.append({'code': r['code'], 'name': r['name'], 'error': r['error']})
                        print(f'[{label}] 重试仍失败: {r["code"]} - {r["error"]}', flush=True)
                    else:
                        result['signals'].extend(r['signals'])
                        retry_recovered += 1
                        print(f'[{label}] 重试成功: {r["code"]}（{len(r["signals"])} 个信号）', flush=True)
        # 更新统计：重试成功的从 failed 移到 success
        result['failed_codes'] = retry_failed
        result['failed'] = len(retry_failed)
        result['success'] = total - result['failed']
        if retry_recovered > 0:
            print(f'[{label}] 重试完成: 恢复 {retry_recovered} 个，仍失败 {len(retry_failed)} 个', flush=True)

    return result


# ============================================================================
# 顶层入口：run_tracking（替代旧 tracking.run_tracking）
# ============================================================================
def _load_held_codes():
    """读取持仓标的代码列表，用于卖出信号限制。

    Returns:
        set: 持仓代码集合；读取失败或为空时返回空 set
    """
    try:
        import pandas as pd
        path = getattr(config, 'POSITION_FILE', None)
        if path is None or not path.exists():
            return set()
        df = pd.read_csv(path, dtype={'code': str})
        codes = set(str(c).strip() for c in df['code'].dropna())
        return codes
    except Exception as e:
        logger.debug(f'[tracker] 读取持仓列表失败: {e}')
        return set()


def run_tracking(track_date, mode, cautious,
                  target_codes=None, min_mv=None, max_mv=None, to_log=True,
                  lookback_days=None, track_mode='full'):
    """运行跟踪任务（替代旧 tracking.run_tracking）。

    数据起始日自动计算：track_date 往前推 lookback_days 个自然日（默认从配置读取，通常 15 天）。
    TA-Lib 形态识别最长需 5 根 K 线，15 天约 10 个交易日已足够覆盖。

    卖出信号策略：仅对持仓标的生成卖出信号（通过 held_codes 控制）。
    指数始终生成完整信号（买入+卖出）。

    Args:
        track_date: 跟踪日 YYYY-MM-DD
        mode: 'index' / 'stock' / 'all'
        cautious: 谨慎模式
        target_codes: 目标个股列表（额外单独跟踪）
        min_mv: 总市值下限（单位：万元），None 不限制
        max_mv: 总市值上限（单位：万元），None 不限制
        to_log: 是否写日志文件（保留参数，新实现通过 print 输出，由 web_app 捕获）
        lookback_days: 向前回溯的自然日数（默认 30 天）
        track_mode: 跟踪模式 'full' / 'directional' / 'configured_only'

    Returns:
        dict: 完整的 summary（同时写入 log/{YYYYMMDD}_summary.json）
    """
    target_codes = target_codes or []

    # 读取持仓列表：卖出信号仅对持仓标的生成
    held_codes = _load_held_codes()
    if held_codes:
        print(f'[tracker] 持仓标的 {len(held_codes)} 个，卖出信号仅对这些标的计算', flush=True)
    else:
        print(f'[tracker] 无持仓数据，卖出信号将不生成（不计算任何卖出）', flush=True)

    # 自动计算数据起始日：track_date 往前推 lookback_days 个自然日
    if lookback_days is None:
        lookback_days = getattr(config, 'TRACKING_LOOKBACK_DAYS', 15)
    from datetime import datetime, timedelta
    track_dt = datetime.strptime(track_date, '%Y-%m-%d')
    start_dt = track_dt - timedelta(days=lookback_days)
    start_date = start_dt.strftime('%Y%m%d')
    print(f'[tracker] 数据起始日自动计算：{start_date}（往前 {lookback_days} 天），'
          f'track_mode={track_mode}', flush=True)

    # 直接使用 config 中的路径，避免字符串拼接导致目录不匹配
    index_tracking_dir = str(config.DAILY_TRACKING_INDEX_DIR)
    a_tracking_dir = str(config.DAILY_TRACKING_A_DIR)
    index_perf_dir = str(config.INDEX_PERFORMANCE_DIR)
    a_perf_dir = str(config.STOCK_PERFORMANCE_DIR)

    # 按 mode 筛选 code_list
    stock_data_df = _load_stock_data()
    index_list = INDEX_CODES
    if stock_data_df.empty:
        stock_list = []
        print('[tracker] 警告: stock_data.csv 为空，A 股个股跟踪范围 0 只', flush=True)
    else:
        mask = pd.Series([True] * len(stock_data_df), index=stock_data_df.index)
        if min_mv is not None:
            mask &= stock_data_df['total_mv'] >= min_mv
        if max_mv is not None:
            mask &= stock_data_df['total_mv'] <= max_mv
        stock_list = stock_data_df[mask]['ts_code'].tolist()
        mv_desc = ''
        if min_mv is not None and max_mv is not None:
            mv_desc = f'[{min_mv/10000:.0f}亿, {max_mv/10000:.0f}亿]'
        elif min_mv is not None:
            mv_desc = f'>= {min_mv/10000:.0f}亿'
        elif max_mv is not None:
            mv_desc = f'<= {max_mv/10000:.0f}亿'
        else:
            mv_desc = '全市场（无市值限制）'
        print(f'[tracker] A 股个股筛选条件：{mv_desc}，共 {len(stock_list)} 只', flush=True)

    # 按模式分组跟踪
    # 先规划所有调用，计算总标的数，再统一输出全局进度
    calls = []  # (args, kwargs) 列表
    if mode == 'all':
        calls.append((
            (index_list, start_date, track_date, cautious, index_tracking_dir, index_perf_dir),
            {'label': '指数', 'track_mode': 'full'}
        ))
        calls.append((
            (stock_list, start_date, track_date, cautious, a_tracking_dir, a_perf_dir),
            {'label': '个股', 'track_mode': track_mode, 'held_codes': held_codes}
        ))
    elif mode == 'index':
        calls.append((
            (index_list, start_date, track_date, cautious, index_tracking_dir, index_perf_dir),
            {'label': '指数', 'track_mode': 'full'}
        ))
    elif mode == 'stock':
        calls.append((
            (stock_list, start_date, track_date, cautious, a_tracking_dir, a_perf_dir),
            {'label': '个股', 'track_mode': track_mode, 'held_codes': held_codes}
        ))
    elif mode == 'position':
        if held_codes:
            pos_list = list(held_codes)
            idx_pos = [c for c in pos_list if is_index_code(c)]
            stk_pos = [c for c in pos_list if not is_index_code(c)]
            print(f'[tracker] === 跟踪持仓标的：指数 {len(idx_pos)} 个，个股 {len(stk_pos)} 个 ===', flush=True)
            if idx_pos:
                calls.append((
                    (idx_pos, start_date, track_date, cautious, index_tracking_dir, index_perf_dir),
                    {'label': '持仓指数', 'track_mode': 'full', 'held_codes': held_codes}
                ))
            if stk_pos:
                calls.append((
                    (stk_pos, start_date, track_date, cautious, a_tracking_dir, a_perf_dir),
                    {'label': '持仓个股', 'track_mode': 'full', 'held_codes': held_codes}
                ))
        else:
            print('[tracker] 无持仓数据，跳过跟踪', flush=True)

    # 目标个股单独跟踪
    if target_codes:
        seen = set()
        target_codes = [c for c in target_codes if not (c in seen or seen.add(c))]
        idx_targets = [c for c in target_codes if is_index_code(c)]
        stk_targets = [c for c in target_codes if not is_index_code(c)]
        print(f'[tracker] === 跟踪目标个股（指数 {len(idx_targets)}，个股 {len(stk_targets)}） ===', flush=True)
        if idx_targets:
            calls.append((
                (idx_targets, start_date, track_date, cautious, index_tracking_dir, index_perf_dir),
                {'label': '目标指数', 'track_mode': 'full'}
            ))
        if stk_targets:
            calls.append((
                (stk_targets, start_date, track_date, cautious, a_tracking_dir, a_perf_dir),
                {'label': '目标个股', 'track_mode': track_mode, 'held_codes': held_codes}
            ))

    grand_total = sum(len(c[0][0]) for c in calls)
    global_progress = {'total': grand_total, 'done': 0, 'failed': 0}
    print(f'[tracker] 本次跟踪总计：{grand_total} 个标的，分 {len(calls)} 组执行', flush=True)

    all_results = []
    for args, kwargs in calls:
        kwargs['global_progress'] = global_progress
        print(f'[tracker] === 开始跟踪 {kwargs.get("label", "")} ===', flush=True)
        all_results.append(track_code_list(*args, **kwargs))

    # 用实际完成数修正最终全局进度（configured_only 等模式可能有跳过）
    actual_total = sum(r['total'] for r in all_results)
    actual_done = sum(r['success'] for r in all_results)
    actual_failed = sum(r['failed'] for r in all_results)
    if actual_total > 0:
        final_pct = actual_done * 100 // actual_total
        print(f'[全局] 进度：{actual_done}/{actual_total}（已完成 {actual_done}，剩余 {actual_total - actual_done}，失败 {actual_failed}，{final_pct}%）', flush=True)

    # 合并所有结果为单个 summary
    summary = _merge_results(all_results, track_date)
    summary['track_mode'] = track_mode
    _write_summary(summary, track_date)
    return summary


def _merge_results(results_list, track_date):
    """合并多次 track_code_list 的结果为单个 summary。"""
    total = sum(r['total'] for r in results_list)
    success = sum(r['success'] for r in results_list)
    failed = sum(r['failed'] for r in results_list)
    all_signals = []
    all_codes = []
    all_failed = []
    for r in results_list:
        all_signals.extend(r['signals'])
        all_codes.extend(r['codes'])
        all_failed.extend(r['failed_codes'])

    # 按 code 去重（mode='all' + target_codes 可能重复）
    seen_codes = set()
    unique_codes = []
    for c in all_codes:
        if c['code'] not in seen_codes:
            seen_codes.add(c['code'])
            unique_codes.append(c)

    index_codes = [{'code': c['code'], 'name': c['name']} for c in unique_codes if c['is_index']]
    stock_codes = [{'code': c['code'], 'name': c['name']} for c in unique_codes if not c['is_index']]

    buy_signals = [s for s in all_signals if s['type'] == 'buy']
    sell_signals = [s for s in all_signals if s['type'] == 'sell']

    # 买入信号分层：
    #   第一层（强烈推荐）：胜率>=70%, 次数>=10, 收益>0
    #   第二层（推荐）    ：胜率>=60%, 次数>=10, 收益>0（排除第一层）
    #   第三层（其他）    ：剩余买入信号
    strong_buy_signals = []
    recommend_buy_signals = []
    other_buy_signals = []
    for sig in buy_signals:
        wr = sig.get('win_rate')
        tc = sig.get('trade_count')
        ret = sig.get('return')
        has_valid = wr is not None and tc is not None and ret is not None
        if has_valid and wr >= 70 and tc >= 10 and ret > 0:
            strong_buy_signals.append(sig)
        elif has_valid and wr >= 60 and tc >= 10 and ret > 0:
            recommend_buy_signals.append(sig)
        else:
            other_buy_signals.append(sig)

    codes_with_signals = len(set(s['code'] for s in all_signals))

    return {
        'track_date': track_date,
        'total': total,
        'success': success,
        'failed': failed,
        'failed_codes': all_failed,
        'index_count': len(index_codes),
        'stock_count': len(stock_codes),
        'codes_with_signals': codes_with_signals,
        'buy_count': len(buy_signals),
        'strong_buy_count': len(strong_buy_signals),
        'recommend_buy_count': len(recommend_buy_signals),
        'other_buy_count': len(other_buy_signals),
        'sell_count': len(sell_signals),
        'buy_signals': buy_signals,
        'strong_buy_signals': strong_buy_signals,
        'recommend_buy_signals': recommend_buy_signals,
        'other_buy_signals': other_buy_signals,
        'sell_signals': sell_signals,
        'has_signals': len(all_signals) > 0,
        'index_codes': index_codes,
        'stock_codes': stock_codes,
    }


def _write_summary(summary, track_date):
    """写入 summary.json，覆盖模式（不再 append，避免污染）。"""
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_file = os.path.join(log_dir, f'{track_date.replace("-", "")}_summary.json')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'[tracker] summary 已写入 {summary_file}', flush=True)
    except Exception as e:
        print(f'[tracker] 写入 summary 失败: {e}', flush=True)

    # 简短状态行
    if not summary['has_signals']:
        print(f'[tracker] 本次跟踪 {summary["total"]} 个标的（指数 {summary["index_count"]}，'
              f'个股 {summary["stock_count"]}），无信号输出。', flush=True)
    else:
        print(f'[tracker] 本次跟踪 {summary["total"]} 个标的，共 {summary["buy_count"] + summary["sell_count"]} 个信号'
              f'（买入 {summary["buy_count"]}，卖出 {summary["sell_count"]}）', flush=True)


# ============================================================================
# CLI 入口（用于 web_app 子进程调用）
# ============================================================================
def main():
    """CLI 入口，从环境变量读取参数（避免命令行参数解析问题）。"""
    import os
    track_date = os.environ.get('TRACK_DATE', '2025-06-24')
    mode = os.environ.get('TRACK_MODE', 'index')
    cautious = os.environ.get('CAUTIOUS', '0') == '1'
    min_mv = float(os.environ['MIN_MV']) if os.environ.get('MIN_MV') else None
    max_mv = float(os.environ['MAX_MV']) if os.environ.get('MAX_MV') else None
    target_codes_str = os.environ.get('TARGET_CODES', '')
    target_codes = [c.strip() for c in target_codes_str.split(',') if c.strip()] if target_codes_str else []
    # 跟踪模式：full=全量 / directional=定向 / configured_only=仅配置
    track_mode = os.environ.get('TRACK_MODE_TYPE', 'full')

    print(f'[tracker] 配置: track_date={track_date}, mode={mode}, '
          f'cautious={cautious}, track_mode={track_mode}, '
          f'min_mv={min_mv}, max_mv={max_mv}, target_codes={len(target_codes)} 个', flush=True)

    summary = run_tracking(
        track_date=track_date,
        mode=mode,
        cautious=cautious,
        target_codes=target_codes,
        min_mv=min_mv,
        max_mv=max_mv,
        track_mode=track_mode,
    )
    print(f'[tracker] 全部完成: 总 {summary["total"]}，成功 {summary["success"]}，'
          f'失败 {summary["failed"]}，信号 {summary["buy_count"] + summary["sell_count"]}', flush=True)


if __name__ == '__main__':
    main()
