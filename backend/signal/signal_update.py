import json
import os
import re
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from config import config
from signal_utils import PATTERN_CN_NAMES
from tools import set_logger
import pattern_scan

logger = __import__('logging').getLogger('trader_system')

PATTERN_LIST = [
    'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU',
    'CDLCOUNTERATTACK', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
    'CDLENGULFING', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
    'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
    'CDLHOMINGPIGEON', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
    'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
    'CDLMATCHINGLOW', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING',
    'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHORTLINE',
    'CDLSPINNINGTOP', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
    'CDLUNIQUE3RIVER', 'CDLXSIDEGAP3METHODS',
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH',
    'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBREAKAWAY',
    'CDLCONCEALBABYSWALL', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
    'CDLHANGINGMAN', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLSHOOTINGSTAR',
    'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUPSIDEGAP2CROWS',
]


def _state_dir_from(output_dirs):
    """从 output_dirs 中提取信号更新状态目录，未指定则使用 config 默认值。"""
    if output_dirs and output_dirs.get('signal_update_state'):
        return Path(output_dirs['signal_update_state'])
    return config.SIGNAL_UPDATE_DIR


def _load_task_state(output_dirs=None):
    """加载当前任务状态（断点续跑）。

    返回的 state 包含：
        - task_id: 任务ID
        - params: 任务参数（types/start_date/end_date/observe_day/cautious 等）
        - completed_keys: 已完成任务标识集合 ['code|pattern|type', ...]
        - total_tasks: 总任务数
        - start_time: 任务开始时间戳
    """
    path = _state_dir_from(output_dirs) / 'current_task.json'
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _save_task_state(state, output_dirs=None):
    """保存当前任务状态（断点续跑用）。

    每完成一个批次后调用，将已完成任务标识集合写入磁盘。
    为了避免频繁写盘，调用方应每批次调用一次（而非每任务）。
    """
    path = _state_dir_from(output_dirs) / 'current_task.json'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f'[信号更新] 保存任务状态失败: {e}')


def _delete_task_state(output_dirs=None):
    """删除任务状态（完成后清理）"""
    path = _state_dir_from(output_dirs) / 'current_task.json'
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def _task_key(code, pattern, pattern_type):
    """生成任务唯一标识，用于断点续跑的去重判断"""
    return f'{code}|{pattern}|{pattern_type}'


# 全局停止事件：web_app 的 stop 接口设置后，主循环检测到并优雅退出
_stop_event = None


def request_stop():
    """请求停止当前正在运行的信号更新任务（由 web_app 调用）。

    设置全局 _stop_event，主循环在下一个批次检查点检测到后退出。
    退出前会保存当前进度，下次启动可从断点续跑。
    """
    global _stop_event
    if _stop_event is not None:
        _stop_event.set()


def is_stop_requested():
    """检查是否收到了停止请求"""
    global _stop_event
    return _stop_event is not None and _stop_event.is_set()


def _save_latest_task(task_id, start_time, end_time, output_dirs=None):
    """保存最新任务信息，供质量报告按本次更新过滤"""
    path = _state_dir_from(output_dirs) / 'latest_task.json'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'task_id': task_id, 'start_time': start_time, 'end_time': end_time}, f)
    except Exception:
        pass


def _load_history(output_dirs=None):
    """加载更新历史"""
    path = _state_dir_from(output_dirs) / 'history.json'
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history, output_dirs=None):
    """保存更新历史（保留最近50条）"""
    path = _state_dir_from(output_dirs) / 'history.json'
    history = history[-50:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_focus_codes():
    """加载重点标的集合（持仓 + 关注信号 + 个股策略配置）。

    这些标的需要 buy+sell 双向信号；其余标的在 buy_only 模式下只跑 buy。
    """
    focus = set()
    # 1. 持仓列表
    try:
        pos_file = Path(config.POSITION_FILE) if hasattr(config, 'POSITION_FILE') else None
        if pos_file and pos_file.exists():
            df = pd.read_csv(pos_file, dtype={'code': str})
            for code in df.get('code', []):
                code = str(code).strip()
                if code:
                    focus.add(code)
    except Exception as e:
        logger.warning(f'[信号更新] 加载持仓列表失败: {e}')
    # 2. 关注信号列表
    try:
        wl_file = Path(getattr(config, 'WATCHLIST_SIGNALS_FILE', '')) if hasattr(config, 'WATCHLIST_SIGNALS_FILE') else None
        if wl_file and wl_file.exists():
            with open(wl_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
            for it in items:
                code = str(it.get('code', '')).strip()
                if code:
                    focus.add(code)
    except Exception as e:
        logger.warning(f'[信号更新] 加载关注信号失败: {e}')
    # 3. 个股策略配置
    try:
        cfg_file = Path(getattr(config, 'STRATEGY_CONFIG_FILE', '')) if hasattr(config, 'STRATEGY_CONFIG_FILE') else None
        if cfg_file and cfg_file.exists():
            df = pd.read_csv(cfg_file, dtype={'code': str})
            for code in df.get('code', []):
                code = str(code).strip()
                if code:
                    focus.add(code)
    except Exception as e:
        logger.warning(f'[信号更新] 加载策略配置失败: {e}')
    return focus


def _get_securities_list(types, min_mv=None, max_mv=None):
    """获取证券列表。

    Args:
        types: 标的类型列表，如 ['index', 'hs', 'cy', 'kc', 'etf']
        min_mv: 最小总市值（万元），None 表示不限制
        max_mv: 最大总市值（万元），None 表示不限制
    """
    result = []
    if 'index' in types:
        result.extend([
            {'code': 'DJI', 'name': '道琼斯指数'},
            {'code': 'FCHI', 'name': '法国CAC40'},
            {'code': 'SPX', 'name': '标普500'},
            {'code': 'GDAXI', 'name': '德国DAX'},
            {'code': 'N225', 'name': '日经225'},
            {'code': '000300.SH', 'name': '沪深300'},
            {'code': '399006.SZ', 'name': '创业板指'},
        ])
    # A股按板块拆分：沪深主板、创业板、科创板
    if 'hs' in types or 'cy' in types or 'kc' in types:
        try:
            df = pd.read_csv(config.STOCK_DATA_FILE)
            # 市值筛选：stock_data.csv 中 total_mv 单位为万元
            if min_mv is not None:
                df = df[df['total_mv'] >= float(min_mv)]
            if max_mv is not None:
                df = df[df['total_mv'] <= float(max_mv)]
            for _, row in df.iterrows():
                code = str(row['ts_code'])
                name = str(row.get('name', ''))
                include = False
                if code.startswith('300') or code.startswith('301'):
                    include = 'cy' in types
                elif code.startswith('688'):
                    include = 'kc' in types
                elif code.endswith('.SH') or code.endswith('.SZ'):
                    # 仅沪深主板（排除北交所.BJ和其他市场）
                    include = 'hs' in types
                if include:
                    result.append({'code': code, 'name': name})
        except Exception as e:
            logger.error(f'[信号更新] 读取A股列表失败: {e}')
    if 'etf' in types:
        etf_file = config.SECURITY_LIST_DIR / 'etf_list.csv'
        if etf_file.exists():
            try:
                df = pd.read_csv(etf_file)
                for _, row in df.iterrows():
                    code = str(row.get('code', '')).strip()
                    name = str(row.get('name', '')).strip()
                    if code and len(code) >= 6:
                        if '.' not in code:
                            market = str(row.get('market', 'SH')).upper()
                            code = f'{code}.{market}'
                        result.append({'code': code, 'name': name})
            except Exception as e:
                logger.error(f'[信号更新] 读取ETF列表失败: {e}')
    return result


def _get_data_dir(code, output_dirs=None):
    """根据代码类型返回数据目录（日线目录 + 策略表现目录）。

    Args:
        output_dirs: 可选输出目录覆盖，键如 index_perf / a_stock_perf / etf_perf / index_data 等。
                     未指定的目录仍使用 config 默认值。
    """
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        data_dir = output_dirs.get('index_data', config.DAILY_TRACKING_INDEX_DIR) if output_dirs else config.DAILY_TRACKING_INDEX_DIR
        perf_dir = output_dirs.get('index_perf', config.INDEX_PERFORMANCE_DIR) if output_dirs else config.INDEX_PERFORMANCE_DIR
        return str(data_dir), str(perf_dir)
    elif code.startswith('51') or code.startswith('513') or code.startswith('515') or \
         code.startswith('15') or code.startswith('16') or code.startswith('512') or \
         code.startswith('510') or code.startswith('516') or code.startswith('517') or \
         code.startswith('518') or code.startswith('519'):
        data_dir = output_dirs.get('etf_data', config.DAILY_TRACKING_ETF_DIR) if output_dirs else config.DAILY_TRACKING_ETF_DIR
        perf_dir = output_dirs.get('etf_perf', config.ETF_PERFORMANCE_DIR) if output_dirs else config.ETF_PERFORMANCE_DIR
        return str(data_dir), str(perf_dir)
    else:
        data_dir = output_dirs.get('a_stock_data', config.DAILY_TRACKING_A_DIR) if output_dirs else config.DAILY_TRACKING_A_DIR
        perf_dir = output_dirs.get('a_stock_perf', config.STOCK_PERFORMANCE_DIR) if output_dirs else config.STOCK_PERFORMANCE_DIR
        return str(data_dir), str(perf_dir)


def _run_one_code(code, pattern_type_pairs, start_date, end_date, observe_day, cautious,
                  data_folder_dir, perf_dir):
    """运行单个 code 的全部形态回测（进程池 worker）。

    性能优化：一次加载该 code 的 DataFrame，复用给所有 118 个形态，
    避免每个形态重复读取 CSV（原来 118 次 IO → 现在 1 次）。

    Args:
        code: 标的代码
        pattern_type_pairs: [(pattern_name, pattern_type), ...] 待跑的形态列表
        其余参数同 _run_one

    Returns:
        {'code': code, 'results': [{'pattern':..,'type':..,'success':..}, ...]}
    """
    import csv as _csv
    results = []

    # 一次性加载该 code 的完整 DataFrame（所有形态复用）
    try:
        cached_df = pattern_scan._load_raw_dataframe(code, data_folder_dir)
    except Exception as e:
        # CSV 加载失败：该 code 的所有形态都标记为失败
        err_msg = f'[信号更新] 加载数据失败 {code}: {e}'
        print(err_msg, file=__import__('sys').stderr)
        for pattern, pattern_type in pattern_type_pairs:
            results.append({'code': code, 'pattern': pattern, 'type': pattern_type,
                            'success': False, 'error': f'加载数据失败: {str(e)[:150]}'})
        return {'code': code, 'results': results}

    # 按日期范围预处理（与 run_single_pattern 内部保持一致）
    try:
        filtered_df = cached_df.loc[start_date:end_date].copy()
        if filtered_df is None or filtered_df.empty:
            raise ValueError(f'{code} 在 {start_date}~{end_date} 无数据')
        for col in filtered_df.columns:
            if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                filtered_df[col] = filtered_df[col].ffill().fillna(0)
    except Exception as e:
        err_msg = f'[信号更新] 数据预处理失败 {code}: {e}'
        print(err_msg, file=__import__('sys').stderr)
        for pattern, pattern_type in pattern_type_pairs:
            results.append({'code': code, 'pattern': pattern, 'type': pattern_type,
                            'success': False, 'error': f'数据预处理失败: {str(e)[:150]}'})
        return {'code': code, 'results': results}

    # code 级形态信号缓存：避免重复 TA-Lib 计算
    unique_patterns = sorted({p for p, _ in pattern_type_pairs})
    cache_path = pattern_scan._signal_cache_path(code, start_date, end_date, data_folder_dir)
    cached_signals = pattern_scan._load_cached_signals(cache_path)
    if cached_signals is None:
        cached_signals = pattern_scan._compute_all_pattern_signals(filtered_df, unique_patterns)
        pattern_scan._save_cached_signals(cache_path, cached_signals)

    # 按 type 分组写入（减少文件打开次数）
    write_buf = {'buy': [], 'sell': []}
    for pattern, pattern_type in pattern_type_pairs:
        try:
            r = pattern_scan.run_single_pattern(
                code=code,
                pattern_name=pattern,
                pattern_type=pattern_type,
                start_date=start_date,
                end_date=end_date,
                data_folder_dir=data_folder_dir,
                observe_day=observe_day,
                cautious=cautious,
                cached_df=cached_df,  # 复用已加载的 DataFrame
                cached_signal=cached_signals.get(pattern),  # 复用已计算的形态信号
            )
            sharpe_val = r.get('sharpe', 0)
            output = {
                '策略名称': f'{pattern_type}_{pattern}',
                '交易次数': r.get('trades', 0),
                '胜率(%)': r.get('win_rate', 0),
                '简易收益率(%)': r.get('return_pct', 0),
                '夏普比率': sharpe_val if sharpe_val is not None else 0,
                '最大回撤(%)': r.get('hold_max_drawdown', 0),
            }
            write_buf[pattern_type].append(output)
            results.append({'code': code, 'pattern': pattern, 'type': pattern_type, 'success': True})
        except Exception as e:
            err_msg = f'[信号更新] 子任务失败 {code} {pattern_type}_{pattern}: {e}'
            print(err_msg, file=__import__('sys').stderr)
            __import__('traceback').print_exc(file=__import__('sys').stderr)
            results.append({'code': code, 'pattern': pattern, 'type': pattern_type,
                            'success': False, 'error': str(e)[:200]})

    # 批量写入 CSV（每个 type 一次追加，而非每个形态一次）
    for side, rows in write_buf.items():
        if not rows:
            continue
        filename = f'{code}_{side}_strategy_performance_test.csv'
        try:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(perf_dir, filename), mode='a', header=False, index=False)
        except Exception as e:
            print(f'[信号更新] 写入CSV失败 {filename}: {e}', file=__import__('sys').stderr)

    # 保存返回值（之后会释放大对象）
    output = {'code': code, 'results': results}

    # 显式释放大对象，避免跨任务累积内存
    # ProcessPoolExecutor 复用 worker 进程，Python 不自动归还内存给 OS
    del cached_df, write_buf, results
    import gc
    gc.collect()

    return output


# 近期速率统计：保留最近 50 个完成时间戳，用于动态计算 ETA
_recent_completion_times = []  # list[float] of time.time()
_RECENT_WINDOW = 50

def _record_completion():
    """记录一个任务完成时刻，用于近期速率计算"""
    global _recent_completion_times
    now = time.time()
    _recent_completion_times.append(now)
    # 只保留最近 N 个
    if len(_recent_completion_times) > _RECENT_WINDOW:
        _recent_completion_times = _recent_completion_times[-_RECENT_WINDOW:]

def _recent_rate():
    """计算近期速率（任务/秒），基于最近 N 次完成时间戳"""
    if len(_recent_completion_times) < 2:
        return None
    times = _recent_completion_times[-_RECENT_WINDOW:]
    elapsed = times[-1] - times[0]
    if elapsed <= 0:
        return None
    return (len(times) - 1) / elapsed

def _reset_recent_rate():
    """任务开始或恢复时清空近期速率统计"""
    global _recent_completion_times
    _recent_completion_times = []

def _report_progress(cb, current, total, success, failed, start_time, message):
    """统一进度汇报，自动计算 ETA。

    Args:
        cb: 回调函数，接受 dict 参数
        current: 已完成任务数
        total: 总任务数
        success: 成功数
        failed: 失败数
        start_time: 任务开始时间戳 (time.time())
        message: 文本消息
    """
    if not cb:
        return
    elapsed = time.time() - start_time if start_time else 0
    # 优先用近期速率（更准确反映当前批次速度），回退到平均速率
    rate = _recent_rate() or (current / elapsed if elapsed > 0 else 0)
    eta = int((total - current) / rate) if rate > 0 and current > 0 else None
    cb({
        'current': current,
        'total': total,
        'message': message,
        'success': success,
        'failed': failed,
        'elapsed': round(elapsed, 1),
        'eta': eta,
    })


def run_signal_update(task_id, params, progress_callback=None, output_dirs=None):
    """运行信号更新任务（支持断点续跑）。

    断点续跑机制：
        - 每完成一个批次，将已完成任务标识集合写入 current_task.json
        - 启动时若检测到未完成的任务状态（且参数一致），跳过已完成的任务
        - 用户主动停止时，保存当前进度，下次启动可选择续跑
        - resume 参数控制是否续跑：True=续跑，False=全新开始（默认）

    Args:
        task_id: 任务ID
        params: 参数字典，包含：types, start_date, end_date, observe_day, cautious, workers, resume
        progress_callback: 进度回调函数 (current, total, message) -> None
        output_dirs: 可选输出目录覆盖字典，用于隔离测试输出。
                     支持的键：a_stock_perf, index_perf, etf_perf,
                     a_stock_data, index_data, etf_data, signal_update_state,
                     market_wide_stats。
                     未指定的目录仍使用 config 默认值。
    """
    import threading
    start_time = time.time()
    types = params.get('types', ['index', 'hs', 'cy', 'kc', 'etf'])
    start_date = params.get('start_date', '20100104')
    end_date = params.get('end_date', '')
    observe_day = params.get('observe_day', 2)
    cautious = params.get('cautious', False)
    workers = params.get('workers', 2)
    # 受全局并发上限约束，防止资源耗尽
    workers = min(int(workers), getattr(config, 'MAX_WORKERS', 4))
    workers = max(1, workers)
    resume = params.get('resume', False)  # 是否断点续跑
    min_mv = params.get('min_mv')  # 最小总市值（万元）
    max_mv = params.get('max_mv')  # 最大总市值（万元）

    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')

    securities = _get_securities_list(types, min_mv, max_mv)
    # 加载重点标的（持仓+关注+策略配置），用于决定是否跑卖出信号
    buy_only_for_non_focus = params.get('buy_only_for_non_focus', True)
    focus_codes = _load_focus_codes() if buy_only_for_non_focus else set()
    if buy_only_for_non_focus:
        # 只统计本次任务涉及的重点标的数（用于日志）
        focus_in_task = sum(1 for s in securities if s['code'] in focus_codes)
        logger.info(f'[信号更新] 任务启动: {task_id}, 标的数: {len(securities)}, 形态数: {len(PATTERN_LIST)}, '
                    f'时间段: {start_date}~{end_date}, 续跑={resume}, '
                    f'仅买入模式=开启(重点标的 {focus_in_task} 个跑双向, 其余只跑买入)')
    else:
        logger.info(f'[信号更新] 任务启动: {task_id}, 标的数: {len(securities)}, 形态数: {len(PATTERN_LIST)}, '
                    f'时间段: {start_date}~{end_date}, 续跑={resume}, 仅买入模式=关闭')

    # 初始化全局停止事件（本任务专用）
    global _stop_event
    _stop_event = threading.Event()

    # ===== 断点续跑：加载已完成的任务清单 =====
    completed_keys_set = set()  # 已完成任务标识集合
    resumed_from = 0  # 续跑起始进度
    if resume:
        prev_state = _load_task_state(output_dirs)
        if prev_state and prev_state.get('params'):
            # 校验参数一致性（types/start_date/end_date/observe_day/cautious/min_mv/max_mv 必须相同）
            p = prev_state['params']
            def _mv_equal(a, b):
                # None、空字符串、0 均视为未限制，统一比较
                def _norm(v):
                    return None if v in (None, '') else float(v)
                return _norm(a) == _norm(b)
            if (set(p.get('types', [])) == set(types)
                    and p.get('start_date') == start_date
                    and p.get('end_date') == end_date
                    and p.get('observe_day') == observe_day
                    and p.get('cautious') == cautious
                    and bool(p.get('buy_only_for_non_focus', False)) == bool(buy_only_for_non_focus)
                    and _mv_equal(p.get('min_mv'), min_mv)
                    and _mv_equal(p.get('max_mv'), max_mv)):
                completed_keys_set = set(prev_state.get('completed_keys', []))
                resumed_from = len(completed_keys_set)
                logger.info(f'[信号更新] 断点续跑: 已完成 {resumed_from} 个任务，将跳过这些任务继续执行')
            else:
                logger.warning(f'[信号更新] 续跑参数不一致，将重新开始任务')
                _delete_task_state(output_dirs)
        else:
            logger.warning(f'[信号更新] 未找到可续跑的任务状态，将重新开始任务')

    _report_progress(progress_callback, 0, len(securities), 0, 0, start_time, '获取标的列表完成')
    _reset_recent_rate()  # 任务开始时清空近期速率统计

    completed = 0
    failed = 0
    total = len(securities)
    results = {'success': [], 'failed': []}

    # ===== 阶段1：准备文件 + 按 code 分组构建任务 =====
    # 性能优化：任务粒度从 (code,pattern,type) 改为 (code, [pattern_type_pairs])
    # 每个 code 一次加载 CSV，复用给所有 118 个形态，IO 从 118 次降到 1 次
    all_tasks = []  # 元素: (code, pattern_type_pairs, start_date, end_date, observe_day, cautious, data_folder_dir, perf_dir)
    total_pattern_count = 0  # 总形态数（用于进度计算，保持与旧版兼容）

    for sec in securities:
        code = sec['code']
        data_folder_dir, perf_dir = _get_data_dir(code, output_dirs)

        # 确保输出目录存在（兼容临时目录场景）
        try:
            os.makedirs(perf_dir, exist_ok=True)
        except Exception:
            pass

        # 续跑模式下：不删除已有CSV，只在文件不存在时创建带表头的新文件
        # 全新模式下：删除并重建CSV（清空旧数据）
        if not resume:
            try:
                perf_file = os.path.join(perf_dir, f'{code}_buy_strategy_performance_test.csv')
                if os.path.exists(perf_file):
                    os.remove(perf_file)
                perf_file = os.path.join(perf_dir, f'{code}_sell_strategy_performance_test.csv')
                if os.path.exists(perf_file):
                    os.remove(perf_file)
            except Exception:
                pass

        # 写入表头（仅在文件不存在或全新模式下执行）
        import csv as _csv
        header = ['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)']
        for side in ('buy', 'sell'):
            perf_path = os.path.join(perf_dir, f'{code}_{side}_strategy_performance_test.csv')
            # 续跑模式：文件不存在才写表头；全新模式：总是写表头
            if not resume or not os.path.exists(perf_path):
                try:
                    with open(perf_path, 'w', newline='', encoding='utf-8') as f:
                        _csv.writer(f).writerow(header)
                except Exception:
                    pass

        # 构建该 code 的 (pattern, type) 对
        # 重点标的跑双向（buy+sell），非重点标的在 buy_only 模式下只跑 buy
        all_pairs = []
        is_focus = code in focus_codes
        for pattern in PATTERN_LIST:
            all_pairs.append((pattern, 'buy'))
            if is_focus or not buy_only_for_non_focus:
                all_pairs.append((pattern, 'sell'))

        # 断点续跑：过滤掉已完成的 (pattern, type)
        if completed_keys_set:
            pending_pairs = [(p, t) for p, t in all_pairs
                             if _task_key(code, p, t) not in completed_keys_set]
        else:
            pending_pairs = all_pairs

        if pending_pairs:
            all_tasks.append((code, pending_pairs, start_date, end_date,
                              observe_day, cautious, data_folder_dir, perf_dir))
            total_pattern_count += len(pending_pairs)

    # total_tasks 保持以"形态数"为单位（与断点记录的 completed_keys 粒度一致）
    total_tasks = total_pattern_count + resumed_from
    completed_tasks = resumed_from  # 从断点继续计数

    _success = completed_tasks - failed
    msg = f'准备完成，共 {total_tasks} 个形态（{len(all_tasks)} 个标的）'
    if resumed_from > 0:
        msg += f'（续跑：已完成 {resumed_from}，待执行 {total_pattern_count}）'
    _report_progress(progress_callback, completed_tasks, total_tasks, _success, failed, start_time, msg)

    # ===== 阶段2：按 code 分组提交任务，每个 worker 跑一个 code 的全部待执行形态 =====
    # 每个 code 任务包含 ~118 个形态，批次大小用标的数控制
    # 批次大小 = workers × 4：一批提交 4 轮任务，减少 executor 重建开销
    # （_run_one_code 末尾已 del + gc.collect()，跨 code 内存累积可控）
    batch_size_codes = max(workers * 4, 8)
    # 单个 code 任务超时：按形态数估算，每个形态最多 8 秒，最少 60 秒，最多 300 秒
    # （原 15 秒/形态最长 915 秒，导致尾部卡住 15 分钟）
    def _code_timeout(pairs_count):
        return min(300, max(60, pairs_count * 8))

    # 用于累积已完成任务标识（批量保存到磁盘）
    pending_completed_keys = []

    # 每批重建 ProcessPoolExecutor，确保 worker 进程不跨批累积内存。
    # max_tasks_per_child 在 Python 3.13 可能不稳定（worker 意外死亡），
    # 改为手动控制生命周期：每 batch_size_codes 个 code 销毁并重建 executor。
    def _executor_context():
        return ProcessPoolExecutor(max_workers=workers)

    executor = _executor_context()
    try:
        for i in range(0, len(all_tasks), batch_size_codes):
            # 检查停止信号（在每个批次开始前检查）
            if is_stop_requested():
                logger.info(f'[信号更新] 收到停止信号，已完成 {completed_tasks}/{total_tasks}，保存断点后退出')
                _save_task_state({
                    'task_id': task_id,
                    'params': {'types': types, 'start_date': start_date, 'end_date': end_date,
                               'observe_day': observe_day, 'cautious': cautious,
                               'buy_only_for_non_focus': buy_only_for_non_focus,
                               'min_mv': min_mv, 'max_mv': max_mv},
                    'completed_keys': list(completed_keys_set),
                    'total_tasks': total_tasks,
                    'start_time': start_time,
                    'stopped_at': time.time(),
                }, output_dirs)
                _report_progress(progress_callback, completed_tasks, total_tasks,
                                 completed_tasks - failed, failed, start_time,
                                 f'已停止（可续跑）: {completed_tasks}/{total_tasks}')
                return {
                    'success': False,
                    'stopped': True,
                    'completed': completed_tasks,
                    'total': total_tasks,
                    'message': f'已停止，已完成 {completed_tasks}/{total_tasks}，可从断点续跑',
                }

            batch = all_tasks[i:i + batch_size_codes]
            batch_end = min(i + batch_size_codes, len(all_tasks))
            logger.info(f'[信号更新] 开始批次 标的{i+1}-{batch_end}/{len(all_tasks)}, 本批 {len(batch)} 个标的')
            futures = {executor.submit(_run_one_code, *t): t for t in batch}
            for future in as_completed(futures):
                task_args = futures[future]
                code = task_args[0]
                pairs = task_args[1]
                timeout = _code_timeout(len(pairs))
                try:
                    r = future.result(timeout=timeout)
                except Exception as e:
                    logger.error(f'[信号更新] 标的任务异常 {code}: {e}')
                    r = {'code': code, 'results': [
                        {'code': code, 'pattern': p, 'type': t, 'success': False,
                         'error': f'任务超时或异常: {str(e)[:100]}'}
                        for p, t in pairs
                    ]}

                # 展开该 code 的所有形态结果
                for item in r.get('results', []):
                    completed_tasks += 1
                    key = _task_key(item['code'], item['pattern'], item['type'])
                    completed_keys_set.add(key)
                    pending_completed_keys.append(key)

                    if item.get('success'):
                        results['success'].append(item)
                    else:
                        results['failed'].append(item)
                        failed += 1

                    # 每个形态完成都记录一次（用于近期速率）
                    _record_completion()

                # 每个 code 完成立即汇报一次进度（不再每 100 次才汇报）
                _report_progress(progress_callback, completed_tasks, total_tasks,
                                 completed_tasks - failed, failed, start_time,
                                 f'进度: {completed_tasks}/{total_tasks} ({code})')

                logger.info(f'[信号更新] 标的完成: {code}, 进度 {completed_tasks}/{total_tasks}, 失败 {failed}')

            # ===== 每个批次完成后保存断点 + 关闭 executor 释放内存 =====
            if pending_completed_keys:
                _save_task_state({
                    'task_id': task_id,
                    'params': {'types': types, 'start_date': start_date, 'end_date': end_date,
                               'observe_day': observe_day, 'cautious': cautious,
                               'buy_only_for_non_focus': buy_only_for_non_focus,
                               'min_mv': min_mv, 'max_mv': max_mv},
                    'completed_keys': list(completed_keys_set),
                    'total_tasks': total_tasks,
                    'start_time': start_time,
                }, output_dirs)
                pending_completed_keys.clear()

            # 关闭当前 executor，worker 进程彻底退出归还内存
            executor.shutdown(wait=True)
            # 重建新 executor 处理下一批，避免跨批累积内存
            executor = _executor_context()

    finally:
        executor.shutdown(wait=True)

    # 任务全部完成后，强制更新进度到100%，避免进度条停留在最后一个50的倍数
    _report_progress(progress_callback, total_tasks, total_tasks,
                     total_tasks - failed, failed, start_time,
                     f'更新完成: {total_tasks}/{total_tasks}')

    # 清理任务状态文件（完成后再也不需要续跑了）
    _delete_task_state(output_dirs)

    # 重置停止事件
    _stop_event = None

    elapsed = time.time() - start_time
    history = _load_history(output_dirs)
    history.append({
        'task_id': task_id,
        'start_time': start_time,
        'end_time': time.time(),
        'elapsed': round(elapsed, 2),
        'types': types,
        'start_date': start_date,
        'end_date': end_date,
        'observe_day': observe_day,
        'cautious': cautious,
        'buy_only_for_non_focus': buy_only_for_non_focus,
        'total_securities': total,
        'completed': total,
        'failed': failed,
        'total_patterns': len(PATTERN_LIST),
        'resumed_from': resumed_from,
    })
    _save_history(history, output_dirs)

    # 保存最新任务信息，供质量报告按本次更新过滤
    _save_latest_task(task_id, start_time, time.time(), output_dirs)

    # 信号更新成功后，自动重算全市场形态统计
    try:
        compute_market_wide_stats(output_dirs)
    except Exception as e:
        logger.warning(f'[信号更新] 全市场统计生成失败（不影响主任务）: {e}')

    logger.info(f'[信号更新] 任务完成: {task_id}, 耗时 {elapsed:.1f}秒, 完成 {total} 个标的, 失败 {failed} 个形态')

    # 失败详情精简（含标的代码/形态/错误原因）
    failed_items = []
    for item in results.get('failed', []):
        failed_items.append({
            'code': item.get('code', ''),
            'pattern': item.get('pattern', ''),
            'type': item.get('type', ''),
            'error': str(item.get('error', ''))[:200],
        })

    return {
        'success': True,
        'elapsed': round(elapsed, 2),
        'total_securities': total,
        'completed': total,
        'failed': failed,
        'failed_items': failed_items,
        'total_patterns': len(PATTERN_LIST),
        'resumed_from': resumed_from,
    }


def compute_market_wide_stats(output_dirs=None):
    """基于信号更新产出的策略表现 CSV，聚合生成全市场形态统计。

    统计口径：
        - 范围：A 股 + 指数（不含 ETF）
        - 按 (code, 策略名称) 去重，保留最新一行
        - 交易次数 = 所有标的求和
        - 胜率/收益率/夏普/最大回撤 按交易次数加权平均（仅交易次数>0的标的参与）
        - 输出文件：config.MARKET_WIDE_STATS_FILE（或 output_dirs['market_wide_stats']）

    Args:
        output_dirs: 可选输出目录覆盖字典，键如 a_stock_perf / index_perf 等。
                     未指定的目录仍使用 config 默认值。
    """
    perf_dirs = [
        Path(output_dirs.get('a_stock_perf', config.STOCK_PERFORMANCE_DIR)) if output_dirs else config.STOCK_PERFORMANCE_DIR,
        Path(output_dirs.get('index_perf', config.INDEX_PERFORMANCE_DIR)) if output_dirs else config.INDEX_PERFORMANCE_DIR,
    ]
    out_file = Path(output_dirs.get('market_wide_stats', config.MARKET_WIDE_STATS_FILE)) if output_dirs else config.MARKET_WIDE_STATS_FILE
    filename_re = re.compile(r'^(.*)_(buy|sell)_strategy_performance_test\.csv$')

    chunks = []
    for perf_dir in perf_dirs:
        if not perf_dir.exists():
            continue
        for csv_path in perf_dir.glob('*_strategy_performance_test.csv'):
            # 过滤带空格/备份后缀的文件
            m = filename_re.match(csv_path.name)
            if not m:
                continue
            code, side = m.group(1), m.group(2)
            try:
                df = pd.read_csv(csv_path, dtype={'策略名称': str})
                if df.empty:
                    continue
                df['code'] = code
                df['side'] = side
                chunks.append(df)
            except Exception as e:
                logger.warning(f'[全市场统计] 读取失败 {csv_path}: {e}')
                continue

    if not chunks:
        logger.warning('[全市场统计] 无策略表现数据，跳过生成')
        return None

    combined = pd.concat(chunks, ignore_index=True)
    # 去重：同一标的同一策略保留最后一行
    combined = combined.drop_duplicates(subset=['code', '策略名称'], keep='last')

    # 解析形态名和信号类型
    combined['pattern'] = combined['策略名称'].str.replace(r'^(buy|sell)_', '', regex=True)
    combined['signal_type'] = combined['side'].map({'buy': '买入', 'sell': '卖出'})

    # 总交易次数（含 0 交易标的）
    trade_sums = combined.groupby(['pattern', 'signal_type'])['交易次数'].sum().reset_index()

    # 指标聚合：仅交易次数 > 0 的标的参与
    active = combined[combined['交易次数'] > 0].copy()

    def _weighted_avg(g):
        total = g['交易次数'].sum()
        if total == 0:
            return pd.Series({
                '胜率(%)': 0.0, '收益率(%)': 0.0, '夏普比率': 0.0, '最大回撤(%)': 0.0
            })
        return pd.Series({
            '胜率(%)': (g['胜率(%)'] * g['交易次数']).sum() / total,
            '收益率(%)': (g['简易收益率(%)'] * g['交易次数']).sum() / total,
            '夏普比率': (g['夏普比率'] * g['交易次数']).sum() / total,
            '最大回撤(%)': (g['最大回撤(%)'] * g['交易次数']).sum() / total,
        })

    metrics = active.groupby(['pattern', 'signal_type']).apply(_weighted_avg).reset_index()

    result = pd.merge(trade_sums, metrics, on=['pattern', 'signal_type'], how='left')
    result[['胜率(%)', '收益率(%)', '夏普比率', '最大回撤(%)']] = result[['胜率(%)', '收益率(%)', '夏普比率', '最大回撤(%)']].fillna(0.0)

    # 中文名称
    result['中文名称'] = result['pattern'].map(PATTERN_CN_NAMES).fillna(result['pattern'])

    result = result.rename(columns={'pattern': '形态名称', 'signal_type': '信号类型'})
    result = result[['形态名称', '中文名称', '交易次数', '胜率(%)', '收益率(%)', '夏普比率', '最大回撤(%)', '信号类型']]
    result = result.sort_values(['信号类型', '形态名称']).reset_index(drop=True)

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_file, index=False, encoding='utf-8-sig')
        logger.info(
            f'[全市场统计] 已生成 {out_file}, '
            f'共 {len(result)} 条记录, 总交易次数 {int(result["交易次数"].sum())}'
        )
        return out_file
    except Exception as e:
        logger.error(f'[全市场统计] 保存失败: {e}')
        return None


def get_update_history():
    """获取更新历史"""
    history = _load_history()
    for h in history:
        h['start_time_str'] = datetime.fromtimestamp(h['start_time']).strftime('%Y-%m-%d %H:%M:%S')
        h['end_time_str'] = datetime.fromtimestamp(h['end_time']).strftime('%Y-%m-%d %H:%M:%S')
    return history


def get_last_update_time():
    """获取上次更新时间"""
    history = _load_history()
    if history:
        return history[-1]['end_time']
    return None


def get_days_since_last_update():
    """获取距上次更新的天数"""
    last_time = get_last_update_time()
    if not last_time:
        return None
    return (time.time() - last_time) / (24 * 3600)


def is_due_for_update(warning_days=180):
    """检查是否临近更新日"""
    days = get_days_since_last_update()
    if days is None:
        return True, '从未更新'
    if days >= warning_days:
        return True, f'距上次更新已{int(days)}天，建议更新'
    remaining = max(0, warning_days - days)
    return False, f'距下次建议更新还有{int(remaining)}天'
