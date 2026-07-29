import json
import logging
import logging.handlers
import os
import queue
import atexit
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_file, send_from_directory

from config import config

# 前端模板目录：frontend/templates/；静态资源目录：项目根/static/
_TEMPLATE_DIR = config.BASE_DIR / 'frontend' / 'templates'
_STATIC_DIR = config.BASE_DIR / 'static'
app = Flask(__name__,
            template_folder=str(_TEMPLATE_DIR),
            static_folder=str(_STATIC_DIR),
            static_url_path='/static')
app.config['SECRET_KEY'] = 'trader-dashboard-secret'
# 模板自动重载：修改 frontend/templates/*.html 后无需重启 Flask，下次请求自动加载最新版
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# subprocess 脚本公用的 sys.path 设置代码（backend 各功能子目录全部加入）
_BACKEND_DIR = config.BASE_DIR / 'backend'
_BACKEND_SUBDIRS = ['', 'web', 'data', 'strategy', 'pattern', 'signal', 'tracking', 'utils']
_SUBPROCESS_PATH_SETUP = '; '.join(
    f'sys.path.insert(0, r"{_BACKEND_DIR / sub}")' for sub in _BACKEND_SUBDIRS
)
_SUBPROCESS_PATH_SETUP = f'import sys; {_SUBPROCESS_PATH_SETUP}'

# 系统日志配置（按天分文件，每天重置）
SYSTEM_LOG_DIR = config.BASE_DIR / 'log'
SYSTEM_LOG_DIR.mkdir(parents=True, exist_ok=True)
TODAY_LOG_FILE = SYSTEM_LOG_DIR / f'system_{datetime.now().strftime("%Y%m%d")}.log'

# 日志保留天数：超过此天数的日志文件自动清理（含 system_*.log、*_tracking.log、*_summary.json）
LOG_RETENTION_DAYS = 30

logger = logging.getLogger('trader_system')
logger.setLevel(logging.DEBUG)

class DailyFileHandler(logging.FileHandler):
    """按天滚动的文件handler：每天自动切换到新日志文件"""
    def __init__(self, log_dir, encoding='utf-8'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.encoding = encoding
        self._current_date = datetime.now().strftime("%Y%m%d")
        self._current_file = self.log_dir / f'system_{self._current_date}.log'
        super().__init__(str(self._current_file), encoding=encoding)

    def emit(self, record):
        # 检查日期是否变化，变化则切换文件
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_date:
            self._current_date = today
            self._current_file = self.log_dir / f'system_{self._current_date}.log'
            # 关闭旧handler，打开新文件
            self.close()
            self.stream = open(str(self._current_file), 'a', encoding=self.encoding)
            # 日期切换时触发旧日志清理
            try:
                cleanup_old_logs(self.log_dir, LOG_RETENTION_DAYS)
            except Exception as _cleanup_err:
                # 清理失败不影响日志写入
                pass
        super().emit(record)


def cleanup_old_logs(log_dir, retention_days=30):
    """清理超过保留天数的旧日志文件。

    清理范围：
        - system_YYYYMMDD.log（系统日志）
        - YYYYMMDD_tracking.log（跟踪日志）
        - YYYYMMDD_summary.json（跟踪结果 JSON）
        - flask.log / system.log（无日期的旧日志，直接删除）

    Args:
        log_dir: 日志目录 Path 对象
        retention_days: 保留天数，超过此天数的文件删除

    Returns:
        dict: {'deleted_count': int, 'deleted_files': [str], 'skipped': int}
    """
    import re
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return {'deleted_count': 0, 'deleted_files': [], 'skipped': 0}

    now = datetime.now()
    # 日期提取正则：匹配 YYYYMMDD 格式（8位数字）
    date_pattern = re.compile(r'(\d{8})')

    deleted_files = []
    skipped = 0

    for f in log_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        # 无日期的旧日志文件（flask.log / system.log），直接删除
        if name in ('flask.log', 'system.log'):
            try:
                f.unlink()
                deleted_files.append(name)
            except OSError:
                skipped += 1
            continue

        # 从文件名提取日期
        match = date_pattern.search(name)
        if not match:
            # 无日期格式的文件，跳过（可能是其他用途）
            skipped += 1
            continue

        try:
            file_date = datetime.strptime(match.group(1), '%Y%m%d')
        except ValueError:
            skipped += 1
            continue

        # 超过保留天数则删除
        age_days = (now - file_date).days
        if age_days > retention_days:
            try:
                f.unlink()
                deleted_files.append(name)
            except OSError:
                skipped += 1

    result = {'deleted_count': len(deleted_files), 'deleted_files': deleted_files, 'skipped': skipped}
    if deleted_files:
        logger.info(f'[日志清理] 清理 {len(deleted_files)} 个超过 {retention_days} 天的旧日志文件')
    return result

# 实际负责写文件的 handlers（在后台线程中运行）
_fh = DailyFileHandler(SYSTEM_LOG_DIR, encoding='utf-8')
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

# 异步日志：主线程只把 LogRecord 放入队列，后台 QueueListener 线程批量写入文件/控制台
_log_queue = queue.Queue(-1)  # 无界队列，避免阻塞业务线程
_queue_listener = logging.handlers.QueueListener(_log_queue, _fh, _ch, respect_handler_level=True)
_queue_listener.start()
atexit.register(_queue_listener.stop)

# logger 只挂载 QueueHandler，所有日志操作非阻塞
logger.addHandler(logging.handlers.QueueHandler(_log_queue))

logger.info('='*60)
logger.info('量化交易系统启动（异步日志已启用）')

# 启动时清理超过保留天数的旧日志文件（每日自动清除机制）
try:
    _cleanup_result = cleanup_old_logs(SYSTEM_LOG_DIR, LOG_RETENTION_DAYS)
    if _cleanup_result['deleted_count'] > 0:
        logger.info(f'[启动清理] 删除 {_cleanup_result["deleted_count"]} 个旧日志文件（保留 {LOG_RETENTION_DAYS} 天）')
except Exception as _e:
    logger.warning(f'[启动清理] 旧日志清理失败: {_e}')

log_streams = {}
running_processes = {}
scan_tasks = {}
target_streams = {}  # 目标个股信号输出
summary_streams = {}  # 跟踪结果结构化汇总
task_meta = {}  # 任务元信息: {task_id: {'success': bool, 'error': str, 'start_time': float, 'end_time': float}}

# 任务超时（秒）：回测/批量回测 10分钟，跟踪 30分钟
TASK_TIMEOUT_BT = 600
TASK_TIMEOUT_PF = 1800
TASK_TIMEOUT_TR = 1800
# 最大并发任务数：超过则拒绝新任务，避免系统资源耗尽
MAX_CONCURRENT_TASKS = 2
# 已完成任务记录保留时长（秒）：超过自动清理，避免内存泄漏
TASK_TTL = 3600


@app.before_request
def _log_request():
    """统一记录所有 API 请求（前端操作）"""
    if request.path.startswith('/api/'):
        # 获取客户端IP
        ip = request.headers.get('X-Real-IP', request.remote_addr or '-')
        method = request.method
        path = request.path
        # 对POST请求记录关键参数
        args = ''
        if method == 'POST':
            try:
                data = request.get_json(silent=True) or {}
                # 只记录关键字段，避免日志过长
                keys = ['code', 'pattern_name', 'pattern_type', 'start_date', 'end_date',
                        'observe_day', 'cautious', 'track_date', 'mode', 'task_id']
                filtered = {k: v for k, v in data.items() if k in keys}
                if 'patterns' in data:
                    filtered['patterns_count'] = len(data['patterns'])
                args = f' params={json.dumps(filtered, ensure_ascii=False)}' if filtered else ''
            except Exception:
                args = ''
        elif method == 'GET' and request.args:
            args = f' params={dict(request.args)}'
        logger.info(f'[请求] {ip} {method} {path}{args}')


@app.after_request
def _log_response(response):
    """统一记录 API 响应状态"""
    if request.path.startswith('/api/'):
        status = response.status_code
        # 只记录非200的响应（异常情况）
        if status >= 400:
            logger.warning(f'[响应] {request.method} {request.path} -> {status}')
    return response


def _count_running_tasks():
    """统计当前运行中的任务数（含 subprocess 和线程任务）"""
    count = 0
    # subprocess 任务
    for proc in running_processes.values():
        if proc is not None and proc.poll() is None:
            count += 1
    # 线程任务（task_meta 中已创建但未结束的）
    for task_id, meta in task_meta.items():
        if meta.get('start_time', 0) > 0 and meta.get('end_time', 0) == 0:
            # 排除已有 subprocess 的（上面已计入）
            if task_id not in running_processes or running_processes.get(task_id) is None:
                count += 1
    return count


def _cleanup_stale_tasks():
    """清理超过TTL的已完成任务记录，避免内存泄漏。

    只清理已结束（end_time>0）且超过TTL的任务；运行中的任务不受影响。
    scan_tasks（形态扫描任务）单独按 started_at + TTL 清理（其结果可能数MB，必须及时回收）。
    """
    import time
    now = time.time()
    to_remove = []
    for task_id, meta in task_meta.items():
        if meta.get('end_time', 0) > 0 and now - meta['end_time'] > TASK_TTL:
            to_remove.append(task_id)
    for task_id in to_remove:
        log_streams.pop(task_id, None)
        running_processes.pop(task_id, None)
        target_streams.pop(task_id, None)
        summary_streams.pop(task_id, None)
        task_meta.pop(task_id, None)
    # 修复：清理 scan_tasks 中过期任务（避免内存泄漏，scan_tasks 每次结果可能数 MB）
    scan_to_remove = []
    for task_id, scan in scan_tasks.items():
        started = scan.get('started_at', 0)
        if started > 0 and now - started > TASK_TTL:
            scan_to_remove.append(task_id)
    for task_id in scan_to_remove:
        scan_tasks.pop(task_id, None)
    if to_remove or scan_to_remove:
        logger.info(f'[清理] 回收 {len(to_remove)} 个过期任务记录，{len(scan_to_remove)} 个扫描任务')


def _ensure_data_coverage(code, start_date, end_date, data_folder_dir=None):
    """确保本地数据覆盖 [start_date, end_date] 范围；不足时自动从网络拉取并保存。

    该函数在 web_app 主线程中调用（子进程禁网），避免在 worker 中发生网络阻塞。

    Args:
        code: 标的代码
        start_date: 开始日期（YYYYMMDD 或 YYYY-MM-DD）
        end_date: 结束日期（YYYYMMDD 或 YYYY-MM-DD）
        data_folder_dir: 数据保存目录（默认由 code 类型自动选择）

    Returns:
        bool: 本地数据已覆盖或成功拉取返回 True，否则 False
    """
    import data_source
    start_str = str(start_date).replace('-', '')
    end_str = str(end_date).replace('-', '')

    if data_source._local_data_has_date_range(code, start_str, end_str):
        logger.info(f'[数据补全] {code} 本地数据已覆盖 {start_str}~{end_str}')
        return True

    logger.info(f'[数据补全] {code} 本地数据未覆盖 {start_str}~{end_str}，尝试从网络拉取')
    try:
        df = data_source.get_kline_df(code, start_str, end_str,
                                      prefer_local=False, allow_network=True)
        if df is not None and not df.empty:
            data_source.save_kline_to_local(code, df, data_folder_dir)
            logger.info(f'[数据补全] {code} 成功拉取并保存 {len(df)} 行数据')
            return True
        else:
            logger.warning(f'[数据补全] {code} 网络拉取未返回数据')
            return False
    except Exception as e:
        logger.warning(f'[数据补全] {code} 拉取失败: {e}')
        return False


def run_subprocess_task(task_id, script_code, tag, timeout=600, on_done=None):
    """统一执行subprocess任务，处理超时、状态维护、日志记录

    Args:
        task_id: 任务ID
        script_code: 要执行的Python脚本字符串
        tag: 日志标签（如 '回测'/'跟踪'/'批量回测'）
        timeout: 超时秒数，超时后kill进程
        on_done: 可选回调 fn(task_id, output, proc_returncode) -> None，在任务结束后调用

    说明：
        - 使用 start_new_session=True 启动子进程，使其成为独立进程组组长；
          停止任务时通过 os.killpg 杀掉整个进程组（含 ProcessPoolExecutor 子进程）。
        - task_meta 增加 'stop_requested' 字段，前端 stop 接口设置后，readline 循环
          检测到会主动 kill 进程组。
        - output 实时累积到 log_streams[task_id]，支持 SSE 增量推送。
    """
    import time
    task_meta[task_id] = {'success': False, 'error': '', 'start_time': time.time(), 'end_time': 0,
                          'stop_requested': False}
    try:
        proc = subprocess.Popen(
            [sys.executable, '-c', script_code],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(config.BASE_DIR),
            start_new_session=True  # 独立进程组，便于 stop 时 killpg
        )
        running_processes[task_id] = proc
        output_lines = []

        # 非阻塞读取，配合超时检查
        start = time.time()
        while True:
            if proc.poll() is not None:
                # 进程已结束，读取剩余输出
                remaining = proc.stdout.read() if proc.stdout else ''
                if remaining:
                    output_lines.append(remaining)
                break
            line = proc.stdout.readline() if proc.stdout else ''
            if line:
                output_lines.append(line)
                # 实时更新 log_streams，支持 SSE 增量推送
                log_streams[task_id] = ''.join(output_lines)
            else:
                time.sleep(0.1)
            # 用户请求停止
            if task_meta[task_id].get('stop_requested'):
                _kill_process_group(proc)
                proc.wait(timeout=10)
                task_meta[task_id].update(success=False, error='用户主动停止', end_time=time.time())
                log_streams[task_id] = ''.join(output_lines) + '\n[系统] 用户主动停止任务'
                logger.info(f'[{tag}] 用户停止任务: {task_id}')
                running_processes[task_id] = None
                if on_done:
                    try: on_done(task_id, ''.join(output_lines), proc.returncode)
                    except Exception as cb_e: logger.error(f'[{tag}] on_done 回调异常: {task_id}: {cb_e}')
                return
            # 超时检查
            if time.time() - start > timeout:
                _kill_process_group(proc)
                proc.wait(timeout=10)
                task_meta[task_id].update(success=False, error=f'任务超时（{timeout}秒），已强制终止', end_time=time.time())
                log_streams[task_id] = ''.join(output_lines) + f'\n[系统] 任务超时（{timeout}秒），已强制终止'
                logger.error(f'[{tag}] 任务超时: {task_id}（{timeout}秒）')
                running_processes[task_id] = None
                return

        proc.wait()
        running_processes[task_id] = None
        output = ''.join(output_lines)
        log_streams[task_id] = output

        if proc.returncode != 0:
            task_meta[task_id].update(success=False, error=f'进程退出码 {proc.returncode}')
            # 失败时完整记录输出到 system.log 供排查（上限 50000 字符防日志爆炸）
            log_tail = output[-50000:] if len(output) > 50000 else output
            logger.error(f'[{tag}] 任务失败: {task_id}\n返回码: {proc.returncode}\n输出(共{len(output)}字符):\n{log_tail}')
        else:
            task_meta[task_id].update(success=True, error='')

        # 先执行 on_done（设置 summary_streams 等），再设置 end_time
        # 避免 SSE 在 summary 准备好之前就判断任务结束并关闭流
        if on_done:
            try:
                on_done(task_id, output, proc.returncode)
            except Exception as cb_e:
                logger.error(f'[{tag}] on_done 回调异常: {task_id}: {cb_e}')

        task_meta[task_id]['end_time'] = time.time()
        logger.info(f'[{tag}] 任务完成: {task_id}（耗时 {task_meta[task_id]["end_time"] - task_meta[task_id]["start_time"]:.1f}秒）')

    except Exception as e:
        task_meta[task_id].update(success=False, error=f'任务异常: {e}', end_time=time.time())
        # 修复：保留历史输出，避免异常分支覆盖 log_streams 导致调试困难
        # （与停止/超时分支一致，使用 ''.join(output_lines) + 异常信息）
        log_streams[task_id] = ''.join(output_lines) + f'\n[Error] {e}'
        logger.error(f'[{tag}] 任务异常: {task_id}\n{traceback.format_exc()}')
        running_processes[task_id] = None


def _kill_process_group(proc):
    """杀掉子进程所在的整个进程组（含 ProcessPoolExecutor 子进程）。"""
    import os as _os, signal as _signal
    if proc is None or proc.poll() is not None:
        return
    try:
        _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
        # 给 3 秒优雅退出，否则强杀
        try:
            proc.wait(timeout=3)
        except Exception:
            _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
    except Exception as e:
        # 进程组可能已退出，fallback 到 kill 单进程
        try:
            proc.kill()
        except Exception:
            pass
        logger.warning(f'[kill_process_group] 杀进程组失败: {e}')

_stock_list_cache = None
_stock_list_cache_mtime = 0
_stock_list_cache_lock = threading.Lock()


def get_stock_list():
    import pandas as pd
    global _stock_list_cache, _stock_list_cache_mtime
    try:
        mtime = config.STOCK_DATA_FILE.stat().st_mtime
        with _stock_list_cache_lock:
            if _stock_list_cache is not None and mtime == _stock_list_cache_mtime:
                return _stock_list_cache
        df = pd.read_csv(config.STOCK_DATA_FILE)
        df = df.fillna('')
        stocks = []
        for _, row in df.iterrows():
            mv = row.get('total_mv', 0)
            try:
                mv = float(mv) if mv else 0.0
            except (ValueError, TypeError):
                mv = 0.0
            stocks.append({
                'code': str(row['ts_code']),
                'name': str(row['name']),
                'industry': str(row.get('industry', '')),
                'total_mv': mv,
            })
        with _stock_list_cache_lock:
            _stock_list_cache = stocks
            _stock_list_cache_mtime = mtime
        return stocks
    except Exception as e:
        with _stock_list_cache_lock:
            if _stock_list_cache is not None:
                return _stock_list_cache
        return []

def get_index_list():
    return [
        {'code': 'DJI', 'name': '道琼斯指数'},
        {'code': 'FCHI', 'name': '法国CAC40'},
        {'code': 'SPX', 'name': '标普500'},
        {'code': 'GDAXI', 'name': '德国DAX'},
        {'code': 'N225', 'name': '日经225'},
        {'code': '000300.SH', 'name': '沪深300'},
        {'code': '399006.SZ', 'name': '创业板指'},
    ]

def get_tracking_logs():
    log_dir = Path(config.LOG_DIR)
    if not log_dir.exists():
        return []
    files = sorted(log_dir.glob('*_tracking.log'), reverse=True)
    result = []
    for f in files[:20]:
        result.append({
            'name': f.name,
            'date': f.name.replace('_tracking.log', ''),
            'path': str(f),
            'size': f.stat().st_size,
        })
    return result

def get_performance_files():
    files = []
    for d in [config.STOCK_PERFORMANCE_DIR, config.INDEX_PERFORMANCE_DIR]:
        if d.exists():
            for f in sorted(d.glob('*_performance_test.csv'), reverse=True)[:50]:
                files.append({
                    'name': f.name,
                    'path': str(f),
                    'size': f.stat().st_size,
                })
    return files

@app.route('/')
def index():
    resp = Response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/api/stocks')
def api_stocks():
    stocks = get_stock_list()
    indexes = get_index_list()
    return jsonify({'stocks': stocks, 'indexes': indexes})


# ============ 扩展证券列表（ETF/可转债/基金）搜索 ============
# 进程级缓存：{type: [{'code','name','market'}, ...]}，避免每次搜索都读 CSV
_securities_cache = {'etf': None, 'bond': None, 'fund': None}
_securities_cache_lock = threading.Lock()
_SEC_CACHE_TTL = 24 * 3600  # 24 小时


def _norm_sec_code(raw_code, market_hint=''):
    """将 akshare 返回的代码统一为 ts_code 风格（如 513310.SH）。
    market_hint: 'sh'/'sz'/'bj' 用于 ETF 等带前缀的代码。"""
    code = str(raw_code).strip().upper()
    # 去掉可能的 sh/sz/bj 前缀
    for prefix in ('SH', 'SZ', 'BJ'):
        if code.startswith(prefix):
            code = code[len(prefix):]
            market_hint = prefix.lower()
            break
    if not code:
        return ''
    # 判断市场：6 位数字按首位判断；否则用 hint
    if len(code) == 6 and code.isdigit():
        if market_hint == 'sh' or code.startswith(('5', '6', '9', '11')):
            suffix = 'SH'
        elif market_hint == 'bj' or code.startswith('8') or code.startswith('4'):
            suffix = 'BJ'
        else:
            suffix = 'SZ'
    else:
        suffix = (market_hint or 'sz').upper()
        if suffix not in ('SH', 'SZ', 'BJ'):
            suffix = 'SZ'
    return f'{code}.{suffix}'


def _load_sec_list(sec_type):
    """加载某类证券列表，优先读本地 CSV 缓存（24h TTL），否则用 akshare 拉取并写缓存。"""
    import pandas as pd
    cache_dir = config.SECURITY_LIST_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f'{sec_type}_list.csv'

    # 缓存有效则直接读
    if csv_path.exists():
        age = time.time() - csv_path.stat().st_mtime
        if age < _SEC_CACHE_TTL:
            try:
                df = pd.read_csv(csv_path, dtype=str)
                if not df.empty and 'code' in df.columns:
                    return df[['code', 'name']].to_dict('records')
            except Exception as e:
                logger.warning(f'[证券列表] 读取缓存 {csv_path} 失败: {e}')

    # 用 akshare 拉取
    try:
        import akshare as ak
        rows = []
        if sec_type == 'etf':
            df = ak.fund_etf_category_sina(symbol='ETF基金')
            for _, r in df.iterrows():
                raw_code = str(r.get('代码', ''))
                name = str(r.get('名称', '')).strip()
                market = 'sh' if raw_code.lower().startswith('sh') else 'sz'
                code = _norm_sec_code(raw_code, market)
                if code and name:
                    rows.append({'code': code, 'name': name})
        elif sec_type == 'bond':
            # 可转债：bond_zh_cov 包含全量在交易可转债
            df = ak.bond_zh_cov()
            for _, r in df.iterrows():
                raw_code = str(r.get('债券代码', ''))
                name = str(r.get('债券简称', '')).strip()
                if raw_code and name:
                    code = _norm_sec_code(raw_code)
                    if code:
                        rows.append({'code': code, 'name': name})
        elif sec_type == 'fund':
            df = ak.fund_open_fund_daily_em()
            for _, r in df.iterrows():
                raw_code = str(r.get('基金代码', ''))
                name = str(r.get('基金简称', '')).strip()
                if raw_code and name:
                    # 基金代码通常 6 位，多为深市，保持原样不补后缀（基金无.SH/.SZ）
                    rows.append({'code': raw_code, 'name': name})
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f'[证券列表] {sec_type} 拉取 {len(rows)} 条，已缓存到 {csv_path}')
            return rows
    except Exception as e:
        logger.error(f'[证券列表] 拉取 {sec_type} 失败: {e}')
    return []


def _get_sec_list(sec_type):
    """带锁的缓存获取"""
    with _securities_cache_lock:
        if _securities_cache[sec_type] is None:
            _securities_cache[sec_type] = _load_sec_list(sec_type)
        return _securities_cache[sec_type]


@app.route('/api/search_securities')
def api_search_securities():
    """搜索 ETF/可转债/基金，供持仓新增使用。
    参数: q=关键词, type=etf|bond|fund|all（默认 all）"""
    import time as _time
    q = request.args.get('q', '').strip().upper()
    sec_type = request.args.get('type', 'all')
    if not q:
        return jsonify({'items': []})
    types = ['etf', 'bond', 'fund'] if sec_type == 'all' else [sec_type]
    results = []
    for t in types:
        items = _get_sec_list(t)
        for it in items:
            if q in str(it.get('code', '')).upper() or q in str(it.get('name', '')).upper():
                results.append({'code': it['code'], 'name': it['name'], 'type': t})
                if len(results) >= 30:
                    break
        if len(results) >= 30:
            break
    return jsonify({'items': results})


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    data = request.get_json()
    code = data.get('code', '')
    start_date = data.get('start_date', '') or ''
    end_date = data.get('end_date', '') or ''
    pattern_name = data.get('pattern_name', '')
    pattern_type = data.get('pattern_type', 'buy')
    observe_day = data.get('observe_day', 2)
    cash = data.get('cash', config.DEFAULT_CASH)
    cautious = bool(data.get('cautious', False))

    if not code:
        return jsonify({'error': '请选择交易标的'}), 400
    if not pattern_name:
        return jsonify({'error': '请选择K线形态'}), 400
    if not start_date or not end_date:
        return jsonify({'error': '请设置完整的回测时间范围（开始日期和结束日期）'}), 400
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')

    logger.info(f'[回测] 请求参数: code={code}, pattern={pattern_name}, type={pattern_type}, '
                f'start={start_date}, end={end_date}, observe_day={observe_day}, cautious={cautious}')

    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        data_folder_dir = str(config.TRAIN_DATA_INDEX_DIR) + '/'
    else:
        data_folder_dir = str(config.TRAIN_DATA_A_DIR) + '/'

    # 自动补全本地数据：确保覆盖用户设置的时间范围
    import data_source
    coverage_ok = _ensure_data_coverage(code, start_date, end_date, data_folder_dir)
    if not coverage_ok:
        # 若完全无本地数据且网络补全失败，直接拒绝任务
        if data_source._find_local_data(code) is None:
            logger.warning(f'[回测] {code} 本地无数据且网络拉取失败，无法回测')
            return jsonify({'error': f'{code} 本地无数据且网络拉取失败，请检查代码或网络后重试'}), 500
        logger.warning(f'[回测] {code} 数据未完全覆盖 {start_date}~{end_date}，将使用现有本地数据继续回测')

    mode = 'pattern_test'
    task_id = f'backtest_{code}_{pattern_name}_{pattern_type}_cautious{int(cautious)}'
    logger.info(f'[回测] 启动任务: {task_id}, 数据目录: {data_folder_dir}')

    # 修复：所有用户输入参数用 repr() 转义，避免双引号注入（与 api_tracking 一致）
    script = (
        f'{_SUBPROCESS_PATH_SETUP}; '
        f'import main; '
        f'main.start_date = {repr(start_date)}; '
        f'main.end_date = {repr(end_date)}; '
        f'main.code = {repr(code)}; '
        f'main.pattern_name = {repr(pattern_name)}; '
        f'main.pattern_type = {repr(pattern_type)}; '
        f'main.mode = {repr(mode)}; '
        f'main.plot = False; '
        f'main.get_new_data = False; '
        f'main.run_pattern_recognition_Strategy('
        f'    main.code, main.start_date, main.end_date, '
        f'    pattern_category="", pattern_name=main.pattern_name, '
        f'    pattern_type=main.pattern_type, observe_day={observe_day}, '
        f'    plot=main.plot, log=True, get_new_data=main.get_new_data, '
        f'    cash={cash}, print_performance=True, cautious={cautious}, '
        f'    data_folder_dir=r"{data_folder_dir}")'
    )

    t = threading.Thread(target=run_subprocess_task, args=(task_id, script, '回测', TASK_TIMEOUT_BT), daemon=True)
    # 并发限制 + 顺带清理过期任务记录
    _cleanup_stale_tasks()
    if _count_running_tasks() >= MAX_CONCURRENT_TASKS:
        logger.warning(f'[回测] 并发任务数已达上限 {MAX_CONCURRENT_TASKS}，拒绝新任务: {task_id}')
        return jsonify({'error': f'当前已有 {_count_running_tasks()} 个任务运行中，超过并发上限 {MAX_CONCURRENT_TASKS}，请等待部分任务完成'}), 429
    t.start()

    return jsonify({'task_id': task_id, 'status': 'running'})


@app.route('/api/market-cycles')
def api_market_cycles():
    """识别标的的上涨/下跌/震荡周期（基于 ZigZag 算法）。

    Query params:
        code: 标的代码（必填）
        threshold: ZigZag 转折点阈值，默认 0.08（8%）
        min_days: 最小持续交易日数，默认 22（约1个月）
        classify_threshold: 上涨/下跌分类阈值，默认 0.05（5%）
        start_date: 起始日期 'YYYY-MM-DD'（可选，默认不限制）
        end_date: 结束日期 'YYYY-MM-DD'（可选，默认不限制）
    """
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'error': '请传入 code 参数'}), 400
    try:
        threshold = float(request.args.get('threshold', 0.08))
        min_days = int(request.args.get('min_days', 22))
        classify_threshold = float(request.args.get('classify_threshold', 0.05))
    except (TypeError, ValueError):
        return jsonify({'error': '参数格式错误'}), 400
    start_date = request.args.get('start_date', '').strip() or None
    end_date = request.args.get('end_date', '').strip() or None

    # 选择数据目录（与回测接口一致）
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        data_folder_dir = str(config.TRAIN_DATA_INDEX_DIR) + '/'
    else:
        data_folder_dir = str(config.TRAIN_DATA_A_DIR) + '/'

    logger.info(f'[周期识别] 请求: code={code}, threshold={threshold}, min_days={min_days}, '
                f'classify={classify_threshold}, range={start_date or "ALL"}~{end_date or "ALL"}')
    try:
        from portfolio_backtest import identify_market_cycles
        result = identify_market_cycles(
            code, data_folder_dir, threshold=threshold,
            min_days=min_days, classify_threshold=classify_threshold,
            start_date=start_date, end_date=end_date,
        )
        up_n = len(result.get('up_cycles', []))
        down_n = len(result.get('down_cycles', []))
        flat_n = len(result.get('flat_cycles', []))
        logger.info(f'[周期识别] 完成: code={code}, 上涨{up_n}段, 下跌{down_n}段, 震荡{flat_n}段')
        return jsonify(result)
    except Exception as e:
        logger.error(f'[周期识别] 异常: code={code}\n{traceback.format_exc()}')
        return jsonify({'error': f'周期识别失败: {e}'}), 500


@app.route('/api/backtest/multi', methods=['POST'])
def api_backtest_multi():
    """多策略叠加回测：单标的 + 多形态并行回测（线程模式，避免子进程导入问题）

    支持两种时间范围模式：
    1. 单段：start_date + end_date（兼容旧版）
    2. 多段：segments = [{start, end, type}, ...]（按周期选择模式）
       多段模式下，每段独立运行回测，结果分段返回 + 汇总统计
    """
    data = request.get_json()
    code = data.get('code', '')
    start_date = data.get('start_date', '') or ''
    end_date = data.get('end_date', '') or ''
    patterns = data.get('patterns', [])
    cash = data.get('cash', config.DEFAULT_CASH)
    cautious = bool(data.get('cautious', False))
    segments = data.get('segments')  # 多段模式：[{start, end, type}, ...]

    if not code:
        return jsonify({'error': '请选择交易标的'}), 400
    if not patterns or len(patterns) == 0:
        return jsonify({'error': '请至少选择一个K线形态'}), 400
    if segments is not None and (not isinstance(segments, list) or len(segments) == 0):
        return jsonify({'error': 'segments 参数格式错误或为空'}), 400

    is_multi_segment = bool(segments)
    seg_count = len(segments) if is_multi_segment else 0

    if not is_multi_segment:
        # 单段模式必须提供起止日期
        if not start_date or not end_date:
            return jsonify({'error': '请设置完整的回测时间范围（开始日期和结束日期）'}), 400
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
    else:
        # 多段模式用 segments 内的日期
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

    logger.info(f'[多策略回测] 请求参数: code={code}, patterns={len(patterns)}个, '
                f'mode={"多段(" + str(seg_count) + "段)" if is_multi_segment else "单段"}, '
                f'start={start_date}, end={end_date}, cautious={cautious}')

    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        data_folder_dir = str(config.TRAIN_DATA_INDEX_DIR) + '/'
    else:
        data_folder_dir = str(config.TRAIN_DATA_A_DIR) + '/'

    task_id = f'multi_bt_{code}_{len(patterns)}p_{int(time.time())}'

    def _run_multi_bt_thread():
        """线程内执行多策略回测（支持单段/多段）"""
        try:
            from portfolio_backtest import run_multi_pattern_backtest
            pattern_names = [f"{p.get('pattern_name')}({p.get('pattern_type')},hold{p.get('observe_day',2)})" for p in patterns]
            logger.info(f'[多策略回测] 形态列表: {", ".join(pattern_names)}')

            # 自动补全本地数据：覆盖整个回测区间（多段取并集）
            if is_multi_segment:
                seg_starts = [seg.get('start', '').replace('-', '') for seg in segments if seg.get('start')]
                seg_ends = [seg.get('end', '').replace('-', '') for seg in segments if seg.get('end')]
                overall_start = min(seg_starts) if seg_starts else start_date
                overall_end = max(seg_ends) if seg_ends else end_date
            else:
                overall_start, overall_end = start_date, end_date
            import data_source
            coverage_ok = _ensure_data_coverage(code, overall_start, overall_end, data_folder_dir)
            if not coverage_ok:
                if data_source._find_local_data(code) is None:
                    logger.warning(f'[多策略回测] {code} 本地无数据且网络拉取失败，无法回测')
                    task_meta[task_id].update(success=False, error='本地无数据且网络拉取失败',
                                              end_time=time.time())
                    log_streams[task_id] = f'Error: {code} 本地无数据且网络拉取失败，无法回测'
                    return
                logger.warning(f'[多策略回测] {code} 数据未完全覆盖 {overall_start}~{overall_end}，将使用现有本地数据继续回测')

            if is_multi_segment:
                # ===== 多段模式：对每段独立回测，资金重置 =====
                segment_results = []
                patterns_sum_total = 0.0
                for i, seg in enumerate(segments, 1):
                    seg_start = seg.get('start', '').replace('-', '')
                    seg_end = seg.get('end', '').replace('-', '')
                    seg_type = seg.get('type', '')
                    seg_label = f'{"上涨" if seg_type == "up" else "下跌" if seg_type == "down" else "震荡"}#{i}'
                    logger.info(f'[多策略回测] 段{i}/{seg_count}: {seg_label} {seg_start}->{seg_end}')
                    try:
                        seg_result = run_multi_pattern_backtest(
                            code, patterns, seg_start, seg_end, data_folder_dir, cash, cautious
                        )
                        seg_combined = seg_result.get('combined', {})
                        seg_td = seg_combined.get('trade_details', [])
                        seg_pnl_sum = round(sum(t.get('pnl_pct', 0) for t in seg_td), 2)
                        patterns_sum_total += seg_combined.get('total_return', 0)
                        logger.info(f'[多策略回测] 段{i}完成: 收益={seg_combined.get("total_return")}% (sum={seg_pnl_sum}%, 笔数={len(seg_td)})')
                        segment_results.append({
                            'idx': i,
                            'label': seg_label,
                            'type': seg_type,
                            'start': seg.get('start'),
                            'end': seg.get('end'),
                            'days': seg.get('days'),
                            'change_pct': seg.get('change_pct'),
                            'result': seg_result,
                        })
                    except Exception as seg_e:
                        logger.error(f'[多策略回测] 段{i}异常: {seg_e}')
                        segment_results.append({
                            'idx': i,
                            'label': seg_label,
                            'type': seg_type,
                            'start': seg.get('start'),
                            'end': seg.get('end'),
                            'days': seg.get('days'),
                            'change_pct': seg.get('change_pct'),
                            'error': str(seg_e),
                        })

                # 汇总统计
                valid_results = [r for r in segment_results if not r.get('error')]
                all_td = []
                for r in valid_results:
                    all_td.extend(r['result'].get('combined', {}).get('trade_details', []))
                total_return = round(sum(t.get('pnl_pct', 0) for t in all_td), 2)
                total_trades = len(all_td)
                won = sum(1 for t in all_td if t.get('pnl_pct', 0) > 0)
                win_rate = round(won / total_trades * 100, 0) if total_trades > 0 else 0
                positive_segments = sum(1 for r in valid_results if r['result'].get('combined', {}).get('total_return', 0) > 0)
                seg_returns = [r['result'].get('combined', {}).get('total_return', 0) for r in valid_results]
                avg_return = round(sum(seg_returns) / len(seg_returns), 2) if seg_returns else 0
                best_seg = max(valid_results, key=lambda r: r['result'].get('combined', {}).get('total_return', -float('inf'))) if valid_results else None
                worst_seg = min(valid_results, key=lambda r: r['result'].get('combined', {}).get('total_return', float('inf'))) if valid_results else None

                # 分组统计
                group_stats = {}
                for t in ['up', 'down', 'flat']:
                    type_results = [r for r in valid_results if r.get('type') == t]
                    if type_results:
                        type_returns = [r['result'].get('combined', {}).get('total_return', 0) for r in type_results]
                        group_stats[t] = {
                            'count': len(type_results),
                            'avg_return': round(sum(type_returns) / len(type_returns), 2),
                            'positive': sum(1 for x in type_returns if x > 0),
                        }

                summary_streams[task_id] = {
                    'mode': 'multi_segment',
                    'code': code,
                    'segment_results': segment_results,
                    'summary': {
                        'total_segments': len(segment_results),
                        'valid_segments': len(valid_results),
                        'positive_segments': positive_segments,
                        'total_return': total_return,
                        'avg_return': avg_return,
                        'total_trades': total_trades,
                        'win_rate': int(win_rate),
                        'best_segment': {'label': best_seg['label'], 'return': best_seg['result']['combined']['total_return']} if best_seg else None,
                        'worst_segment': {'label': worst_seg['label'], 'return': worst_seg['result']['combined']['total_return']} if worst_seg else None,
                        'group_stats': group_stats,
                    },
                }
                task_meta[task_id].update(success=True, error='', end_time=time.time())
                logger.info(f'[多策略回测] 多段任务完成: {task_id}')
                logger.info(f'[多策略回测] 汇总: {len(valid_results)}/{len(segment_results)}段有效 | 盈利{positive_segments}段 | 总收益={total_return}% (各段avg={avg_return}%) | 胜率={win_rate}% ({won}/{total_trades})')
                logger.info(f'[多策略回测] 校验: 各段收益和={round(patterns_sum_total,2)}% vs 总交易明细sum={total_return}% {"[OK]" if abs(patterns_sum_total - total_return) < 0.01 else "[不一致]"}')
            else:
                # ===== 单段模式（兼容旧版） =====
                result = run_multi_pattern_backtest(
                    code, patterns, start_date, end_date, data_folder_dir, cash, cautious
                )
                summary_streams[task_id] = result
                task_meta[task_id].update(success=True, error='', end_time=time.time())
                combined = result.get('combined', {})
                td = combined.get('trade_details', [])
                sum_pnl = round(sum(t.get('pnl_pct', 0) for t in td), 2)
                combined_won = sum(1 for t in td if t.get('pnl_pct', 0) > 0)
                logger.info(f'[多策略回测] 任务完成: {task_id}')
                logger.info(f'[多策略回测] 有效形态 {result.get("valid_count", 0)}/{result.get("pattern_count", 0)}')
                logger.info(f'[多策略回测] 组合总收益: {combined.get("total_return")}% (交易明细sum={sum_pnl}%, 笔数={len(td)})')
                logger.info(f'[多策略回测] 真实账户收益: {combined.get("real_return", "N/A")}% (含复利，仅供参考)')
                logger.info(f'[多策略回测] 最大回撤: {combined.get("max_drawdown")}% | 夏普: {combined.get("sharpe_ratio")} | 胜率: {combined.get("win_rate")}% (won={combined_won}/{len(td)})')
                logger.info(f'[多策略回测] 买入持有: 原始={combined.get("buy_hold_return")}% | 调整后={combined.get("adj_buy_hold_return")}%')
                logger.info(f'[多策略回测] 资金利用率: {combined.get("total_hold_days",0)}/{combined.get("total_trading_days",0)}天')
                patterns_sum = 0.0
                for p in result.get('patterns', []):
                    p_td = p.get('trade_details', [])
                    p_sum = round(sum(t.get('pnl_pct', 0) for t in p_td), 2)
                    patterns_sum += p_sum
                    logger.info(f'[多策略回测] 形态 {p.get("pattern_cn", p.get("pattern_name"))}: 收益={p.get("total_return")}% (明细sum={p_sum}%, 笔数={len(p_td)})')
                logger.info(f'[多策略回测] 校验: 各形态收益和={round(patterns_sum,2)}% vs 组合总收益={combined.get("total_return")}% {"[OK]" if abs(patterns_sum - combined.get("total_return", 0)) < 0.01 else "[不一致]"}')
                for i, t in enumerate(td, 1):
                    logger.info(f'[多策略回测] 交易#{i}: {t.get("buy_date","")}->{t.get("sell_date","")} 盈亏={t.get("pnl_pct")}% 持有={t.get("hold_days")}天 形态={t.get("trigger_patterns","")}')
        except Exception as e:
            task_meta[task_id].update(success=False, error=str(e), end_time=time.time())
            log_streams[task_id] = f'Error: {e}'
            logger.error(f'[多策略回测] 任务异常: {task_id}\n{traceback.format_exc()}')

    _cleanup_stale_tasks()
    if _count_running_tasks() >= MAX_CONCURRENT_TASKS:
        logger.warning(f'[多策略回测] 并发任务数已达上限 {MAX_CONCURRENT_TASKS}，拒绝新任务: {task_id}')
        return jsonify({'error': f'当前已有 {_count_running_tasks()} 个任务运行中，超过并发上限 {MAX_CONCURRENT_TASKS}，请等待部分任务完成'}), 429

    # 并发检查通过后再注册任务，避免被拒绝的请求残留为“运行中”
    task_meta[task_id] = {'success': False, 'error': '', 'start_time': time.time(), 'end_time': 0,
                          'stop_requested': False}
    t = threading.Thread(target=_run_multi_bt_thread, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'running'})


@app.route('/api/tracking', methods=['POST'])
def api_tracking():
    data = request.get_json()
    track_date = data.get('track_date', '2025-01-01')
    mode = data.get('mode', 'index')
    cautious = bool(data.get('cautious', False))
    target_codes = data.get('target_codes', []) or []
    # 跟踪模式：full=全量 / directional=定向（默认）/ configured_only=仅配置
    track_mode = data.get('track_mode', 'directional')
    if track_mode not in ('full', 'directional', 'configured_only'):
        track_mode = 'directional'
    # A 股个股市值范围筛选（前端输入单位为亿元，转为万元）
    min_mv_yi = data.get('min_mv')  # 亿元
    max_mv_yi = data.get('max_mv')  # 亿元
    min_mv = float(min_mv_yi) * 10000 if min_mv_yi not in (None, '', 0) else None
    max_mv = float(max_mv_yi) * 10000 if max_mv_yi not in (None, '', 0) else None

    logger.info(f'[跟踪] 请求参数: track_date={track_date}, mode={mode}, '
                f'cautious={cautious}, track_mode={track_mode}, target_codes={target_codes}, '
                f'min_mv={min_mv}万元, max_mv={max_mv}万元')

    task_id = f'tracking_{track_date}_{mode}_{track_mode}_cautious{int(cautious)}_{int(time.time())}'

    # 构造子进程脚本：调用 signal_tracker.run_tracking
    # 用环境变量传参避免命令行参数解析问题
    env_overrides = (
        f"import os; "
        f"os.environ['TRACK_DATE'] = {repr(track_date)}; "
        f"os.environ['TRACK_MODE'] = {repr(mode)}; "
        f"os.environ['CAUTIOUS'] = {repr('1' if cautious else '0')}; "
        f"os.environ['MIN_MV'] = {repr(str(min_mv) if min_mv is not None else '')}; "
        f"os.environ['MAX_MV'] = {repr(str(max_mv) if max_mv is not None else '')}; "
        f"os.environ['TARGET_CODES'] = {repr(','.join(target_codes))}; "
        f"os.environ['TRACK_MODE_TYPE'] = {repr(track_mode)}; "
    )
    script = (
        f'{_SUBPROCESS_PATH_SETUP}; '
        f'{env_overrides}'
        f'import signal_tracker; signal_tracker.main()'
    )

    def on_tracking_done(task_id, output, returncode):
        """跟踪任务完成回调：读取结构化汇总 JSON。"""
        summary_file = config.LOG_DIR / f'{track_date.replace("-", "")}_summary.json'
        if summary_file.exists():
            try:
                import json
                with open(summary_file, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                    # 计算信号后续涨幅
                    all_sigs = s.get('buy_signals', []) + s.get('sell_signals', [])
                    if all_sigs:
                        _compute_signal_followup(all_sigs)
                    # 构造前端 renderTrackSummary 所需的结构
                    buy_sigs = s.get('buy_signals', [])
                    sell_sigs = s.get('sell_signals', [])
                    for sig in buy_sigs:
                        sig['type'] = 'buy'
                        sig['track_date'] = track_date
                    for sig in sell_sigs:
                        sig['type'] = 'sell'
                        sig['track_date'] = track_date
                    s['all_signals'] = buy_sigs + sell_sigs
                    s['dates'] = [track_date]
                    s['daily_summaries'] = {
                        track_date: {
                            'track_date': track_date,
                            'buy_count': s.get('buy_count', 0),
                            'sell_count': s.get('sell_count', 0),
                            'total': s.get('total', 0),
                            'success': s.get('success', 0),
                            'failed': s.get('failed', 0),
                            'has_signals': s.get('has_signals', False),
                        }
                    }
                    s['stats'] = {
                        'total_buy': s.get('buy_count', 0),
                        'total_sell': s.get('sell_count', 0),
                        'days_with_data': 1,
                        'total_signals': len(buy_sigs) + len(sell_sigs),
                    }
                    summary_streams[task_id] = s
                    strong_count = s.get('strong_buy_count', 0)
                    recommend_count = s.get('recommend_buy_count', 0)
                    other_count = s.get('other_buy_count', 0)
                    logger.info(f'[跟踪回调] summary 已设置: 强烈推荐={strong_count}, 推荐={recommend_count}, 其他买入={other_count}, 卖出={len(s.get("sell_signals",[]))}, has_signals={s.get("has_signals")}')
            except Exception as e:
                summary_streams[task_id] = {'error': f'读取汇总失败: {e}'}
                logger.error(f'[跟踪回调] 读取汇总失败: {e}')
        else:
            summary_streams[task_id] = {'has_signals': False, 'total': 0,
                                         'buy_count': 0, 'sell_count': 0,
                                         'error': '无汇总文件（tracking可能未正常结束）'}
            logger.warning(f'[跟踪回调] summary 文件不存在: {summary_file}')

    t = threading.Thread(
        target=run_subprocess_task,
        args=(task_id, script, '跟踪', TASK_TIMEOUT_TR, on_tracking_done),
        daemon=True,
    )
    _cleanup_stale_tasks()
    if _count_running_tasks() >= MAX_CONCURRENT_TASKS:
        logger.warning(f'[跟踪] 并发任务数已达上限 {MAX_CONCURRENT_TASKS}，拒绝新任务: {task_id}')
        return jsonify({'error': f'当前已有 {_count_running_tasks()} 个任务运行中，超过并发上限 {MAX_CONCURRENT_TASKS}，请等待部分任务完成'}), 429
    t.start()

    return jsonify({'task_id': task_id, 'status': 'running'})


# ============================================================================
# 信号后续涨幅计算
# ============================================================================
def _compute_signal_followup(signals):
    """为信号列表计算出现后的累计涨幅。

    对每个信号，从本地 CSV 读取该 code 的 K 线数据，
    计算信号日次日到最新交易日的累计涨幅、最大涨幅、最大回撤。

    会在 sig dict 上直接添加字段：
        followup_return: 至今累计涨幅（%）
        followup_days: 经过的交易日数
        followup_max: 期间最大涨幅（%）
        followup_min: 期间最大跌幅（%）
    """
    import pandas as pd

    # 按 code 分组，避免重复读 CSV
    code_cache = {}

    def get_df(code, signal_date=None):
        """获取 code 的 K 线数据。本地数据不足时自动从网络拉取最新数据。"""
        if code not in code_cache:
            try:
                import data_source
                df = data_source._read_local_data(code, '20200101', '29991231')
                # 本地数据不足（最新日期 <= 信号日），尝试从网络拉取最新
                if df is not None and not df.empty:
                    df['trade_date'] = df['trade_date'].astype(str)
                    local_max = df['trade_date'].max()
                    if signal_date and local_max <= signal_date:
                        logger.info(f'[后续涨幅] {code} 本地数据截止 {local_max}，尝试拉取最新')
                        fresh = data_source.get_kline_df(code, signal_date, '29991231',
                                                          prefer_local=False, allow_network=True)
                        if fresh is not None and not fresh.empty:
                            fresh['trade_date'] = fresh['trade_date'].astype(str)
                            df = pd.concat([df, fresh]).drop_duplicates('trade_date').sort_values('trade_date').reset_index(drop=True)
                elif df is None or df.empty:
                    # 本地完全没有，直接网络拉取
                    df = data_source.get_kline_df(code, '20200101', '29991231',
                                                    prefer_local=False, allow_network=True)
                    if df is not None and not df.empty:
                        df['trade_date'] = df['trade_date'].astype(str)
                if df is not None and not df.empty:
                    df = df.sort_values('trade_date').reset_index(drop=True)
                code_cache[code] = df
            except Exception as e:
                logger.warning(f'[后续涨幅] 读取 {code} 数据失败: {e}')
                code_cache[code] = None
        return code_cache[code]

    for sig in signals:
        try:
            code = sig.get('code', '')
            # track_date 格式 YYYY-MM-DD，转为 YYYYMMDD
            td = sig.get('track_date', '')
            if not td:
                td = sig.get('trade_date', '')
            td = td.replace('-', '')
            if not code or not td:
                continue

            base_price = sig.get('close')
            if base_price is None or base_price <= 0:
                continue

            df = get_df(code, signal_date=td)
            if df is None or df.empty:
                continue

            # 找到信号日之后的行（trade_date > 信号日）
            after = df[df['trade_date'].astype(str) > td]
            if after.empty:
                # 信号日就是最新交易日，没有后续数据
                continue

            closes = after['close'].astype(float).tolist()
            returns = [(c - base_price) / base_price * 100 for c in closes]

            sig['followup_return'] = round(returns[-1], 2)  # 至今累计涨幅
            sig['followup_days'] = len(closes)
            sig['followup_max'] = round(max(returns), 2)  # 最大涨幅
            sig['followup_min'] = round(min(returns), 2)   # 最大跌幅
        except Exception as e:
            logger.warning(f'[后续涨幅] 计算失败 {sig.get("code")}: {e}')

    return signals


# ============================================================================
# 近7天信号：自动检查已有 summary，缺失日期补测，合并返回
# ============================================================================

def _get_recent_trading_days(days=7):
    """获取最近 N 个交易日（跳过周末）"""
    today = datetime.now()
    trading_days = []
    d = today
    while len(trading_days) < days:
        if d.weekday() < 5:  # Mon-Fri
            trading_days.append(d.strftime('%Y-%m-%d'))
        d -= timedelta(days=1)
    trading_days.reverse()  # 最旧在前
    return trading_days


def run_recent_signals_task(task_id, trading_days, mode, cautious, target_codes, min_mv, max_mv, track_mode='directional'):
    """近7天信号任务：逐日检查 summary.json，缺失则补测，最后合并返回。"""
    task_meta[task_id] = {'success': False, 'error': '', 'start_time': time.time(),
                          'end_time': 0, 'stop_requested': False}
    output_lines = []

    def log(msg):
        ts = time.strftime('%H:%M:%S')
        output_lines.append(f'[{ts}] {msg}\n')
        log_streams[task_id] = ''.join(output_lines)

    log(f'[近7天] 共 {len(trading_days)} 个交易日：{trading_days[0]} ~ {trading_days[-1]}，track_mode={track_mode}')

    # 检查哪些日期已有 summary
    missing_days = []
    for day in trading_days:
        summary_file = config.LOG_DIR / f'{day.replace("-", "")}_summary.json'
        if summary_file.exists():
            log(f'[近7天] {day} 已有跟踪记录，跳过')
        else:
            missing_days.append(day)

    log(f'[近7天] 已有 {len(trading_days) - len(missing_days)} 天，需补测 {len(missing_days)} 天')

    # 逐日补测
    for i, day in enumerate(missing_days):
        if task_meta[task_id].get('stop_requested'):
            log('[近7天] 用户主动停止')
            task_meta[task_id].update(success=False, error='用户主动停止', end_time=time.time())
            return

        log(f'[近7天] 补测 {day}（{i+1}/{len(missing_days)}）...')

        env_overrides = (
            f"import os; "
            f"os.environ['TRACK_DATE'] = {repr(day)}; "
            f"os.environ['TRACK_MODE'] = {repr(mode)}; "
            f"os.environ['CAUTIOUS'] = {repr('1' if cautious else '0')}; "
            f"os.environ['MIN_MV'] = {repr(str(min_mv) if min_mv is not None else '')}; "
            f"os.environ['MAX_MV'] = {repr(str(max_mv) if max_mv is not None else '')}; "
            f"os.environ['TARGET_CODES'] = {repr(','.join(target_codes))}; "
            f"os.environ['TRACK_MODE_TYPE'] = {repr(track_mode)}; "
        )
        script = (
            f'{_SUBPROCESS_PATH_SETUP}; '
            f'{env_overrides}'
            f'import signal_tracker; signal_tracker.main()'
        )

        try:
            proc = subprocess.Popen(
                [sys.executable, '-c', script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(config.BASE_DIR),
                start_new_session=True
            )
            running_processes[task_id] = proc

            while True:
                if proc.poll() is not None:
                    remaining = proc.stdout.read() if proc.stdout else ''
                    if remaining:
                        output_lines.append(remaining)
                        log_streams[task_id] = ''.join(output_lines)
                    break
                line = proc.stdout.readline() if proc.stdout else ''
                if line:
                    output_lines.append(line)
                    log_streams[task_id] = ''.join(output_lines)
                else:
                    time.sleep(0.1)
                if task_meta[task_id].get('stop_requested'):
                    _kill_process_group(proc)
                    proc.wait(timeout=10)
                    break

            proc.wait()
            running_processes[task_id] = None

            summary_file = config.LOG_DIR / f'{day.replace("-", "")}_summary.json'
            if summary_file.exists():
                log(f'[近7天] {day} 补测完成')
            else:
                log(f'[近7天] {day} 补测失败（无 summary）')
        except Exception as e:
            log(f'[近7天] {day} 异常: {e}')
            running_processes[task_id] = None

    # 读取所有 summary 并合并
    all_signals = []
    daily_summaries = {}
    for day in trading_days:
        summary_file = config.LOG_DIR / f'{day.replace("-", "")}_summary.json'
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                    daily_summaries[day] = {
                        'track_date': s.get('track_date', day),
                        'buy_count': s.get('buy_count', 0),
                        'sell_count': s.get('sell_count', 0),
                        'total': s.get('total', 0),
                        'success': s.get('success', 0),
                        'failed': s.get('failed', 0),
                        'has_signals': s.get('has_signals', False),
                    }
                    for sig in s.get('buy_signals', []):
                        sig['track_date'] = day
                        sig['type'] = 'buy'
                        all_signals.append(sig)
                    for sig in s.get('sell_signals', []):
                        sig['track_date'] = day
                        sig['type'] = 'sell'
                        all_signals.append(sig)
            except Exception as e:
                log(f'[近7天] 读取 {day} summary 失败: {e}')

    # 标记连续信号（同一标的同方向 ≥2 次）
    code_signal_count = {}
    for sig in all_signals:
        key = (sig.get('code'), sig.get('type'))
        code_signal_count[key] = code_signal_count.get(key, 0) + 1
    for sig in all_signals:
        key = (sig.get('code'), sig.get('type'))
        if code_signal_count.get(key, 0) >= 2:
            sig['is_consecutive'] = True

    # 买入信号分层：
    #   第一层（强烈推荐）：胜率>=70%, 次数>=10, 收益>0
    #   第二层（推荐）    ：胜率>=60%, 次数>=10, 收益>0（排除第一层）
    #   第三层（其他）    ：剩余买入信号
    buy_signals = [s for s in all_signals if s.get('type') == 'buy']
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

    total_buy = sum(s.get('buy_count', 0) for s in daily_summaries.values())
    total_sell = sum(s.get('sell_count', 0) for s in daily_summaries.values())

    merged = {
        'dates': trading_days,
        'daily_summaries': daily_summaries,
        'all_signals': all_signals,
        'strong_buy_signals': strong_buy_signals,
        'recommend_buy_signals': recommend_buy_signals,
        'other_buy_signals': other_buy_signals,
        'strong_buy_count': len(strong_buy_signals),
        'recommend_buy_count': len(recommend_buy_signals),
        'other_buy_count': len(other_buy_signals),
        'stats': {
            'total_buy': total_buy,
            'total_sell': total_sell,
            'days_with_data': len(daily_summaries),
            'total_signals': len(all_signals),
        }
    }

    # 计算信号后续涨幅
    if all_signals:
        _compute_signal_followup(all_signals)

    summary_streams[task_id] = merged
    task_meta[task_id].update(success=True, error='', end_time=time.time())
    log(f'[近7天] 全部完成：{len(daily_summaries)} 天有数据，共 {len(all_signals)} 个信号'
        f'（买入 {total_buy}，卖出 {total_sell}）')


@app.route('/api/recent_signals', methods=['POST'])
def api_recent_signals():
    """近7天信号：自动检查已有 summary，缺失日期补测，合并返回。"""
    data = request.get_json()
    mode = data.get('mode', 'all')
    cautious = bool(data.get('cautious', False))
    target_codes = data.get('target_codes', []) or []
    days = int(data.get('days', 7))
    track_mode = data.get('track_mode', 'directional')
    if track_mode not in ('full', 'directional', 'configured_only'):
        track_mode = 'directional'

    min_mv_yi = data.get('min_mv')
    max_mv_yi = data.get('max_mv')
    min_mv = float(min_mv_yi) * 10000 if min_mv_yi not in (None, '', 0) else None
    max_mv = float(max_mv_yi) * 10000 if max_mv_yi not in (None, '', 0) else None

    trading_days = _get_recent_trading_days(days)

    logger.info(f'[近7天] 请求参数: mode={mode}, cautious={cautious}, track_mode={track_mode}, '
                f'days={days}, trading_days={trading_days}')

    task_id = f'recent_{int(time.time())}'

    t = threading.Thread(
        target=run_recent_signals_task,
        args=(task_id, trading_days, mode, cautious, target_codes, min_mv, max_mv, track_mode),
        daemon=True,
    )
    _cleanup_stale_tasks()
    if _count_running_tasks() >= MAX_CONCURRENT_TASKS:
        return jsonify({'error': f'当前已有 {_count_running_tasks()} 个任务运行中，'
                                 f'超过并发上限 {MAX_CONCURRENT_TASKS}，请等待'}), 429
    t.start()

    return jsonify({'task_id': task_id, 'status': 'running', 'trading_days': trading_days})


@app.route('/api/tracking/<task_id>/events')
def api_tracking_events(task_id):
    """SSE 流：实时推送跟踪进度和信号。

    每秒推送一次当前累积的 output（含进度行）和 summary（如有）。
    任务完成后推送 final 事件并关闭流。
    """
    import time as _time

    def generate():
        last_output_len = 0
        while True:
            meta = task_meta.get(task_id, {})
            proc = running_processes.get(task_id)
            # 任务是否结束：以 task_meta 的 end_time 为准（兼容 subprocess 任务和线程任务）
            task_finished = meta.get('end_time', 0) > 0
            is_running = not task_finished
            # 是否可停止：有正在运行的子进程时才可停止
            stoppable = proc is not None and proc.poll() is None

            output = log_streams.get(task_id, '')
            # 增量推送：只发新增的输出
            new_output = output[last_output_len:] if len(output) > last_output_len else ''
            last_output_len = len(output)

            summary = summary_streams.get(task_id)

            payload = {
                'output': new_output,
                'running': is_running,
                'stoppable': stoppable,
                'stopped': meta.get('stop_requested', False),
                'summary': summary,
                'success': meta.get('success', False) if task_finished else None,
                'error': meta.get('error', '') if task_finished else '',
            }
            yield f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'

            if task_finished:
                # 推送 final 事件（此时 task_meta 已确保更新完成）
                yield f'event: done\ndata: {json.dumps({"success": meta.get("success", False), "error": meta.get("error", "")}, ensure_ascii=False)}\n\n'
                return
            _time.sleep(1)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

def _extract_target_signals(log_content, target_codes):
    """从跟踪日志中筛选目标个股的信号块

    日志格式（每行带时间戳前缀）：
        [2025-03-18 10:30:00] [INFO] 跟踪日：2025-03-18 - FCHI - 法国CAC40: 买入原因：xxx
        [2025-03-18 10:30:00] [INFO] 触发形态：xxx
        [2025-03-18 10:30:00] [INFO] 策略在该股的交易次数：xxx
    一个完整信号块 = 买入原因/卖出原因 行 + 后续的 触发形态/策略表现 行
    """
    lines = log_content.split('\n')
    # 收集目标个股的信号块（每个信号块以"买入原因："或"卖出原因："开头）
    blocks = []
    current_block = []
    current_code = None
    for line in lines:
        # 检查是否是信号开始行（包含目标代码 + 买入/卖出原因）
        is_signal_start = any(k in line for k in ['买入原因：', '卖出原因：'])
        if is_signal_start:
            # 保存上一个块
            if current_block and current_code:
                blocks.append((current_code, current_block))
            current_block = [line]
            # 确定这个信号属于哪个目标代码
            current_code = None
            for code in target_codes:
                if code in line:
                    current_code = code
                    break
        elif current_block:
            # 当前块还在继续，收集相关行
            if any(k in line for k in ['触发形态：', '策略在该股的交易次数：', '策略在该股的', '最大回撤']):
                current_block.append(line)
            elif '跟踪日：' in line and not any(k in line for k in ['买入原因：', '卖出原因：']):
                # 遇到新的跟踪日输出但不是信号开始，结束当前块
                if current_code:
                    blocks.append((current_code, current_block))
                current_block = []
                current_code = None
            else:
                # 其他行也结束当前块
                if current_code:
                    blocks.append((current_code, current_block))
                current_block = []
                current_code = None
    # 收集最后一个块
    if current_block and current_code:
        blocks.append((current_code, current_block))

    # 去重：相同代码+相同信号块内容只保留一份
    seen = set()
    result_lines = []
    for code, block in blocks:
        block_text = '\n'.join(block)
        key = (code, block_text)
        if key in seen:
            continue
        seen.add(key)
        result_lines.extend(block)
        result_lines.append('')  # 信号块之间空行分隔

    return '\n'.join(result_lines) if result_lines else '当日目标个股无信号'

@app.route('/api/performance', methods=['POST'])
def api_performance():
    data = request.get_json()
    code = data.get('code', '')
    start_date = data.get('start_date', '20100104')
    end_date = data.get('end_date', '20231229')
    buy_sell = data.get('buy_sell', 'buy')
    # 数据获取改为系统自动判断：renew_performances 入口会检查本地csv，缺失时父进程预拉取
    get_new_data = False
    cautious = bool(data.get('cautious', False))

    if not code:
        return jsonify({'error': '请选择交易标的'}), 400

    logger.info(f'[批量回测] 请求参数: code={code}, buy_sell={buy_sell}, start={start_date}, end={end_date}, cautious={cautious}')

    task_id = f'perf_{code}_{buy_sell}_cautious{int(cautious)}'

    # 修复：用户输入参数用 repr() 转义，避免双引号注入（与 api_tracking 一致）
    script = (
        f'{_SUBPROCESS_PATH_SETUP}; '
        f'from renew_strategy_performance import renew_performances; '
        f'import pandas as pd; '
        f'from config import config; '
        f'code = {repr(code)}; '
        f'buy_sell = {repr(buy_sell)}; '
        f'start = {repr(start_date)}; '
        f'end = {repr(end_date)}; '
        f'if len(code) < 9 or code in ["000300.SH", "399006.SZ"]: '
        f'    data_dir = str(config.TRAIN_DATA_INDEX_DIR) + "/"; '
        f'    perf_file = str(config.INDEX_PERFORMANCE_DIR) + f"/{code}_{buy_sell}_strategy_performance_test.csv"; '
        f'else: '
        f'    data_dir = str(config.TRAIN_DATA_A_DIR) + "/"; '
        f'    perf_file = str(config.STOCK_PERFORMANCE_DIR) + f"/{code}_{buy_sell}_strategy_performance_test.csv"; '
        f'renew_performances(code, buy_sell, start, end, perf_file, data_dir, get_new_data={str(get_new_data)}, cautious={cautious})'
    )

    t = threading.Thread(target=run_subprocess_task, args=(task_id, script, '批量回测', TASK_TIMEOUT_PF), daemon=True)
    # 并发限制 + 顺带清理过期任务记录
    _cleanup_stale_tasks()
    if _count_running_tasks() >= MAX_CONCURRENT_TASKS:
        logger.warning(f'[批量回测] 并发任务数已达上限 {MAX_CONCURRENT_TASKS}，拒绝新任务: {task_id}')
        return jsonify({'error': f'当前已有 {_count_running_tasks()} 个任务运行中，超过并发上限 {MAX_CONCURRENT_TASKS}，请等待部分任务完成'}), 429
    t.start()

    return jsonify({'task_id': task_id, 'status': 'running'})

def _sanitize_json(obj):
    """递归清理 dict/list 中的 NaN/Infinity，替换为 None（JSON null）。

    Python json.dumps 默认 allow_nan=True 会序列化 NaN/Infinity，
    但 JavaScript JSON.parse 无法解析这些值，导致前端 fetch 报错无限重试。
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


@app.route('/api/task/<task_id>/status')
def api_task_status(task_id):
    meta = task_meta.get(task_id, {})
    # 以 task_meta 的 end_time 判断任务是否结束（兼容 subprocess 和线程任务）
    task_finished = meta.get('end_time', 0) > 0
    is_running = not task_finished
    proc = running_processes.get(task_id)
    stoppable = proc is not None and proc.poll() is None
    output = log_streams.get(task_id, '')
    target_output = target_streams.get(task_id, '')
    summary = _sanitize_json(summary_streams.get(task_id))
    # 异常分支兜底：确保 target_streams 有值
    if not is_running and target_output == '' and meta and not meta.get('success', False):
        target_output = meta.get('error', '') or '任务失败'
    resp = jsonify({
        'task_id': task_id,
        'running': is_running,
        'success': meta.get('success', False) if not is_running else None,
        'error': meta.get('error', '') if not is_running else '',
        'output': output[-5000:] if output else '',
        'target_output': target_output,
        'summary': summary,
        'stoppable': stoppable,
        'stopped': meta.get('stop_requested', False),
    })
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/task/<task_id>/stop', methods=['POST'])
def api_task_stop(task_id):
    """停止运行中的任务。

    设置 stop_requested 标志，run_subprocess_task 的 readline 循环检测到后会
    kill 整个进程组。同时直接 killpg 兜底（避免 readline 阻塞在长任务上）。
    对于多日跟踪等线程任务（running_processes 可能为 None），仅设置 stop_requested。
    """
    meta = task_meta.get(task_id, {})
    if meta.get('end_time', 0) > 0:
        return jsonify({'error': '任务已结束'}), 400
    meta['stop_requested'] = True
    # 如果有子进程在运行，直接 kill 进程组兜底
    proc = running_processes.get(task_id)
    if proc is not None and proc.poll() is None:
        _kill_process_group(proc)
        logger.info(f'[停止] kill 子进程组: {task_id}')
    logger.info(f'[停止] 用户请求停止任务: {task_id}')
    return jsonify({'task_id': task_id, 'status': 'stop_requested'})


@app.route('/api/data/refresh', methods=['POST'])
def api_data_refresh():
    """集中补全指定标的的本地日K数据。

    请求体 JSON：
        {
            "codes": ["000001.SZ", "000002.SZ", ...],
            "start_date": "20260709",   # 可选，默认最近 30 天
            "end_date": "20260724"      # 可选，默认今天
        }
    响应：
        {
            "task_id": "data_refresh_<ts>",
            "total": 100,
            "success": 95,
            "failed": 5,
            "failed_codes": [{"code": "xxx", "error": "无数据返回"}]
        }
    """
    data = request.get_json() or {}
    codes = data.get('codes', [])
    if not codes or not isinstance(codes, list):
        return jsonify({'error': 'codes 参数必填且必须为列表'}), 400
    # 去重 + 过滤北交所
    codes = list(dict.fromkeys(c for c in codes if not str(c).endswith('.BJ')))
    if not codes:
        return jsonify({'error': '没有可更新的标的（已过滤 .BJ 北交所股票）'}), 400

    from datetime import datetime, timedelta
    end_date = data.get('end_date') or datetime.now().strftime('%Y%m%d')
    start_date = data.get('start_date') or (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    # 标准化为 YYYYMMDD
    end_date = str(end_date).replace('-', '')
    start_date = str(start_date).replace('-', '')

    logger.info(f'[数据补全] 请求 {len(codes)} 个标的，{start_date}~{end_date}')

    # 派发到子进程执行，避免主 Flask 进程被阻塞
    env_overrides = (
        f"import os; "
        f"os.environ['REFRESH_CODES'] = {repr(','.join(codes))}; "
        f"os.environ['REFRESH_START'] = {repr(start_date)}; "
        f"os.environ['REFRESH_END'] = {repr(end_date)}; "
    )
    script = (
        f'{_SUBPROCESS_PATH_SETUP}; '
        f'{env_overrides}'
        f'import data_refresh; data_refresh.main()'
    )
    task_id = f'data_refresh_{int(time.time())}'

    def on_refresh_done(task_id, output, returncode):
        """数据补全任务完成回调：解析末尾的结果 JSON 行。"""
        summary_streams[task_id] = {
            'output': output[-3000:],
            'refresh_done': True,
        }

    t = threading.Thread(
        target=run_subprocess_task,
        args=(task_id, script, '数据补全', 600, on_refresh_done),
        daemon=True,
    )
    _cleanup_stale_tasks()
    t.start()
    return jsonify({'task_id': task_id, 'total': len(codes),
                     'message': f'开始补全 {len(codes)} 个标的数据'})


@app.route('/api/data/refresh/<task_id>/events')
def api_data_refresh_events(task_id):
    """SSE 流：实时推送数据补全进度。"""
    import time as _time

    def generate():
        last_output_len = 0
        while True:
            meta = task_meta.get(task_id, {})
            task_finished = meta.get('end_time', 0) > 0
            output = log_streams.get(task_id, '')
            new_output = output[last_output_len:] if len(output) > last_output_len else ''
            last_output_len = len(output)

            payload = {
                'output': new_output,
                'running': not task_finished,
                'success': meta.get('success', False) if task_finished else None,
                'error': meta.get('error', '') if task_finished else '',
            }
            yield f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'

            if task_finished:
                yield f'event: done\ndata: {json.dumps({"success": meta.get("success", False), "error": meta.get("error", "")}, ensure_ascii=False)}\n\n'
                return
            _time.sleep(1)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/logs')
def api_logs():
    logs = get_tracking_logs()
    return jsonify({'logs': logs})


# ============================================================================
# 盈湖管理接口
# ============================================================================
# 盈湖统计缓存：stats API 直接返回此缓存，仅在用户点击「刷新统计」时重新计算
_mdb_stats_cache = {
    'yinghu_db': None,     # yinghu_db.get_db_stats() 结果
    'result_db': None,       # result_db.get_stats() 结果
    'updated_at': None,     # 最近一次刷新时间
}


def _refresh_mdb_stats_cache():
    """重新计算盈湖统计并更新内存缓存。"""
    import yinghu_db
    import result_db
    _mdb_stats_cache['yinghu_db'] = yinghu_db.get_db_stats()
    _mdb_stats_cache['result_db'] = result_db.get_stats()
    _mdb_stats_cache['updated_at'] = datetime.now().isoformat(timespec='seconds')


@app.route('/api/yinghu/stats')
def api_yinghu_stats():
    """获取盈湖统计信息（返回内存缓存，不实时计算）。"""
    try:
        # 缓存为空时首次填充（首次启动后第一次请求）
        if _mdb_stats_cache['yinghu_db'] is None:
            _refresh_mdb_stats_cache()
        return jsonify({
            'yinghu_db': {
                **_mdb_stats_cache['yinghu_db'],
                'start_date': config.YINGHU_DB_START_DATE,
                'updated_at': _mdb_stats_cache['updated_at'],
            },
            'result_db': _mdb_stats_cache['result_db'],
        })
    except Exception as e:
        logger.error(f'[盈湖] 获取统计失败: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/yinghu/refresh-stats', methods=['POST'])
def api_yinghu_refresh_stats():
    """重新计算盈湖统计并更新缓存（用户点击「刷新统计」时触发）。"""
    try:
        _refresh_mdb_stats_cache()
        return jsonify({
            'success': True,
            'updated_at': _mdb_stats_cache['updated_at'],
            'yinghu_db': {
                **_mdb_stats_cache['yinghu_db'],
                'start_date': config.YINGHU_DB_START_DATE,
            },
            'result_db': _mdb_stats_cache['result_db'],
        })
    except Exception as e:
        logger.error(f'[盈湖] 刷新统计失败: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/yinghu/securities')
def api_yinghu_securities():
    """查询盈湖标的列表，可按板块过滤。"""
    try:
        import yinghu_db
        board = request.args.get('board')
        include_st = request.args.get('include_st', '1') == '1'
        result = yinghu_db.list_securities(board=board, include_st=include_st)
        return jsonify({'total': len(result), 'securities': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/yinghu/init', methods=['POST'])
def api_yinghu_init():
    """启动盈湖初始化任务。

    请求体 JSON：
        {
            "start_date": "20100101",         # 可选
            "end_date": "20260724",           # 可选，默认今天
            "board": "hs",                    # 可选，仅初始化某板块
            "codes": ["000001.SZ", ...],      # 可选，仅初始化指定代码
            "include_etf": true,
            "include_index": true,
            "workers": 4
        }
    """
    data = request.get_json() or {}
    from datetime import datetime as _dt

    start_date = data.get('start_date') or config.YINGHU_DB_START_DATE
    end_date = data.get('end_date') or _dt.now().strftime('%Y%m%d')
    start_date = str(start_date).replace('-', '')
    end_date = str(end_date).replace('-', '')
    board = data.get('board')
    codes = data.get('codes')
    include_etf = data.get('include_etf', True)
    include_index = data.get('include_index', True)
    workers = data.get('workers')

    # 构造 Python 代码字符串，由 run_subprocess_task 通过 python -c 执行
    args_parts = [
        f'start_date={repr(start_date)}',
        f'end_date={repr(end_date)}',
    ]
    if board:
        args_parts.append(f'board_filter={repr(board)}')
    if codes:
        args_parts.append(f'code_filter={repr(codes)}')
    args_parts.append(f'include_etf={repr(include_etf)}')
    args_parts.append(f'include_index={repr(include_index)}')
    if workers:
        args_parts.append(f'workers={int(workers)}')

    script = (
        f'{_SUBPROCESS_PATH_SETUP}; '
        f'from yinghu_db_init import run_init; '
        f'run_init({", ".join(args_parts)})'
    )

    task_id = f'yinghu_init_{int(time.time())}'
    logger.info(f'[盈湖] 启动初始化任务: {task_id}')

    _cleanup_stale_tasks()
    t = threading.Thread(
        target=run_subprocess_task,
        args=(task_id, script, '盈湖初始化', 3600),  # 1 小时超时
        daemon=True,
    )
    t.start()
    return jsonify({
        'task_id': task_id,
        'start_date': start_date,
        'end_date': end_date,
        'board': board,
        'message': '盈湖初始化任务已启动',
    })


@app.route('/api/yinghu/init/<task_id>/events')
def api_yinghu_init_events(task_id):
    """SSE 流：实时推送盈湖初始化进度。"""
    import time as _time

    def generate():
        last_output_len = 0
        while True:
            meta = task_meta.get(task_id, {})
            task_finished = meta.get('end_time', 0) > 0
            output = log_streams.get(task_id, '')
            new_output = output[last_output_len:] if len(output) > last_output_len else ''
            last_output_len = len(output)

            payload = {
                'output': new_output,
                'running': not task_finished,
                'success': meta.get('success', False) if task_finished else None,
                'error': meta.get('error', '') if task_finished else '',
            }
            yield f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'

            if task_finished:
                yield f'event: done\ndata: {json.dumps({"success": meta.get("success", False), "error": meta.get("error", "")}, ensure_ascii=False)}\n\n'
                return
            _time.sleep(1)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/yinghu/cleanup-result-db', methods=['POST'])
def api_yinghu_cleanup_result():
    """手动清理结果库过期缓存。"""
    try:
        import result_db
        deleted = result_db.cleanup_expired()
        return jsonify({'deleted': deleted, 'message': f'已清理 {deleted} 条过期缓存'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/code-detail')
def api_tracking_code_detail():
    """获取指定标的在指定跟踪日的所有信号明细。

    GET /api/tracking/code-detail?track_date=2025-06-24&code=000001.SZ

    从 log/{YYYYMMDD}_tracking.log 中提取该 code 的所有信号块
    （买入原因/触发形态/策略表现/卖出提示），按时间顺序返回。
    """
    import re
    track_date = request.args.get('track_date', '')
    code = request.args.get('code', '')
    if not track_date or not code:
        return jsonify({'error': 'track_date 和 code 参数必填'}), 400

    log_file = config.LOG_DIR / f'{track_date.replace("-", "")}_tracking.log'
    if not log_file.exists():
        return jsonify({'error': f'当日无跟踪日志: {log_file.name}'}), 404

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        return jsonify({'error': f'读取日志失败: {e}'}), 500

    # 解析日志行，提取该 code 的信号块
    # 日志行格式：asctime 跟踪日：track_date - code - code_name: message
    signals = []  # [{type, lines: [str]}]
    current_block = None
    current_code = None
    code_name = ''
    for line in log_content.split('\n'):
        m = re.match(r'^\S+\s+\S+\s+跟踪日：\S+\s+-\s+(\S+)\s+-\s+(.+?):\s+(.+)$', line)
        if not m:
            # 非跟踪日志行，结束当前块
            if current_block and current_code == code:
                signals.append(current_block)
            current_block = None
            current_code = None
            continue

        line_code, line_name, msg = m.groups()
        if line_code != code:
            # 其他 code 的行，结束当前块
            if current_block and current_code == code:
                signals.append(current_block)
            current_block = None
            current_code = None
            continue

        # 命中目标 code
        if not code_name:
            code_name = line_name

        # 信号开始：买入原因 / 卖出提示
        if msg.startswith('买入原因：') or msg.startswith('卖出提示：'):
            # 保存上一个块
            if current_block and current_code == code:
                signals.append(current_block)
            sig_type = 'buy' if msg.startswith('买入原因：') else 'sell'
            current_block = {'type': sig_type, 'lines': [msg]}
            current_code = line_code
        elif current_block and current_code == code:
            # 当前块还在继续，收集 触发形态 / 策略表现
            if msg.startswith('触发形态：') or msg.startswith('策略在该股的交易次数') or msg.startswith('策略在该股的'):
                current_block['lines'].append(msg)
            else:
                # 其他消息，结束当前块
                signals.append(current_block)
                current_block = None
                current_code = None
        # 否则忽略非信号起始行

    # 收集最后一个块
    if current_block and current_code == code:
        signals.append(current_block)

    return jsonify({
        'code': code,
        'name': code_name,
        'track_date': track_date,
        'signals': signals,
        'count': len(signals),
    })

@app.route('/api/log/<path:log_path>')
def api_log_detail(log_path):
    import os
    safe_path = os.path.join(config.LOG_DIR, os.path.basename(log_path))
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content[-50000:]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-log')
def api_system_log():
    """获取当天系统日志内容（前端操作 + 后端处理结果）"""
    date = request.args.get('date', '')
    if date:
        # 兼容 YYYYMMDD 和 YYYY-MM-DD
        date = date.replace('-', '')
        log_file = SYSTEM_LOG_DIR / f'system_{date}.log'
    else:
        log_file = TODAY_LOG_FILE
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # 返回最后 20000 字符（约 500 行）
        return jsonify({'content': content[-20000:], 'file': log_file.name})
    except FileNotFoundError:
        return jsonify({'content': '', 'file': log_file.name, 'error': '日志文件不存在'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-logs')
def api_system_logs():
    """列出所有系统日志文件"""
    import os
    files = []
    if SYSTEM_LOG_DIR.exists():
        for f in sorted(SYSTEM_LOG_DIR.glob('system_*.log'), reverse=True):
            stat = f.stat()
            files.append({
                'name': f.name,
                'size': stat.st_size,
                'size_kb': round(stat.st_size / 1024, 1),
                'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return jsonify({'files': files})


@app.route('/api/log/frontend', methods=['POST'])
def api_log_frontend():
    """前端操作上报：将前端用户操作写入当天系统日志。

    前端在关键操作（切换Tab、添加/删除标的、修改参数、点击按钮等）时调用此接口，
    实现前端操作的完整日志覆盖，便于事后排查问题。

    请求体 JSON 格式：
        {
            "action": "操作类型（如 switchTab/addTarget/removeTarget）",
            "detail": "操作详情（可选，如标的代码、Tab名等）",
            "page": "当前页面/模块（可选，如 backtest/tracking/signal_update）"
        }
    """
    data = request.get_json(silent=True) or {}
    action = data.get('action', '').strip()
    detail = data.get('detail', '').strip()
    page = data.get('page', '').strip()

    if not action:
        return jsonify({'error': '缺少 action 参数'}), 400

    # 获取客户端IP
    ip = request.headers.get('X-Real-IP', request.remote_addr or '-')
    # 构造日志消息
    msg_parts = [f'[前端操作] {ip} action={action}']
    if page:
        msg_parts.append(f'page={page}')
    if detail:
        # detail 可能含敏感信息，限制长度避免日志过长
        msg_parts.append(f'detail={detail[:200]}')
    logger.info(' '.join(msg_parts))
    return jsonify({'success': True})


@app.route('/api/log/clear', methods=['POST'])
def api_log_clear():
    """清空当天系统日志文件内容。

    前端日志面板提供"清空日志"按钮，调用此接口清空当天 system_YYYYMMDD.log 文件。
    不删除文件本身，只清空内容，后续日志继续写入同一文件。
    """
    try:
        # 也可指定 date 参数清空指定日期的日志
        date = (request.get_json(silent=True) or {}).get('date', '')
        if date:
            date = date.replace('-', '')
            log_file = SYSTEM_LOG_DIR / f'system_{date}.log'
        else:
            log_file = SYSTEM_LOG_DIR / f'system_{datetime.now().strftime("%Y%m%d")}.log'

        if not log_file.exists():
            return jsonify({'success': False, 'error': '日志文件不存在'})

        # 清空文件内容（保留文件本身）
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('')

        logger.info(f'[日志清空] 用户手动清空了当天日志 {log_file.name}')
        return jsonify({'success': True, 'file': log_file.name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/log/cleanup', methods=['POST'])
def api_log_cleanup():
    """手动触发旧日志清理（调用 cleanup_old_logs）。

    前端日志面板提供"清理旧日志"按钮，调用此接口立即清理超过保留天数的旧日志。
    可通过 days 参数自定义保留天数（默认 30 天）。
    """
    try:
        data = request.get_json(silent=True) or {}
        days = int(data.get('days', LOG_RETENTION_DAYS))
        result = cleanup_old_logs(SYSTEM_LOG_DIR, days)
        logger.info(f'[手动清理] 手动触发日志清理: {result["deleted_count"]} 个文件被删除')
        return jsonify({
            'success': True,
            'deleted_count': result['deleted_count'],
            'deleted_files': result['deleted_files'][:50],  # 限制返回数量
            'skipped': result['skipped'],
            'retention_days': days,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/log/config')
def api_log_config():
    """返回日志配置信息（保留天数、当前日志文件等）"""
    return jsonify({
        'retention_days': LOG_RETENTION_DAYS,
        'log_dir': str(SYSTEM_LOG_DIR),
        'today_log_file': f'system_{datetime.now().strftime("%Y%m%d")}.log',
    })

@app.route('/api/performance-files')
def api_performance_files():
    files = get_performance_files()
    return jsonify({'files': files})

@app.route('/api/performance-file/<path:file_path>')
def api_performance_detail(file_path):
    import os
    safe_path = os.path.join(config.BASE_DIR, '数据', os.path.relpath(file_path, '/'))
    from config import config as cfg
    try:
        import pandas as pd
        df = pd.read_csv(safe_path)
        return jsonify({
            'columns': list(df.columns),
            'data': df.head(200).to_dict(orient='records'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _mask_token(token):
    """对 Tushare Token 做脱敏展示，未配置时返回空字符串。"""
    if not token:
        return ''
    return token[:8] + '****'


@app.route('/api/config')
def api_config():
    return jsonify({
        'tushare_token': _mask_token(config.TUSHARE_TOKEN),
        'default_cash': config.DEFAULT_CASH,
        'default_commission': config.DEFAULT_COMMISSION,
        'data_dir': str(config.DATA_DIR),
    })

@app.route('/api/patterns')
def api_patterns():
    from pattern_data import BUY_PATTERNS, SELL_PATTERNS, PATTERN_CN_NAMES, PATTERN_DESCRIPTIONS
    buy_patterns = [{'en': p, 'cn': PATTERN_CN_NAMES.get(p, p), 'desc': PATTERN_DESCRIPTIONS.get(p, '')} for p in BUY_PATTERNS]
    sell_patterns = [{'en': p, 'cn': PATTERN_CN_NAMES.get(p, p), 'desc': PATTERN_DESCRIPTIONS.get(p, '')} for p in SELL_PATTERNS]
    return jsonify({
        'buy_patterns': buy_patterns,
        'sell_patterns': sell_patterns,
    })


@app.route('/api/dashboard')
def api_dashboard():
    """首页仪表盘聚合接口：一次性返回首页所需全部元数据，减少首屏 HTTP 请求数。"""
    result = {}
    try:
        stocks = get_stock_list()
        indexes = get_index_list()
        result['stocks'] = stocks
        result['indexes'] = indexes
    except Exception as e:
        logger.error(f'[dashboard] stocks 加载失败: {e}')
        result['stocks'] = []
        result['indexes'] = []
    try:
        from pattern_data import BUY_PATTERNS, SELL_PATTERNS, PATTERN_CN_NAMES, PATTERN_DESCRIPTIONS
        result['buy_patterns'] = [{'en': p, 'cn': PATTERN_CN_NAMES.get(p, p), 'desc': PATTERN_DESCRIPTIONS.get(p, '')} for p in BUY_PATTERNS]
        result['sell_patterns'] = [{'en': p, 'cn': PATTERN_CN_NAMES.get(p, p), 'desc': PATTERN_DESCRIPTIONS.get(p, '')} for p in SELL_PATTERNS]
    except Exception as e:
        logger.error(f'[dashboard] patterns 加载失败: {e}')
        result['buy_patterns'] = []
        result['sell_patterns'] = []
    try:
        result['files'] = get_performance_files()
    except Exception as e:
        logger.error(f'[dashboard] performance-files 加载失败: {e}')
        result['files'] = []
    try:
        result['logs'] = get_tracking_logs()
    except Exception as e:
        logger.error(f'[dashboard] logs 加载失败: {e}')
        result['logs'] = []
    try:
        result['config'] = {
            'tushare_token': _mask_token(config.TUSHARE_TOKEN),
            'default_cash': config.DEFAULT_CASH,
            'default_commission': config.DEFAULT_COMMISSION,
            'data_dir': str(config.DATA_DIR),
        }
    except Exception as e:
        logger.error(f'[dashboard] config 加载失败: {e}')
        result['config'] = {}
    return jsonify(result)


# ========== 关注信号管理 API ==========
def _read_watchlist():
    """读取关注信号 JSON 文件，返回 list。文件不存在或损坏返回空列表。"""
    import json as _json
    p = config.WATCHLIST_SIGNALS_FILE
    if not p.exists():
        return []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = _json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f'[watchlist] 读取失败: {e}')
        return []


def _write_watchlist(records):
    """写入关注信号 JSON 文件。"""
    import json as _json
    try:
        config.WATCHLIST_SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.WATCHLIST_SIGNALS_FILE, 'w', encoding='utf-8') as f:
            _json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f'[watchlist] 写入失败: {e}')
        raise


@app.route('/api/watchlist/signals', methods=['GET'])
def api_watchlist_list():
    """获取关注信号列表。支持 type 过滤（buy/sell）和 q 关键词搜索。"""
    records = _read_watchlist()
    sig_type = request.args.get('type', '').strip()
    q = request.args.get('q', '').strip().lower()
    if sig_type in ('buy', 'sell'):
        records = [r for r in records if r.get('signal_type') == sig_type]
    if q:
        records = [r for r in records if q in (str(r.get('name', '')) + str(r.get('code', '')) + str(r.get('pattern_cn', '')) + str(r.get('pattern', ''))).lower()]
    # 只读参数 has_note
    if request.args.get('has_note') == '1':
        records = [r for r in records if r.get('note', '').strip()]
    # 按创建时间倒序
    records.sort(key=lambda r: r.get('created_at', ''), reverse=True)
    return jsonify({'signals': records, 'count': len(records)})


@app.route('/api/watchlist/signals', methods=['POST'])
def api_watchlist_add():
    """添加关注信号。请求体字段：
    code, name, pattern, pattern_cn, signal_type(buy/sell),
    track_date, close, pct_chg, pattern_desc, win_rate, return_pct, trade_count,
    market_win_rate, market_return_pct, market_trade_count, is_index
    """
    data = request.get_json(silent=True) or {}
    required = ['code', 'name', 'pattern', 'signal_type']
    for f in required:
        if not data.get(f):
            return jsonify({'error': f'缺少必填字段: {f}'}), 400

    records = _read_watchlist()
    # 去重：同 code + pattern + signal_type + track_date 视为同一关注
    track_date = data.get('track_date', '')
    existing = [r for r in records if r.get('code') == data['code']
                and r.get('pattern') == data['pattern']
                and r.get('signal_type') == data['signal_type']
                and r.get('track_date', '') == track_date]
    if existing:
        return jsonify({'error': '该信号已关注', 'signal': existing[0]}), 409

    import uuid
    from datetime import datetime
    new_rec = {
        'id': str(uuid.uuid4())[:8],
        'code': data['code'],
        'name': data['name'],
        'pattern': data['pattern'],
        'pattern_cn': data.get('pattern_cn', data['pattern']),
        'pattern_desc': data.get('pattern_desc', ''),
        'signal_type': data['signal_type'],
        'track_date': track_date,
        'close': data.get('close'),
        'pct_chg': data.get('pct_chg'),
        'is_index': bool(data.get('is_index', False)),
        'win_rate': data.get('win_rate'),
        'return_pct': data.get('return_pct'),
        'trade_count': data.get('trade_count'),
        'market_win_rate': data.get('market_win_rate'),
        'market_return_pct': data.get('market_return_pct'),
        'market_trade_count': data.get('market_trade_count'),
        'note': '',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    records.append(new_rec)
    try:
        _write_watchlist(records)
        logger.info(f'[watchlist] 添加关注: {new_rec["code"]} {new_rec["pattern"]} {new_rec["signal_type"]}')
        return jsonify({'ok': True, 'signal': new_rec})
    except Exception as e:
        return jsonify({'error': f'保存失败: {e}'}), 500


@app.route('/api/watchlist/signals/<signal_id>', methods=['DELETE'])
def api_watchlist_delete(signal_id):
    """删除关注信号。"""
    records = _read_watchlist()
    new_records = [r for r in records if r.get('id') != signal_id]
    if len(new_records) == len(records):
        return jsonify({'error': '未找到该关注信号'}), 404
    try:
        _write_watchlist(new_records)
        logger.info(f'[watchlist] 删除关注: {signal_id}')
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': f'删除失败: {e}'}), 500


@app.route('/api/watchlist/signals/<signal_id>/note', methods=['PATCH'])
def api_watchlist_update_note(signal_id):
    """更新关注信号备注。请求体：{note: "..."}"""
    data = request.get_json(silent=True) or {}
    note = (data.get('note') or '').strip()
    records = _read_watchlist()
    target = None
    for r in records:
        if r.get('id') == signal_id:
            r['note'] = note
            target = r
            break
    if not target:
        return jsonify({'error': '未找到该关注信号'}), 404
    try:
        _write_watchlist(records)
        logger.info(f'[watchlist] 更新备注: {signal_id} -> {note[:30]}')
        return jsonify({'ok': True, 'signal': target})
    except Exception as e:
        return jsonify({'error': f'更新失败: {e}'}), 500


@app.route('/api/watchlist/check', methods=['GET'])
def api_watchlist_check():
    """检查某信号是否已关注。参数：code, pattern, signal_type, track_date"""
    code = request.args.get('code', '').strip()
    pattern = request.args.get('pattern', '').strip()
    sig_type = request.args.get('signal_type', '').strip()
    track_date = request.args.get('track_date', '').strip()
    records = _read_watchlist()
    exists = any(r.get('code') == code and r.get('pattern') == pattern
                 and r.get('signal_type') == sig_type
                 and r.get('track_date', '') == track_date for r in records)
    return jsonify({'watched': exists})


@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常捕获，记录所有未处理的错误"""
    logger.error(f'[未捕获异常] {request.method} {request.url}\n{traceback.format_exc()}')
    return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


def _read_positions():
    """读取持仓CSV"""
    import pandas as pd
    if not config.POSITION_FILE.exists():
        return []
    try:
        df = pd.read_csv(config.POSITION_FILE, dtype={'code': str})
        df = df.fillna('')
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f'[持仓] 读取失败: {e}')
        return []

def _write_positions(records):
    """写入持仓CSV"""
    import pandas as pd
    config.POSITION_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=['code', 'name', 'cost_price', 'shares', 'buy_date', 'notes'])
    df.to_csv(config.POSITION_FILE, index=False, encoding='utf-8-sig')

@app.route('/api/positions')
def api_positions():
    return jsonify({'positions': _read_positions()})

@app.route('/api/positions', methods=['POST'])
def api_position_add():
    data = request.get_json()
    code = str(data.get('code', '')).strip()
    name = str(data.get('name', '')).strip()
    cost_price = float(data.get('cost_price', 0) or 0)
    shares = int(data.get('shares', 0) or 0)
    buy_date = str(data.get('buy_date', '')).strip()
    notes = str(data.get('notes', '')).strip()

    if not code:
        return jsonify({'error': '股票代码不能为空'}), 400
    if shares <= 0:
        return jsonify({'error': '股数必须大于0'}), 400

    records = _read_positions()
    # 去重：同代码则更新
    records = [r for r in records if str(r.get('code')) != code]
    records.append({
        'code': code, 'name': name, 'cost_price': cost_price,
        'shares': shares, 'buy_date': buy_date, 'notes': notes,
    })
    _write_positions(records)
    logger.info(f'[持仓] 新增/更新: code={code}, name={name}, shares={shares}, cost={cost_price}')
    return jsonify({'status': 'ok', 'positions': records})

@app.route('/api/positions/<code>', methods=['DELETE'])
def api_position_delete(code):
    records = _read_positions()
    new_records = [r for r in records if str(r.get('code')) != code]
    if len(new_records) == len(records):
        return jsonify({'error': '未找到该持仓记录'}), 404
    _write_positions(new_records)
    logger.info(f'[持仓] 删除: code={code}')
    return jsonify({'status': 'ok', 'positions': new_records})

@app.route('/api/positions/<code>', methods=['PATCH'])
def api_position_patch(code):
    """单字段更新持仓（行内编辑支持）

    请求体: {"cost_price": 1680.0} 或 {"shares": 100} 或 {"notes": "..."}
    支持 cost_price / shares / notes / name 任意单字段或多字段组合更新。
    """
    data = request.get_json() or {}
    records = _read_positions()
    target = None
    for r in records:
        if str(r.get('code')) == code:
            target = r
            break
    if target is None:
        return jsonify({'error': '未找到该持仓记录'}), 404

    # 仅允许更新白名单字段
    updated_fields = []
    if 'cost_price' in data:
        try:
            new_cost = float(data['cost_price'])
            if new_cost <= 0:
                return jsonify({'error': '成本价必须大于0'}), 400
            target['cost_price'] = new_cost
            updated_fields.append('cost_price')
        except (ValueError, TypeError):
            return jsonify({'error': '成本价格式无效'}), 400
    if 'shares' in data:
        try:
            new_shares = int(data['shares'])
            if new_shares <= 0:
                return jsonify({'error': '股数必须大于0'}), 400
            target['shares'] = new_shares
            updated_fields.append('shares')
        except (ValueError, TypeError):
            return jsonify({'error': '股数格式无效'}), 400
    if 'notes' in data:
        target['notes'] = str(data['notes']).strip()
        updated_fields.append('notes')
    if 'name' in data:
        target['name'] = str(data['name']).strip()
        updated_fields.append('name')

    if not updated_fields:
        return jsonify({'error': '未提供任何可更新字段'}), 400

    _write_positions(records)
    logger.info(f'[持仓] PATCH: code={code}, fields={updated_fields}')
    return jsonify({'status': 'ok', 'positions': records, 'updated_fields': updated_fields})


def _get_position_quote(code):
    """读取持仓个股的最新行情（工作日显示当天，周末显示上一工作日）

    1. 优先读本地 daily Parquet/CSV（每日跟踪→训练→测试目录）
    2. 若本地最新日期 < 期望最新交易日（工作日=今天；周末=最近周五），通过 data_source 拉取最新
    3. 若本地 pct_chg 缺失，用前一日 close 反推
    返回 {'current': float|None, 'pct_chg': float|None, 'trade_date': str}
    """
    import pandas as pd
    from datetime import datetime, timedelta
    code = str(code)
    # 期望最新交易日：周一~周五=今天；周六=上周五；周日=上周五
    today = datetime.now().date()
    weekday = today.weekday()  # 0=周一, 6=周日
    if weekday >= 5:  # 周末
        days_back = weekday - 4  # 周六=1, 周日=2
        expected_latest = today - timedelta(days=days_back)
    else:
        expected_latest = today
    expected_latest_str = expected_latest.strftime('%Y%m%d')
    # 决定搜索目录（指数 vs A股）
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        dirs = [config.DAILY_TRACKING_INDEX_DIR, config.TRAIN_DATA_INDEX_DIR, config.TEST_DATA_INDEX_DIR]
    else:
        dirs = [config.DAILY_TRACKING_A_DIR, config.TRAIN_DATA_A_DIR, config.TEST_DATA_A_DIR]
    local_quote = None
    for d in dirs:
        # 优先 Parquet，回退 CSV
        pq_path = d / f'{code}_daily.parquet'
        csv_path = d / f'{code}_daily.csv'
        data_path = None
        is_parquet = False
        if pq_path.exists() and pq_path.stat().st_size > 0:
            data_path = pq_path
            is_parquet = True
        elif csv_path.exists():
            data_path = csv_path
        if not data_path:
            continue
        try:
            if is_parquet:
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path, dtype={'ts_code': str})
            if df.empty:
                continue
            df = df.sort_values(by='trade_date', ascending=False)
            latest = df.iloc[0]
            current = float(latest['close']) if pd.notna(latest.get('close')) else None
            pct_chg = float(latest['pct_chg']) if pd.notna(latest.get('pct_chg')) else None
            # 本地 pct_chg 缺失时，用前一日 close 反推
            if (pct_chg is None or pd.isna(pct_chg)) and current is not None and len(df) > 1:
                try:
                    prev_close = df.iloc[1].get('close')
                    if pd.notna(prev_close) and float(prev_close) > 0:
                        pct_chg = round((current - float(prev_close)) / float(prev_close) * 100, 4)
                except Exception:
                    pass
            trade_date = str(latest.get('trade_date', ''))
            if current is not None:
                local_quote = {'current': current, 'pct_chg': pct_chg, 'trade_date': trade_date}
                break
        except Exception as e:
            logger.warning(f'[持仓行情] 读取 {data_path} 失败: {e}')
            continue
    # 本地数据已是最新期望交易日：直接返回
    if local_quote and local_quote['trade_date'] >= expected_latest_str:
        return local_quote
    # 本地无数据 或 本地日期落后于最新交易日：调用 data_source 拉取最新
    try:
        fresh = _fetch_quote_from_data_source(code)
        if fresh and fresh.get('current') is not None:
            # fresh 比 local 更新则采用 fresh
            if not local_quote or fresh.get('trade_date', '') >= local_quote['trade_date']:
                return fresh
    except Exception as e:
        logger.warning(f'[持仓行情] data_source 兜底 {code} 失败: {e}')
    # 兜底：返回本地数据（可能为 None）
    if local_quote:
        return local_quote
    return {'current': None, 'pct_chg': None, 'trade_date': ''}


def _fetch_quote_from_data_source(code):
    """从 data_source（akshare 免费数据源）获取最新一条行情（本地无数据时兜底）

    优先用 akshare，tushare 作为最后备选。pct_chg 缺失时用前一日 close 反推。
    """
    from datetime import datetime, timedelta
    import pandas as pd
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
    try:
        import data_source
        df = data_source.get_kline_df(code, start_date, end_date, prefer_local=False)
        if df is not None and not df.empty:
            df = df.sort_values(by='trade_date', ascending=False)
            latest = df.iloc[0]
            current = float(latest['close']) if pd.notna(latest.get('close')) else None
            pct_chg = float(latest['pct_chg']) if pd.notna(latest.get('pct_chg')) else None
            # pct_chg 缺失时用前一日 close 反推（akshare 部分接口不返回 pct_chg）
            if (pct_chg is None or pd.isna(pct_chg)) and current is not None and len(df) > 1:
                try:
                    prev_close = df.iloc[1].get('close')
                    if pd.notna(prev_close) and float(prev_close) > 0:
                        pct_chg = round((current - float(prev_close)) / float(prev_close) * 100, 4)
                except Exception:
                    pass
            trade_date = str(latest.get('trade_date', ''))
            return {'current': current, 'pct_chg': pct_chg, 'trade_date': trade_date}
    except Exception as e:
        logger.warning(f'[持仓行情] data_source 获取 {code} 失败: {e}')

    # data_source 不支持 ETF/可转债/基金，走实时行情批量接口
    spot = _fetch_spot_quote(code)
    if spot is not None:
        return spot

    # data_source 也失败，最后尝试 tushare（可能因 token 权限不足而失败）
    return _fetch_quote_from_tushare(code)


# ============ ETF/可转债/基金实时行情（批量缓存） ============
# 进程级缓存：避免每次持仓刷新都全量拉取 akshare
_spot_cache = {'etf': {'data': {}, 'ts': 0}, 'bond': {'data': {}, 'ts': 0}, 'fund': {'data': {}, 'ts': 0}}
_SPOT_CACHE_TTL = 60  # 60 秒缓存（盘中刷新够用）
_spot_cache_lock = threading.Lock()


def _load_etf_spot():
    """拉取全部 ETF 实时行情，返回 {code6: {current, pct_chg, trade_date}}"""
    import time as _t
    import pandas as pd
    with _spot_cache_lock:
        if _t.time() - _spot_cache['etf']['ts'] < _SPOT_CACHE_TTL and _spot_cache['etf']['data']:
            return _spot_cache['etf']['data']
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        result = {}
        today = datetime.now().strftime('%Y%m%d')
        for _, r in df.iterrows():
            code6 = str(r.get('代码', '')).strip()
            if not code6:
                continue
            current = r.get('最新价')
            pct = r.get('涨跌幅')
            try:
                current = float(current) if pd.notna(current) else None
            except (ValueError, TypeError):
                current = None
            try:
                pct = float(pct) if pd.notna(pct) else None
            except (ValueError, TypeError):
                pct = None
            result[code6] = {'current': current, 'pct_chg': pct, 'trade_date': today}
        with _spot_cache_lock:
            _spot_cache['etf'] = {'data': result, 'ts': _t.time()}
        logger.info(f'[持仓行情] ETF 实时行情拉取 {len(result)} 条')
        return result
    except Exception as e:
        logger.warning(f'[持仓行情] ETF 实时行情拉取失败: {e}')
        return {}


def _load_bond_spot():
    """拉取全部可转债实时行情，返回 {code6: {current, pct_chg, trade_date}}"""
    import time as _t
    import pandas as pd
    with _spot_cache_lock:
        if _t.time() - _spot_cache['bond']['ts'] < _SPOT_CACHE_TTL and _spot_cache['bond']['data']:
            return _spot_cache['bond']['data']
    try:
        import akshare as ak
        # bond_zh_hs_cov_spot: 全量可转债实时行情（含现价、涨跌幅）
        df = ak.bond_zh_hs_cov_spot()
        result = {}
        today = datetime.now().strftime('%Y%m%d')
        for _, r in df.iterrows():
            code6 = str(r.get('code', '')).strip()
            if not code6:
                continue
            current = r.get('trade')  # 现价
            pct = r.get('changepercent')  # 涨跌幅
            try:
                current = float(current) if pd.notna(current) else None
            except (ValueError, TypeError):
                current = None
            try:
                pct = float(pct) if pd.notna(pct) else None
            except (ValueError, TypeError):
                pct = None
            result[code6] = {'current': current, 'pct_chg': pct, 'trade_date': today}
        with _spot_cache_lock:
            _spot_cache['bond'] = {'data': result, 'ts': _t.time()}
        logger.info(f'[持仓行情] 可转债实时行情拉取 {len(result)} 条')
        return result
    except Exception as e:
        logger.warning(f'[持仓行情] 可转债实时行情拉取失败: {e}')
        return {}


def _load_fund_spot():
    """拉取全部开放式基金净值，返回 {code: {current, pct_chg, trade_date}}"""
    import time as _t
    import pandas as pd
    with _spot_cache_lock:
        if _t.time() - _spot_cache['fund']['ts'] < _SPOT_CACHE_TTL and _spot_cache['fund']['data']:
            return _spot_cache['fund']['data']
    try:
        import akshare as ak
        df = ak.fund_open_fund_daily_em()
        result = {}
        for _, r in df.iterrows():
            code = str(r.get('基金代码', '')).strip()
            if not code:
                continue
            # 列名含日期，如 '2026-07-10-单位净值'，取最新单位净值
            nav = None
            pct = None
            for col in df.columns:
                if '单位净值' in str(col):
                    v = r.get(col)
                    try:
                        v = float(v) if pd.notna(v) else None
                    except (ValueError, TypeError):
                        v = None
                    if v is not None:
                        nav = v  # 取最后一个非空（最新日期列）
                if '日增长率' in str(col):
                    v = r.get(col)
                    try:
                        v = float(v) if pd.notna(v) else None
                    except (ValueError, TypeError):
                        v = None
                    pct = v
            # 交易日期从列名提取
            trade_date = ''
            for col in df.columns:
                if '单位净值' in str(col) and r.get(col) is not None:
                    m = __import__('re').match(r'(\d{4}-\d{2}-\d{2})', str(col))
                    if m:
                        trade_date = m.group(1).replace('-', '')
                        break
            if nav is not None:
                result[code] = {'current': nav, 'pct_chg': pct, 'trade_date': trade_date}
        with _spot_cache_lock:
            _spot_cache['fund'] = {'data': result, 'ts': _t.time()}
        logger.info(f'[持仓行情] 基金净值拉取 {len(result)} 条')
        return result
    except Exception as e:
        logger.warning(f'[持仓行情] 基金净值拉取失败: {e}')
        return {}


def _fetch_spot_quote(code):
    """对 ETF/可转债/基金代码，从实时行情缓存中取现价和涨跌幅。
    返回 {'current','pct_chg','trade_date'} 或 None（非这三类或未命中）。
    """
    code = str(code)
    code6 = code.split('.')[0] if '.' in code else code
    # ETF：5 开头（沪）或 1 开头（深）的 6 位代码
    if len(code6) == 6 and code6.isdigit() and code6.startswith(('5', '15', '16', '51', '52', '56', '58')):
        etf_data = _load_etf_spot()
        if code6 in etf_data:
            return etf_data[code6]
    # 可转债：11/12 开头
    if len(code6) == 6 and code6.isdigit() and code6.startswith(('11', '12')):
        bond_data = _load_bond_spot()
        if code6 in bond_data:
            return bond_data[code6]
    # 基金：非 6 位或非 SH/SZ 后缀（基金代码通常 5-6 位，无市场后缀）
    if '.' not in code and (len(code6) != 6 or not code6.startswith(('5', '6', '0', '3', '11', '12'))):
        fund_data = _load_fund_spot()
        if code in fund_data:
            return fund_data[code]
    return None




def _fetch_quote_from_tushare(code):
    """从 tushare API 获取最新一条行情（最后兜底）"""
    from datetime import datetime, timedelta
    import pandas as pd
    if not config.TUSHARE_TOKEN:
        return {'current': None, 'pct_chg': None, 'trade_date': ''}
    try:
        import tushare as ts
        ts.set_token(config.TUSHARE_TOKEN)
        pro = ts.pro_api()
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        if len(code) < 9:
            df = pro.index_global(ts_code=code, start_date=start_date, end_date=end_date)
        elif code in ['000300.SH', '399006.SZ']:
            df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
        else:
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return {'current': None, 'pct_chg': None, 'trade_date': ''}
        df = df.sort_values(by='trade_date', ascending=False)
        latest = df.iloc[0]
        current = float(latest['close']) if pd.notna(latest.get('close')) else None
        pct_chg = float(latest['pct_chg']) if pd.notna(latest.get('pct_chg')) else None
        trade_date = str(latest.get('trade_date', ''))
        return {'current': current, 'pct_chg': pct_chg, 'trade_date': trade_date}
    except Exception as e:
        logger.warning(f'[持仓行情] tushare获取 {code} 失败: {e}')
        return {'current': None, 'pct_chg': None, 'trade_date': ''}


@app.route('/api/positions/quote')
def api_positions_quote():
    """批量获取持仓的现价和当日涨跌幅（读本地daily csv，不调外部API）

    使用线程池并行获取，加快多持仓刷新速度。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    positions = _read_positions()
    codes = [str(p.get('code', '')) for p in positions if p.get('code')]
    quotes = {}
    if not codes:
        return jsonify({'quotes': quotes})
    # 并行获取（受全局 MAX_WORKERS 约束）
    with ThreadPoolExecutor(max_workers=min(getattr(config, 'MAX_WORKERS', 4), len(codes))) as executor:
        futures = {executor.submit(_get_position_quote, code): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                quotes[code] = future.result()
            except Exception as e:
                logger.warning(f'[持仓行情] 获取 {code} 失败: {e}')
                quotes[code] = {'current': None, 'pct_chg': None, 'trade_date': ''}
    return jsonify({'quotes': quotes})


def _load_kline_data(code):
    """读取标的完整日K数据（OHLC+量），用于K线图展示

    查找顺序：每日跟踪→训练→测试目录；本地无数据时调用tushare API获取近1年数据。
    返回 list[{date, open, high, low, close, vol}]，按日期升序。
    """
    import pandas as pd
    from datetime import datetime, timedelta
    code = str(code)
    # 决定搜索目录（指数 vs A股）
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        dirs = [config.DAILY_TRACKING_INDEX_DIR, config.TRAIN_DATA_INDEX_DIR, config.TEST_DATA_INDEX_DIR]
    else:
        dirs = [config.DAILY_TRACKING_A_DIR, config.TRAIN_DATA_A_DIR, config.TEST_DATA_A_DIR]

    df = None
    for d in dirs:
        # 优先 Parquet，回退 CSV
        pq_path = d / f'{code}_daily.parquet'
        csv_path = d / f'{code}_daily.csv'
        data_path = None
        is_parquet = False
        if pq_path.exists() and pq_path.stat().st_size > 0:
            data_path = pq_path
            is_parquet = True
        elif csv_path.exists():
            data_path = csv_path
        if not data_path:
            continue
        try:
            if is_parquet:
                tmp = pd.read_parquet(data_path)
            else:
                tmp = pd.read_csv(data_path, dtype={'ts_code': str})
            if not tmp.empty:
                df = tmp
                break
        except Exception as e:
            logger.warning(f'[K线] 读取 {data_path} 失败: {e}')
            continue

    # 本地无数据或数据不足 60 根，通过 data_source 拉取近 1 年数据
    if df is None or len(df) < 60:
        try:
            import data_source
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            fresh = data_source.get_kline_df(code, start_date, end_date,
                                              prefer_local=False, allow_network=True)
            if fresh is not None and not fresh.empty:
                if df is not None and not df.empty:
                    df['trade_date'] = df['trade_date'].astype(str)
                    fresh['trade_date'] = fresh['trade_date'].astype(str)
                    df = pd.concat([df, fresh]).drop_duplicates('trade_date').sort_values('trade_date').reset_index(drop=True)
                else:
                    df = fresh
            if df is None or df.empty:
                return []
        except Exception as e:
            logger.warning(f'[K线] data_source 获取 {code} 失败: {e}')
            return []

    if df is None or df.empty:
        return []

    # 标准化列名：不同接口返回的列名可能不同
    # index_global: ts_code,trade_date,open,close,high,low,pre_close,change,pct_chg,swing,vol
    # pro.daily: ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
    # A股daily(每日跟踪): ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
    try:
        df = df.sort_values(by='trade_date', ascending=True)
        # 限制最近 500 条，避免数据过多影响前端渲染
        if len(df) > 500:
            df = df.tail(500)
        kline = []
        for _, row in df.iterrows():
            trade_date = str(row.get('trade_date', ''))
            # 转换为 YYYY-MM-DD 格式（lightweight-charts 要求）
            if len(trade_date) == 8:
                date_str = f'{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}'
            else:
                date_str = trade_date
            open_p = float(row.get('open', 0)) if pd.notna(row.get('open')) else 0
            high_p = float(row.get('high', 0)) if pd.notna(row.get('high')) else 0
            low_p = float(row.get('low', 0)) if pd.notna(row.get('low')) else 0
            close_p = float(row.get('close', 0)) if pd.notna(row.get('close')) else 0
            vol = float(row.get('vol', 0)) if pd.notna(row.get('vol')) else 0
            # 跳过无效数据（OHLC 全 0）
            if open_p == 0 and high_p == 0 and low_p == 0 and close_p == 0:
                continue
            kline.append({
                'date': date_str,
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p,
                'vol': vol,
            })
        return kline
    except Exception as e:
        logger.error(f'[K线] 解析 {code} 数据失败: {e}')
        return []


@app.route('/api/position/<code>/kline')
def api_position_kline(code):
    """获取持仓标的的日K线数据（OHLC+量）"""
    kline = _load_kline_data(code)
    # 同时返回标的名称（从持仓数据中查找）
    name = ''
    for p in _read_positions():
        if str(p.get('code')) == str(code):
            name = p.get('name', '')
            break
    return jsonify({'code': code, 'name': name, 'kline': kline})


@app.route('/api/kline/<code>')
def api_kline(code):
    """通用 K 线数据接口（不依赖持仓，用于每日信号页面点击查看 K 线图）

    支持通过 query 参数 track_date 高亮跟踪日（前端用）。
    """
    kline = _load_kline_data(code)
    # 尝试从 stock_data.csv 或指数映射中获取名称
    name = ''
    if len(str(code)) < 9 or str(code) in ['000300.SH', '399006.SZ']:
        index_name_map = {
            'DJI': '道琼斯', 'FCHI': '法国CAC40', 'SPX': '标普500',
            'N225': '日经225', 'GDAXI': '德国DAX',
            '000300.SH': '沪深300', '399006.SZ': '创业板指',
        }
        name = index_name_map.get(str(code), '')
    else:
        try:
            import pandas as pd
            if config.STOCK_DATA_FILE.exists():
                sd = pd.read_csv(config.STOCK_DATA_FILE)
                m = sd[sd['ts_code'] == str(code)]['name']
                if len(m) > 0:
                    name = str(m.values[0])
        except Exception:
            pass
    return jsonify({'code': code, 'name': name, 'kline': kline})


# ============================================================================
# 个股策略配置 CRUD + 单股全量扫描（定向跟踪功能）
# ============================================================================
def _get_code_name(code):
    """查询 code 名称（用于策略配置展示）"""
    code = str(code)
    index_name_map = {
        'DJI': '道琼斯', 'FCHI': '法国CAC40', 'SPX': '标普500',
        'N225': '日经225', 'GDAXI': '德国DAX',
        '000300.SH': '沪深300', '399006.SZ': '创业板指',
    }
    if code in index_name_map:
        return index_name_map[code]
    try:
        import pandas as pd
        if config.STOCK_DATA_FILE.exists():
            sd = pd.read_csv(config.STOCK_DATA_FILE)
            m = sd[sd['ts_code'] == code]['name']
            if len(m) > 0:
                return str(m.values[0])
    except Exception:
        pass
    return ''


@app.route('/api/strategy_configs', methods=['GET'])
def api_strategy_configs_list():
    """列出全部策略配置"""
    try:
        import strategy_config
        configs = strategy_config.load_all_configs()
        # 聚合：按 code 分组统计
        by_code = {}
        for c in configs:
            code = c['code']
            if code not in by_code:
                by_code[code] = {
                    'code': code, 'name': c.get('name', ''),
                    'patterns': [], 'enabled_count': 0, 'total_count': 0,
                }
            by_code[code]['patterns'].append(c)
            by_code[code]['total_count'] += 1
            if c.get('enabled', True):
                by_code[code]['enabled_count'] += 1
        return jsonify({'configs': list(by_code.values()), 'total': len(configs)})
    except Exception as e:
        logger.error(f'[策略配置] 列出失败: {e}\n{traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500


# 注意：静态路径 /codes 必须在 /<code> 之前注册，否则会被 <code> 捕获
@app.route('/api/strategy_configs/codes', methods=['GET'])
def api_strategy_configured_codes():
    """获取所有已配置策略的股票代码集合（用于前端判断是否定向）"""
    try:
        import strategy_config
        codes = sorted(strategy_config.load_configured_codes())
        return jsonify({'codes': codes, 'count': len(codes)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_configs/<code>', methods=['GET'])
def api_strategy_configs_get(code):
    """查询某股的策略配置"""
    try:
        import strategy_config
        configs = strategy_config.load_configs_by_code(code)
        name = ''
        if configs:
            name = configs[0].get('name', '')
        else:
            name = _get_code_name(code)
        return jsonify({'code': code, 'name': name, 'configs': configs})
    except Exception as e:
        logger.error(f'[策略配置] 查询 {code} 失败: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_configs/<code>', methods=['POST'])
def api_strategy_configs_save(code):
    """保存某股的策略配置（覆盖式）

    Body: {'name': str, 'patterns': [{'pattern','pattern_type','observe_day','win_rate','return_pct','sharpe','enabled'}]}
    """
    data = request.get_json()
    name = data.get('name', '') or _get_code_name(code)
    patterns = data.get('patterns', [])
    if not patterns:
        return jsonify({'error': '请至少选择一个策略'}), 400
    try:
        import strategy_config
        strategy_config.save_configs(code, name, patterns)
        logger.info(f'[策略配置] 保存 {code} {name} 的 {len(patterns)} 条配置')
        return jsonify({'success': True, 'code': code, 'name': name, 'count': len(patterns)})
    except Exception as e:
        logger.error(f'[策略配置] 保存 {code} 失败: {e}\n{traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_configs/<code>', methods=['DELETE'])
def api_strategy_configs_delete(code):
    """删除某股的全部策略配置"""
    try:
        import strategy_config
        deleted = strategy_config.delete_configs_by_code(code)
        logger.info(f'[策略配置] 删除 {code} 的 {deleted} 条配置')
        return jsonify({'success': True, 'code': code, 'deleted': deleted})
    except Exception as e:
        logger.error(f'[策略配置] 删除 {code} 失败: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_configs/<code>/toggle', methods=['POST'])
def api_strategy_configs_toggle(code):
    """启用/禁用单条配置

    Body: {'pattern': str, 'pattern_type': 'buy'/'sell', 'enabled': bool}
    """
    data = request.get_json()
    pattern = data.get('pattern', '')
    pattern_type = data.get('pattern_type', 'buy')
    enabled = bool(data.get('enabled', True))
    if not pattern:
        return jsonify({'error': '缺少 pattern 参数'}), 400
    try:
        import strategy_config
        ok = strategy_config.toggle_config_enabled(code, pattern, pattern_type, enabled)
        if not ok:
            return jsonify({'error': '未找到对应配置'}), 404
        return jsonify({'success': True, 'code': code, 'pattern': pattern,
                         'pattern_type': pattern_type, 'enabled': enabled})
    except Exception as e:
        logger.error(f'[策略配置] 切换 {code} {pattern} 失败: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_scan', methods=['POST'])
def api_strategy_scan():
    """单股全量扫描：用于策略配置 tab，复用 strategy_config.scan_stock_full

    Body: {'code': str, 'name': str, 'start_date': 'YYYYMMDD', 'end_date': 'YYYYMMDD',
           'observe_day': int, 'cautious': bool}
    Returns: {'task_id': str}（线程任务，通过 /api/strategy_scan/<task_id> 查询进度）
    """
    import uuid
    data = request.get_json()
    code = data.get('code', '')
    name = data.get('name', '') or _get_code_name(code)
    start_date = data.get('start_date', '20230101')
    end_date = data.get('end_date', '20231231')
    observe_day = int(data.get('observe_day', 2))
    cautious = bool(data.get('cautious', False))

    if not code:
        return jsonify({'error': '请选择股票'}), 400

    logger.info(f'[策略扫描] 请求: code={code}, name={name}, start={start_date}, '
                f'end={end_date}, observe_day={observe_day}, cautious={cautious}')

    task_id = f'strat_scan_{uuid.uuid4().hex[:8]}'
    scan_tasks[task_id] = {
        'task_id': task_id, 'status': 'running', 'progress': 0,
        'total': 0, 'current_pattern': '', 'results': [],
        'buy_hold_return': None, 'code': code, 'name': name,
        'started_at': time.time(), 'error': None,
    }

    def _run():
        try:
            import strategy_config
            task = scan_tasks[task_id]

            def progress_cb(done, total, c, pattern):
                task['progress'] = int(done / total * 100) if total > 0 else 0
                task['current_pattern'] = pattern
                task['total'] = total

            result = strategy_config.scan_stock_full(
                code=code, name=name, start_date=start_date, end_date=end_date,
                observe_day=observe_day, cautious=cautious, progress_cb=progress_cb,
            )
            task['results'] = result.get('results', [])
            task['buy_hold_return'] = result.get('buy_hold_return')
            task['status'] = 'done'
            task['progress'] = 100
            logger.info(f'[策略扫描] 完成 {code}: {len(task["results"])} 条结果')
        except Exception as e:
            scan_tasks[task_id]['status'] = 'error'
            scan_tasks[task_id]['error'] = str(e)
            logger.error(f'[策略扫描] 异常 {code}: {e}\n{traceback.format_exc()}')

    _cleanup_stale_tasks()
    if _count_running_tasks() >= MAX_CONCURRENT_TASKS:
        return jsonify({'error': f'当前已有 {_count_running_tasks()} 个任务运行中，'
                                 f'超过并发上限 {MAX_CONCURRENT_TASKS}，请等待'}), 429
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'task_id': task_id, 'status': 'running'})


@app.route('/api/strategy_scan/<task_id>', methods=['GET'])
def api_strategy_scan_status(task_id):
    """查询策略扫描任务进度和结果"""
    task = scan_tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'task_id': task['task_id'], 'status': task['status'],
        'progress': task.get('progress', 0), 'total': task.get('total', 0),
        'current_pattern': task.get('current_pattern', ''),
        'results': task.get('results', []),
        'buy_hold_return': task.get('buy_hold_return'),
        'code': task.get('code', ''), 'name': task.get('name', ''),
        'error': task.get('error'),
    })


# ============ 信号更新接口 ============
signal_update_task = None
signal_update_progress = {'current': 0, 'total': 0, 'message': '', 'status': 'idle',
                          'success': 0, 'failed': 0, 'eta': None, 'elapsed': 0}


@app.route('/api/signal-update/start', methods=['POST'])
def api_signal_update_start():
    """启动信号更新任务（支持断点续跑）。

    请求体新增 resume 参数：
        - resume=false（默认）：全新开始，会清空已有CSV重新生成
        - resume=true：断点续跑，跳过已完成的任务，仅执行未完成部分
    """
    global signal_update_task
    if signal_update_task and signal_update_progress['status'] == 'running':
        return jsonify({'error': '已有任务正在运行'}), 400

    data = request.get_json() or {}
    template_id = data.get('template_id')
    custom_params = data.get('params', {})

    import signal_templates as st
    if template_id:
        template = st.get_template(template_id)
        if not template:
            return jsonify({'error': '模板不存在'}), 404
        params = {**template['params'], **custom_params}
    else:
        params = custom_params

    from datetime import datetime
    if not params.get('start_date'):
        params['start_date'] = '20100104'
    if not params.get('end_date'):
        params['end_date'] = datetime.now().strftime('%Y%m%d')
    if not params.get('types'):
        params['types'] = ['index', 'hs', 'cy', 'kc', 'etf']
    if not params.get('observe_day'):
        params['observe_day'] = 2
    if 'cautious' not in params:
        params['cautious'] = False
    # 仅买入模式：非重点标的只跑 buy 信号，重点标的（持仓+关注+策略配置）跑双向
    # 默认开启，节省约一半时间；前端可关闭
    if 'buy_only_for_non_focus' not in params:
        params['buy_only_for_non_focus'] = bool(data.get('buy_only_for_non_focus', True))
    if not params.get('workers'):
        params['workers'] = data.get('workers', 8)
    # A 股个股市值范围筛选（前端输入单位为亿元，转为万元）
    min_mv_yi = data.get('min_mv')  # 亿元
    max_mv_yi = data.get('max_mv')  # 亿元
    if min_mv_yi not in (None, '', 0):
        params['min_mv'] = float(min_mv_yi) * 10000
    if max_mv_yi not in (None, '', 0):
        params['max_mv'] = float(max_mv_yi) * 10000
    # resume 参数：是否断点续跑
    if 'resume' not in params:
        params['resume'] = bool(data.get('resume', False))

    # 续跑前检查：是否存在可续跑的任务状态
    if params['resume']:
        import signal_update as su
        prev_state = su._load_task_state()
        if not prev_state or not prev_state.get('params'):
            return jsonify({'error': '没有可续跑的任务状态，请重新开始任务'}), 400
        # 续跑时直接使用上次任务的参数（保证一致性），避免默认值填充导致不匹配
        p = prev_state['params']
        params['types'] = p.get('types', params['types'])
        params['start_date'] = p.get('start_date', params['start_date'])
        params['end_date'] = p.get('end_date', params['end_date'])
        params['observe_day'] = p.get('observe_day', params['observe_day'])
        params['cautious'] = p.get('cautious', params['cautious'])
        params['buy_only_for_non_focus'] = p.get('buy_only_for_non_focus', params.get('buy_only_for_non_focus', True))
        # 返回续跑信息（已完成数量）
        prev_completed = len(prev_state.get('completed_keys', []))
        logger.info(f'[信号更新] 续跑模式：已完成 {prev_completed} 个任务')

    task_id = f'signal_update_{int(time.time())}'
    signal_update_progress.update({
        'current': 0, 'total': 100, 'message': '准备启动任务',
        'status': 'running', 'task_id': task_id, 'resume': params['resume'],
    })

    def _progress_cb(data):
        signal_update_progress.update({
            'current': data.get('current', 0),
            'total': data.get('total', 0),
            'message': data.get('message', ''),
            'success': data.get('success', 0),
            'failed': data.get('failed', 0),
            'eta': data.get('eta'),
            'elapsed': data.get('elapsed', 0),
        })

    def _run_update():
        try:
            import signal_update as su
            result = su.run_signal_update(task_id, params, _progress_cb)
            # 如果是被用户停止的（stopped=True），不生成报告，不标记为成功
            if result and result.get('stopped'):
                signal_update_progress['status'] = 'stopped'
                signal_update_progress['result'] = result
                signal_update_progress['message'] = result.get('message', '已停止，可续跑')
                logger.info(f'[信号更新] 任务已停止，断点已保存: {result.get("completed")}/{result.get("total")}')
                return
            # 正常完成：先生成数据质量报告，再标记任务成功
            try:
                import signal_quality as sq
                # 读取最新任务的时间戳，只统计本次更新的文件
                latest_path = config.SIGNAL_UPDATE_DIR / 'latest_task.json'
                since_ts = None
                if latest_path.exists():
                    with open(latest_path, 'r', encoding='utf-8') as _f:
                        _info = json.load(_f)
                        since_ts = _info.get('start_time')
                sq.generate_quality_report(since_timestamp=since_ts)
                logger.info(f'[信号更新] 数据质量报告已自动生成（本次更新）')
            except Exception as qe:
                logger.error(f'[信号更新] 自动生成数据质量报告失败: {qe}')
            signal_update_progress['status'] = 'success'
            signal_update_progress['result'] = result
        except Exception as e:
            logger.error(f'[信号更新] 任务失败: {e}\n{traceback.format_exc()}')
            signal_update_progress['status'] = 'error'
            signal_update_progress['error'] = str(e)

    t = threading.Thread(target=_run_update, daemon=True)
    signal_update_task = t  # 保存线程引用，用于状态判断
    t.start()
    return jsonify({'task_id': task_id, 'status': 'running', 'params': params})


@app.route('/api/signal-update/status', methods=['GET'])
def api_signal_update_status():
    """查询信号更新任务状态"""
    return jsonify(signal_update_progress)


@app.route('/api/signal-update/stop', methods=['POST'])
def api_signal_update_stop():
    """停止信号更新任务（协作式停止，保存断点）。

    调用 signal_update.request_stop() 设置停止事件，
    主循环在下一个批次检查点检测到后优雅退出，并保存当前进度。
    前端可随后用 resume=true 续跑。
    """
    import signal_update as su
    su.request_stop()
    signal_update_progress['status'] = 'stopping'
    signal_update_progress['message'] = '正在停止（保存断点中...）'
    logger.info(f'[信号更新] 收到停止请求，等待主循环优雅退出')
    return jsonify({'status': 'stopping', 'message': '正在停止，断点将保存，可稍后续跑'})


@app.route('/api/signal-update/resume-info', methods=['GET'])
def api_signal_update_resume_info():
    """查询是否有可续跑的任务状态（供前端判断是否显示"续跑"按钮）"""
    import signal_update as su
    state = su._load_task_state()
    if not state or not state.get('params'):
        return jsonify({'available': False})
    return jsonify({
        'available': True,
        'params': state['params'],
        'completed_count': len(state.get('completed_keys', [])),
        'total_tasks': state.get('total_tasks', 0),
        'task_id': state.get('task_id', ''),
        'start_time': state.get('start_time', 0),
        'stopped_at': state.get('stopped_at', 0),
    })


@app.route('/api/signal-update/history', methods=['GET'])
def api_signal_update_history():
    """获取信号更新历史"""
    import signal_update as su
    history = su.get_update_history()
    days_since, msg = su.is_due_for_update()
    return jsonify({
        'history': history,
        'days_since_last_update': su.get_days_since_last_update(),
        'due_for_update': days_since,
        'update_message': msg,
    })


# 标的策略表现 CSV 文件名缓存（避免每次搜索都遍历目录）
_perf_files_cache = {'dirs': None, 'mtime': 0, 'codes': None}


def _get_perf_file_codes():
    """获取所有有策略表现 CSV 的标的代码集合，带 5 分钟缓存。"""
    import time as _time
    cache_ttl = 300  # 5 分钟
    now = _time.time()
    if (_perf_files_cache['codes'] is not None
            and now - _perf_files_cache['mtime'] < cache_ttl):
        return _perf_files_cache['codes']

    import re
    from pathlib import Path
    pattern = re.compile(r'^(.+?)_(buy|sell)_strategy_performance_test\.csv$')
    codes = {}  # code -> {'buy': bool, 'sell': bool}
    for perf_dir in [config.STOCK_PERFORMANCE_DIR, config.INDEX_PERFORMANCE_DIR,
                     config.ETF_PERFORMANCE_DIR]:
        p = Path(str(perf_dir))
        if not p.exists():
            continue
        for f in p.iterdir():
            m = pattern.match(f.name)
            if m:
                code = m.group(1)
                side = m.group(2)
                if code not in codes:
                    codes[code] = {'buy': False, 'sell': False}
                codes[code][side] = True
    _perf_files_cache['codes'] = codes
    _perf_files_cache['mtime'] = now
    return codes


@app.route('/api/signal-update/search-securities', methods=['GET'])
def api_signal_update_search_securities():
    """搜索有策略表现数据的标的（按代码或名称模糊匹配）。"""
    q = request.args.get('q', '').strip().lower()
    limit = min(int(request.args.get('limit', 30)), 100)
    if not q:
        return jsonify({'items': []})

    perf_codes = _get_perf_file_codes()
    if not perf_codes:
        return jsonify({'items': []})

    # 加载名册
    candidates = []  # [(code, name, type)]
    # 指数
    for idx in get_index_list():
        candidates.append((idx['code'], idx['name'], 'index'))
    # A 股
    for s in get_stock_list():
        candidates.append((s['code'], s['name'], s.get('type', 'hs')))
    # ETF
    try:
        from pathlib import Path
        etf_file = config.SECURITY_LIST_DIR / 'etf_list.csv'
        if etf_file.exists():
            import pandas as _pd
            df = _pd.read_csv(etf_file)
            for _, row in df.iterrows():
                code = str(row.get('code', '')).strip()
                name = str(row.get('name', '')).strip()
                if code and len(code) >= 6:
                    if '.' not in code:
                        market = str(row.get('market', 'SH')).upper()
                        code = f'{code}.{market}'
                    candidates.append((code, name, 'etf'))
    except Exception:
        pass

    # 过滤：有策略表现 CSV + 模糊匹配
    results = []
    for code, name, sec_type in candidates:
        if code not in perf_codes:
            continue
        if q in code.lower() or q in str(name).lower():
            info = perf_codes[code]
            results.append({
                'code': code, 'name': name, 'type': sec_type,
                'has_buy': info['buy'], 'has_sell': info['sell'],
            })
        if len(results) >= limit:
            break

    return jsonify({'items': results})


def _get_perf_dir_for_code(code):
    """根据 code 返回策略表现目录（轻量实现，避免 import signal_update 触发 backtrader/talib 加载）。

    与 signal_update._get_data_dir 的 perf_dir 逻辑保持一致。
    """
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        return str(config.INDEX_PERFORMANCE_DIR)
    if (code.startswith('51') or code.startswith('513') or code.startswith('515')
            or code.startswith('15') or code.startswith('16') or code.startswith('512')
            or code.startswith('510') or code.startswith('516') or code.startswith('517')
            or code.startswith('518') or code.startswith('519')):
        return str(config.ETF_PERFORMANCE_DIR)
    return str(config.STOCK_PERFORMANCE_DIR)


def _safe_float(v, default=0.0):
    """安全转 float：处理 NaN/None，避免 int(NaN) 崩溃。"""
    try:
        if v is None:
            return default
        f = float(v)
        if not (f == f):  # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0):
    """安全转 int：处理 NaN/None。"""
    try:
        if v is None:
            return default
        f = float(v)
        if not (f == f):  # NaN check
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


@app.route('/api/signal-update/security-performance', methods=['GET'])
def api_signal_update_security_performance():
    """获取单个标的的策略表现明细（买入+卖出）。"""
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'error': '缺少 code 参数'}), 400

    # 轻量获取 perf_dir，避免 import signal_update（会触发 pattern_scan → backtrader/talib 加载，
    # 在 Flask 主进程中首次 import 可能导致段错误）
    try:
        from pattern_data import PATTERN_CN_NAMES
    except Exception:
        PATTERN_CN_NAMES = {}
    import pandas as _pd
    from pathlib import Path

    perf_dir = _get_perf_dir_for_code(code)
    perf_path = Path(perf_dir)

    def _load_side(side):
        csv_path = perf_path / f'{code}_{side}_strategy_performance_test.csv'
        if not csv_path.exists():
            return []
        try:
            df = _pd.read_csv(csv_path, dtype={'策略名称': str})
            # 去重（续跑追加可能产生重复行，保留最后一行）
            df = df.drop_duplicates(subset=['策略名称'], keep='last')
            rows = []
            for _, r in df.iterrows():
                strategy_name = str(r.get('策略名称', ''))
                # 从 "buy_CDL3INSIDE" 提取形态名
                pattern_key = strategy_name.split('_', 1)[-1] if '_' in strategy_name else strategy_name
                cn_name = PATTERN_CN_NAMES.get(pattern_key, pattern_key)
                rows.append({
                    '策略名称': strategy_name,
                    '中文名称': cn_name,
                    '交易次数': _safe_int(r.get('交易次数', 0)),
                    '胜率(%)': _safe_float(r.get('胜率(%)', 0)),
                    '简易收益率(%)': _safe_float(r.get('简易收益率(%)', 0)),
                    '夏普比率': _safe_float(r.get('夏普比率', 0)),
                    '最大回撤(%)': _safe_float(r.get('最大回撤(%)', 0)),
                })
            return rows
        except Exception as e:
            logger.warning(f'[API] 读取策略表现失败 {csv_path}: {e}')
            return []

    buy_rows = _load_side('buy')
    sell_rows = _load_side('sell')

    # 查名称
    name = ''
    for idx in get_index_list():
        if idx['code'] == code:
            name = idx['name']
            break
    if not name:
        for s in get_stock_list():
            if s['code'] == code:
                name = s['name']
                break

    # 判断类型
    sec_type = 'hs'
    if code in [i['code'] for i in get_index_list()]:
        sec_type = 'index'
    elif code.startswith('51') or code.startswith('15') or code.startswith('16'):
        sec_type = 'etf'

    return jsonify({
        'code': code,
        'name': name,
        'type': sec_type,
        'rows': {'buy': buy_rows, 'sell': sell_rows},
        'total_buy': len(buy_rows),
        'total_sell': len(sell_rows),
    })


@app.route('/api/signal-update/templates', methods=['GET'])
def api_signal_update_templates():
    """获取回测参数模板列表"""
    import signal_templates as st
    templates = st.get_templates()
    return jsonify({'templates': templates})


@app.route('/api/signal-update/templates/<template_id>', methods=['GET'])
def api_signal_update_template(template_id):
    """获取单个回测参数模板"""
    import signal_templates as st
    template = st.get_template(template_id)
    if not template:
        return jsonify({'error': '模板不存在'}), 404
    return jsonify(template)


@app.route('/api/signal-update/templates', methods=['POST'])
def api_signal_update_create_template():
    """创建自定义回测参数模板"""
    data = request.get_json() or {}
    name = data.get('name')
    description = data.get('description', '')
    params = data.get('params', {})

    if not name:
        return jsonify({'error': '模板名称不能为空'}), 400

    import signal_templates as st
    template = st.create_template(name, description, params)
    return jsonify(template)


@app.route('/api/signal-update/templates/<template_id>', methods=['PUT'])
def api_signal_update_update_template(template_id):
    """更新回测参数模板"""
    data = request.get_json() or {}
    import signal_templates as st
    template, error = st.update_template(
        template_id,
        name=data.get('name'),
        description=data.get('description'),
        params=data.get('params'),
    )
    if error:
        return jsonify({'error': error}), 400
    if not template:
        return jsonify({'error': '模板不存在'}), 404
    return jsonify(template)


@app.route('/api/signal-update/templates/<template_id>', methods=['DELETE'])
def api_signal_update_delete_template(template_id):
    """删除回测参数模板"""
    import signal_templates as st
    success, error = st.delete_template(template_id)
    if error:
        return jsonify({'error': error}), 400
    if not success:
        return jsonify({'error': '模板不存在'}), 404
    return jsonify({'success': True})


@app.route('/api/signal-update/quality', methods=['GET'])
def api_signal_update_quality():
    """获取数据质量报告"""
    import signal_quality as sq
    report = sq.get_quality_report()
    return jsonify({'report': report})


@app.route('/api/signal-update/quality/generate', methods=['POST'])
def api_signal_update_quality_generate():
    """生成数据质量报告（只统计本次更新）"""
    import signal_quality as sq
    # 读取最近一次信号更新的时间戳，只统计本次更新的文件
    latest_path = config.SIGNAL_UPDATE_DIR / 'latest_task.json'
    since_ts = None
    if latest_path.exists():
        with open(latest_path, 'r', encoding='utf-8') as _f:
            _info = json.load(_f)
            since_ts = _info.get('start_time')
    report = sq.generate_quality_report(since_timestamp=since_ts)
    return jsonify({'report': report})


@app.route('/api/signal-update/quality/download', methods=['GET'])
def api_signal_update_quality_download():
    """下载数据质量报告（Excel）"""
    xlsx_path = config.SIGNAL_UPDATE_DIR / 'quality_report.xlsx'
    if not xlsx_path.exists():
        return jsonify({'error': '报告不存在，请先生成报告'}), 404
    return send_file(
        str(xlsx_path),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name=f'quality_report_{datetime.now().strftime("%Y%m%d")}.xlsx',
        as_attachment=True
    )


def create_app():
    return app


# ============================================================================
# 项目文档 Wiki 路由
# ============================================================================
_DOCS_DIR = config.BASE_DIR / 'docs'
_DOCS_LIST = [
    {'name': 'WIKI', 'title': '综合 Wiki', 'icon': '📚'},
    {'name': 'architecture', 'title': '系统架构', 'icon': '🏗️'},
    {'name': 'data-source', 'title': '数据源与盈湖', 'icon': '💾'},
    {'name': 'strategy-signals', 'title': '策略与信号', 'icon': '📊'},
    {'name': 'ai-model', 'title': 'AI 模型', 'icon': '🤖'},
    {'name': 'deployment', 'title': '部署运维', 'icon': '🚀'},
]


@app.route('/api/docs')
def api_docs_list():
    """返回文档列表"""
    return jsonify({'docs': _DOCS_LIST})


@app.route('/api/docs/<doc_name>')
def api_docs_get(doc_name):
    """返回单个文档的 markdown 原文"""
    # 安全检查：只允许字母、数字、连字符
    if not doc_name or not all(c.isalnum() or c == '-' for c in doc_name):
        return jsonify({'error': '非法文档名'}), 400
    doc_path = _DOCS_DIR / f'{doc_name}.md'
    if not doc_path.exists():
        return jsonify({'error': f'文档不存在: {doc_name}'}), 404
    try:
        content = doc_path.read_text('utf-8')
        return jsonify({'content': content, 'name': doc_name})
    except Exception as e:
        return jsonify({'error': f'读取失败: {e}'}), 500


@app.route('/docs/assets/<path:filename>')
def docs_assets(filename):
    """文档图片资源服务"""
    return send_from_directory(_DOCS_DIR / 'assets', filename)


if __name__ == '__main__':
    # macOS 默认 spawn 多进程，子进程会重新 import 主模块。
    # 若通过 python web_app.py 直接启动，子进程执行到此会尝试启动 Flask 服务器，
    # 导致端口冲突、子进程崩溃，进而使信号更新等 ProcessPoolExecutor 任务全部失败。
    # 因此子进程到达此处时跳过 app.run()，让主模块顶层代码正常结束，随后即可执行父进程分发的任务。
    from multiprocessing import current_process
    if current_process().name == 'MainProcess':
        print(f'启动量化交易系统前端面板...')
        print(f'访问地址: http://{config.FLASK_HOST}:{config.FLASK_PORT}')
        print(f'系统日志目录: {SYSTEM_LOG_DIR}')
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG,
        )
    # 子进程：不启动服务器，也不退出进程
