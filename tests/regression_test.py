"""回归测试：建立基准 + 全流程校验。

用法：
    python3 regression_test.py              # 运行回归并比对基准
    python3 regression_test.py --save-baseline  # 重新生成基准
    python3 regression_test.py --smoke      # 只跑冒烟测试，不比对基准

基准文件：baseline/baseline.json
"""
import sys
import json
import time
import shutil
import tempfile
import argparse
import traceback
import urllib.request
from pathlib import Path
from datetime import datetime

# backend 各功能子目录加入 sys.path
_BACKEND_DIR = Path(__file__).parent.parent / 'backend'
for sub in ['', 'web', 'data', 'strategy', 'pattern', 'signal', 'tracking', 'utils']:
    sys.path.insert(0, str(_BACKEND_DIR / sub))

from config import config
from pattern_scan import run_single_pattern
from signal_update import run_signal_update
from tracking import tracking
from signal_utils import BUY_PATTERNS

# 基准用例：覆盖 buy/sell、不同胜率/收益率
BASELINE_CASES = [
    {'code': '000001.SZ', 'pattern': 'CDLCLOSINGMARUBOZU', 'pattern_type': 'buy'},
    {'code': '000002.SZ', 'pattern': 'CDLDOJI', 'pattern_type': 'buy'},
    {'code': '000004.SZ', 'pattern': 'CDLBELTHOLD', 'pattern_type': 'buy'},
    {'code': '000021.SZ', 'pattern': 'CDLEVENINGSTAR', 'pattern_type': 'sell'},
    {'code': '000021.SZ', 'pattern': 'CDLSHOOTINGSTAR', 'pattern_type': 'sell'},
]

START = '20260624'
END = '20260724'
DATA_DIR = config.DAILY_TRACKING_A_DIR
WEB_BASE = f'http://{config.FLASK_HOST}:{config.FLASK_PORT}'

BASELINE_DIR = Path(__file__).parent / 'baseline'
BASELINE_FILE = BASELINE_DIR / 'baseline.json'

# 指标比对容差
METRIC_TOLERANCE = {
    'return_pct': 0.01,
    'annualized_return': 0.01,
    'sharpe': 0.001,
    'hold_max_drawdown': 0.001,
}


def _web_request(path, data=None, method='GET', binary=False):
    url = WEB_BASE + path
    headers = {'Content-Type': 'application/json'}
    body = json.dumps(data).encode('utf-8') if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        return (resp.status, raw) if binary else (resp.status, raw.decode('utf-8'))


def _extract_signals(trade_details):
    """从交易详情中提取信号日期（买入/卖出发生日）。"""
    dates = []
    for t in trade_details:
        for key in ('buy_date', 'open_date', 'entry_date', 'sell_date', 'close_date', 'exit_date'):
            if t.get(key):
                dates.append(str(t[key]))
    return sorted(set(dates))


def _run_single_case(case):
    """运行单个基准用例并返回标准化结果。"""
    res = run_single_pattern(
        code=case['code'],
        pattern_name=case['pattern'],
        pattern_type=case['pattern_type'],
        start_date=START,
        end_date=END,
        data_folder_dir=str(DATA_DIR),
        observe_day=2,
        cash=100000000,
        cautious=False,
    )
    return {
        'code': case['code'],
        'pattern': case['pattern'],
        'pattern_type': case['pattern_type'],
        'observe_day': 2,
        'start_date': START,
        'end_date': END,
        'signals': _extract_signals(res.get('trade_details', [])),
        'metrics': {
            'trades': res.get('trades'),
            'win_rate': res.get('win_rate'),
            'return_pct': res.get('return_pct'),
            'annualized_return': res.get('annualized_return'),
            'capital_occupation': res.get('capital_occupation'),
            'sharpe': res.get('sharpe'),
            'hold_max_drawdown': res.get('hold_max_drawdown'),
        },
        'trade_details': res.get('trade_details', []),
    }


def save_baseline():
    """保存当前计算结果作为基准。"""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = {
        'created_at': datetime.now().isoformat(),
        'python_version': sys.version,
        'cases': [_run_single_case(c) for c in BASELINE_CASES],
    }
    with open(BASELINE_FILE, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f'[基准] 已保存 {len(baseline["cases"])} 个用例到 {BASELINE_FILE}')
    return baseline


def _compare_metric(name, expected, actual, tolerance):
    """对比单个指标，返回差异描述或 None。"""
    if expected is None or actual is None:
        if expected != actual:
            return f'{name}: 基准={expected}, 实际={actual}'
        return None
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) > tolerance:
            return f'{name}: 基准={expected}, 实际={actual}, 差异>{tolerance}'
        return None
    if expected != actual:
        return f'{name}: 基准={expected}, 实际={actual}'
    return None


def check_baseline():
    """重新计算基准用例并与 baseline.json 比对。"""
    if not BASELINE_FILE.exists():
        print(f'[基准] 基准文件不存在，先执行 --save-baseline')
        return False, []

    with open(BASELINE_FILE, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    failures = []
    for i, case in enumerate(BASELINE_CASES):
        expected = baseline['cases'][i]
        actual = _run_single_case(case)
        case_id = f"{case['code']} {case['pattern']} ({case['pattern_type']})"

        # 1. 信号日期必须完全一致
        if set(expected.get('signals', [])) != set(actual.get('signals', [])):
            failures.append(f'{case_id}: 信号日期不一致')

        # 2. 交易次数必须一致
        exp_metrics = expected.get('metrics', {})
        act_metrics = actual.get('metrics', {})
        if exp_metrics.get('trades') != act_metrics.get('trades'):
            failures.append(f'{case_id}: 交易次数不一致 基准={exp_metrics.get("trades")} 实际={act_metrics.get("trades")}')

        # 3. 浮点指标在容差内一致
        for metric, tol in METRIC_TOLERANCE.items():
            diff = _compare_metric(
                metric,
                exp_metrics.get(metric),
                act_metrics.get(metric),
                tol,
            )
            if diff:
                failures.append(f'{case_id}: {diff}')

    if failures:
        print('[基准比对] 失败：')
        for f in failures:
            print(f'  - {f}')
        return False, failures
    print(f'[基准比对] {len(BASELINE_CASES)} 个用例全部通过')
    return True, []


# ==================== 冒烟测试 ====================

def test_vectorized_backtest():
    print(f'[回归] 单形态向量化回测: 000001.SZ CDL3INSIDE')
    t0 = time.time()
    res = run_single_pattern(
        code='000001.SZ', pattern_name='CDL3INSIDE', pattern_type='buy',
        start_date=START, end_date=END, data_folder_dir=str(DATA_DIR),
        observe_day=2, cash=100000000, cautious=False,
    )
    print(f'  首次耗时: {time.time() - t0:.3f}s, 交易次数: {res.get("trades")}')
    assert 'win_rate' in res

    t0 = time.time()
    run_single_pattern(
        code='000001.SZ', pattern_name='CDL3INSIDE', pattern_type='buy',
        start_date=START, end_date=END, data_folder_dir=str(DATA_DIR),
        observe_day=2, cash=100000000, cautious=False,
    )
    print(f'  缓存后耗时: {time.time() - t0:.3f}s')


def test_signal_update():
    print('[回归] 信号更新（仅指数，使用临时目录隔离）')
    temp_root = Path(tempfile.mkdtemp(prefix='regression_signal_update_'))
    try:
        task_id = f'regression_{int(time.time())}'
        output_dirs = {
            'index_perf': str(temp_root / 'index_perf'),
            'index_data': str(config.DAILY_TRACKING_INDEX_DIR),
            'signal_update_state': str(temp_root / 'state'),
            'market_wide_stats': str(temp_root / 'market_wide_pattern_stats.csv'),
        }
        result = run_signal_update(
            task_id,
            {'types': ['index'], 'start_date': START, 'end_date': END,
             'observe_day': 2, 'cautious': False, 'workers': 2, 'resume': False},
            output_dirs=output_dirs,
        )
        assert result.get('success'), f'信号更新失败: {result}'
        print(f'  完成: success={result.get("success")}, temp_root={temp_root}')
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_tracking():
    print('[回归] 单标的跟踪（仅本地数据）')
    tracking(
        code_list=['000001.SZ'], get_new_data=False, save_data=False,
        start_date=START, track_date='2026-07-24', observe_day=2,
        folder=str(DATA_DIR), to_log=False, cautious=False,
    )


def test_web_api():
    print('[回归] Web API')
    try:
        status, _ = _web_request('/')
        assert status == 200
        print('  [/] 首页 OK')
    except urllib.error.URLError as e:
        print(f'  服务未启动，跳过 Web API 测试: {e}')
        return

    status, _ = _web_request('/api/config')
    assert status == 200
    print('  [/api/config] OK')

    status, body = _web_request('/api/backtest', {
        'code': '000001.SZ', 'pattern_name': 'CDLDOJI', 'pattern_type': 'buy',
        'start_date': START, 'end_date': END, 'observe_day': 2, 'cautious': False,
    }, method='POST')
    assert status == 200
    task_id = json.loads(body)['task_id']
    print(f'  [/api/backtest] OK task_id={task_id}')
    _wait_task(task_id)

    status, body = _web_request('/api/backtest/multi', {
        'code': '000001.SZ',
        'patterns': [
            {'pattern_name': 'CDLDOJI', 'pattern_type': 'buy', 'observe_day': 2},
            {'pattern_name': 'CDLHAMMER', 'pattern_type': 'buy', 'observe_day': 2},
        ],
        'start_date': START, 'end_date': END,
        'cash': 100000000, 'cautious': False,
    }, method='POST')
    assert status == 200
    multi_id = json.loads(body)['task_id']
    print(f'  [/api/backtest/multi] OK task_id={multi_id}')
    _wait_task(multi_id)

    status, _ = _web_request('/api/logs')
    assert status == 200
    print('  [/api/logs] OK')

    status, body = _web_request('/api/tracking', {
        'track_date': '2026-07-24',
        'mode': 'stock',
        'track_mode': 'directional',
        'target_codes': ['000001.SZ'],
        'cautious': False,
    }, method='POST')
    assert status == 200
    print(f'  [/api/tracking] OK task_id={json.loads(body)["task_id"]}')
    # 注：signal-update/quality 相关 Web API 已在 test_signal_update 中用临时目录直接测试，
    # 此处不再通过 Web API 触发，避免污染生产数据目录。


def _wait_task(task_id, max_iter=60):
    output = ''
    for _ in range(max_iter):
        time.sleep(0.5)
        status, body = _web_request(f'/api/task/{task_id}/status')
        if status == 200:
            info = json.loads(body)
            output = info.get('output', '')
            if info.get('success') or info.get('error'):
                print(f'  任务完成: success={info.get("success")}, error={info.get("error")}')
                if not info.get('success') and output:
                    print(f'  任务输出(末尾):\n{output[-2000:]}')
                break
    else:
        raise TimeoutError(f'等待任务 {task_id} 完成超时')


def run_smoke_tests():
    tests = [
        ('向量化回测', test_vectorized_backtest),
        ('信号更新', test_signal_update),
        ('跟踪', test_tracking),
        ('Web API', test_web_api),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f'[回归失败] {name}: {e}')
            traceback.print_exc()
            failed.append(name)
    return failed


def main():
    parser = argparse.ArgumentParser(description='量化交易系统回归测试')
    parser.add_argument('--save-baseline', action='store_true', help='保存当前结果作为基准')
    parser.add_argument('--smoke', action='store_true', help='只跑冒烟测试，不比对基准')
    args = parser.parse_args()

    if args.save_baseline:
        save_baseline()
        return 0

    failed = []

    if not args.smoke:
        ok, baseline_failures = check_baseline()
        if not ok:
            failed.extend([f'基准比对: {f}' for f in baseline_failures])

    smoke_failed = run_smoke_tests()
    failed.extend([f'冒烟测试: {n}' for n in smoke_failed])

    if failed:
        print(f'\n[回归测试] 失败项 ({len(failed)}):')
        for f in failed:
            print(f'  - {f}')
        return 1

    print('\n[回归测试] 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
