"""Numba 加速回测对拍测试脚本。

验证 numba_backtest.backtest_with_numba 与 pattern_scan._backtest_pattern_vectorized
输出（trade_details / equity_curve / holding_days_total）完全一致。

测试覆盖：
1. 多个真实标的（A股 + 指数 + ETF）
2. 买入信号 + 卖出信号
3. 谨慎模式开启 + 关闭
4. 多个 observe_day 值
5. 多种时间段长度（短/中/长）

运行：
    python3 test_numba_backtest.py
"""
import sys
import time
import traceback
from pathlib import Path

# backend 各功能子目录加入 sys.path
_BACKEND_DIR = Path(__file__).parent.parent / 'backend'
for sub in ['', 'web', 'data', 'strategy', 'pattern', 'signal', 'tracking', 'utils']:
    sys.path.insert(0, str(_BACKEND_DIR / sub))

import numpy as np
import pandas as pd
from config import config
import pattern_scan
import numba_backtest


# 测试用例：(code, name, start_date, end_date, observe_day, cautious)
TEST_CASES = [
    # A股 - 短期
    ('000001.SZ', '平安银行', '20230101', '20240101', 2, False),
    ('000001.SZ', '平安银行', '20230101', '20240101', 2, True),
    ('000001.SZ', '平安银行', '20230101', '20240101', 5, False),
    # A股 - 中期
    ('600519.SH', '贵州茅台', '20200101', '20240101', 2, False),
    ('600519.SH', '贵州茅台', '20200101', '20240101', 2, True),
    # A股 - 长期
    ('000858.SZ', '五粮液', '20100101', '20240101', 2, False),
    # 创业板
    ('300750.SZ', '宁德时代', '20200101', '20240101', 3, False),
    ('300750.SZ', '宁德时代', '20200101', '20240101', 3, True),
    # 科创板
    ('688981.SH', '中芯国际', '20210101', '20240101', 2, False),
    # 指数
    ('000300.SH', '沪深300', '20150101', '20240101', 2, False),
    ('399006.SZ', '创业板指', '20150101', '20240101', 5, False),
    # ETF
    ('510300.SH', '沪深300ETF', '20150101', '20240101', 2, False),
    # 极短时间段（仅几根 bar）
    ('000001.SZ', '平安银行-极短', '20230101', '20230201', 2, False),
    # 包含停牌/NaN 的边界情况
    ('000002.SZ', '万科A', '20200101', '20240101', 2, False),
]


# 测试的形态子集（覆盖谨慎模式受影响的6个 + 常规几个）
TEST_PATTERNS = [
    'CDLDOJI',               # 十字星 - 普通买入
    'CDLENGULFING',          # 吞没形态 - 普通买入
    'CDLHAMMER',             # 锤子线 - 普通买入
    'CDLSEPARATINGLINES',    # 分手线 - 谨慎模式受影响
    'CDLTASUKIGAP',          # 跳空缺口 - 谨慎模式受影响
    'CDLINVERTEDHAMMER',     # 倒锤子 - 谨慎模式受影响
    'CDLDRAGONFLYDOJI',      # 蜻蜓十字 - 谨慎模式受影响
    'CDLTAKURI',             # 探水竿 - 谨慎模式受影响
    'CDLMARUBOZU',           # 光头光脚 - 谨慎模式受影响
    'CDLSHOOTINGSTAR',       # 射击之星 - 卖出
    'CDLDARKCLOUDCOVER',     # 乌云盖顶 - 卖出
]


def _load_df(code, start_date, end_date):
    """加载已过滤的 DataFrame，复用 pattern_scan 的预处理逻辑。"""
    df = pattern_scan._load_raw_dataframe(code, None)
    filtered_df = df.loc[start_date:end_date].copy()
    for col in filtered_df.columns:
        if filtered_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            filtered_df[col] = filtered_df[col].ffill().fillna(0)
    return filtered_df


def run_one_comparison(code, pattern_name, pattern_type, start_date, end_date,
                      observe_day, cautious):
    """跑一次对比，返回 (passed, msg, orig_time_ms, numba_time_ms)。"""
    try:
        df = _load_df(code, start_date, end_date)
    except Exception as e:
        return False, f'加载数据失败 {code}: {e}', 0, 0
    if df.empty:
        return True, f'空数据跳过 {code}', 0, 0

    # 计算形态信号
    try:
        signal = pattern_scan._compute_pattern_signal(df, pattern_name)
    except Exception as e:
        return False, f'计算形态信号失败 {code} {pattern_name}: {e}', 0, 0

    # 原版（pandas/numpy）
    try:
        t0 = time.time()
        orig_td, orig_eq, orig_hdays = pattern_scan._backtest_pattern_vectorized(
            df, signal, pattern_name, pattern_type,
            observe_day, 100000000, cautious,
        )
        orig_ms = (time.time() - t0) * 1000
    except Exception as e:
        return False, f'原版回测崩溃 {code} {pattern_name}: {e}', 0, 0

    # Numba 版
    try:
        t0 = time.time()
        numba_td, numba_eq, numba_hdays = numba_backtest.backtest_with_numba(
            df, signal, pattern_name, pattern_type,
            observe_day, 100000000, cautious,
        )
        numba_ms = (time.time() - t0) * 1000
    except Exception as e:
        return False, f'Numba 回测崩溃 {code} {pattern_name}: {e}\n{traceback.format_exc()}', orig_ms, 0

    # 对比 trade_details
    is_eq, msg = numba_backtest._eq_compare(numba_td, orig_td, 'numba', 'origin')
    if not is_eq:
        return False, f'trade_details 不一致 {code} {pattern_type}_{pattern_name}: {msg}', orig_ms, numba_ms

    # 对比 holding_days_total（必须完全相等）
    if orig_hdays != numba_hdays:
        return False, f'holding_days_total 不一致 {code} {pattern_type}_{pattern_name}: numba={numba_hdays}, origin={orig_hdays}', orig_ms, numba_ms

    # 对比 equity_curve（容差 1e-6）
    if len(orig_eq) != len(numba_eq):
        return False, f'equity_curve 长度不一致 {code} {pattern_type}_{pattern_name}: numba={len(numba_eq)}, origin={len(orig_eq)}', orig_ms, numba_ms
    if len(orig_eq) > 0:
        max_diff = float(np.max(np.abs(numba_eq - orig_eq)))
        if max_diff > 1e-6:
            # 找出最大差异位置
            diff_idx = int(np.argmax(np.abs(numba_eq - orig_eq)))
            return False, f'equity_curve 不一致 {code} {pattern_type}_{pattern_name}: max_diff={max_diff:.2e} @ idx={diff_idx} (numba={numba_eq[diff_idx]:.4f}, origin={orig_eq[diff_idx]:.4f})', orig_ms, numba_ms

    return True, '', orig_ms, numba_ms


def main():
    if not numba_backtest._NUMBA_AVAILABLE:
        print('❌ numba 未安装，无法运行对拍测试')
        sys.exit(1)

    print(f'numba 版本: {numba_backtest.njit.__module__}')
    print(f'测试用例数: {len(TEST_CASES)} 标的 × {len(TEST_PATTERNS)} 形态 × 2 类型 = {len(TEST_CASES) * len(TEST_PATTERNS) * 2} 组')
    print()

    # 预热 Numba（首次调用会编译，约 2-3 秒）
    print('预热 Numba JIT 编译...')
    df = _load_df('000001.SZ', '20230101', '20240101')
    signal = pattern_scan._compute_pattern_signal(df, 'CDLDOJI')
    t0 = time.time()
    numba_backtest.backtest_with_numba(
        df, signal, 'CDLDOJI', 'buy', 2, 100000000, False,
    )
    print(f'  首次编译+运行耗时: {(time.time() - t0)*1000:.0f}ms\n')

    passed = 0
    failed = 0
    skipped = 0
    total_orig_ms = 0
    total_numba_ms = 0
    failures = []

    for case in TEST_CASES:
        code, name, start_date, end_date, observe_day, cautious = case
        for pattern_name in TEST_PATTERNS:
            for pattern_type in ('buy', 'sell'):
                # 谨慎模式只对 buy 信号生效
                if cautious and pattern_type == 'sell':
                    continue

                passed_ok, msg, orig_ms, numba_ms = run_one_comparison(
                    code, pattern_name, pattern_type, start_date, end_date,
                    observe_day, cautious,
                )

                total_orig_ms += orig_ms
                total_numba_ms += numba_ms

                label = f'{code} {pattern_type}_{pattern_name} obs={observe_day} cautious={cautious}'
                if passed_ok:
                    if not msg:
                        passed += 1
                        speedup = orig_ms / numba_ms if numba_ms > 0 else float('inf')
                        # 只打印显著慢或加速比 > 5 的
                        if speedup > 5 or speedup < 0.5:
                            print(f'  ✓ {label}: {orig_ms:.1f}ms → {numba_ms:.1f}ms ({speedup:.1f}x)')
                    else:
                        skipped += 1
                else:
                    failed += 1
                    failures.append((label, msg))
                    print(f'  ✗ {label}')
                    print(f'      {msg}')

    print()
    print('=' * 70)
    print(f'对拍结果: 通过 {passed} / 失败 {failed} / 跳过 {skipped} / 总计 {passed + failed + skipped}')
    if total_numba_ms > 0:
        print(f'累计耗时: 原版 {total_orig_ms:.0f}ms, Numba {total_numba_ms:.0f}ms, '
              f'整体加速 {total_orig_ms / total_numba_ms:.2f}x')
    print('=' * 70)
    if failed > 0:
        print(f'\n失败用例（{len(failures)} 个）:')
        for label, msg in failures:
            print(f'  - {label}')
            print(f'      {msg}')
        sys.exit(1)
    else:
        print('\n✅ 全部对拍通过，Numba 加速版与原版输出完全一致')


if __name__ == '__main__':
    main()
