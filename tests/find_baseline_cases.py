"""扫描本地数据，找出有真实交易信号的股票+形态组合，用于建立回归基准。"""
import sys
import json
import time
import traceback
from pathlib import Path

# backend 各功能子目录加入 sys.path
_BACKEND_DIR = Path(__file__).parent.parent / 'backend'
for sub in ['', 'web', 'data', 'strategy', 'pattern', 'signal', 'tracking', 'utils']:
    sys.path.insert(0, str(_BACKEND_DIR / sub))

from config import config
from pattern_scan import run_single_pattern
from signal_utils import BUY_PATTERNS, SELL_PATTERNS

START = '20260624'
END = '20260724'
DATA_DIR = config.DAILY_TRACKING_A_DIR
MIN_TRADES = 1


def find_cases(codes, patterns, pattern_type):
    cases = []
    for code in codes:
        for pattern in patterns:
            try:
                res = run_single_pattern(
                    code=code, pattern_name=pattern, pattern_type=pattern_type,
                    start_date=START, end_date=END, data_folder_dir=str(DATA_DIR),
                    observe_day=2, cash=100000000, cautious=False,
                )
                if res.get('trades', 0) >= MIN_TRADES:
                    cases.append({
                        'code': code,
                        'pattern': pattern,
                        'pattern_type': pattern_type,
                        'trades': res.get('trades'),
                        'return_pct': res.get('return_pct'),
                        'win_rate': res.get('win_rate'),
                    })
                    print(f'[发现] {code} {pattern} ({pattern_type}): 交易{res["trades"]}次 收益{res["return_pct"]}% 胜率{res["win_rate"]}%')
            except Exception as e:
                print(f'[跳过] {code} {pattern}: {e}')
    return cases


def main():
    # 先扫描前 20 只 A 股
    csv_files = sorted([f for f in DATA_DIR.iterdir() if f.suffix == '.csv'])[:20]
    codes = [f.name.replace('_daily.csv', '') for f in csv_files]
    print(f'扫描 {len(codes)} 只标的，日期 {START}~{END}')

    buy_cases = find_cases(codes, BUY_PATTERNS, 'buy')
    sell_cases = find_cases(codes, SELL_PATTERNS, 'sell')
    all_cases = buy_cases + sell_cases

    out_path = Path(__file__).parent / 'baseline' / 'candidate_cases.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f'\n找到 {len(all_cases)} 个有信号的用例，已保存到 {out_path}')


if __name__ == '__main__':
    main()
