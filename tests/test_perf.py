import time
import pandas as pd
from config import config
from signal_tracker import track_code_list
import data_source


def main():
    start_date = '20260709'
    end_date = '20260724'

    df = pd.read_csv(str(config.STOCK_DATA_FILE))
    all_codes = df[df['total_mv'] >= 200 * 10000]['ts_code'].tolist()
    codes = []
    for code in all_codes:
        if data_source._local_data_has_date_range(code, start_date, end_date):
            codes.append(code)

    print(f'Testing with {len(codes)} stocks (all with local data)')

    t0 = time.time()
    result = track_code_list(
        code_list=codes,
        start_date=start_date,
        track_date='2026-07-24',
        cautious=False,
        data_folder_dir=str(config.DAILY_TRACKING_A_DIR),
        perf_dir=str(config.STOCK_PERFORMANCE_DIR),
        label='perfAll',
        track_mode='full'
    )
    t1 = time.time()
    print(f'Time: {t1 - t0:.2f}s')
    print(f'Total: {result["total"]}, Success: {result["success"]}, Failed: {result["failed"]}, Signals: {len(result["signals"])}')


if __name__ == '__main__':
    main()
