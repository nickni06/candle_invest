"""个股策略配置模块

功能：
1. 存储层：CSV 读写个股策略配置（增删改查）
2. 扫描层：对单股跑全量形态回测，输出可筛选结果供用户勾选
3. 查询层：跟踪时查询某只股票已配置的策略

配置文件：数据/策略配置/stock_strategy_config.csv
字段：code, name, pattern, pattern_cn, pattern_type, observe_day,
      win_rate, return_pct, sharpe, config_date, enabled
"""
import csv
import os
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import config
from signal_utils import BUY_PATTERNS, SELL_PATTERNS, PATTERN_CN_NAMES, PATTERN_DESCRIPTIONS


# 文件锁，防止并发写入冲突
_config_lock = threading.Lock()

# CSV 字段顺序
CONFIG_COLUMNS = [
    'code', 'name', 'pattern', 'pattern_cn', 'pattern_type',
    'observe_day', 'win_rate', 'return_pct', 'sharpe',
    'config_date', 'enabled',
]


def _ensure_dir():
    """确保配置目录存在"""
    config.STRATEGY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _is_index_code(code):
    """判断是否指数代码"""
    if len(code) < 9 or code in ['000300.SH', '399006.SZ']:
        return True
    return False


def load_all_configs():
    """读取全部策略配置

    Returns:
        list[dict]: 每条配置一个 dict，字段同 CONFIG_COLUMNS
    """
    if not config.STRATEGY_CONFIG_FILE.exists():
        return []
    try:
        df = pd.read_csv(config.STRATEGY_CONFIG_FILE, dtype=str)
        # 兼容空文件
        if df.empty:
            return []
        # 数值字段转类型
        result = []
        for _, row in df.iterrows():
            item = row.to_dict()
            item['observe_day'] = int(float(item.get('observe_day') or 2))
            item['win_rate'] = float(item.get('win_rate') or 0)
            item['return_pct'] = float(item.get('return_pct') or 0)
            item['sharpe'] = float(item.get('sharpe') or 0)
            item['enabled'] = str(item.get('enabled', 'true')).lower() == 'true'
            result.append(item)
        return result
    except Exception as e:
        print(f'[策略配置] 读取配置失败: {e}', flush=True)
        return []


def load_configs_by_code(code):
    """查询某只股票已配置的启用策略

    Args:
        code: 股票代码

    Returns:
        list[dict]: 该股票已启用的策略配置列表
    """
    all_configs = load_all_configs()
    return [c for c in all_configs if c['code'] == code and c.get('enabled', True)]


def load_configured_codes():
    """获取所有已配置策略的股票代码集合

    Returns:
        set[str]: 已配置（且至少有一个 enabled=true）的股票代码集合
    """
    all_configs = load_all_configs()
    return {c['code'] for c in all_configs if c.get('enabled', True)}


def save_configs(code, name, patterns):
    """保存个股策略配置（覆盖式）

    Args:
        code: 股票代码
        name: 股票名称
        patterns: list[dict]，每条包含：
            - pattern: 形态英文名
            - pattern_type: buy/sell
            - observe_day: 持有天数
            - win_rate: 配置时胜率
            - return_pct: 配置时收益
            - sharpe: 夏普比率
            - enabled: 是否启用

    说明：
        - 会先删除该 code 的所有旧配置，再写入新的
        - 其他 code 的配置不受影响
    """
    _ensure_dir()
    with _config_lock:
        # 读取现有配置（排除当前 code）
        existing = []
        if config.STRATEGY_CONFIG_FILE.exists():
            try:
                df = pd.read_csv(config.STRATEGY_CONFIG_FILE, dtype=str)
                if not df.empty:
                    existing = df[df['code'] != code].to_dict('records')
            except Exception:
                pass

        # 构造新配置
        today = datetime.now().strftime('%Y-%m-%d')
        new_rows = []
        for p in patterns:
            pattern_en = p.get('pattern', '')
            new_rows.append({
                'code': code,
                'name': name,
                'pattern': pattern_en,
                'pattern_cn': PATTERN_CN_NAMES.get(pattern_en, pattern_en),
                'pattern_type': p.get('pattern_type', 'buy'),
                'observe_day': int(p.get('observe_day', 2)),
                'win_rate': float(p.get('win_rate', 0) or 0),
                'return_pct': float(p.get('return_pct', 0) or 0),
                'sharpe': float(p.get('sharpe', 0) or 0),
                'config_date': today,
                'enabled': 'true' if p.get('enabled', True) else 'false',
            })

        # 合并并写入
        all_rows = existing + new_rows
        df_out = pd.DataFrame(all_rows, columns=CONFIG_COLUMNS)
        df_out.to_csv(config.STRATEGY_CONFIG_FILE, index=False, encoding='utf-8-sig')
        print(f'[策略配置] 保存 {code} {name} 的 {len(new_rows)} 条策略配置', flush=True)


def delete_configs_by_code(code):
    """删除某只股票的所有策略配置

    Args:
        code: 股票代码

    Returns:
        int: 删除的条数
    """
    if not config.STRATEGY_CONFIG_FILE.exists():
        return 0
    with _config_lock:
        try:
            df = pd.read_csv(config.STRATEGY_CONFIG_FILE, dtype=str)
            if df.empty:
                return 0
            before = len(df)
            df = df[df['code'] != code]
            after = len(df)
            df.to_csv(config.STRATEGY_CONFIG_FILE, index=False, encoding='utf-8-sig')
            deleted = before - after
            if deleted > 0:
                print(f'[策略配置] 删除 {code} 的 {deleted} 条配置', flush=True)
            return deleted
        except Exception as e:
            print(f'[策略配置] 删除配置失败: {e}', flush=True)
            return 0


def toggle_config_enabled(code, pattern, pattern_type, enabled):
    """单条配置启用/禁用切换

    Args:
        code: 股票代码
        pattern: 形态英文名
        pattern_type: buy/sell
        enabled: True/False

    Returns:
        bool: 是否成功
    """
    if not config.STRATEGY_CONFIG_FILE.exists():
        return False
    with _config_lock:
        try:
            df = pd.read_csv(config.STRATEGY_CONFIG_FILE, dtype=str)
            if df.empty:
                return False
            mask = (df['code'] == code) & (df['pattern'] == pattern) & (df['pattern_type'] == pattern_type)
            if not mask.any():
                return False
            df.loc[mask, 'enabled'] = 'true' if enabled else 'false'
            df.to_csv(config.STRATEGY_CONFIG_FILE, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            print(f'[策略配置] 切换启用状态失败: {e}', flush=True)
            return False


def scan_stock_full(code, name, start_date, end_date,
                    observe_day=2, cash=None, cautious=False, progress_cb=None):
    """对单只股票跑全量形态回测

    复用 pattern_scan.scan_stock，但返回结构更适合前端展示和勾选。

    Args:
        code: 股票代码
        name: 股票名称
        start_date: 回测开始日期 YYYYMMDD
        end_date: 回测结束日期 YYYYMMDD
        observe_day: 默认持有天数
        cash: 初始资金
        cautious: 谨慎模式
        progress_cb: 进度回调 fn(done, total, code, pattern)

    Returns:
        dict: {
            'code': str,
            'name': str,
            'results': list[dict],  # 每个形态的回测结果
            'buy_hold_return': float,  # 买入持有基准收益
        }
    """
    from pattern_scan import scan_stock, calc_buy_hold_return

    if cash is None:
        cash = config.DEFAULT_CASH

    # 判断数据目录
    if _is_index_code(code):
        data_folder_dir = str(config.TRAIN_DATA_INDEX_DIR) + '/'
    else:
        data_folder_dir = str(config.TRAIN_DATA_A_DIR) + '/'

    # 跑全量扫描
    results = scan_stock(
        code=code,
        start_date=start_date,
        end_date=end_date,
        observe_day=observe_day,
        cash=cash,
        data_folder_dir=data_folder_dir,
        scan_buy=True,
        scan_sell=True,
        progress_cb=progress_cb,
        cautious=cautious,
    )

    # 买入持有基准
    buy_hold = calc_buy_hold_return(code, start_date, end_date, data_folder_dir)

    # 标注中文名和说明
    for r in results:
        if 'pattern_cn' not in r or not r['pattern_cn']:
            r['pattern_cn'] = PATTERN_CN_NAMES.get(r.get('pattern', ''), r.get('pattern', ''))
        if 'pattern_desc' not in r:
            r['pattern_desc'] = PATTERN_DESCRIPTIONS.get(r.get('pattern', ''), '')
        r['name'] = name

    return {
        'code': code,
        'name': name,
        'results': results,
        'buy_hold_return': buy_hold,
    }


def get_tracking_patterns(code):
    """跟踪时查询某股票应该跑哪些形态

    Returns:
        tuple: (pattern_names, pattern_types, observe_days)
            - pattern_names: list[str] 形态英文名列表
            - pattern_types: list[str] 'buy'/'sell'
            - observe_days: list[int] 持有天数
        若该股票无配置，返回 (None, None, None)，表示跑全量
    """
    configs = load_configs_by_code(code)
    if not configs:
        return None, None, None
    pattern_names = [c['pattern'] for c in configs]
    pattern_types = [c['pattern_type'] for c in configs]
    observe_days = [c.get('observe_day', 2) for c in configs]
    return pattern_names, pattern_types, observe_days


if __name__ == '__main__':
    # CLI 测试
    print('=== 策略配置模块测试 ===')
    configs = load_all_configs()
    print(f'当前配置数: {len(configs)}')
    for c in configs[:5]:
        print(f"  {c['code']} {c['name']} - {c['pattern_cn']} ({c['pattern_type']}) 胜率{c['win_rate']}%")
