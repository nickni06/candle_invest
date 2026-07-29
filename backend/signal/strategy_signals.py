"""策略信号计算模块（纯 TA-Lib 实现，脱离 Backtrader）。

从旧 tracking.py 的 patternUp_combine_Strategy_index/stock 提取业务逻辑，
保留：59 个买入形态、59 个卖出形态、谨慎模式 6 形态额外条件。
不再对信号做胜率/收益率过滤，所有匹配形态均展示给用户。

不再依赖 Backtrader 的 next() 状态机和日志解析，直接在 DataFrame 上计算信号，
返回结构化结果。性能比 Backtrader 快 10x+，且可在多进程 worker 中安全调用。

业务逻辑保留点（与旧代码一致）：
1. 所有匹配形态均输出，不再只保留最后一个
2. CDL2CROWS 笔误保留：触发 CDL2CROWS 时记为 CDLADVANCEBLOCK
3. track_date 过滤：只输出 track_date 当天的信号
4. stock 策略三分支互斥：买入/observe_day卖出/形态卖出三选一
5. index 策略不实际交易，同日可同时输出买卖信号
6. observe_day 语义：买入后持有 N 个交易日，第 N+1 日卖出
"""
import os
import json
import logging
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
import talib

from config import config

# 复用项目里已有的形态中文名 / 含义描述字典
try:
    from signal_utils import PATTERN_CN_NAMES, PATTERN_DESCRIPTIONS
except Exception:
    PATTERN_CN_NAMES = {}
    PATTERN_DESCRIPTIONS = {}

logger = logging.getLogger('trader_system')

# ============================================================================
# AI 评分（Model Server 批量调用，离线静默降级）
# ============================================================================
_AI_SERVER_URL = 'http://127.0.0.1:8766'
_AI_TIMEOUT = 5  # 单次请求超时 5 秒，超时则降级
# 缓存：仅在成功时缓存（True），失败不缓存以便下次重试
_ai_server_available = None
_ai_health_check_ts = 0  # 上次探测时间戳（秒），失败时 60s 内不重复探测


def _ai_health_check():
    """探测 Model Server 是否可达。

    缓存策略：
    - 成功 → 永久缓存 True（直到进程结束）
    - 失败 → 60s 内不重复探测，60s 后允许重试（避免每次信号都打 HTTP）
    """
    import time
    global _ai_server_available, _ai_health_check_ts
    if _ai_server_available is True:
        return True
    now = time.time()
    if _ai_server_available is False and (now - _ai_health_check_ts) < 60:
        return False
    try:
        req = urllib.request.Request(f'{_AI_SERVER_URL}/health', method='GET')
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                _ai_server_available = True
                _ai_health_check_ts = now
                logger.info(f'[AI] Model Server 健康检查通过 ({_AI_SERVER_URL})')
                return True
    except Exception as e:
        logger.warning(f'[AI] Model Server 不可达: {type(e).__name__}: {e}')
    _ai_server_available = False
    _ai_health_check_ts = now
    return False


def _ai_batch_predict(klines):
    """批量调用 AI Model Server，返回 (xgb_probs, cnn_probs)。

    Args:
        klines: list of (20, 5) OHLCV 数组

    Returns:
        (xgb_probs, cnn_probs): 各为 list[float] 或 None（服务不可用时）
    """
    if not klines:
        return None, None
    if not _ai_health_check():
        logger.warning(f'[AI] 跳过批量推理：Model Server 不可达，klines={len(klines)} 个')
        return None, None

    xgb_probs = None
    cnn_probs = None

    # XGBoost 批量
    try:
        payload = json.dumps({'klines': klines}).encode('utf-8')
        req = urllib.request.Request(
            f'{_AI_SERVER_URL}/predict/xgb/batch',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=_AI_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data.get('ai_available'):
                xgb_probs = data.get('probs')
                logger.info(f'[AI] XGBoost 推理成功: {len(xgb_probs) if xgb_probs else 0} 个样本')
            else:
                logger.warning(f'[AI] XGBoost 推理返回 ai_available=false: {data}')
    except Exception as e:
        logger.warning(f'[AI] XGBoost 批量推理异常: {type(e).__name__}: {e}')

    # CNN 批量
    try:
        payload = json.dumps({'klines': klines}).encode('utf-8')
        req = urllib.request.Request(
            f'{_AI_SERVER_URL}/predict/cnn/batch',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=_AI_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data.get('ai_available'):
                cnn_probs = data.get('probs')
                logger.info(f'[AI] CNN 推理成功: {len(cnn_probs) if cnn_probs else 0} 个样本')
            else:
                logger.warning(f'[AI] CNN 推理返回 ai_available=false: {data}')
    except Exception as e:
        logger.warning(f'[AI] CNN 批量推理异常: {type(e).__name__}: {e}')

    return xgb_probs, cnn_probs


def _collect_kline_windows(df, target_idx, window_size=20):
    """收集信号触发点的前序 20 根 K 线窗口（不含触发日）。

    Args:
        df: K 线 DataFrame，含 open/high/low/close/vol 列
        target_idx: 信号触发日的行索引
        window_size: 窗口大小（默认 20）

    Returns:
        list of (20, 5) numpy 数组，每个对应一个信号触发点
        若数据不足，返回空列表
    """
    if target_idx < window_size:
        return []
    # 只取一个窗口（信号跟踪日 target_idx 当天触发，取前 20 根）
    window = df.iloc[target_idx - window_size: target_idx]
    if len(window) < window_size:
        return []
    arr = window[['open', 'high', 'low', 'close', 'vol']].values.astype(np.float64)
    return [arr.tolist()]


def _enrich_signals_with_ai(signals, df, target_idx):
    """给信号列表批量补充 AI 评分字段。

    所有信号共享同一个 K 线窗口（同一天触发），所以只需调用一次批量推理。
    离线时静默降级，xgb_buy_prob/cnn_buy_prob 设为 None。

    Args:
        signals: 信号 dict 列表
        df: K 线 DataFrame
        target_idx: 信号触发日的 pandas 索引值（可能是日期/Timestamp）
    """
    if not signals:
        return

    logger.info(f'[AI] 开始为 {len(signals)} 个信号补充 AI 评分')

    # 把 target_idx（pandas 索引值）转成整数位置
    try:
        target_pos = df.index.get_loc(target_idx)
        if not isinstance(target_pos, int):
            # 若返回布尔数组，取第一个 True
            target_pos = int(np.where(target_pos)[0][0])
        else:
            target_pos = int(target_pos)
    except Exception as e:
        logger.warning(f'[AI] target_idx 转换失败: {e}, 用最后一天兜底')
        # 兜底：用最后一天
        target_pos = len(df) - 1

    # 收集 K 线窗口（所有信号共享同一天的前 20 根）
    klines = _collect_kline_windows(df, target_pos)
    if not klines:
        # 数据不足，所有信号标记 AI 不可用
        logger.warning(f'[AI] K线窗口数据不足 target_pos={target_pos}, 标记不可用')
        for sig in signals:
            sig['xgb_buy_prob'] = None
            sig['cnn_buy_prob'] = None
            sig['ai_available'] = False
        return

    # 批量推理（共享同一窗口）
    xgb_probs, cnn_probs = _ai_batch_predict(klines)
    xgb_prob = xgb_probs[0] if xgb_probs else None
    cnn_prob = cnn_probs[0] if cnn_probs else None
    ai_available = (xgb_prob is not None or cnn_prob is not None)
    logger.info(f'[AI] 评分完成: xgb={xgb_prob}, cnn={cnn_prob}, available={ai_available}')

    for sig in signals:
        # 买入信号：xgb_buy_prob = 看涨概率
        # 卖出信号：xgb_buy_prob 实为看跌概率（模型输出即"正样本概率"）
        # 这里统一用 buy_prob 字段名，前端按信号 type 解释
        if sig.get('type') == 'buy':
            sig['xgb_buy_prob'] = xgb_prob
            sig['cnn_buy_prob'] = cnn_prob
        else:
            # 卖出信号：展示看跌概率 = 1 - 看涨概率
            sig['xgb_buy_prob'] = (1 - xgb_prob) if xgb_prob is not None else None
            sig['cnn_buy_prob'] = (1 - cnn_prob) if cnn_prob is not None else None
        sig['ai_available'] = ai_available

# ============================================================================
# 全市场形态统计（历史基准）
# 文件：market_wide_pattern_stats.csv，由 signal_update.py 在信号更新成功后自动生成，
# 基于 A 股 + 指数的所有策略表现 CSV 按交易次数加权聚合。
# 列：形态名称,中文名称,交易次数,胜率(%),收益率(%),夏普比率,最大回撤(%),信号类型
# ============================================================================
_MARKET_PERF_PATH = str(config.MARKET_WIDE_STATS_FILE)
_market_perf_cache = {'df': None, 'mtime': 0, 'path': ''}


def _load_market_perf():
    """加载全市场形态统计 CSV，带 mtime 缓存。文件不存在返回空 DataFrame。"""
    if not os.path.exists(_MARKET_PERF_PATH):
        return pd.DataFrame()
    try:
        mtime = os.path.getmtime(_MARKET_PERF_PATH)
    except OSError:
        return pd.DataFrame()
    if _market_perf_cache['df'] is not None and _market_perf_cache['mtime'] == mtime \
            and _market_perf_cache['path'] == _MARKET_PERF_PATH:
        return _market_perf_cache['df']
    try:
        df = pd.read_csv(_MARKET_PERF_PATH)
        _market_perf_cache['df'] = df
        _market_perf_cache['mtime'] = mtime
        _market_perf_cache['path'] = _MARKET_PERF_PATH
        return df
    except Exception as e:
        logger.warning(f'[信号] 加载全市场统计失败: {e}')
        return pd.DataFrame()


def _query_market_perf(pattern_name, side):
    """查询某形态的全市场统计胜率/收益率/交易次数。

    Args:
        pattern_name: 形态名（如 CDLHAMMER）
        side: 'buy' / 'sell'

    Returns:
        {'win_rate', 'return', 'trade_count', 'sharpe', 'max_drawdown'} 或 None
    """
    df = _load_market_perf()
    if df.empty:
        return None
    side_cn = '买入' if side == 'buy' else '卖出'
    try:
        rows = df[(df['形态名称'] == pattern_name) & (df['信号类型'] == side_cn)]
        if len(rows) == 0:
            return None
        r = rows.iloc[0]
        return {
            'win_rate': float(r['胜率(%)']),
            'return': float(r['收益率(%)']),
            'sharpe': float(r['夏普比率']),
            'max_drawdown': float(r['最大回撤(%)']),
            'trade_count': float(r['交易次数']),
        }
    except Exception:
        return None

# ============================================================================
# 形态列表（与旧代码完全一致）
# ============================================================================
# 主要买入信号（39 个）
PRIMARY_BUY_PATTERNS = [
    'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLCOUNTERATTACK',
    'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLGAPSIDESIDEWHITE',
    'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
    'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLINVERTEDHAMMER', 'CDLKICKING',
    'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
    'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING',
    'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHORTLINE',
    'CDLSPINNINGTOP', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLUNIQUE3RIVER',
    'CDLXSIDEGAP3METHODS',
]
# 可能的买入信号（20 个，后判断会覆盖前者）
SECONDARY_BUY_PATTERNS = [
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
    'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBREAKAWAY', 'CDLCONCEALBABYSWALL',
    'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHANGINGMAN',
    'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLSHOOTINGSTAR', 'CDLSTALLEDPATTERN',
    'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUPSIDEGAP2CROWS',
]
ALL_BUY_PATTERNS = PRIMARY_BUY_PATTERNS + SECONDARY_BUY_PATTERNS

# 主要卖出信号（20 个）
PRIMARY_SELL_PATTERNS = [
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
    'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBREAKAWAY', 'CDLCONCEALBABYSWALL',
    'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHANGINGMAN',
    'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLSHOOTINGSTAR', 'CDLSTALLEDPATTERN',
    'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUPSIDEGAP2CROWS',
]
# 可能的卖出信号（39 个，后判断会覆盖前者）
SECONDARY_SELL_PATTERNS = [
    'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLCOUNTERATTACK',
    'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLGAPSIDESIDEWHITE',
    'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
    'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLINVERTEDHAMMER', 'CDLKICKING',
    'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
    'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING',
    'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHORTLINE',
    'CDLSPINNINGTOP', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLUNIQUE3RIVER',
    'CDLXSIDEGAP3METHODS',
]
ALL_SELL_PATTERNS = PRIMARY_SELL_PATTERNS + SECONDARY_SELL_PATTERNS

# 谨慎模式额外条件的 6 个形态
CAUTIOUS_PATTERNS = {
    'CDLMARUBOZU', 'CDLDRAGONFLYDOJI', 'CDLINVERTEDHAMMER',
    'CDLSEPARATINGLINES', 'CDLTAKURI', 'CDLTASUKIGAP',
}

# CDL2CROWS 触发时记为 CDLADVANCEBLOCK（保留旧代码笔误）
_PATTERN_NAME_OVERRIDE = {'CDL2CROWS': 'CDLADVANCEBLOCK'}

# 策略表现筛选阈值
BUY_PERF_THRESHOLD = {'win_rate': 50, 'return': 0}
SELL_PERF_THRESHOLD = {'win_rate': 50, 'return': 0}  # 卖出信号阈值与买入一致（胜率>50% 且 收益率>0%）


# ============================================================================
# 策略表现 CSV 加载（带容错和缓存）
# ============================================================================
_perf_cache = {}  # (csv_path, mtime) -> DataFrame


def load_strategy_perf(csv_path):
    """加载策略表现 CSV，带 mtime 缓存。文件不存在返回空 DataFrame（含表头）。"""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)'])
    try:
        mtime = os.path.getmtime(csv_path)
    except OSError:
        return pd.DataFrame(columns=['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)'])
    key = (csv_path, mtime)
    if key in _perf_cache:
        return _perf_cache[key]
    try:
        df = pd.read_csv(csv_path)
        _perf_cache[key] = df
        return df
    except Exception as e:
        logger.warning(f'[信号] 读取策略表现失败 {csv_path}: {e}')
        return pd.DataFrame(columns=['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)'])


def _query_perf(perf_df, pattern_name, buy_side=True):
    """查询策略表现，返回 (win_rate, return, sharpe, max_drawdown, trade_count) 或全 None。

    Args:
        perf_df: 策略表现 DataFrame
        pattern_name: 形态名（如 CDL3INSIDE，不含 buy_/sell_ 前缀）
        buy_side: True 查买入（匹配 'buy_' + pattern），False 查卖出（匹配 'sell_' + pattern）
    """
    if perf_df.empty:
        return None
    strategy_name = ('buy_' + pattern_name) if buy_side else ('sell_' + pattern_name)
    try:
        rows = perf_df[perf_df['策略名称'] == strategy_name]
        if len(rows) == 0:
            return None
        r = rows.iloc[0]
        return {
            'win_rate': float(r['胜率(%)']),
            'return': float(r['简易收益率(%)']),
            'sharpe': float(r['夏普比率']),
            'max_drawdown': float(r['最大回撤(%)']),
            'trade_count': float(r['交易次数']),
        }
    except Exception:
        return None


# ============================================================================
# 谨慎模式额外条件（6 个形态）
# ============================================================================
def _cautious_filter(pattern, df, i):
    """谨慎模式下对 6 个特定形态施加额外条件。返回 True 表示通过过滤可触发信号。"""
    if pattern == 'CDLMARUBOZU':
        # 前两根 K 线涨幅均 < 1.5%
        return (df['close'].iloc[i-1] / df['open'].iloc[i-1] < 1.015
                and df['close'].iloc[i-2] / df['open'].iloc[i-2] < 1.015)
    elif pattern in ('CDLDRAGONFLYDOJI', 'CDLTAKURI'):
        # 处于下降趋势：前 3 根 close 递减 + 当日 close < 前一日 close
        return (df['close'].iloc[i-1] < df['open'].iloc[i-1]
                and df['close'].iloc[i-1] < df['close'].iloc[i-2]
                and df['close'].iloc[i-2] < df['close'].iloc[i-3]
                and df['close'].iloc[i] < df['close'].iloc[i-1])
    elif pattern == 'CDLINVERTEDHAMMER':
        # 当日实体接近最低价（实体/最低价 < 1.003）
        c, o, l = df['close'].iloc[i], df['open'].iloc[i], df['low'].iloc[i]
        return ((c > o and o / l < 1.003) or (c < o and c / l < 1.003))
    elif pattern == 'CDLSEPARATINGLINES':
        # 前 2 根 close 上涨 + 当日阳线
        return (df['close'].iloc[i-2] > df['close'].iloc[i-3]
                and df['close'].iloc[i] > df['open'].iloc[i])
    elif pattern == 'CDLTASUKIGAP':
        # 前一日开盘和最低价均高于前两日最高价
        return (df['open'].iloc[i-1] > df['high'].iloc[i-2]
                and df['low'].iloc[i-1] > df['high'].iloc[i-2])
    return True  # 非谨慎模式形态不过滤


# ============================================================================
# 信号计算核心
# ============================================================================
def _compute_all_patterns(df, patterns):
    """批量计算所有 TA-Lib CDL 形态，返回 {pattern_name: np.ndarray}。"""
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    results = {}
    for p in patterns:
        try:
            func = getattr(talib, p)
            results[p] = func(o, h, l, c)
        except Exception as e:
            logger.warning(f'[信号] TA-Lib 计算失败 {p}: {e}')
            results[p] = np.zeros(len(df))
    return results


def _resolve_pattern_name(pattern):
    """处理 CDL2CROWS -> CDLADVANCEBLOCK 的笔误映射。"""
    return _PATTERN_NAME_OVERRIDE.get(pattern, pattern)


def _enrich_signal(sig):
    """给信号 dict 补充中文名、含义描述、历史基准胜率/收益率（全市场形态统计）。

    输入 sig 必须含 'pattern' 和 'type'（'buy'/'sell'）。
    若 pattern 为 None（如「达到持有天数」卖出），中文名/含义/市场数据置为 None。
    """
    pattern = sig.get('pattern')
    if not pattern:
        sig.setdefault('pattern_cn', None)
        sig.setdefault('pattern_desc', None)
        sig.setdefault('market_win_rate', None)
        sig.setdefault('market_return', None)
        sig.setdefault('market_trade_count', None)
        return sig
    sig['pattern_cn'] = PATTERN_CN_NAMES.get(pattern, pattern)
    sig['pattern_desc'] = PATTERN_DESCRIPTIONS.get(pattern, '')
    mp = _query_market_perf(pattern, sig['type'])
    if mp:
        sig['market_win_rate'] = mp['win_rate']
        sig['market_return'] = mp['return']
        sig['market_trade_count'] = mp['trade_count']
    else:
        sig['market_win_rate'] = None
        sig['market_return'] = None
        sig['market_trade_count'] = None
    return sig


def _find_matched_patterns(df, idx, buy_results, sell_results,
                           buy_patterns_to_compute, sell_patterns_to_compute,
                           cautious, code, held_codes):
    """先找出当日实际匹配的形态列表；若一个都没匹配，则可跳过策略表现 CSV 加载。"""
    matched_buy = []
    for pattern in buy_patterns_to_compute:
        arr = buy_results.get(pattern)
        if arr is None:
            continue
        if arr[idx] > 0:
            if pattern in CAUTIOUS_PATTERNS and cautious:
                if not _cautious_filter(pattern, df, idx):
                    continue
            matched_buy.append(pattern)

    matched_sell = []
    _should_compute_sell = True
    if held_codes is not None and code is not None:
        if code not in held_codes and len(held_codes) > 0:
            _should_compute_sell = False

    if _should_compute_sell:
        for pattern in sell_patterns_to_compute:
            arr = sell_results.get(pattern)
            if arr is None:
                continue
            if arr[idx] < 0:
                matched_sell.append(pattern)

    return matched_buy, matched_sell


def compute_signals_for_code(df, code, code_name, track_date,
                              cautious, is_index, perf_dir, track_patterns=None,
                              held_codes=None, data_folder_dir=None,
                              perf_start_date='20100104', observe_day=2):
    """计算单个标的在 track_date 当天的所有信号。

    每日信号是给用户的「当日操作建议」，系统不知道用户是否持仓，
    因此直接判断 track_date 当天的形态信号，不回溯历史持仓状态机。

    Args:
        df: 日 K DataFrame（按 trade_date 升序，列含 open/high/low/close/trade_date）
        code: 标的代码
        code_name: 标的名称
        track_date: 跟踪日 YYYY-MM-DD
        cautious: 是否谨慎模式
        is_index: 是否指数（保留参数，个股/指数逻辑现已统一）
        perf_dir: 策略表现 CSV 所在目录（实时计算失败时的回退数据源）
        track_patterns: 定向跟踪的形态列表，None=全量；否则 list[dict]，
            每条 {'pattern': str, 'pattern_type': 'buy'/'sell'}。
            传入后只判断这些形态，跳过其他形态。
        held_codes: 持仓代码列表 set；非空时，不在列表中的标的不计算卖出信号。
            卖出信号仅针对持仓个股生成。
        data_folder_dir: 数据目录，用于实时计算策略绩效（与策略回测一致）。
            传入时对匹配形态实时调用 run_single_pattern，确保结果与策略回测一致。
        perf_start_date: 实时绩效计算的起始日（默认 20100104，与信号更新一致）
        observe_day: 观察日天数（默认 2，与信号更新/策略回测一致）

    Returns:
        {
            'code': str, 'name': str, 'is_index': bool,
            'signals': [...],
            'error': str,
        }
    """
    result = {'code': code, 'name': code_name, 'is_index': is_index, 'signals': [], 'error': ''}

    if df is None or df.empty:
        result['error'] = '数据为空'
        return result

    try:
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date_str'] = df['trade_date'].astype(str)

        # 找到 track_date 在 df 中的位置
        track_date_str = track_date.replace('-', '')
        mask = df['trade_date_str'] == track_date_str
        if not mask.any():
            # 跟踪日不在数据中：数据过期或缺失，不能 fallback 到最后一根 K 线
            # （否则会导致不管选哪天都是相同信号）
            latest_date = df["trade_date_str"].iloc[-1]
            data_range = f'{df["trade_date_str"].iloc[0]}~{latest_date}'

            # 海外指数当日数据可能尚未生成（时差原因，今天的K线要等晚上才能拿到）
            overseas_codes = {'DJI', 'FCHI', 'SPX', 'GDAXI', 'N225'}
            if code in overseas_codes:
                # 计算最新数据日期与跟踪日的差距
                try:
                    from datetime import datetime as _dt
                    latest_dt = _dt.strptime(latest_date, '%Y%m%d')
                    track_dt = _dt.strptime(track_date_str, '%Y%m%d')
                    gap_days = (track_dt - latest_dt).days
                except Exception:
                    gap_days = 0
                if 0 < gap_days <= 3:
                    result['error'] = (
                        f'跟踪日 {track_date_str} 数据尚未生成（最新数据：{latest_date}）。'
                        f'海外指数当日K线通常在收盘后2-3小时更新（北京时间晚间），请稍后重试'
                    )
                    return result
            # 默认提示
            result['error'] = (
                f'跟踪日 {track_date_str} 不在数据中（数据范围：{data_range}），'
                f'系统已尝试拉取最新数据但仍未覆盖，请稍后重试或检查数据源'
            )
            return result
        target_idx = mask.idxmax()

        if target_idx < 3:
            result['error'] = f'数据不足（{target_idx+1} 行），至少需要 4 行'
            return result

        # 批量计算所有形态
        # 定向跟踪：若指定了 track_patterns，只计算这些形态
        if track_patterns:
            buy_patterns_to_compute = list({p['pattern'] for p in track_patterns if p.get('pattern_type') == 'buy'})
            sell_patterns_to_compute = list({p['pattern'] for p in track_patterns if p.get('pattern_type') == 'sell'})
        else:
            buy_patterns_to_compute = ALL_BUY_PATTERNS
            sell_patterns_to_compute = ALL_SELL_PATTERNS
        buy_pattern_results = _compute_all_patterns(df, buy_patterns_to_compute)
        sell_pattern_results = _compute_all_patterns(df, sell_patterns_to_compute)

        # 先判断是否有任何匹配形态；没有则跳过绩效计算
        matched_buy, matched_sell = _find_matched_patterns(
            df, target_idx, buy_pattern_results, sell_pattern_results,
            buy_patterns_to_compute, sell_patterns_to_compute,
            cautious, code, held_codes)
        if not matched_buy and not matched_sell:
            result['signals'] = []
            return result

        # 实时计算匹配形态的策略绩效（与策略回测使用同一 run_single_pattern，确保结果一致）
        buy_perf_df = pd.DataFrame(columns=['策略名称', '交易次数', '胜率(%)', '简易收益率(%)', '夏普比率', '最大回撤(%)'])
        sell_perf_df = buy_perf_df.copy()
        if data_folder_dir and (matched_buy or matched_sell):
            try:
                import pattern_scan
                # 加载完整历史 DataFrame（一次加载，所有匹配形态复用）
                full_df = pattern_scan._load_raw_dataframe(code, data_folder_dir)
                end_date_str = track_date.replace('-', '')
                perf_rows = []
                for pattern in matched_buy:
                    try:
                        r = pattern_scan.run_single_pattern(
                            code=code, pattern_name=pattern, pattern_type='buy',
                            start_date=perf_start_date, end_date=end_date_str,
                            data_folder_dir=data_folder_dir, observe_day=observe_day,
                            cautious=cautious, cached_df=full_df)
                        perf_rows.append({
                            '策略名称': f'buy_{pattern}',
                            '交易次数': r.get('trades', 0),
                            '胜率(%)': r.get('win_rate', 0),
                            '简易收益率(%)': r.get('return_pct', 0),
                            '夏普比率': r.get('sharpe', 0) or 0,
                            '最大回撤(%)': r.get('hold_max_drawdown', 0),
                        })
                    except Exception as e:
                        logger.debug(f'[信号] 实时计算买入绩效失败 {code} {pattern}: {e}')
                for pattern in matched_sell:
                    try:
                        r = pattern_scan.run_single_pattern(
                            code=code, pattern_name=pattern, pattern_type='sell',
                            start_date=perf_start_date, end_date=end_date_str,
                            data_folder_dir=data_folder_dir, observe_day=observe_day,
                            cautious=cautious, cached_df=full_df)
                        perf_rows.append({
                            '策略名称': f'sell_{pattern}',
                            '交易次数': r.get('trades', 0),
                            '胜率(%)': r.get('win_rate', 0),
                            '简易收益率(%)': r.get('return_pct', 0),
                            '夏普比率': r.get('sharpe', 0) or 0,
                            '最大回撤(%)': r.get('hold_max_drawdown', 0),
                        })
                    except Exception as e:
                        logger.debug(f'[信号] 实时计算卖出绩效失败 {code} {pattern}: {e}')
                if perf_rows:
                    all_perf_df = pd.DataFrame(perf_rows)
                    buy_perf_df = all_perf_df[all_perf_df['策略名称'].str.startswith('buy_')]
                    sell_perf_df = all_perf_df[all_perf_df['策略名称'].str.startswith('sell_')]
                del full_df
            except Exception as e:
                logger.debug(f'[信号] 加载完整历史数据失败 {code}: {e}，回退到 CSV')
                buy_perf_df = load_strategy_perf(os.path.join(perf_dir, f'{code}_buy_strategy_performance_test.csv'))
                sell_perf_df = load_strategy_perf(os.path.join(perf_dir, f'{code}_sell_strategy_performance_test.csv'))
        else:
            # 未传 data_folder_dir，回退到 CSV 快照
            buy_perf_df = load_strategy_perf(os.path.join(perf_dir, f'{code}_buy_strategy_performance_test.csv'))
            sell_perf_df = load_strategy_perf(os.path.join(perf_dir, f'{code}_sell_strategy_performance_test.csv'))

        # 计算信号（个股/指数统一逻辑：纯形态判断，不回溯状态机）
        signals = _compute_signals(
            df, target_idx, buy_pattern_results, sell_pattern_results,
            buy_perf_df, sell_perf_df, cautious, is_index,
            buy_patterns_to_compute=buy_patterns_to_compute,
            sell_patterns_to_compute=sell_patterns_to_compute,
            code=code, held_codes=held_codes)

        # 跟踪日当天的行情数据（用于前端展示当前价格和 K 线图定位）
        track_row = df.iloc[target_idx]
        track_close = float(track_row['close'])
        # 涨跌幅：相对前一日收盘
        track_pct_chg = None
        if target_idx > 0:
            prev_close = float(df.iloc[target_idx - 1]['close'])
            if prev_close > 0:
                track_pct_chg = round((track_close - prev_close) / prev_close * 100, 2)

        # 给每个信号补全 code/name/is_index + 中文名/含义/全市场统计数据 + 当天行情
        for sig in signals:
            sig['code'] = code
            sig['name'] = code_name
            sig['is_index'] = is_index
            sig['trade_date'] = track_date_str  # YYYYMMDD
            sig['close'] = track_close
            sig['pct_chg'] = track_pct_chg
            _enrich_signal(sig)

        # 批量补充 AI 评分（CNN + XGBoost，离线静默降级）
        _enrich_signals_with_ai(signals, df, target_idx)

        result['signals'] = signals
    except Exception as e:
        import traceback
        result['error'] = f'{type(e).__name__}: {e}'
        logger.debug(f'[信号] 计算异常 {code}: {traceback.format_exc()}')

    return result


def _compute_signals(df, idx, buy_results, sell_results,
                     buy_perf_df, sell_perf_df, cautious, is_index,
                     buy_patterns_to_compute=None, sell_patterns_to_compute=None,
                     code=None, held_codes=None):
    """统一信号计算逻辑（个股/指数一致）。

    每日信号是给用户的「当日操作建议」，不假设用户持仓状态。
    直接判断 track_date 当天的形态：
    - 出现看涨形态（>0）且通过策略表现筛选 → 输出买入信号
    - 出现看跌形态（<0）且通过策略表现筛选 → 输出卖出信号
    - 同日可同时输出买卖信号（由用户根据自身持仓决定如何操作）

    Args:
        df: 日 K DataFrame
        idx: track_date 在 df 中的位置
        buy_results/sell_results: 形态计算结果 dict
        buy_perf_df/sell_perf_df: 策略表现 DataFrame
        cautious: 是否谨慎模式
        is_index: 是否指数（保留参数，个股/指数逻辑现已统一）
        buy_patterns_to_compute: 要判断的买入形态列表（None=ALL_BUY_PATTERNS）
        sell_patterns_to_compute: 要判断的卖出形态列表（None=ALL_SELL_PATTERNS）
        code: 标的代码，用于判断是否持仓
        held_codes: 持仓代码 set；非空时，不在集合中的标的不计算卖出信号
    """
    signals = []

    buy_patterns = buy_patterns_to_compute if buy_patterns_to_compute is not None else ALL_BUY_PATTERNS
    sell_patterns = sell_patterns_to_compute if sell_patterns_to_compute is not None else ALL_SELL_PATTERNS

    # 卖出信号仅对持仓个股生成
    _should_compute_sell = True
    if held_codes is not None and code is not None:
        # held_codes 非空时，只有在持仓列表中的标的才计算卖出信号
        if code not in held_codes and len(held_codes) > 0:
            _should_compute_sell = False

    # 买入信号：收集所有匹配的形态（不再只保留最后一个，也不做绩效过滤）
    matched_buy_patterns = []
    for pattern in buy_patterns:
        arr = buy_results.get(pattern)
        if arr is None:
            continue
        if arr[idx] > 0:  # 看涨形态
            if pattern in CAUTIOUS_PATTERNS and cautious:
                if not _cautious_filter(pattern, df, idx):
                    continue
            matched_buy_patterns.append(pattern)

    for pattern in matched_buy_patterns:
        resolved = _resolve_pattern_name(pattern)
        perf = _query_perf(buy_perf_df, resolved, buy_side=True)
        sig = {
            'type': 'buy',
            'pattern': resolved,
        }
        if perf is not None:
            sig.update({
                'win_rate': perf['win_rate'],
                'return': perf['return'],
                'sharpe': perf['sharpe'],
                'max_drawdown': perf['max_drawdown'],
                'trade_count': perf['trade_count'],
            })
        else:
            sig.update({
                'win_rate': None, 'return': None, 'sharpe': None,
                'max_drawdown': None, 'trade_count': None,
                'no_perf_data': True,
            })
        signals.append(sig)

    # 卖出信号：仅对持仓标的计算，收集所有匹配的形态
    matched_sell_patterns = []
    if _should_compute_sell:
        for pattern in sell_patterns:
            arr = sell_results.get(pattern)
            if arr is None:
                continue
            if arr[idx] < 0:  # 看跌形态
                matched_sell_patterns.append(pattern)

        for pattern in matched_sell_patterns:
            resolved = _resolve_pattern_name(pattern)
            perf = _query_perf(sell_perf_df, resolved, buy_side=False)
            sig = {
                'type': 'sell',
                'pattern': resolved,
                'reason': '形态卖出',
            }
            if perf is not None:
                sig.update({
                    'win_rate': perf['win_rate'],
                    'return': perf['return'],
                    'sharpe': perf['sharpe'],
                    'max_drawdown': perf['max_drawdown'],
                    'trade_count': perf['trade_count'],
                })
            else:
                sig.update({
                    'win_rate': None, 'return': None, 'sharpe': None,
                    'max_drawdown': None, 'trade_count': None,
                    'no_perf_data': True,
                })
            signals.append(sig)

    return signals


def _pass_buy_filter(perf):
    """买入信号筛选（已废弃，保留函数签名仅作兼容）：不再过滤信号。"""
    return True


def _pass_sell_filter(perf):
    """卖出信号筛选（已废弃，保留函数签名仅作兼容）：不再过滤信号。"""
    return True
