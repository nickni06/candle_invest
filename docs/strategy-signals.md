# 策略与信号

## 目录

- [1. K 线形态识别](#1-k-线形态识别)
- [2. 信号生成](#2-信号生成)
- [3. 向量化回测](#3-向量化回测)
- [4. 信号跟踪调度](#4-信号跟踪调度)
- [5. 全市场统计（历史基准）](#5-全市场统计历史基准)
- [6. 策略表现 CSV](#6-策略表现-csv)

---

## 1. K 线形态识别

### 1.1 形态列表

系统使用 TA-Lib 的 61 个 CDL 函数识别 K 线形态，分为：

| 分类 | 数量 | 说明 |
|------|------|------|
| 主要买入形态 | 39 | PRIMARY_BUY_PATTERNS |
| 次要买入形态 | 20 | SECONDARY_BUY_PATTERNS（后判断覆盖前者） |
| 主要卖出形态 | 20 | PRIMARY_SELL_PATTERNS |
| 次要卖出形态 | 39 | SECONDARY_SELL_PATTERNS |

**去重后唯一形态 59 个**（买入和卖出共用同一批形态，靠 TA-Lib 返回值的正负区分方向）。

### 1.2 形态触发规则

- TA-Lib 返回值 `> 0`（如 100） → 买入信号
- TA-Lib 返回值 `< 0`（如 -100） → 卖出信号
- TA-Lib 返回值 `= 0` → 无信号

### 1.3 谨慎模式

谨慎模式下，6 个特定形态需额外条件：

| 形态 | 额外条件 |
|------|---------|
| CDLMARUBOZU | 前两根 K 线涨幅均 < 1.5% |
| CDLDRAGONFLYDOJI | 处于下降趋势（前 3 根 close 递减） |
| CDLTAKURI | 同上 |
| CDLINVERTEDHAMMER | 当日实体接近最低价（实体/最低价 < 1.003） |
| CDLSEPARATINGLINES | 前 2 根 close 上涨 + 当日阳线 |
| CDLTASUKIGAP | 前一日开盘和最低价均高于前两日最高价 |

### 1.4 笔误保留

`CDL2CROWS` 触发时记为 `CDLADVANCEBLOCK`（与旧代码一致，保留兼容性）。

## 2. 信号生成

### 2.1 信号计算入口

```python
# backend/signal/strategy_signals.py
def compute_signals_for_code(df, code, code_name, track_date,
                               cautious, is_index, perf_dir, ...):
    """计算单个标的在 track_date 当天的所有信号。"""
    # 1. 找到 track_date 在 df 中的位置
    # 2. 批量计算所有 TA-Lib 形态
    # 3. 查找匹配形态（matched_buy / matched_sell）
    # 4. 对匹配形态实时计算策略绩效（run_single_pattern）
    # 5. 补充「历史基准」字段（全市场统计）
    # 6. 返回结构化信号列表
```

### 2.2 信号输出格式

```python
{
    'code': '600519.SH',
    'name': '贵州茅台',
    'is_index': False,
    'signals': [
        {
            'type': 'buy',                    # buy / sell
            'pattern': 'CDLHAMMER',           # 形态名
            'pattern_cn': '锤子线',           # 中文名
            'pattern_desc': '...',            # 含义描述
            'win_rate': 55.3,                 # 胜率%
            'return_pct': 2.1,                # 收益率%
            'trades': 15,                     # 交易次数
            'sharpe': 0.8,                    # 夏普比率
            'hold_max_drawdown': -3.2,        # 最大回撤%
            'market_win_rate': 44.85,         # 历史基准胜率
            'market_return': 1.2,            # 历史基准收益率
            'market_trade_count': 15165,      # 全市场交易次数
        }
    ],
    'error': ''
}
```

### 2.3 信号过滤规则

**所有匹配形态均输出**，不按胜率/收益率过滤。用户在前端自行判断。

### 2.4 买入信号止损规则

- **3% 固定止损**：买入后若 `close[0]/buy_price - 1 ≤ -3%` 立即卖出
- **卖出信号不设止损**

## 3. 向量化回测

### 3.1 输入

- 单个股票的 OHLCV DataFrame（带 `trade_date` 日期索引）
- 形态信号序列（TA-Lib 输出）
- 参数：observe_day、cash、cautious

### 3.2 输出

绩效字典（非 DataFrame）：

```python
{
    'pattern': 'CDLHAMMER',
    'pattern_cn': '锤子线',
    'type': 'buy',
    'trades': 15,                # 总交易次数
    'win_rate': 55.3,            # 胜率%
    'return_pct': 2.1,           # 简易收益率%
    'annualized_return': 8.5,    # 年化收益%
    'capital_occupation': 0.45,   # 资金占用率
    'sharpe': 0.8,               # 夏普比率
    'hold_max_drawdown': -3.2,   # 最大回撤%
    'trade_details': [           # 交易明细
        {
            'buy_date': '2025-01-05',
            'buy_price': 1850.0,
            'sell_date': '2025-01-07',
            'sell_price': 1895.0,
            'return_pct': 2.43,
            'hold_days': 2,
            'win': True
        }
    ]
}
```

### 3.3 observe_day 语义

- `observe_day = 2`：买入后持有 2 个交易日，第 3 日卖出
- 全项目统一为 2

## 4. 信号跟踪调度

### 4.1 扫描频率

- **每日收盘后扫描一次**（非盘中实时）
- 通过 `track_date` 参数指定跟踪日

### 4.2 扫描范围

| 模式 | 说明 |
|------|------|
| `all` | 全 A 股 + 7 个指数 |
| `index` | 仅 7 个指数 |
| `held` | 仅持仓标的 |
| `target` | 用户指定的目标个股 |

### 4.3 并行架构

```
signal_tracker.run_tracking()
  ├─ 父进程预拉取（ProcessPoolExecutor, 4 workers）
  │   └─ yinghu_db.check_coverage() → 未命中 → get_kline_df()
  ├─ 信号计算（ProcessPoolExecutor, 4-8 workers）
  │   └─ worker: compute_signals_for_code()
  └─ 汇总 → summary.json
```

### 4.4 macOS 多进程

```python
# macOS 用 forkserver（避免 akshare V8 引擎 fork 崩溃）
# Linux 用 fork（性能更好）
if platform.system() == 'Darwin':
    multiprocessing.set_start_method('forkserver', force=True)
```

## 5. 全市场统计（历史基准）

### 5.1 数据来源

`market_wide_pattern_stats.csv`（由信号更新时自动生成）

### 5.2 统计方式

- 覆盖 A 股 + 指数（不含 ETF）
- 交易次数：求和
- 胜率/收益率：按交易次数加权平均
- 去重：按 (code, 策略名) 去重，保留最新

### 5.3 前端展示

- 信号卡片和详情弹窗使用「历史基准」描述
- 胜率 `toFixed(0)`，收益率 `toFixed(2)`

## 6. 策略表现 CSV

### 6.1 文件路径

```
策略表现/
├── A股/
│   └── 个股策略表现/
│       ├── 600519.SH_perf.csv
│       └── ...
└── 指数/
    └── 个股策略表现/
        └── ...

策略表现/market_wide_pattern_stats.csv    # 全市场聚合
```

### 6.2 CSV 列定义

```
策略名称,交易次数,胜率(%),简易收益率(%),夏普比率,最大回撤(%)
buy_CDLHAMMER,15,55.3,2.1,0.8,-3.2
sell_CDLDOJI,8,50.0,-1.2,-0.3,-2.5
```

### 6.3 更新规则

- 全量更新（非续跑）时删除并重建 CSV
- 续跑模式追加写入
- None 夏普比率转为 0
