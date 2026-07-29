# 数据源与盈湖

## 目录

- [1. 盈湖（Yinghu DB）](#1-盈湖yinghu-db)
- [2. 外部数据源优先级](#2-外部数据源优先级)
- [3. 数据读取流程](#3-数据读取流程)
- [4. 数据入库校验](#4-数据入库校验)
- [5. 结果库（Result DB）](#5-结果库result-db)
- [6. 配置参考](#6-配置参考)

---

## 1. 盈湖（Yinghu DB）

盈湖是项目的核心数据存储，包含 2010 年至今的全市场 A 股日 K 数据（北交所除外）。

### 1.1 存储结构

```
盈湖/
├── kline/                          # K 线数据根目录
│   ├── A股/                        # A 股个股
│   │   ├── 000001.SZ/
│   │   │   ├── 201001.parquet      # 2010年1月数据
│   │   │   ├── 201002.parquet
│   │   │   └── ...
│   │   └── 600519.SH/
│   │       └── ...
│   └── 指数/                       # 指数数据
│       ├── 000300.SH/
│       └── ...
├── yinghu.db                       # 元数据库（SQLite + WAL）
└── backup/                         # 每日快照，保留 30 天
```

### 1.2 分区规则

- **按月分区**：`{YYYYMM}.parquet`（如 `202607.parquet`）
- **单文件大小**：50-100MB（通过 Parquet 列式压缩）
- **归档机制**：近 24 个月数据在线，更早数据归档压缩

### 1.3 元数据库

`yinghu.db`（SQLite + WAL 模式）存储：

| 表 | 用途 |
|----|------|
| `securities` | 标的列表（code、name、板块） |
| `data_status` | 数据状态（每个 code 每月的数据覆盖范围） |

### 1.4 核心接口

```python
from data.yinghu_db import get_kline, check_coverage, save_kline

# 读取 K 线
df = get_kline('600519.SH', '20200101', '20260727')

# 检查数据覆盖
if check_coverage('600519.SH', '20200101', '20260727'):
    print("盈湖已覆盖该时间范围")

# 保存新拉取的数据入盈湖
save_kline('600519.SH', df)
```

## 2. 外部数据源优先级

当盈湖未覆盖所需数据时，按以下优先级从外部数据源拉取：

```
akshare  →  tushare  →  腾讯财经  →  新浪财经
  (1)        (2)         (3)         (4)
```

| 优先级 | 数据源 | 优点 | 缺点 |
|--------|--------|------|------|
| 1 | akshare | 免费、覆盖全 | 限频、JS 引擎线程不安全 |
| 2 | tushare | 专业、稳定 | 需 token、积分限制 |
| 3 | 腾讯财经 | 免费 | 接口偶发除权除息附加字段 |
| 4 | 新浪财经 | 免费、实时 | 仅适合补最后一根 K 线 |

### 2.1 重试机制

- 每个数据源最多重试 3 次，指数退避
- 单标超时 60 秒
- 并行度限制 4-8 进程

### 2.2 数据源兜底

```python
# backend/data/data_source.py
def get_kline_df(code, start_date, end_date,
                 prefer_local=True, allow_network=True):
    # 1. 盈湖优先
    if check_coverage(code, start_date, end_date):
        return get_kline(code, start_date, end_date)

    # 2. 本地旧目录（CSV/Parquet）
    df = _load_local(code, start_date, end_date)
    if _df_has_end_date(df, end_date):
        return df

    # 3. 外部数据源（按优先级）
    if allow_network:
        for source in [akshare, tushare, tencent, sina]:
            df = source(code, start_date, end_date)
            if df is not None:
                save_kline(code, df)  # 入盈湖
                return df

    return df  # 回退到 partial 数据
```

## 3. 数据读取流程

```
get_kline_df(code, start, end)
  │
  ├─ 盈湖 check_coverage → True → get_kline → return
  │
  ├─ 盈湖未命中
  │   ├─ 本地旧目录 → 有 end_date → return
  │   └─ 无 end_date
  │       ├─ akshare → 成功 → 入盈湖 → return
  │       ├─ tushare → 成功 → 入盈湖 → return
  │       ├─ 腾讯    → 成功 → 入盈湖 → return
  │       └─ 新浪    → 成功 → 入盈湖 → return
  │
  └─ 全部失败 → 回退 partial 数据
```

**关键改进**：盈湖返回数据后，会检查是否包含 end_date。如果不包含（如盈湖数据只到昨天，但跟踪日是今天），不会直接返回旧数据，而是继续走网络拉取。

## 4. 数据入库校验

数据入盈湖前必须通过 3 项校验：

| 校验项 | 规则 | 失败处理 |
|--------|------|---------|
| 列完整性 | 必须含 open/high/low/close/vol/trade_date | 拒绝入库 |
| 价格合理性 | high ≥ max(open, close) ≥ min(open, close) ≥ low | 拒绝入库 |
| 日期连续性 | trade_date 单调递增、无重复 | 去重后入库 |

## 5. 结果库（Result DB）

结果库缓存策略回测和信号跟踪的结果，避免重复计算。

### 5.1 缓存键

```
SHA1(code + 策略名 + 日期范围 + observe_day + cautious模式)
```

源数据变更或策略逻辑变更时自动失效。

### 5.2 存储结构

```
结果库/
├── result.db          # 索引数据库（SQLite + WAL）
└── data/              # 数据目录
    ├── ab/            # 按 hash 前两位分桶
    │   └── ab1234...json
    └── cd/
        └── cd5678...json
```

### 5.3 保留策略

- 保留期：365 天
- 超过自动清理

## 6. 配置参考

```python
# backend/config.py
class Config:
    YINGHU_DB_DIR = BASE_DIR / '盈湖'
    YINGHU_DB_KLINE_DIR = YINGHU_DB_DIR / 'kline'
    YINGHU_DB_META = YINGHU_DB_DIR / 'yinghu.db'
    YINGHU_DB_START_DATE = '20100101'
    YINGHU_DB_ARCHIVE_MONTHS = 24

    RESULT_DB_DIR = BASE_DIR / '结果库'
    RESULT_DB_META = RESULT_DB_DIR / 'result.db'
    RESULT_DB_RETENTION_DAYS = 365
```
