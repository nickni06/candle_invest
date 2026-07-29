# Candle Invest Wiki

> 基于 Backtrader + TA-Lib + AI 的 A 股量化交易系统知识库（单文件版）
>
> 本文件整合架构、数据、策略、信号、AI 模型、部署、API 全部内容，顶部目录支持锚点快速定位。
> 图片请放在 `docs/assets/` 目录，用相对路径引用：`![描述](./assets/xxx.png)`

---

## 📑 完整目录

- [1. 项目概览](#1-项目概览)
- [2. 快速开始](#2-快速开始)
- [3. 系统架构](#3-系统架构)
  - [3.1 整体架构图](#31-整体架构图)
  - [3.2 模块分层](#32-模块分层)
  - [3.3 数据流](#33-数据流)
  - [3.4 并发模型](#34-并发模型)
  - [3.5 缓存策略](#35-缓存策略)
  - [3.6 关键设计决策](#36-关键设计决策)
- [4. 数据源与盈湖](#4-数据源与盈湖)
  - [4.1 盈湖存储结构](#41-盈湖存储结构)
  - [4.2 外部数据源优先级](#42-外部数据源优先级)
  - [4.3 数据读取流程](#43-数据读取流程)
  - [4.4 数据入库校验](#44-数据入库校验)
  - [4.5 结果库（Result DB）](#45-结果库result-db)
- [5. 策略与信号](#5-策略与信号)
  - [5.1 K 线形态识别](#51-k-线形态识别)
  - [5.2 信号生成](#52-信号生成)
  - [5.3 向量化回测](#53-向量化回测)
  - [5.4 信号跟踪调度](#54-信号跟踪调度)
  - [5.5 全市场统计（历史基准）](#55-全市场统计历史基准)
  - [5.6 策略表现 CSV](#56-策略表现-csv)
- [6. AI 模型（CNN + XGBoost）](#6-ai-模型cnn--xgboost)
  - [6.1 设计目标](#61-设计目标)
  - [6.2 整体架构](#62-整体架构)
  - [6.3 样本采集](#63-样本采集)
  - [6.4 特征工程（131 维）](#64-特征工程131-维)
  - [6.5 XGBoost 阶段1](#65-xgboost-阶段1)
  - [6.6 CNN 阶段2](#66-cnn-阶段2)
  - [6.7 Model Server 推理服务](#67-model-server-推理服务)
  - [6.8 训练命令速查](#68-训练命令速查)
- [7. 部署与运维](#7-部署与运维)
  - [7.1 环境准备](#71-环境准备)
  - [7.2 启动服务](#72-启动服务)
  - [7.3 配置参考](#73-配置参考)
  - [7.4 日志管理](#74-日志管理)
  - [7.5 数据备份](#75-数据备份)
  - [7.6 常见问题](#76-常见问题)
- [8. API 接口参考](#8-api-接口参考)
  - [8.1 主服务 API（端口 8765）](#81-主服务-api端口-8765)
  - [8.2 AI 推理服务 API（端口 8766）](#82-ai-推理服务-api端口-8766)
  - [8.3 调用示例](#83-调用示例)
- [9. 项目结构](#9-项目结构)
- [10. 关键约束](#10-关键约束)
- [11. 文档配图说明](#11-文档配图说明)

---

## 1. 项目概览

**Candle Invest** 是一套面向 A 股的量化交易策略研究与回测系统，核心能力：

- **61 个 K 线形态识别**（基于 TA-Lib CDL 函数）
- **向量化回测**（pandas/numpy，比 Backtrader 快 10x+）
- **信号跟踪**（多进程并行扫描全市场，每日收盘后运行）
- **组合回测**（多策略多标的组合绩效）
- **持仓管理**（前端可视化）
- **AI 模型**（CNN + XGBoost 融合买卖点信号）
- **盈湖数据存储**（全市场日 K，按月分区 Parquet + SQLite 元数据）

| 项目 | 说明 |
|------|------|
| 后端 | Python 3.13, Flask, Backtrader, TA-Lib, Numba |
| 数据存储 | Parquet（按月分区）, SQLite（WAL 模式） |
| 数据源 | akshare → tushare → 腾讯财经 → 新浪财经（多级兜底） |
| 前端 | HTML + Bootstrap 5 + 原生 JS（SSE 实时进度推送） |
| AI | XGBoost + 1D-CNN + ONNX Runtime |

---

## 2. 快速开始

```bash
# 1. 克隆仓库
git clone git@github.com:nickni06/candle_invest.git
cd candle_invest

# 2. 创建虚拟环境并安装依赖
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN

# 4. 安装 TA-Lib C 库（macOS）
brew install ta-lib

# 5. 初始化盈湖
python3 backend/data/yinghu_db_init.py

# 6. 启动 Web 服务
python3 backend/web/run_app.py
# 访问 http://127.0.0.1:8765

# 7.（可选）启动 AI 推理服务
python3 backend/ai/model_server.py
# 访问 http://127.0.0.1:8766/health
```

> 📌 **不要用 `python3 web_app.py` 启动**，macOS 多进程会冲突。必须用 `backend/web/run_app.py`。

---

## 3. 系统架构

### 3.1 整体架构图

<!-- 建议在此处插入架构图：![系统架构](./assets/architecture.png) -->

系统采用分层架构，前后端分离，AI 推理服务独立部署。

```
┌──────────────────────────────────────────────────────────┐
│                     用户浏览器                            │
│              http://127.0.0.1:8765                        │
└────────────────┬─────────────────────────────────────────┘
                 │ HTTP / SSE
┌────────────────▼─────────────────────────────────────────┐
│              Flask Web 服务 (端口 8765)                   │
│              backend/web/web_app.py                      │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ 策略回测  │ 信号跟踪  │ 持仓管理  │  盈湖管理 / 日志  │  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────────────┘  │
└───────┼──────────┼──────────┼──────────┼────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌────────────┬────────────┬──────────┬─────────────────────┐
│ pattern_   │ signal_    │ 持仓CSV  │     yinghu_db       │
│ scan       │ tracker    │          │   (盈湖数据访问层)   │
│ (形态回测)  │ (信号跟踪)  │          │                     │
└─────┬──────└─────┬──────┘          └─────────┬───────────┘
      │            │                            │
      ▼            ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│              盈湖（Parquet 按月分区）                      │
│              盈湖/kline/{板块}/{code}/{YYYYMM}.parquet    │
│              + 元数据库 yinghu.db (SQLite + WAL)          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│           AI 推理服务 (端口 8766，独立进程)                │
│           backend/ai/model_server.py                    │
│  ┌─────────────┬─────────────────┐                      │
│  │  XGBoost     │     CNN         │                      │
│  │  (阶段1)     │   (阶段2, 可选)  │                      │
│  └─────────────┴─────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模块分层

#### 3.2.1 数据层 (`backend/data/`)

| 模块 | 职责 |
|------|------|
| `data_source.py` | 统一数据源：akshare → tushare → 腾讯 → 新浪 |
| `yinghu_db.py` | 盈湖访问层：Parquet 读写、元数据管理、覆盖检查 |
| `yinghu_db_init.py` | 盈湖初始化（建表、建目录） |
| `result_db.py` | 结果库缓存（SHA1 键，365 天保留） |
| `data_refresh.py` | 集中数据补全模块 |

#### 3.2.2 策略层 (`backend/pattern/`, `backend/strategy/`)

| 模块 | 职责 |
|------|------|
| `pattern_scan.py` | 向量化回测引擎 + 形态信号计算 |
| `patternStrategy.py` | Backtrader 策略封装（旧版，保留兼容） |
| `numba_backtest.py` | Numba 加速版回测 |
| `portfolio_backtest.py` | 组合回测 |

#### 3.2.3 信号层 (`backend/signal/`)

| 模块 | 职责 |
|------|------|
| `signal_tracker.py` | 信号跟踪调度（多进程并行） |
| `strategy_signals.py` | 信号计算核心（纯 TA-Lib） |
| `signal_update.py` | 信号更新（批量回测） |
| `signal_utils.py` | 形态列表、中文名、描述 |

#### 3.2.4 AI 层 (`backend/ai/`)

| 模块 | 职责 |
|------|------|
| `cnn_model.py` | 1D-CNN 模型定义（参数量 4.4K） |
| `features.py` | 特征工程（131 维 = 118 TA-Lib + 13 手工） |
| `sample_collector.py` | 样本采集（滑动窗口 + 标签） |
| `train_xgb.py` | 阶段1 XGBoost 训练 |
| `train.py` | 阶段2 CNN 训练 |
| `model_server.py` | AI 推理 Flask 服务（端口 8766） |
| `inference.py` | CNN 推理（PyTorch / ONNX 双模式） |
| `inference_xgb.py` | XGBoost 推理 |
| `export_onnx.py` | ONNX 导出 + 动态量化 |

#### 3.2.5 Web 层 (`backend/web/`)

| 模块 | 职责 |
|------|------|
| `run_app.py` | 启动入口 |
| `web_app.py` | Flask 路由与 API（主服务 8765） |

### 3.3 数据流

#### 3.3.1 信号跟踪流程

```
用户点击「信号跟踪」
  ↓
web_app.py 启动 subprocess（避免阻塞主服务）
  ↓
signal_tracker.run_tracking()
  ├─ 父进程预拉取数据（多进程并行）
  │   └─ yinghu_db.check_coverage() → 未命中 → get_kline_df() 网络拉取
  ├─ ProcessPoolExecutor (workers=4-8)
  │   └─ worker: strategy_signals.compute_signals_for_code()
  │       ├─ 计算所有 TA-Lib 形态
  │       ├─ 查找匹配形态
  │       └─ 查询全市场统计 → 补充「历史基准」字段
  └─ 汇总结果 → summary.json
```

#### 3.3.2 策略回测流程

```
用户选择标的 + 形态 + 时间范围
  ↓
web_app.py → pattern_scan.run_single_pattern()
  ├─ 加载 K 线（盈湖优先，lru_cache + mtime 签名）
  ├─ 计算 TA-Lib 形态信号
  ├─ 向量化回测（pandas/numpy，可选 numba 加速）
  └─ 返回：交易明细 + 胜率/收益/夏普/回撤
```

#### 3.3.3 AI 训练流程

```
阶段1: XGBoost
  batch_collect.py
    → 盈湖 get_kline()
    → 滑动窗口 20 根 K 线
    → 计算 ATR + 未来 5 日涨幅 → 标签
    → 保存 .npy
  ↓
  dataset_xgb.py
    → 每个窗口提取 131 维特征
    → xgb_X.npy + xgb_y.npy
  ↓
  train_xgb.py
    → 按标的分组划分（防泄露）
    → XGBoost 训练（早停 + scale_pos_weight）
    → xgb_model.json

阶段2: CNN（可选）
  train.py
    → .npy 样本 → CNN 模型
    → cnn_model_best.pth
  ↓
  export_onnx.py
    → cnn_model.onnx（20KB，量化后）
```

### 3.4 并发模型

#### 3.4.1 多进程启动方式

```python
# backend/signal/signal_tracker.py
if platform.system() == 'Darwin':
    multiprocessing.set_start_method('forkserver', force=True)  # macOS
elif platform.system() == 'Linux':
    multiprocessing.set_start_method('fork', force=True)       # Linux
```

**为什么 macOS 用 forkserver？**
- akshare 内部用 libmini_racer（V8 JS 引擎），fork 会复制不安全状态导致崩溃
- forkserver 每个子进程独立初始化 V8，安全且比 spawn 快

#### 3.4.2 进程池配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SCAN_WORKERS` | 0（自动） | 形态扫描并行进程数 |
| `TRACKING_POOL_WORKERS` | 0（自动） | 信号跟踪并行进程数 |
| `TRACKING_PREFETCH_WORKERS` | 4 | 预拉取进程数 |
| `MAX_WORKERS` | max(4, min(CPU, 8)) | 全局上限 |

### 3.5 缓存策略

#### 3.5.1 数据缓存（3 层）

```
进程内 lru_cache  →  文件级 mtime 签名  →  盈湖 Parquet
   (最快)              (跨进程共享)          (持久化)
```

- `_load_raw_dataframe_cached()`：lru_cache(64)，key 含 mtime 签名
- 文件变更自动失效（mtime+size 变化 → 签名变化 → cache miss）

#### 3.5.2 结果库缓存

- 键：`SHA1(code + 策略名 + 日期范围 + observe_day + cautious)`
- 保留期：365 天
- 源数据/策略变更 → 自动失效

### 3.6 关键设计决策

#### 3.6.1 为什么放弃 Backtrader 的 next()？

| 问题 | Backtrader | 向量化回测 |
|------|-----------|-----------|
| 性能 | 慢（逐根 K 线回调） | 快 10x+（numpy 向量化） |
| 多进程安全 | 不安全（状态机） | 安全（无状态） |
| 日志解析 | 依赖正则解析日志 | 直接返回结构化 dict |

#### 3.6.2 为什么 AI 服务独立进程？

- PyTorch / ONNX Runtime 占用 500MB+ 内存
- 避免每个子进程都加载模型（7 workers × 500MB = 3.5GB）
- Model Server 加载一次，worker 通过 HTTP 调用

#### 3.6.3 为什么 CNN 模型只有 4.4K 参数？

- A 股形态本质是局部模式（3-5 根 K 线）
- 20 根输入已足够覆盖上下文
- 2 层 Conv1d（感受野 5）+ 全局池化，参数量刚好
- 大模型容易过拟合 A 股的噪声

---

## 4. 数据源与盈湖

### 4.1 盈湖存储结构

盈湖是项目的核心数据存储，包含 2010 年至今的全市场 A 股日 K 数据（北交所除外）。

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

**分区规则**：
- 按月分区：`{YYYYMM}.parquet`（如 `202607.parquet`）
- 单文件大小：50-100MB（Parquet 列式压缩）
- 归档机制：近 24 个月数据在线，更早数据归档压缩

**元数据库** `yinghu.db`（SQLite + WAL）：

| 表 | 用途 |
|----|------|
| `securities` | 标的列表（code、name、板块） |
| `data_status` | 数据状态（每个 code 每月的数据覆盖范围） |

**核心接口**：

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

### 4.2 外部数据源优先级

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

**重试机制**：
- 每个数据源最多重试 3 次，指数退避
- 单标超时 60 秒
- 并行度限制 4-8 进程

### 4.3 数据读取流程

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

> 📌 **关键改进**：盈湖返回数据后，会检查是否包含 end_date。如果不包含（如盈湖数据只到昨天，但跟踪日是今天），不会直接返回旧数据，而是继续走网络拉取。

### 4.4 数据入库校验

数据入盈湖前必须通过 3 项校验：

| 校验项 | 规则 | 失败处理 |
|--------|------|---------|
| 列完整性 | 必须含 open/high/low/close/vol/trade_date | 拒绝入库 |
| 价格合理性 | high ≥ max(open, close) ≥ min(open, close) ≥ low | 拒绝入库 |
| 日期连续性 | trade_date 单调递增、无重复 | 去重后入库 |

### 4.5 结果库（Result DB）

结果库缓存策略回测和信号跟踪的结果，避免重复计算。

**缓存键**：
```
SHA1(code + 策略名 + 日期范围 + observe_day + cautious模式)
```
源数据变更或策略逻辑变更时自动失效。

**存储结构**：
```
结果库/
├── result.db          # 索引数据库（SQLite + WAL）
└── data/              # 数据目录
    ├── ab/            # 按 hash 前两位分桶
    │   └── ab1234...json
    └── cd/
        └── cd5678...json
```

**保留策略**：365 天，超过自动清理。

---

## 5. 策略与信号

### 5.1 K 线形态识别

#### 5.1.1 形态列表

系统使用 TA-Lib 的 61 个 CDL 函数识别 K 线形态，分为：

| 分类 | 数量 | 说明 |
|------|------|------|
| 主要买入形态 | 39 | PRIMARY_BUY_PATTERNS |
| 次要买入形态 | 20 | SECONDARY_BUY_PATTERNS（后判断覆盖前者） |
| 主要卖出形态 | 20 | PRIMARY_SELL_PATTERNS |
| 次要卖出形态 | 39 | SECONDARY_SELL_PATTERNS |

**去重后唯一形态 59 个**（买入和卖出共用同一批形态，靠 TA-Lib 返回值的正负区分方向）。

#### 5.1.2 形态触发规则

- TA-Lib 返回值 `> 0`（如 100） → 买入信号
- TA-Lib 返回值 `< 0`（如 -100） → 卖出信号
- TA-Lib 返回值 `= 0` → 无信号

#### 5.1.3 谨慎模式

谨慎模式下，6 个特定形态需额外条件：

| 形态 | 额外条件 |
|------|---------|
| CDLMARUBOZU | 前两根 K 线涨幅均 < 1.5% |
| CDLDRAGONFLYDOJI | 处于下降趋势（前 3 根 close 递减） |
| CDLTAKURI | 同上 |
| CDLINVERTEDHAMMER | 当日实体接近最低价（实体/最低价 < 1.003） |
| CDLSEPARATINGLINES | 前 2 根 close 上涨 + 当日阳线 |
| CDLTASUKIGAP | 前一日开盘和最低价均高于前两日最高价 |

#### 5.1.4 笔误保留

`CDL2CROWS` 触发时记为 `CDLADVANCEBLOCK`（与旧代码一致，保留兼容性）。

### 5.2 信号生成

#### 5.2.1 信号计算入口

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

#### 5.2.2 信号输出格式

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

#### 5.2.3 信号过滤规则

**所有匹配形态均输出**，不按胜率/收益率过滤。用户在前端自行判断。

#### 5.2.4 买入信号止损规则

- **3% 固定止损**：买入后若 `close[0]/buy_price - 1 ≤ -3%` 立即卖出
- **卖出信号不设止损**

### 5.3 向量化回测

#### 5.3.1 输入

- 单个股票的 OHLCV DataFrame（带 `trade_date` 日期索引）
- 形态信号序列（TA-Lib 输出）
- 参数：observe_day、cash、cautious

#### 5.3.2 输出

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

#### 5.3.3 observe_day 语义

- `observe_day = 2`：买入后持有 2 个交易日，第 3 日卖出
- 全项目统一为 2

### 5.4 信号跟踪调度

#### 5.4.1 扫描频率

- **每日收盘后扫描一次**（非盘中实时）
- 通过 `track_date` 参数指定跟踪日

#### 5.4.2 扫描范围

| 模式 | 说明 |
|------|------|
| `all` | 全 A 股 + 7 个指数 |
| `index` | 仅 7 个指数 |
| `held` | 仅持仓标的 |
| `target` | 用户指定的目标个股 |

#### 5.4.3 并行架构

```
signal_tracker.run_tracking()
  ├─ 父进程预拉取（ProcessPoolExecutor, 4 workers）
  │   └─ yinghu_db.check_coverage() → 未命中 → get_kline_df()
  ├─ 信号计算（ProcessPoolExecutor, 4-8 workers）
  │   └─ worker: compute_signals_for_code()
  └─ 汇总 → summary.json
```

#### 5.4.4 macOS 多进程

```python
# macOS 用 forkserver（避免 akshare V8 引擎 fork 崩溃）
# Linux 用 fork（性能更好）
if platform.system() == 'Darwin':
    multiprocessing.set_start_method('forkserver', force=True)
```

### 5.5 全市场统计（历史基准）

#### 5.5.1 数据来源

`market_wide_pattern_stats.csv`（由信号更新时自动生成）

#### 5.5.2 统计方式

- 覆盖 A 股 + 指数（不含 ETF）
- 交易次数：求和
- 胜率/收益率：按交易次数加权平均
- 去重：按 (code, 策略名) 去重，保留最新

#### 5.5.3 前端展示

- 信号卡片和详情弹窗使用「历史基准」描述
- 胜率 `toFixed(0)`，收益率 `toFixed(2)`

### 5.6 策略表现 CSV

#### 5.6.1 文件路径

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

#### 5.6.2 CSV 列定义

```
策略名称,交易次数,胜率(%),简易收益率(%),夏普比率,最大回撤(%)
buy_CDLHAMMER,15,55.3,2.1,0.8,-3.2
sell_CDLDOJI,8,50.0,-1.2,-0.3,-2.5
```

#### 5.6.3 更新规则

- 全量更新（非续跑）时删除并重建 CSV
- 续跑模式追加写入
- None 夏普比率转为 0

---

## 6. AI 模型（CNN + XGBoost）

### 6.1 设计目标

为「盈湖」系统提供 **模糊 K 线形态识别 + 背景过滤** 能力，生成高胜率买卖信号。

| 指标 | 目标 |
|------|------|
| CNN 参数量 | < 50 万（实际 4.4K） |
| CNN 推理速度 | < 1ms/样本（CPU） |
| 融合方式 | CNN 输出作为 XGBoost 的特征 |
| 部署格式 | ONNX（支持 macOS MPS / Linux） |

### 6.2 整体架构

<!-- 建议在此处插入 AI 架构图：![AI 架构](./assets/ai-architecture.png) -->

```
┌─────────────────────────────────────────────────────┐
│                     输入                             │
│         20 根 K 线 OHLCV 序列                       │
└──────────────┬──────────────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────┐    ┌───────────────────┐
│ 1D-CNN   │    │  特征工程 features │
│ (阶段2)  │    │  131 维特征向量    │
│ 输出置信度│    │  (118 TA + 13 手工)│
│ 0~1     │    └─────────┬─────────┘
└────┬─────┘              │
     │                    │
     └────────┬───────────┘
              ▼
    ┌──────────────────┐
    │    XGBoost        │
    │  132 维输入       │
    │  (131 + CNN置信度) │
    └────────┬─────────┘
             │
             ▼
      最终信号概率 0~1
```

**两阶段渐进路线**：

| 阶段 | 输入 | 模型 | 目的 |
|------|------|------|------|
| 阶段1 | 131 维特征 | XGBoost | 快速验证 AI 是否有用 |
| 阶段2 | 131 + CNN 置信度 = 132 维 | XGBoost | CNN 锦上添花 |

> 📌 **如果阶段1效果已够好，阶段2可不做。**

### 6.3 样本采集

#### 6.3.1 标签规则

| 参数 | 值 | 说明 |
|------|------|------|
| N（未来窗口） | 5 个交易日 | 短线 |
| ATR 周期 | 100 | 中长期波动率 |
| ATR 倍数 | 1.0 | 正样本涨幅阈值 = ATR × 1.0 |
| 回撤约束 | > -8% | 软过滤（硬条件之一，非二次过滤） |

```
正样本(1): 未来5日涨幅 > ATR(100)×1.0 且 期间最大回撤 > -8%
负样本(0): 未来5日涨幅 ≤ 0
模糊样本: 丢弃（避免污染标签边界）
```

#### 6.3.2 滑动窗口

- 输入：20 根 K 线（不含触发日）
- 步长：1 个交易日
- 归一化：OHLC 除以末根 close；vol 窗口内 z-score

#### 6.3.3 数据量估算

```
5000 只 A 股 × 4000 个交易日 = 2000 万样本
压缩后约 4-6GB（Parquet）
```

### 6.4 特征工程（131 维）

#### 6.4.1 特征清单

| 类别 | 数量 | 说明 |
|------|------|------|
| TA-Lib 买入信号强度 | 59 | `max(arr, 0) / 100`，范围 [0, 1] |
| TA-Lib 卖出信号强度 | 59 | `abs(min(arr, 0)) / 100`，范围 [0, 1] |
| 手工特征 | 13 | ATR / RSI / 均线乖离 / 量比 等 |

#### 6.4.2 手工特征列表

| # | 特征 | 计算方式 |
|---|------|---------|
| 1 | atr_14_ratio | ATR(14) / close |
| 2 | atr_100_ratio | ATR(100) / close |
| 3 | rsi_6 | RSI(6) / 100 |
| 4 | rsi_14 | RSI(14) / 100 |
| 5 | ma5_bias | (close - MA5) / MA5 |
| 6 | ma10_bias | (close - MA10) / MA10 |
| 7 | ma20_bias | (close - MA20) / MA20 |
| 8 | ma60_bias | (close - MA60) / MA60 |
| 9 | vol_ratio_5 | vol / MA(vol, 5) |
| 10 | vol_ratio_20 | vol / MA(vol, 20) |
| 11 | close_position_20 | (close - low_20) / (high_20 - low_20) |
| 12 | return_5d | close / close_5d_ago - 1 |
| 13 | return_20d | close / close_20d_ago - 1 |

#### 6.4.3 关键设计：TA-Lib 拆分 buy/sell

每个形态拆为两个特征：
- `ta_buy_CDLHAMMER`：正信号强度
- `ta_sell_CDLHAMMER`：负信号强度

这样模型能区分方向，而不是只看到一个混合值。

### 6.5 XGBoost 阶段1

#### 6.5.1 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_estimators | 500 | 最大树数 |
| max_depth | 6 | 树最大深度 |
| learning_rate | 0.05 | 学习率 |
| subsample | 0.8 | 行采样率 |
| colsample_bytree | 0.8 | 列采样率 |
| patience | 30 | F1 早停耐心值 |
| scale_pos_weight | n_neg/n_pos | 类别不平衡处理 |

#### 6.5.2 数据划分

**按标的分组划分**（防止数据泄露）：
- 80% 标的 → 训练集
- 20% 标的 → 验证集
- 同一股票的数据不会同时出现在训练集和验证集

#### 6.5.3 评估指标

- Accuracy / Precision / Recall / F1 / AUC
- 特征重要性 Top-20

### 6.6 CNN 阶段2

#### 6.6.1 模型结构

```
Conv1d(5→32, k=3, pad=1) → BN → ReLU       # 感受野 3
Conv1d(32→32, k=3, pad=1) → BN → ReLU      # 感受野 5
AdaptiveAvgPool1d(1) → Flatten             # 全局平均池化
Linear(32→16) → ReLU → Dropout(0.3)
Linear(16→1)                               # 输出 logit
```

- 参数量：4.4K（远低于 50 万上限）
- 推理速度：CPU < 0.1ms/样本

#### 6.6.2 输入格式

- 形状：`(B, 5, 20)` = (batch, channels, length)
- channels = 5：open / high / low / close / vol
- 归一化：OHLC 除以末根 close；vol z-score

#### 6.6.3 ONNX 导出

```bash
python3 backend/ai/export_onnx.py
```

- 输出：`cnn_model.onnx`（约 20KB）
- 支持动态量化（INT8）
- 导出后自动做一致性验证（PyTorch vs ONNX 输出对比）

### 6.7 Model Server 推理服务

#### 6.7.1 架构

```
Flask Web 服务 (8765)  ←──用户──→  浏览器
         │
         │ HTTP
         ▼
AI Model Server (8766)  ←─ 独立进程
  ├─ XGBoost 推理器
  └─ CNN 推理器（ONNX 优先）
```

**为什么独立进程？**
- PyTorch / ONNX Runtime 占用 500MB+ 内存
- 避免信号跟踪多进程 fork 时每个子进程都加载模型
- 支持独立重启模型服务

#### 6.7.2 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/predict/xgb` | POST | 纯 XGBoost 推理 |
| `/predict/cnn` | POST | 纯 CNN 推理 |
| `/predict/fusion` | POST | CNN + XGBoost 融合推理 |
| `/predict/batch` | POST | 批量推理（回测用） |

#### 6.7.3 融合逻辑

```python
# 阶段1: CNN 未加载 → 纯 XGBoost
result = xgb.predict(features_131)

# 阶段2: CNN 已加载 → 软融合
cnn_prob = cnn.predict(kline_20)
xgb_prob = xgb.predict(features_131)
fused_prob = 0.7 * xgb_prob + 0.3 * cnn_prob
```

### 6.8 训练命令速查

#### 6.8.1 阶段1：XGBoost（无需 torch）

```bash
# 1. 采集样本（默认 50 只 A 股）
python3 backend/ai/batch_collect.py

# 2. 构建 XGBoost 数据集
python3 backend/ai/dataset_xgb.py \
  --sample_dir backend/ai/data/train \
  --output_dir backend/ai/data/xgb

# 3. 训练
python3 backend/ai/train_xgb.py \
  --data_dir backend/ai/data/xgb \
  --output_dir backend/ai/outputs
```

#### 6.8.2 阶段2：CNN（需 torch）

```bash
# 1. 安装依赖
pip install torch onnx onnxruntime

# 2. 训练（复用阶段1的样本）
python3 backend/ai/train.py

# 3. 导出 ONNX
python3 backend/ai/export_onnx.py
```

#### 6.8.3 启动推理服务

```bash
python3 backend/ai/model_server.py
# 访问 http://127.0.0.1:8766/health
```

#### 6.8.4 自定义参数

```bash
# 指定标的数和时间范围
python3 backend/ai/batch_collect.py --n_codes 100 --start_date 20150101

# 指定标的列表
python3 backend/ai/batch_collect.py --code_list 600519.SH,000001.SZ

# 调整 XGBoost 超参
python3 backend/ai/train_xgb.py \
  --data_dir backend/ai/data/xgb \
  --n_estimators 1000 \
  --max_depth 8 \
  --learning_rate 0.03
```

---

## 7. 部署与运维

### 7.1 环境准备

#### 7.1.1 系统要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| Python | 3.10+ | 3.13 |
| OS | macOS / Linux | macOS（开发）/ Linux（生产） |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 10GB | 50GB+（盈湖数据） |

#### 7.1.2 安装依赖

```bash
# 克隆仓库
git clone git@github.com:nickni06/candle_invest.git
cd candle_invest

# 创建虚拟环境
python3.13 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN
```

#### 7.1.3 TA-Lib 安装（C 库）

**macOS**：
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux**：
```bash
# 安装 C 库
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make && make install

# 安装 Python 包
pip install TA-Lib
```

#### 7.1.4 初始化盈湖

```bash
python3 backend/data/yinghu_db_init.py
```

### 7.2 启动服务

#### 7.2.1 主服务（Flask Web）

```bash
python3 backend/web/run_app.py
```

- 访问地址：http://127.0.0.1:8765
- **不要用 `python3 web_app.py`**（macOS 多进程冲突）

#### 7.2.2 AI 推理服务（可选）

```bash
python3 backend/ai/model_server.py
```

- 访问地址：http://127.0.0.1:8766
- 独立进程，不影响主服务

#### 7.2.3 信号更新（命令行）

```bash
python3 backend/signal/signal_update.py
```

#### 7.2.4 信号跟踪（命令行）

```bash
python3 backend/signal/signal_tracker.py
```

### 7.3 配置参考

#### 7.3.1 核心配置（`backend/config.py`）

```python
class Config:
    # 路径
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / '数据'
    YINGHU_DB_DIR = BASE_DIR / '盈湖'

    # Flask
    FLASK_HOST = '127.0.0.1'
    FLASK_PORT = 8765
    FLASK_DEBUG = False

    # 并发
    SCAN_WORKERS = 0              # 0=自动
    TRACKING_POOL_WORKERS = 0     # 0=自动
    TRACKING_PREFETCH_WORKERS = 4
    MAX_WORKERS = max(4, min(cpu_count(), 8))

    # 超时
    PREFETCH_TIMEOUT_SECONDS = 60
    WORKER_TIMEOUT_SECONDS = 30

    # AI
    AI_MODEL_SERVER_PORT = 8766
    AI_SAMPLE_SEQ_LEN = 20
    AI_SAMPLE_FORWARD_DAYS = 5
    AI_BUY_THRESHOLD = 0.6
```

#### 7.3.2 环境变量（`.env`）

```bash
TUSHARE_TOKEN=your_token_here
```

> ⚠️ **禁止在代码中硬编码 token。**

### 7.4 日志管理

#### 7.4.1 日志文件

| 文件 | 说明 | 保留期 |
|------|------|--------|
| `log/system_*.log` | 系统日志 | 30 天 |
| `log/*_tracking.log` | 跟踪日志 | 30 天 |
| `log/*_summary.json` | 跟踪汇总 | 30 天 |
| `log/flask.log` | Flask 日志 | 30 天 |

#### 7.4.2 自动清理

系统自动清理超过 30 天的日志文件。

#### 7.4.3 手动管理

前端提供「清空日志」和「清理旧日志」按钮。

#### 7.4.4 前端操作日志

所有前端操作通过 `/api/log/frontend` 端点记录（使用 `sendBeacon` 非阻塞上报）。

关键操作必埋点：Tab 切换、回测启动、跟踪操作、信号更新。

### 7.5 数据备份

#### 7.5.1 盈湖备份

- 每日自动备份到 `盈湖/backup/`
- 保留 30 天快照

#### 7.5.2 手动备份

```bash
# 打包盈湖 + 元数据
tar -czf yinghu_backup_$(date +%Y%m%d).tar.gz 盈湖/ 数据/kline_meta.db
```

#### 7.5.3 迁移到其他机器

```bash
# 源机器打包
tar -czf yinghu_data.tar.gz 盈湖/ 数据/kline_meta.db

# 目标机器解压
tar -xzf yinghu_data.tar.gz
```

### 7.6 常见问题

#### Q1: macOS 启动报 `AttributeError: module 'signal' has no attribute 'SIGINT'`

**原因**：`backend/signal/__init__.py` 空文件导致包名冲突，覆盖了 stdlib 的 `signal` 模块。

**解决**：删除 `backend/signal/__init__.py`（已修复）。

#### Q2: 信号跟踪报「无汇总文件」

**原因**：`signal_tracker.main()` 异常退出，未写 summary.json。

**解决**：检查 `log/*_tracking.log` 中的 traceback。

#### Q3: 信号跟踪报「跟踪日不在数据中」

**原因**：盈湖数据未覆盖跟踪日（如当天数据未入库）。

**解决**：`data_source.get_kline_df()` 已修复，会自动从外部拉取缺失数据并入库。

#### Q4: TA-Lib 安装失败

**macOS**：
```bash
brew install ta-lib
pip install TA-Lib --no-binary :all:
```

**Linux**：确认已安装 `gcc` 和 `make`，再按 7.1.3 节步骤安装 C 库。

#### Q5: 多进程在 macOS 崩溃

**原因**：akshare 的 libmini_racer（V8 引擎）在 fork 时不安全。

**解决**：`signal_tracker.py` 已设置 macOS 用 `forkserver`，无需手动处理。

#### Q6: 前端修改 HTML 后不生效

**原因**：Jinja2 模板缓存。

**解决**：`web_app.py` 已设置 `TEMPLATES_AUTO_RELOAD=True`，修改后刷新浏览器即可。

#### Q7: XGBoost 训练报 `ModuleNotFoundError: No module named 'xgboost'`

```bash
pip install xgboost scikit-learn
```

#### Q8: CNN 训练报 `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch onnx onnxruntime
```

阶段1（XGBoost）不需要 torch，可先跳过。

---

## 8. API 接口参考

### 8.1 主服务 API（端口 8765）

#### 8.1.1 策略回测

```
POST /api/backtest
```

**请求**：
```json
{
  "code": "600519.SH",
  "pattern": "CDLHAMMER",
  "pattern_type": "buy",
  "start_date": "20250101",
  "end_date": "20260727",
  "observe_day": 2,
  "cautious": false
}
```

**响应**：
```json
{
  "status": "success",
  "result": {
    "pattern": "CDLHAMMER",
    "trades": 15,
    "win_rate": 55.3,
    "return_pct": 2.1,
    "sharpe": 0.8,
    "hold_max_drawdown": -3.2,
    "trade_details": [...]
  }
}
```

#### 8.1.2 信号跟踪

```
POST /api/tracking/start
```

**请求**：
```json
{
  "track_date": "2026-07-27",
  "mode": "all",
  "cautious": false,
  "target_codes": []
}
```

**响应**（SSE 流式）：
```
data: {"progress": 10, "total": 50}

data: {"progress": 50, "total": 50}

data: {"status": "done", "summary": {...}}
```

#### 8.1.3 信号更新

```
POST /api/signal-update/start
```

#### 8.1.4 持仓管理

```
GET  /api/positions                 # 获取持仓列表
POST /api/positions/add              # 添加持仓
POST /api/positions/remove           # 删除持仓
```

#### 8.1.5 日志

```
POST /api/log/frontend               # 前端操作日志上报（sendBeacon）
GET  /api/logs                       # 获取系统日志
POST /api/logs/clear                 # 清空日志
POST /api/logs/clean-old              # 清理旧日志（>30天）
```

#### 8.1.6 盈湖管理

```
GET  /api/yinghu/stats               # 盈湖统计信息
POST /api/yinghu/refresh             # 增量更新
POST /api/yinghu/full-refresh        # 全量重跑
GET  /api/yinghu/quality-report      # 数据质量报告
```

### 8.2 AI 推理服务 API（端口 8766）

#### 8.2.1 健康检查

```
GET /health
```

**响应**：
```json
{
  "status": "ok",
  "xgb_loaded": true,
  "cnn_loaded": false,
  "n_features": 131
}
```

#### 8.2.2 XGBoost 推理

```
POST /predict/xgb
```

**请求**（二选一）：

方式 A - 特征向量：
```json
{
  "features": [0.0, 1.0, 0.0, ..., 0.5]
}
```

方式 B - K 线序列：
```json
{
  "kline": [[open, high, low, close, vol], ...]
}
```

**响应**：
```json
{
  "buy_prob": 0.72,
  "sell_prob": 0.28,
  "raw_prob": 0.72
}
```

#### 8.2.3 CNN 推理

```
POST /predict/cnn
```

**请求**：
```json
{
  "kline": [[open, high, low, close, vol], ...]
}
```

**响应**：
```json
{
  "buy_prob": 0.65,
  "sell_prob": 0.35,
  "raw_prob": 0.65
}
```

#### 8.2.4 融合推理（CNN + XGBoost）

```
POST /predict/fusion
```

**请求**：
```json
{
  "kline": [[open, high, low, close, vol], ...]
}
```

**响应**（阶段1，CNN 未加载）：
```json
{
  "buy_prob": 0.72,
  "sell_prob": 0.28,
  "raw_prob": 0.72,
  "fusion_mode": "xgb_only"
}
```

**响应**（阶段2，CNN 已加载）：
```json
{
  "buy_prob": 0.70,
  "sell_prob": 0.30,
  "raw_prob": 0.70,
  "xgb_prob": 0.72,
  "cnn_prob": 0.65,
  "fusion_mode": "soft"
}
```

#### 8.2.5 批量推理

```
POST /predict/batch
```

**请求**：
```json
{
  "features": [[...131维...], [...131维...], ...]
}
```

**响应**：
```json
{
  "buy_prob": [0.72, 0.15, ...],
  "sell_prob": [0.28, 0.85, ...],
  "raw_prob": [0.72, 0.15, ...]
}
```

### 8.3 调用示例

#### 8.3.1 Python 调用示例

```python
import requests

# XGBoost 推理
resp = requests.post('http://127.0.0.1:8766/predict/xgb', json={
    'kline': [
        [1850, 1860, 1845, 1855, 12000],
        [1855, 1870, 1850, 1865, 15000],
        # ... 共 20 根
    ]
})
print(resp.json())
# {'buy_prob': 0.72, 'sell_prob': 0.28, 'raw_prob': 0.72}
```

#### 8.3.2 curl 调用示例

```bash
# 健康检查
curl http://127.0.0.1:8766/health

# XGBoost 推理
curl -X POST http://127.0.0.1:8766/predict/xgb \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 1.0, 0.0, 0.5, ...]}'
```

---

## 9. 项目结构

```
candle_invest/
├── backend/                  # 后端
│   ├── config.py             # 全局配置
│   ├── data/                 # 数据层（盈湖、数据源、结果库）
│   ├── signal/               # 信号跟踪
│   ├── pattern/              # 形态扫描与回测
│   ├── strategy/             # 策略层
│   ├── ai/                   # AI 模型（CNN+XGB）
│   ├── web/                  # Flask Web 服务
│   ├── tracking/             # 旧版跟踪（已废弃）
│   └── utils/                # 工具
├── frontend/templates/       # 前端模板（Jinja2）
├── docs/                     # Wiki 文档
│   ├── WIKI.md               # 本文件（单文件综合 Wiki）
│   ├── README.md             # 文档导航
│   ├── architecture.md       # 系统架构（详版）
│   ├── data-source.md        # 数据源（详版）
│   ├── strategy-signals.md   # 策略与信号（详版）
│   ├── ai-model.md           # AI 模型（详版）
│   ├── deployment.md         # 部署运维（详版）
│   ├── api-reference.md      # API 接口（详版）
│   └── assets/               # 文档图片
│       └── .gitkeep
├── 盈湖/                     # K 线数据（不入库）
├── 数据/                     # 缓存/CSV/策略表现（不入库）
├── 结果库/                   # 回测结果缓存（不入库）
├── requirements.txt
└── README.md
```

---

## 10. 关键约束

- **TA-Lib 必须安装**（61 个 CDL 函数依赖）
- **配置集中在 config.py**，禁止硬编码 token/路径
- **A股数据优先读盈湖**，避免 Tushare API 限频
- **observe_day = 2**（全项目统一）
- **买入信号含 3% 固定止损**（卖出信号不设止损）
- **多进程 macOS 用 forkserver，Linux 用 fork**
- **结果库保留 1 年**，盈湖保留 24 个月在线
- **数据入库前必须通过列完整性、价格合理性、日期连续性校验**
- **信号过滤规则：所有匹配形态均输出**，不按胜率/收益率过滤
- **历史基准数据来源于 market_wide_pattern_stats.csv**
- **胜率 toFixed(0)，收益率 toFixed(2)**
- **Flask 启动 threaded=False, use_reloader=False**，避免 C 扩展多线程段错误
- **禁止 `from signal import xxx`**（避免与 `backend/signal/` 包冲突）
- **前端必须用 `python3 backend/web/run_app.py` 启动**

---

## 11. 文档配图说明

文档中引用的图片统一放在 `docs/assets/` 目录，使用相对路径：

```markdown
![系统架构图](./assets/architecture.png)
```

**支持的格式**：PNG / JPG / GIF / SVG。PDF 不入库（.gitignore 已排除）。

**建议配图清单**（可后续补充）：

| 图片 | 文件名 | 说明 |
|------|--------|------|
| 系统架构图 | `architecture.png` | 第 3.1 节 |
| 数据流图 | `data-flow.png` | 第 3.3 节 |
| AI 架构图 | `ai-architecture.png` | 第 6.2 节 |
| CNN 模型结构 | `cnn-model.png` | 第 6.6 节 |
| 信号跟踪流程 | `signal-tracking.png` | 第 5.4 节 |
| 前端界面截图 | `frontend-*.png` | 按需 |

**插入图片示例**：

```markdown
### 3.1 整体架构图

![系统架构图](./assets/architecture.png)

系统采用分层架构...
```

---

> 📖 **分章节详版文档**：如需查看某章节的完整细节，请前往 `docs/` 目录下对应的单章节文件（如 `architecture.md`、`ai-model.md` 等）。
>
> 📝 **文档维护**：本 Wiki 与分章节文档同步维护。新增功能时，请同步更新本文件和对应章节文件。
