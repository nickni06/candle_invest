# 系统架构

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 模块分层](#2-模块分层)
- [3. 数据流](#3-数据流)
- [4. 并发模型](#4-并发模型)
- [5. 缓存策略](#5-缓存策略)
- [6. 关键设计决策](#6-关键设计决策)

---

## 1. 整体架构

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

## 2. 模块分层

### 2.1 数据层 (`backend/data/`)

| 模块 | 职责 |
|------|------|
| `data_source.py` | 统一数据源：akshare → tushare → 腾讯 → 新浪 |
| `yinghu_db.py` | 盈湖访问层：Parquet 读写、元数据管理、覆盖检查 |
| `yinghu_db_init.py` | 盈湖初始化（建表、建目录） |
| `result_db.py` | 结果库缓存（SHA1 键，365 天保留） |
| `data_refresh.py` | 集中数据补全模块 |

### 2.2 策略层 (`backend/pattern/`, `backend/strategy/`)

| 模块 | 职责 |
|------|------|
| `pattern_scan.py` | 向量化回测引擎 + 形态信号计算 |
| `patternStrategy.py` | Backtrader 策略封装（旧版，保留兼容） |
| `numba_backtest.py` | Numba 加速版回测 |
| `portfolio_backtest.py` | 组合回测 |

### 2.3 信号层 (`backend/signal/`)

| 模块 | 职责 |
|------|------|
| `signal_tracker.py` | 信号跟踪调度（多进程并行） |
| `strategy_signals.py` | 信号计算核心（纯 TA-Lib） |
| `signal_update.py` | 信号更新（批量回测） |
| `signal_utils.py` | 形态列表、中文名、描述 |

### 2.4 AI 层 (`backend/ai/`)

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

### 2.5 Web 层 (`backend/web/`)

| 模块 | 职责 |
|------|------|
| `run_app.py` | 启动入口 |
| `web_app.py` | Flask 路由与 API（主服务 8765） |

## 3. 数据流

### 3.1 信号跟踪流程

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

### 3.2 策略回测流程

```
用户选择标的 + 形态 + 时间范围
  ↓
web_app.py → pattern_scan.run_single_pattern()
  ├─ 加载 K 线（盈湖优先，lru_cache + mtime 签名）
  ├─ 计算 TA-Lib 形态信号
  ├─ 向量化回测（pandas/numpy，可选 numba 加速）
  └─ 返回：交易明细 + 胜率/收益/夏普/回撤
```

### 3.3 AI 训练流程

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

## 4. 并发模型

### 4.1 多进程启动方式

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

### 4.2 进程池配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SCAN_WORKERS` | 0（自动） | 形态扫描并行进程数 |
| `TRACKING_POOL_WORKERS` | 0（自动） | 信号跟踪并行进程数 |
| `TRACKING_PREFETCH_WORKERS` | 4 | 预拉取进程数 |
| `MAX_WORKERS` | max(4, min(CPU, 8)) | 全局上限 |

## 5. 缓存策略

### 5.1 数据缓存（3 层）

```
进程内 lru_cache  →  文件级 mtime 签名  →  盈湖 Parquet
   (最快)              (跨进程共享)          (持久化)
```

- `_load_raw_dataframe_cached()`：lru_cache(64)，key 含 mtime 签名
- 文件变更自动失效（mtime+size 变化 → 签名变化 → cache miss）

### 5.2 结果库缓存

- 键：`SHA1(code + 策略名 + 日期范围 + observe_day + cautious)`
- 保留期：365 天
- 源数据/策略变更 → 自动失效

## 6. 关键设计决策

### 6.1 为什么放弃 Backtrader 的 next()？

| 问题 | Backtrader | 向量化回测 |
|------|-----------|-----------|
| 性能 | 慢（逐根 K 线回调） | 快 10x+（numpy 向量化） |
| 多进程安全 | 不安全（状态机） | 安全（无状态） |
| 日志解析 | 依赖正则解析日志 | 直接返回结构化 dict |

### 6.2 为什么 AI 服务独立进程？

- PyTorch / ONNX Runtime 占用 500MB+ 内存
- 避免每个子进程都加载模型（7 workers × 500MB = 3.5GB）
- Model Server 加载一次，worker 通过 HTTP 调用

### 6.3 为什么 CNN 模型只有 4.4K 参数？

- A 股形态本质是局部模式（3-5 根 K 线）
- 20 根输入已足够覆盖上下文
- 2 层 Conv1d（感受野 5）+ 全局池化，参数量刚好
- 大模型容易过拟合 A 股的噪声
