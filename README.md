# 量化交易策略回测系统

基于 Backtrader + TA-Lib 的 A 股量化交易策略研究与回测系统，支持 60+ 种 K 线形态识别、信号跟踪、组合回测和持仓管理。

## 快速开始

```bash
# 启动 Web 服务
python3 backend/web/run_app.py

# 访问地址
# http://127.0.0.1:8765
```

## 目录结构

```
项目根目录/
├── backend/                    # 后端代码（Python）
│   ├── config.py               # 全局配置（路径、参数、并发限制等）
│   ├── web/                    # Web 服务层
│   │   ├── run_app.py          # 启动入口
│   │   └── web_app.py          # Flask 路由与 API
│   ├── data/                   # 数据层
│   │   ├── data_source.py      # 统一数据源（akshare→tushare→腾讯→新浪）
│   │   ├── yinghu_db.py        # 盈湖数据库访问层（Parquet + SQLite）
│   │   ├── yinghu_db_init.py   # 盈湖初始化脚本
│   │   ├── result_db.py        # 结果库缓存层（SHA1 键 + 365 天保留）
│   │   └── data_refresh.py     # 集中数据补全模块
│   ├── strategy/               # 策略层
│   │   ├── main.py             # 策略回测入口（Backtrader Cerebro）
│   │   ├── baseStrategy.py     # 策略基类（仓位、止损、持有期）
│   │   ├── strategy.py         # 各策略实现（突破、均线等）
│   │   ├── patternStrategy.py  # 形态策略（带谨慎模式过滤）
│   │   ├── strategy_config.py  # 个股策略配置管理
│   │   ├── numba_backtest.py   # Numba 加速回测核心
│   │   ├── indicator.py        # 自定义 Backtrader 指标
│   │   ├── cautious_mode.py    # 谨慎模式额外条件
│   │   ├── portfolio_backtest.py       # 多策略组合回测
│   │   ├── renew_strategy_performance.py  # 策略绩效刷新
│   │   └── stockPoolStrategy.py        # 股票池策略
│   ├── pattern/                # 形态识别层
│   │   ├── pattern_data.py     # 形态字典（中文名、描述、买卖分类）
│   │   ├── pattern_names.py    # 61 个 CDL 形态名列表
│   │   └── pattern_scan.py     # TA-Lib 形态扫描与向量化回测
│   ├── signal/                 # 信号层
│   │   ├── signal_tracker.py   # 信号跟踪调度器（多进程并行）
│   │   ├── signal_update.py    # 信号绩效批量更新
│   │   ├── signal_utils.py     # 信号工具函数
│   │   ├── signal_templates.py # 信号更新模板管理
│   │   ├── signal_quality.py   # 数据质量报告生成
│   │   └── strategy_signals.py # 纯 TA-Lib 信号计算（无 Backtrader）
│   ├── tracking/               # 跟踪工具层
│   │   ├── tracking.py         # 旧版单标的跟踪（兼容保留）
│   │   └── tools.py            # 日志、数据存取、Tushare 重试
│   └── utils/                  # 通用工具
│       └── convert_csv_to_parquet.py  # CSV 转 Parquet 工具
├── frontend/                   # 前端代码
│   └── templates/
│       └── index.html          # 单页应用（仪表盘、信号跟踪、回测等）
├── tests/                      # 测试与回归
│   ├── regression_test.py      # 全流程回归测试
│   ├── test_numba_backtest.py # Numba 加速对拍测试
│   ├── test_perf.py            # 性能测试
│   └── find_baseline_cases.py  # 回归基准用例扫描
├── docs/                       # 文档资料
│   ├── 海龟交易法则.pdf
│   └── 兴业银行_全形态回测结果.csv
├── 盈湖/                       # 盈湖数据库（全市场日 K 存储）
│   ├── yinghu.db               # 元数据库（SQLite + WAL）
│   └── kline/                  # K 线数据（按月分区 Parquet）
├── 结果库/                     # 结果缓存库
│   └── result.db
├── 数据/                       # 业务数据
│   ├── A股/                    # A 股数据与策略表现
│   ├── 信号更新/               # 信号更新记录与模板
│   ├── 持仓/                   # 持仓记录
│   └── 证券列表/               # 债券/ETF/Fund 列表
├── 策略表现/                   # 策略绩效 CSV
├── log/                        # 日志（按天分文件，保留 30 天）
└── 方案/                       # 策略方案文档
```

## 核心功能

| 功能 | 入口 | 说明 |
|------|------|------|
| 仪表盘 | 首页 | 关注信号卡片 + 持仓管理 |
| 信号跟踪 | 信号跟踪 Tab | 多标的并行扫描 K 线形态，生成买卖信号 |
| 策略回测 | 策略回测 Tab | 单形态/组合形态回测，输出绩效指标 |
| 信号更新 | 信号更新 Tab | 全市场批量回测，生成策略表现 CSV |
| 策略配置 | 策略配置 Tab | 个股定向形态配置 |
| 盈湖管理 | 盈湖 Tab | 数据库初始化、统计、结果库清理 |

## 技术栈

- **后端**: Python 3.13, Flask, Backtrader, TA-Lib, Numba
- **数据存储**: Parquet（按月分区）, SQLite（WAL 模式）
- **数据源**: akshare → tushare → 腾讯财经 → 新浪财经（多级兜底）
- **前端**: HTML + Bootstrap 5 + 原生 JS（SSE 实时进度推送）

## 关键设计

- **盈湖数据库**: 全市场股票日 K 统一存储（2010 年至今），按月分区 Parquet，SQLite 索引快速查询
- **结果库**: SHA1 缓存键（标的+策略+日期范围+参数），源数据变更自动失效
- **多进程**: macOS 使用 forkserver，4-8 进程并行，单标超时 60s
- **3% 止损**: 买入信号自带固定止损规则，卖出信号不止损
- **断点续跑**: 信号更新支持 current_task.json 记录已完成任务
