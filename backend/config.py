import os
from pathlib import Path

# backend/config.py → 项目根目录
BASE_DIR = Path(__file__).parent.parent

class Config:
    BASE_DIR = Path(__file__).parent.parent
    # Tushare Token 必须从环境变量读取，禁止硬编码
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')

    DATA_DIR = BASE_DIR / '数据'
    A_MARKET_DIR = DATA_DIR / 'A股'
    INDEX_DIR = DATA_DIR / '指数'
    ETF_DIR = DATA_DIR / 'ETF'

    DAILY_TRACKING_A_DIR = A_MARKET_DIR / '每日跟踪'
    DAILY_TRACKING_INDEX_DIR = INDEX_DIR / '每日跟踪'
    DAILY_TRACKING_ETF_DIR = ETF_DIR / '每日跟踪'
    STOCK_PERFORMANCE_DIR = A_MARKET_DIR / '个股策略表现'
    INDEX_PERFORMANCE_DIR = INDEX_DIR / '个股策略表现'
    ETF_PERFORMANCE_DIR = ETF_DIR / '个股策略表现'
    STOCK_DATA_FILE = A_MARKET_DIR / 'stock_data.csv'

    TRAIN_DATA_A_DIR = A_MARKET_DIR / '训练测试库/训练'
    TEST_DATA_A_DIR = A_MARKET_DIR / '训练测试库/测试'
    TRAIN_DATA_INDEX_DIR = INDEX_DIR / '训练测试库/训练'
    TEST_DATA_INDEX_DIR = INDEX_DIR / '训练测试库/测试'
    TRAIN_DATA_ETF_DIR = ETF_DIR / '训练测试库/训练'
    TEST_DATA_ETF_DIR = ETF_DIR / '训练测试库/测试'

    LOG_DIR = BASE_DIR / 'log'
    STRATEGY_DICT_FILE = BASE_DIR / '策略表现/策略字典.csv'

    SIGNAL_UPDATE_DIR = DATA_DIR / '信号更新'
    SIGNAL_UPDATE_TASK_FILE = SIGNAL_UPDATE_DIR / 'current_task.json'
    SIGNAL_UPDATE_HISTORY_FILE = SIGNAL_UPDATE_DIR / 'history.json'
    SIGNAL_UPDATE_QUALITY_FILE = SIGNAL_UPDATE_DIR / 'quality_report.json'

    # 本地 K 线元数据索引（SQLite），用于加速数据覆盖范围检查
    DATA_META_DB = DATA_DIR / 'kline_meta.db'

    # 全市场形态统计文件：基于信号更新产出的策略表现 CSV 聚合生成
    # 与 策略字典.csv 同目录，便于统一管理策略表现相关产物
    MARKET_WIDE_STATS_FILE = BASE_DIR / '策略表现' / 'market_wide_pattern_stats.csv'

    # 关注信号持久化文件（JSON 格式，存储用户从信号跟踪页关注的信号）
    WATCHLIST_SIGNALS_FILE = DATA_DIR / 'watchlist_signals.json'

    POSITION_DIR = DATA_DIR / '持仓'
    POSITION_FILE = POSITION_DIR / 'positions.csv'  # 持仓管理CSV

    # 扩展证券列表缓存（ETF/可转债/基金），用于持仓搜索，24小时过期
    SECURITY_LIST_DIR = DATA_DIR / '证券列表'

    # 个股策略配置（定向跟踪用）
    STRATEGY_CONFIG_DIR = DATA_DIR / '策略配置'
    STRATEGY_CONFIG_FILE = STRATEGY_CONFIG_DIR / 'stock_strategy_config.csv'

    DEFAULT_CASH = 100000000
    DEFAULT_COMMISSION = 0.0001

    # 形态扫描并行回测的进程数（0=自动，按CPU核心数；1=串行）
    SCAN_WORKERS = 0
    # 跟踪任务并行回测进程数（0=自动；1=串行；建议 4-8）
    TRACKING_POOL_WORKERS = 0
    # 跟踪任务父进程并行预拉取进程数（akshare JS 引擎进程隔离，可并行；4-8 为宜）
    TRACKING_PREFETCH_WORKERS = 4
    # 全局并发 worker 上限，防止 CPU/内存资源耗尽
    MAX_WORKERS = max(4, min((os.cpu_count() or 4), 8))
    # 预拉取单标超时（秒），网络拉取通常比计算慢
    PREFETCH_TIMEOUT_SECONDS = 60
    # 单 worker 任务超时（秒），防止永久挂起
    WORKER_TIMEOUT_SECONDS = 30
    # 信号跟踪默认向前回溯的自然日数（AI 评分需 20 根 K 线窗口，35 天约 24 个交易日）
    TRACKING_LOOKBACK_DAYS = 35

    # ============================================================================
    # 盈湖（Yinghu DB）：全市场股票日K的统一存储，按月分区 Parquet
    # ============================================================================
    # 盈湖根目录
    YINGHU_DB_DIR = BASE_DIR / '盈湖'
    # 盈湖 K 线数据根目录（按板块/代码/月份分区）
    YINGHU_DB_KLINE_DIR = YINGHU_DB_DIR / 'kline'
    # 盈湖元数据库（SQLite + WAL）
    YINGHU_DB_META = YINGHU_DB_DIR / 'yinghu.db'
    # 盈湖备份目录（每日快照，保留 30 天）
    YINGHU_DB_BACKUP_DIR = YINGHU_DB_DIR / 'backup'
    # 盈湖数据起始日期（2010-01-01 起的全市场数据）
    YINGHU_DB_START_DATE = '20100101'
    # 盈湖归档阈值（月份超过该数量后归档压缩）
    YINGHU_DB_ARCHIVE_MONTHS = 24

    # ============================================================================
    # 结果库（Result DB）：策略回测/信号跟踪结果缓存，保留 1 年
    # ============================================================================
    RESULT_DB_DIR = BASE_DIR / '结果库'
    # 结果库索引数据库（SQLite + WAL）
    RESULT_DB_META = RESULT_DB_DIR / 'result.db'
    # 结果库数据目录（按 hash 前两位分桶，避免单目录文件过多）
    RESULT_DB_DATA_DIR = RESULT_DB_DIR / 'data'
    # 结果库保留天数（超过自动清理）
    RESULT_DB_RETENTION_DAYS = 365

    FLASK_HOST = '127.0.0.1'
    FLASK_PORT = 8765
    FLASK_DEBUG = False

    # ============================================================================
    # AI 模型（CNN + XGBoost 融合买卖点信号）
    # ============================================================================
    # AI 模块根目录（backend/ai/）
    AI_DIR = BASE_DIR / 'backend' / 'ai'
    # AI 训练样本目录（按 <code>_samples.npy / <code>_labels.npy 分文件存储）
    AI_SAMPLE_DIR = AI_DIR / 'data' / 'train'
    # AI 模型输出目录（.pth / .onnx / .json / .pkl）
    AI_MODEL_DIR = AI_DIR / 'outputs'
    # AI 推理服务端口（与 Flask 主服务独立，避免 PyTorch 占用主进程内存）
    AI_MODEL_SERVER_PORT = 8766
    # AI 样本默认参数（与 sample_collector.py 保持一致）
    AI_SAMPLE_SEQ_LEN = 20          # 输入 K 线根数
    AI_SAMPLE_FORWARD_DAYS = 5       # 未来窗口 N
    AI_SAMPLE_ATR_PERIOD = 100       # ATR 计算周期
    AI_SAMPLE_ATR_MULTIPLIER = 1.0   # 正样本涨幅阈值 = ATR × 1.0
    AI_SAMPLE_MAX_DRAWDOWN = -0.08   # 回撤过滤阈值（软过滤，不作硬标签）
    # AI 信号阈值（超过才触发买卖）
    AI_BUY_THRESHOLD = 0.6
    AI_SELL_THRESHOLD = 0.6
    # AI 采样默认标的数（从 stock_data.csv 随机抽取，0=全部 A 股）
    AI_SAMPLE_DEFAULT_CODES = 50
    # AI 样本默认时间范围
    AI_SAMPLE_START_DATE = '20100101'
    AI_SAMPLE_END_DATE = '20260727'

config = Config()
