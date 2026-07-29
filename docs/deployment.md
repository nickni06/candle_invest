# 部署与运维

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 启动服务](#2-启动服务)
- [3. 配置参考](#3-配置参考)
- [4. 日志管理](#4-日志管理)
- [5. 数据备份](#5-数据备份)
- [6. 常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 系统要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| Python | 3.10+ | 3.13 |
| OS | macOS / Linux | macOS（开发）/ Linux（生产） |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 10GB | 50GB+（盈湖数据） |

### 1.2 安装依赖

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

### 1.3 TA-Lib 安装（C 库）

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

### 1.4 初始化盈湖

```bash
python3 backend/data/yinghu_db_init.py
```

## 2. 启动服务

### 2.1 主服务（Flask Web）

```bash
python3 backend/web/run_app.py
```

- 访问地址：http://127.0.0.1:8765
- **不要用 `python3 web_app.py`**（macOS 多进程冲突）

### 2.2 AI 推理服务（可选）

```bash
python3 backend/ai/model_server.py
```

- 访问地址：http://127.0.0.1:8766
- 独立进程，不影响主服务

### 2.3 信号更新（命令行）

```bash
python3 backend/signal/signal_update.py
```

### 2.4 信号跟踪（命令行）

```bash
python3 backend/signal/signal_tracker.py
```

## 3. 配置参考

### 3.1 核心配置（`backend/config.py`）

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

### 3.2 环境变量（`.env`）

```bash
TUSHARE_TOKEN=your_token_here
```

**禁止在代码中硬编码 token。**

## 4. 日志管理

### 4.1 日志文件

| 文件 | 说明 | 保留期 |
|------|------|--------|
| `log/system_*.log` | 系统日志 | 30 天 |
| `log/*_tracking.log` | 跟踪日志 | 30 天 |
| `log/*_summary.json` | 跟踪汇总 | 30 天 |
| `log/flask.log` | Flask 日志 | 30 天 |

### 4.2 自动清理

系统自动清理超过 30 天的日志文件。

### 4.3 手动管理

前端提供「清空日志」和「清理旧日志」按钮。

### 4.4 前端操作日志

所有前端操作通过 `/api/log/frontend` 端点记录（使用 `sendBeacon` 非阻塞上报）。

关键操作必埋点：Tab 切换、回测启动、跟踪操作、信号更新。

## 5. 数据备份

### 5.1 盈湖备份

- 每日自动备份到 `盈湖/backup/`
- 保留 30 天快照

### 5.2 手动备份

```bash
# 打包盈湖 + 元数据
tar -czf yinghu_backup_$(date +%Y%m%d).tar.gz 盈湖/ 数据/kline_meta.db
```

### 5.3 迁移到其他机器

```bash
# 源机器打包
tar -czf yinghu_data.tar.gz 盈湖/ 数据/kline_meta.db

# 目标机器解压
tar -xzf yinghu_data.tar.gz
```

## 6. 常见问题

### Q1: macOS 启动报 `AttributeError: module 'signal' has no attribute 'SIGINT'`

**原因**：`backend/signal/__init__.py` 空文件导致包名冲突，覆盖了 stdlib 的 `signal` 模块。

**解决**：删除 `backend/signal/__init__.py`（已修复）。

### Q2: 信号跟踪报「无汇总文件」

**原因**：`signal_tracker.main()` 异常退出，未写 summary.json。

**解决**：检查 `log/*_tracking.log` 中的 traceback。

### Q3: 信号跟踪报「跟踪日不在数据中」

**原因**：盈湖数据未覆盖跟踪日（如当天数据未入库）。

**解决**：`data_source.get_kline_df()` 已修复，会自动从外部拉取缺失数据并入库。

### Q4: TA-Lib 安装失败

**macOS**：
```bash
brew install ta-lib
pip install TA-Lib --no-binary :all:
```

**Linux**：确认已安装 `gcc` 和 `make`，再按 1.3 节步骤安装 C 库。

### Q5: 多进程在 macOS 崩溃

**原因**：akshare 的 libmini_racer（V8 引擎）在 fork 时不安全。

**解决**：`signal_tracker.py` 已设置 macOS 用 `forkserver`，无需手动处理。

### Q6: 前端修改 HTML 后不生效

**原因**：Jinja2 模板缓存。

**解决**：`web_app.py` 已设置 `TEMPLATES_AUTO_RELOAD=True`，修改后刷新浏览器即可。

### Q7: XGBoost 训练报 `ModuleNotFoundError: No module named 'xgboost'`

```bash
pip install xgboost scikit-learn
```

### Q8: CNN 训练报 `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch onnx onnxruntime
```

阶段1（XGBoost）不需要 torch，可先跳过。
