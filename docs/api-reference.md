# API 接口参考

## 目录

- [1. 主服务 API（端口 8765）](#1-主服务-api端口-8765)
- [2. AI 推理服务 API（端口 8766）](#2-ai-推理服务-api端口-8766)
- [3. 请求/响应示例](#3-请求响应示例)

---

## 1. 主服务 API（端口 8765）

### 1.1 策略回测

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

### 1.2 信号跟踪

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

### 1.3 信号更新

```
POST /api/signal-update/start
```

### 1.4 持仓管理

```
GET  /api/positions                 # 获取持仓列表
POST /api/positions/add              # 添加持仓
POST /api/positions/remove           # 删除持仓
```

### 1.5 日志

```
POST /api/log/frontend               # 前端操作日志上报（sendBeacon）
GET  /api/logs                       # 获取系统日志
POST /api/logs/clear                 # 清空日志
POST /api/logs/clean-old             # 清理旧日志（>30天）
```

### 1.6 盈湖管理

```
GET  /api/yinghu/stats               # 盈湖统计信息
POST /api/yinghu/refresh             # 增量更新
POST /api/yinghu/full-refresh        # 全量重跑
GET  /api/yinghu/quality-report      # 数据质量报告
```

## 2. AI 推理服务 API（端口 8766）

### 2.1 健康检查

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

### 2.2 XGBoost 推理

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

### 2.3 CNN 推理

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

### 2.4 融合推理（CNN + XGBoost）

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

### 2.5 批量推理

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

## 3. 请求/响应示例

### 3.1 Python 调用示例

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

### 3.2 curl 调用示例

```bash
# 健康检查
curl http://127.0.0.1:8766/health

# XGBoost 推理
curl -X POST http://127.0.0.1:8766/predict/xgb \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 1.0, 0.0, 0.5, ...]}'
```
