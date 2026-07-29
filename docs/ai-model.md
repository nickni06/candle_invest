# AI 模型（CNN + XGBoost 融合买卖点）

## 目录

- [1. 设计目标](#1-设计目标)
- [2. 整体架构](#2-整体架构)
- [3. 样本采集](#3-样本采集)
- [4. 特征工程](#4-特征工程)
- [5. XGBoost 阶段1](#5-xgboost-阶段1)
- [6. CNN 阶段2](#6-cnn-阶段2)
- [7. Model Server 推理服务](#7-model-server-推理服务)
- [8. 训练命令速查](#8-训练命令速查)

---

## 1. 设计目标

为「盈湖」系统提供 **模糊 K 线形态识别 + 背景过滤** 能力，生成高胜率买卖信号。

| 指标 | 目标 |
|------|------|
| CNN 参数量 | < 50 万（实际 4.4K） |
| CNN 推理速度 | < 1ms/样本（CPU） |
| 融合方式 | CNN 输出作为 XGBoost 的特征 |
| 部署格式 | ONNX（支持 macOS MPS / Linux） |

## 2. 整体架构

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

### 两阶段渐进路线

| 阶段 | 输入 | 模型 | 目的 |
|------|------|------|------|
| 阶段1 | 131 维特征 | XGBoost | 快速验证 AI 是否有用 |
| 阶段2 | 131 + CNN 置信度 = 132 维 | XGBoost | CNN 锦上添花 |

**如果阶段1效果已够好，阶段2可不做。**

## 3. 样本采集

### 3.1 标签规则

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

### 3.2 滑动窗口

- 输入：20 根 K 线（不含触发日）
- 步长：1 个交易日
- 归一化：OHLC 除以末根 close；vol 窗口内 z-score

### 3.3 数据量估算

```
5000 只 A 股 × 4000 个交易日 = 2000 万样本
压缩后约 4-6GB（Parquet）
```

## 4. 特征工程

### 4.1 特征清单（131 维）

| 类别 | 数量 | 说明 |
|------|------|------|
| TA-Lib 买入信号强度 | 59 | `max(arr, 0) / 100`，范围 [0, 1] |
| TA-Lib 卖出信号强度 | 59 | `abs(min(arr, 0)) / 100`，范围 [0, 1] |
| 手工特征 | 13 | ATR / RSI / 均线乖离 / 量比 等 |

### 4.2 手工特征列表

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

### 4.3 关键设计：TA-Lib 拆分 buy/sell

每个形态拆为两个特征：
- `ta_buy_CDLHAMMER`：正信号强度
- `ta_sell_CDLHAMMER`：负信号强度

这样模型能区分方向，而不是只看到一个混合值。

## 5. XGBoost 阶段1

### 5.1 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_estimators | 500 | 最大树数 |
| max_depth | 6 | 树最大深度 |
| learning_rate | 0.05 | 学习率 |
| subsample | 0.8 | 行采样率 |
| colsample_bytree | 0.8 | 列采样率 |
| patience | 30 | F1 早停耐心值 |
| scale_pos_weight | n_neg/n_pos | 类别不平衡处理 |

### 5.2 数据划分

**按标的分组划分**（防止数据泄露）：
- 80% 标的 → 训练集
- 20% 标的 → 验证集
- 同一股票的数据不会同时出现在训练集和验证集

### 5.3 评估指标

- Accuracy / Precision / Recall / F1 / AUC
- 特征重要性 Top-20

## 6. CNN 阶段2

### 6.1 模型结构

```
Conv1d(5→32, k=3, pad=1) → BN → ReLU       # 感受野 3
Conv1d(32→32, k=3, pad=1) → BN → ReLU      # 感受野 5
AdaptiveAvgPool1d(1) → Flatten             # 全局平均池化
Linear(32→16) → ReLU → Dropout(0.3)
Linear(16→1)                               # 输出 logit
```

- 参数量：4.4K（远低于 50 万上限）
- 推理速度：CPU < 0.1ms/样本

### 6.2 输入格式

- 形状：`(B, 5, 20)` = (batch, channels, length)
- channels = 5：open / high / low / close / vol
- 归一化：OHLC 除以末根 close；vol z-score

### 6.3 ONNX 导出

```bash
python3 backend/ai/export_onnx.py
```

- 输出：`cnn_model.onnx`（约 20KB）
- 支持动态量化（INT8）
- 导出后自动做一致性验证（PyTorch vs ONNX 输出对比）

## 7. Model Server 推理服务

### 7.1 架构

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

### 7.2 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/predict/xgb` | POST | 纯 XGBoost 推理 |
| `/predict/cnn` | POST | 纯 CNN 推理 |
| `/predict/fusion` | POST | CNN + XGBoost 融合推理 |
| `/predict/batch` | POST | 批量推理（回测用） |

### 7.3 融合逻辑

```python
# 阶段1: CNN 未加载 → 纯 XGBoost
result = xgb.predict(features_131)

# 阶段2: CNN 已加载 → 软融合
cnn_prob = cnn.predict(kline_20)
xgb_prob = xgb.predict(features_131)
fused_prob = 0.7 * xgb_prob + 0.3 * cnn_prob
```

## 8. 训练命令速查

### 阶段1：XGBoost（无需 torch）

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

### 阶段2：CNN（需 torch）

```bash
# 1. 安装依赖
pip install torch onnx onnxruntime

# 2. 训练（复用阶段1的样本）
python3 backend/ai/train.py

# 3. 导出 ONNX
python3 backend/ai/export_onnx.py
```

### 启动推理服务

```bash
python3 backend/ai/model_server.py
# 访问 http://127.0.0.1:8766/health
```

### 自定义参数

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
