# Candle Invest Wiki

> 基于 Backtrader + TA-Lib 的 A 股量化交易系统知识库

> 📘 **单文件综合 Wiki**：[WIKI.md](./WIKI.md) — 整合全部章节，顶部带完整目录锚点快速定位，支持图片插入。一个文件看全部。

## 📚 文档导航

| 章节 | 文件 | 说明 |
|------|------|------|
| 🏗️ 系统架构 | [architecture.md](./architecture.md) | 整体架构、模块分层、数据流 |
| 📊 数据源 | [data-source.md](./data-source.md) | 盈湖、数据源优先级、缓存策略 |
| 📈 策略与信号 | [strategy-signals.md](./strategy-signals.md) | 形态识别、信号生成、回测 |
| 🤖 AI 模型 | [ai-model.md](./ai-model.md) | CNN + XGBoost 融合买卖点 |
| 🚀 部署运维 | [deployment.md](./deployment.md) | 启动、配置、日志、常见问题 |
| 🔌 API 接口 | [api-reference.md](./api-reference.md) | REST API 完整参考 |

## 🚀 快速开始

### 启动 Web 服务

```bash
cd backend
python3 web/run_app.py
# 访问 http://127.0.0.1:8765
```

### 启动 AI 推理服务（可选）

```bash
python3 backend/ai/model_server.py
# 访问 http://127.0.0.1:8766/health
```

### 训练 AI 模型

```bash
# 阶段1：XGBoost
python3 backend/ai/batch_collect.py                          # 采集样本
python3 backend/ai/dataset_xgb.py --sample_dir backend/ai/data/train --output_dir backend/ai/data/xgb
python3 backend/ai/train_xgb.py --data_dir backend/ai/data/xgb --output_dir backend/ai/outputs

# 阶段2：CNN（需 torch）
python3 backend/ai/train.py
python3 backend/ai/export_onnx.py
```

## 🗂️ 项目结构速览

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
│   ├── README.md             # 本文件
│   ├── architecture.md
│   ├── data-source.md
│   ├── strategy-signals.md
│   ├── ai-model.md
│   ├── deployment.md
│   ├── api-reference.md
│   └── assets/               # 文档图片
├── 盈湖/                     # K 线数据（不入库）
├── 数据/                     # 缓存/CSV/策略表现（不入库）
├── 结果库/                   # 回测结果缓存（不入库）
├── requirements.txt
└── README.md
```

## 📌 关键约束

- **TA-Lib 必须安装**（61 个 CDL 函数依赖）
- **配置集中在 config.py**，禁止硬编码 token/路径
- **A股数据优先读盈湖**，避免 Tushare API 限频
- **observe_day = 2**（全项目统一）
- **买入信号含 3% 固定止损**（卖出信号不设止损）
- **多进程 macOS 用 fork，Linux 用 fork**
- **结果库保留 1 年**，盈湖保留 24 个月在线

## 🖼️ 文档配图说明

文档中引用的图片统一放在 `docs/assets/` 目录，使用相对路径：

```markdown
![系统架构图](./assets/architecture.png)
```

支持的格式：PNG / JPG / GIF / SVG。PDF 不入库（.gitignore 已排除）。
