"""盈湖交易系统 - AI 模糊形态识别模块。

提供基于 1D-CNN 的 K 线模糊形态识别能力，作为 TA-Lib 标准 60+ 形态的补充：
- cnn_model: 轻量级 1D-CNN 模型（输入 20×5，输出 0~1 置信度）
- sample_collector: 样本采集（盈湖 K 线 → 20×5 窗口 + ATR 标签）
- dataset: PyTorch Dataset + 训练/验证划分
- train: 训练脚本（MPS/CPU 自适应）
- inference: 推理示例
- export_onnx: ONNX 导出 + 动态量化
"""
