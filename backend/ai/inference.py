"""推理示例：输入 20×5 K 线序列，输出 0~1 置信度。

支持两种输入方式:
    1. numpy 数组 (20, 5) 或 (5, 20)
    2. .npy 文件路径

用法:
    python inference.py --model model.pth --input sample.npy
    python inference.py --model model.onnx --input sample.npy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.cnn_model import CandleCNN, SEQ_LEN, NUM_FEATURES, get_device  # noqa: E402


def load_input(data, seq_len: int = SEQ_LEN,
               num_features: int = NUM_FEATURES) -> np.ndarray:
    """加载并校验输入数据，转为 (1, 5, 20) float32。

    Args:
        data: numpy 数组或 .npy 文件路径。
        seq_len: 期望序列长度。
        num_features: 期望特征数。

    Returns:
        (1, num_features, seq_len) float32 数组。
    """
    if isinstance(data, str):
        arr = np.load(data)
    else:
        arr = np.asarray(data)

    if arr.shape == (seq_len, num_features):
        # (20, 5) -> (1, 5, 20) 转置
        arr = arr.T
    elif arr.shape == (num_features, seq_len):
        # (5, 20) 直接用
        pass
    elif arr.shape == (seq_len, num_features, 1) or arr.shape == (1, seq_len, num_features):
        arr = arr.reshape(seq_len, num_features).T
    else:
        raise ValueError(f'输入形状 {arr.shape} 不支持，期望 ({seq_len},{num_features}) 或 ({num_features},{seq_len})')

    arr = arr.astype(np.float32)
    if arr.shape != (num_features, seq_len):
        arr = arr.reshape(num_features, seq_len)
    return arr[np.newaxis, ...]  # (1, 5, 20)


def infer_torch(model_path: str, input_data) -> float:
    """用 PyTorch 模型推理。

    Args:
        model_path: .pth 模型权重路径。
        input_data: numpy 数组或 .npy 文件路径。

    Returns:
        0~1 置信度（float）。
    """
    import torch

    device = get_device()
    model = CandleCNN().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    x = load_input(input_data)
    x_tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        logit = model(x_tensor)
        confidence = torch.sigmoid(logit).item()
    return confidence


def infer_onnx(model_path: str, input_data) -> float:
    """用 ONNX 模型推理。

    Args:
        model_path: .onnx 模型路径。
        input_data: numpy 数组或 .npy 文件路径。

    Returns:
        0~1 置信度（float）。
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    x = load_input(input_data)
    outputs = sess.run(None, {input_name: x})
    logit = outputs[0][0, 0]
    # ONNX 导出时已包含 Sigmoid，直接返回
    if logit < 0 or logit > 1:
        # 未包含 Sigmoid，手动处理
        confidence = 1.0 / (1.0 + np.exp(-logit))
    else:
        confidence = float(logit)
    return confidence


def main():
    parser = argparse.ArgumentParser(description='1D-CNN K 线形态识别推理')
    parser.add_argument('--model', required=True, help='模型路径 (.pth 或 .onnx)')
    parser.add_argument('--input', required=True, help='输入数据：.npy 文件路径')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二分类阈值，默认 0.5')
    args = parser.parse_args()

    ext = os.path.splitext(args.model)[1].lower()
    if ext == '.pth':
        confidence = infer_torch(args.model, args.input)
    elif ext == '.onnx':
        confidence = infer_onnx(args.model, args.input)
    else:
        raise ValueError(f'不支持的模型格式: {ext}，仅支持 .pth 或 .onnx')

    prediction = 1 if confidence >= args.threshold else 0
    print(f'置信度: {confidence:.4f}')
    print(f'预测: {prediction} (阈值 {args.threshold})')
    return confidence


if __name__ == '__main__':
    main()
