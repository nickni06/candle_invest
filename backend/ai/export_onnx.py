"""ONNX 导出 + 动态量化。

将训练好的 PyTorch 模型导出为 ONNX 格式，并应用 ONNX Runtime 动态量化减小体积，
便于集成到主系统（C# / Java / 其他 Python 服务均可加载 ONNX）。

特性:
    - 导出含 Sigmoid 的 ONNX（推理直接得 0~1 置信度）
    - 动态 batch 维度（支持单样本和批量推理）
    - ONNX Runtime 动态量化（量化 MatMul/Linear，体积可减 30-50%）
    - 一致性验证：对比 PyTorch 和 ONNX 输出

用法:
    python export_onnx.py --model candle_cnn.pth --output candle_cnn.onnx
    python export_onnx.py --model candle_cnn.pth --output candle_cnn.onnx --no_quantize
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.cnn_model import CandleCNN, SEQ_LEN, NUM_FEATURES, get_device  # noqa: E402


class CandleCNNWithSigmoid(torch.nn.Module):
    """包装模型，导出时包含 Sigmoid（ONNX 推理直接得 0~1 置信度）。"""

    def __init__(self, base_model: CandleCNN):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.base(x)
        return torch.sigmoid(logit)


def export_fp32_onnx(model: CandleCNN, output_path: str, device) -> str:
    """导出 FP32 ONNX 模型（含 Sigmoid，动态 batch）。

    Args:
        model: 已加载权重的 PyTorch 模型。
        output_path: ONNX 输出路径。
        device: torch.device。

    Returns:
        实际输出的 ONNX 文件路径。
    """
    wrapped = CandleCNNWithSigmoid(model).to(device).eval()
    dummy_input = torch.randn(1, NUM_FEATURES, SEQ_LEN).to(device)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 使用旧版 TorchScript-based 导出（dynamo 在小模型上易失败）
    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['confidence'],
        dynamic_axes={
            'input': {0: 'batch'},
            'confidence': {0: 'batch'},
        },
        dynamo=False,  # 强制使用传统导出路径，避免 dynamo 兼容问题
    )
    return output_path


def quantize_onnx_dynamic(input_onnx: str, output_onnx: str) -> str:
    """用 ONNX Runtime 动态量化 ONNX 模型。

    量化 MatMul/Linear 权重为 INT8，激活保持 FP32（动态量化）。
    比 torch 动态量化在导出阶段更稳定。

    Args:
        input_onnx: FP32 ONNX 输入路径。
        output_onnx: 量化后 ONNX 输出路径。

    Returns:
        量化后的 ONNX 文件路径。
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_onnx,
        op_types_to_quantize=['MatMul', 'Gemm'],  # Linear 在 ONNX 里是 Gemm/MatMul
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return output_onnx


def verify_onnx(torch_model: CandleCNN, onnx_path: str,
                n_samples: int = 100) -> dict:
    """验证 ONNX 模型与 PyTorch 模型输出一致性。

    Args:
        torch_model: PyTorch 模型（已 eval）。
        onnx_path: ONNX 模型路径。
        n_samples: 验证样本数。

    Returns:
        dict: max_abs_diff, mean_abs_diff, all_close
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    test_inputs = np.random.randn(n_samples, NUM_FEATURES, SEQ_LEN).astype(np.float32)

    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model.forward_confidence(
            torch.from_numpy(test_inputs)
        ).numpy()

    onnx_out = sess.run(None, {input_name: test_inputs})[0].squeeze(-1)

    diff = np.abs(torch_out - onnx_out)
    return {
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
        'all_close': bool(np.allclose(torch_out, onnx_out, atol=1e-4)),
        'n_samples': n_samples,
    }


def export_onnx(model_path: str, output_path: str, quantize: bool = True) -> dict:
    """导出 ONNX 模型，可选动态量化。

    流程:
        1. 加载 PyTorch 权重
        2. 导出 FP32 ONNX（含 Sigmoid，动态 batch）
        3. （可选）用 ONNX Runtime 动态量化
        4. 一致性验证

    Args:
        model_path: PyTorch 模型路径 (.pth)。
        output_path: 最终 ONNX 输出路径 (.onnx)。
        quantize: 是否应用动态量化。

    Returns:
        dict: 含文件大小、量化状态、验证结果。
    """
    device = get_device()
    model = CandleCNN().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 1. 先导出 FP32 ONNX
    if quantize:
        # 量化时先导出到临时 FP32 文件，再量化到目标路径
        fp32_path = output_path.replace('.onnx', '_fp32.onnx')
    else:
        fp32_path = output_path

    print(f'[export] 步骤1: 导出 FP32 ONNX → {fp32_path}')
    export_fp32_onnx(model, fp32_path, device)

    fp32_size = os.path.getsize(fp32_path)

    # 2. （可选）动态量化
    if quantize:
        print(f'[export] 步骤2: ONNX Runtime 动态量化 → {output_path}')
        quantize_onnx_dynamic(fp32_path, output_path)
        # 删除中间 FP32 文件
        if os.path.exists(fp32_path) and fp32_path != output_path:
            os.remove(fp32_path)
        quant_status = 'dynamic_int8 (MatMul/Gemm, per-channel)'
    else:
        quant_status = 'none (FP32)'

    final_size = os.path.getsize(output_path)

    # 3. 一致性验证
    print(f'[export] 步骤3: 一致性验证（PyTorch vs ONNX）')
    verification = verify_onnx(model, output_path)

    result = {
        'model_path': model_path,
        'onnx_path': output_path,
        'fp32_size_bytes': fp32_size if quantize else final_size,
        'onnx_size_bytes': final_size,
        'fp32_size_kb': round(fp32_size / 1024, 2) if quantize else round(final_size / 1024, 2),
        'onnx_size_kb': round(final_size / 1024, 2),
        'quantization': quant_status,
        'verification': verification,
    }

    print(f'\n[export] ===== 导出完成 =====')
    print(f'  ONNX 文件: {output_path}')
    if quantize:
        print(f'  FP32 大小: {result["fp32_size_kb"]} KB')
        print(f'  量化后大小: {result["onnx_size_kb"]} KB '
              f'(压缩率 {(1 - final_size/fp32_size)*100:.1f}%)')
    else:
        print(f'  大小: {result["onnx_size_kb"]} KB')
    print(f'  量化方式: {quant_status}')
    print(f'  一致性: max_diff={verification["max_abs_diff"]:.6f}, '
          f'mean_diff={verification["mean_abs_diff"]:.6f}, '
          f'all_close={verification["all_close"]}')
    return result


def main():
    parser = argparse.ArgumentParser(description='导出 ONNX 模型')
    parser.add_argument('--model', required=True, help='PyTorch 模型路径 (.pth)')
    parser.add_argument('--output', default='candle_cnn.onnx', help='ONNX 输出路径')
    parser.add_argument('--no_quantize', action='store_true', help='禁用量化')
    args = parser.parse_args()

    export_onnx(args.model, args.output, quantize=not args.no_quantize)


if __name__ == '__main__':
    main()
