"""AI 模型推理服务（Flask）。

独立于主 Flask 服务（web_app.py:8765），监听 8766 端口。
原因:
    1. PyTorch / ONNX Runtime 占用内存大，不与主服务共享进程
    2. 模型加载一次，避免 signal_tracker 多进程 fork 时每个子进程都加载
    3. 支持独立重启模型服务（不影响主服务）

支持的接口:
    GET  /health              健康检查
    POST /predict/xgb         纯 XGBoost 推理
    POST /predict/cnn         纯 CNN 推理
    POST /predict/fusion      CNN + XGBoost 融合推理（阶段2）
    POST /predict/batch       批量推理（用于回测）

请求格式（POST /predict/xgb）:
    {"features": [131维特征数组]}
    或
    {"kline": [[open,high,low,close,vol], ...]}  # 20根K线
响应:
    {"buy_prob": 0.72, "sell_prob": 0.28, "raw_prob": 0.72}

启动:
    python backend/ai/model_server.py
    # 或指定端口
    AI_MODEL_SERVER_PORT=8800 python backend/ai/model_server.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.features import extract_features, FEATURE_NAMES, N_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ai_model_server')

# ============================================================================
# 推理器（延迟加载，避免启动慢）
# ============================================================================
_xgb_infer = None
_cnn_infer = None


def get_xgb_infer():
    """单例 XGBoost 推理器。"""
    global _xgb_infer
    if _xgb_infer is None:
        try:
            from config import config
            model_path = str(config.AI_MODEL_DIR / 'xgb_model.json')
        except Exception:
            model_path = str(_THIS_DIR / 'outputs' / 'xgb_model.json')
        from ai.inference_xgb import XGBInference
        _xgb_infer = XGBInference(model_path)
    return _xgb_infer


def get_cnn_infer():
    """单例 CNN 推理器。"""
    global _cnn_infer
    if _cnn_infer is None:
        try:
            from config import config
            onnx_path = str(config.AI_MODEL_DIR / 'candle_cnn.onnx')
            pth_path = str(config.AI_MODEL_DIR / 'candle_cnn.pth')
        except Exception:
            onnx_path = str(_THIS_DIR / 'outputs' / 'candle_cnn.onnx')
            pth_path = str(_THIS_DIR / 'outputs' / 'candle_cnn.pth')

        from ai.inference import CNNInference
        # 优先 ONNX，回退 PyTorch
        try:
            _cnn_infer = CNNInference(onnx_path if os.path.exists(onnx_path) else pth_path,
                                       use_onnx=os.path.exists(onnx_path))
        except Exception as e:
            logger.warning(f'CNN 模型加载失败: {e}')
            return None
    return _cnn_infer


# ============================================================================
# Flask 应用
# ============================================================================
def create_app():
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'ok',
            'xgb_loaded': _xgb_infer is not None,
            'cnn_loaded': _cnn_infer is not None,
            'n_features': N_FEATURES,
        })

    @app.route('/predict/xgb', methods=['POST'])
    def predict_xgb():
        """XGBoost 单样本预测。

        请求体:
            {"features": [131维]} 或 {"kline": [[20根 OHLCV]]}
        """
        try:
            data = request.get_json(force=True)
            if 'features' in data:
                feats = np.array(data['features'], dtype=np.float32)
                if feats.shape != (N_FEATURES,):
                    return jsonify({'error': f'特征维度错误: {feats.shape}, 预期 ({N_FEATURES},)'}), 400
            elif 'kline' in data:
                kline = np.array(data['kline'], dtype=np.float64)
                if kline.shape != (20, 5):
                    return jsonify({'error': f'K线维度错误: {kline.shape}, 预期 (20, 5)'}), 400
                df = pd.DataFrame(kline, columns=['open', 'high', 'low', 'close', 'vol'])
                feats = extract_features(df)[-1]
            else:
                return jsonify({'error': '需要 features 或 kline 字段'}), 400

            infer = get_xgb_infer()
            if infer is None:
                return jsonify({'error': 'XGBoost 模型未加载'}), 503

            result = infer.predict_features(feats)
            return jsonify(result)
        except Exception as e:
            logger.exception(f'/predict/xgb 异常')
            return jsonify({'error': str(e)}), 500

    @app.route('/predict/cnn', methods=['POST'])
    def predict_cnn():
        """CNN 单样本预测。

        请求体:
            {"kline": [[20根 OHLCV]]}
        """
        try:
            data = request.get_json(force=True)
            if 'kline' not in data:
                return jsonify({'error': '需要 kline 字段'}), 400

            kline = np.array(data['kline'], dtype=np.float32)
            if kline.shape != (20, 5):
                return jsonify({'error': f'K线维度错误: {kline.shape}, 预期 (20, 5)'}), 400

            infer = get_cnn_infer()
            if infer is None:
                return jsonify({'error': 'CNN 模型未加载'}), 503

            # CNN 输入需要归一化（与训练一致）
            from ai.sample_collector import normalize_window
            norm_kline = normalize_window(kline)
            prob = infer.predict(norm_kline)
            return jsonify({
                'buy_prob': float(prob),
                'sell_prob': 1.0 - float(prob),
                'raw_prob': float(prob),
            })
        except Exception as e:
            logger.exception(f'/predict/cnn 异常')
            return jsonify({'error': str(e)}), 500

    @app.route('/predict/fusion', methods=['POST'])
    def predict_fusion():
        """CNN + XGBoost 融合推理（阶段2）。

        请求体:
            {"kline": [[20根 OHLCV]]}

        融合逻辑:
            1. CNN 对原始 K 线序列输出置信度 cnn_prob
            2. features.py 提取 131 维特征
            3. XGBoost 输入 = 131 维特征 + [cnn_prob] = 132 维
            4. XGBoost 输出最终概率

        当前阶段1实现:
            如果 CNN 未加载，直接用 XGBoost 131 维特征推理；
            CNN 加载后，需要重新训练 XGBoost（132 维输入），此处自动切换。
        """
        try:
            data = request.get_json(force=True)
            if 'kline' not in data:
                return jsonify({'error': '需要 kline 字段'}), 400

            kline = np.array(data['kline'], dtype=np.float64)
            if kline.shape != (20, 5):
                return jsonify({'error': f'K线维度错误: {kline.shape}, 预期 (20, 5)'}), 400

            df = pd.DataFrame(kline, columns=['open', 'high', 'low', 'close', 'vol'])
            feats_131 = extract_features(df)[-1]  # (131,)

            # 尝试 CNN 推理
            cnn_prob = None
            cnn_infer = get_cnn_infer()
            if cnn_infer is not None:
                from ai.sample_collector import normalize_window
                norm_kline = normalize_window(kline.astype(np.float32))
                cnn_prob = float(cnn_infer.predict(norm_kline))

            # XGBoost 推理
            xgb_infer = get_xgb_infer()
            if xgb_infer is None:
                return jsonify({'error': 'XGBoost 模型未加载'}), 503

            if cnn_prob is not None:
                # 阶段2: CNN 输出作为 XGBoost 的额外特征
                # 注意: 需要训练 132 维输入的 XGBoost 才能用
                # 这里做软融合: 加权平均
                feats_132 = np.concatenate([feats_131, [cnn_prob]])
                # 用 131 维模型推理（临时方案）
                xgb_result = xgb_infer.predict_features(feats_131)
                xgb_prob = xgb_result['raw_prob']
                # 软融合: 0.7 * XGB + 0.3 * CNN
                fused_prob = 0.7 * xgb_prob + 0.3 * cnn_prob
                return jsonify({
                    'buy_prob': float(fused_prob),
                    'sell_prob': 1.0 - float(fused_prob),
                    'raw_prob': float(fused_prob),
                    'xgb_prob': float(xgb_prob),
                    'cnn_prob': float(cnn_prob),
                    'fusion_mode': 'soft',
                })
            else:
                # 阶段1: 纯 XGBoost
                xgb_result = xgb_infer.predict_features(feats_131)
                return jsonify({
                    **xgb_result,
                    'fusion_mode': 'xgb_only',
                })
        except Exception as e:
            logger.exception(f'/predict/fusion 异常')
            return jsonify({'error': str(e)}), 500

    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        """批量推理（用于回测）。

        请求体:
            {"features": [[131维], [131维], ...]}
        """
        try:
            data = request.get_json(force=True)
            feats = np.array(data['features'], dtype=np.float32)
            if feats.ndim != 2 or feats.shape[1] != N_FEATURES:
                return jsonify({'error': f'特征维度错误: {feats.shape}'}), 400

            infer = get_xgb_infer()
            if infer is None:
                return jsonify({'error': 'XGBoost 模型未加载'}), 503

            result = infer.predict_features(feats)
            return jsonify(result)
        except Exception as e:
            logger.exception(f'/predict/batch 异常')
            return jsonify({'error': str(e)}), 500

    @app.route('/predict/xgb/batch', methods=['POST'])
    def predict_xgb_batch():
        """XGBoost 批量推理（用于信号跟踪/回测批量评分）。

        请求体:
            {"klines": [[[open,high,low,close,vol], ...20根], ...]}  # 多个20根K线窗口
        响应:
            {"probs": [0.72, 0.48, ...], "ai_available": true}
        """
        try:
            data = request.get_json(force=True)
            klines = data.get('klines')
            if not klines or not isinstance(klines, list):
                return jsonify({'error': '需要 klines 字段（数组）'}), 400

            infer = get_xgb_infer()
            if infer is None:
                return jsonify({'error': 'XGBoost 模型未加载', 'ai_available': False}), 503

            # 批量提取特征
            feats_list = []
            for kl in klines:
                arr = np.array(kl, dtype=np.float64)
                if arr.shape != (20, 5):
                    return jsonify({'error': f'K线维度错误: {arr.shape}, 预期 (20, 5)'}), 400
                df = pd.DataFrame(arr, columns=['open', 'high', 'low', 'close', 'vol'])
                feats_list.append(extract_features(df)[-1])

            feats_mat = np.array(feats_list, dtype=np.float32)  # (N, 131)
            result = infer.predict_features(feats_mat)
            # predict_features 返回 {'buy_prob': list, ...}
            probs = result.get('raw_prob')
            if not isinstance(probs, list):
                probs = [float(probs)]
            return jsonify({'probs': probs, 'ai_available': True})
        except Exception as e:
            logger.exception(f'/predict/xgb/batch 异常')
            return jsonify({'error': str(e), 'ai_available': False}), 500

    @app.route('/predict/cnn/batch', methods=['POST'])
    def predict_cnn_batch():
        """CNN 批量推理（用于信号跟踪/回测批量评分）。

        请求体:
            {"klines": [[[open,high,low,close,vol], ...20根], ...]}  # 多个20根K线窗口
        响应:
            {"probs": [0.65, 0.52, ...], "ai_available": true}
        """
        try:
            data = request.get_json(force=True)
            klines = data.get('klines')
            if not klines or not isinstance(klines, list):
                return jsonify({'error': '需要 klines 字段（数组）'}), 400

            infer = get_cnn_infer()
            if infer is None:
                return jsonify({'error': 'CNN 模型未加载', 'ai_available': False}), 503

            from ai.sample_collector import normalize_window
            arr_list = []
            for kl in klines:
                arr = np.array(kl, dtype=np.float32)
                if arr.shape != (20, 5):
                    return jsonify({'error': f'K线维度错误: {arr.shape}, 预期 (20, 5)'}), 400
                arr_list.append(normalize_window(arr))

            batch = np.array(arr_list, dtype=np.float32)  # (N, 20, 5)
            probs = infer.predict_batch(batch).tolist()
            return jsonify({'probs': probs, 'ai_available': True})
        except Exception as e:
            logger.exception(f'/predict/cnn/batch 异常')
            return jsonify({'error': str(e), 'ai_available': False}), 500

    return app


def main():
    try:
        from config import config
        port = getattr(config, 'AI_MODEL_SERVER_PORT', 8766)
    except Exception:
        port = int(os.environ.get('AI_MODEL_SERVER_PORT', '8766'))

    app = create_app()
    print(f'[model_server] 启动 AI 模型推理服务，端口 {port}', flush=True)
    print(f'[model_server] 接口:', flush=True)
    print(f'  GET  /health', flush=True)
    print(f'  POST /predict/xgb     (纯 XGBoost)', flush=True)
    print(f'  POST /predict/cnn     (纯 CNN)', flush=True)
    print(f'  POST /predict/fusion  (CNN+XGB 融合)', flush=True)
    print(f'  POST /predict/batch   (批量 XGBoost 特征)', flush=True)
    print(f'  POST /predict/xgb/batch (批量 XGBoost K线)', flush=True)
    print(f'  POST /predict/cnn/batch (批量 CNN K线)', flush=True)
    app.run(host='127.0.0.1', port=port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
