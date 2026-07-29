"""XGBoost 数据集构建：从 sample_collector 的 .npy 样本生成特征矩阵 + 标签。

与 CNN 的 dataset.py 不同：
    - CNN 输入: (20, 5) OHLCV 原始序列 → 适合卷积
    - XGB 输入: 131 维特征向量（118 TA-Lib + 13 手工）→ 适合树模型

流程:
    sample_collector .npy (M, 20, 5)
        → features.py.extract_features 逐窗口计算 131 维特征
        → 与标签对齐
        → 输出 X (M, 131), y (M,)

存储格式:
    <output_dir>/xgb_X.npy  # (总样本数, 131) float32
    <output_dir>/xgb_y.npy  # (总样本数,) int32
    <output_dir>/xgb_codes.npy  # (总样本数,) object 记录每个样本的标的代码（用于按标的划分）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.cnn_model import SEQ_LEN, NUM_FEATURES  # noqa: E402
from ai.features import extract_features, N_FEATURES  # noqa: E402
from ai.sample_collector import collect_samples_for_code  # noqa: E402


def build_xgb_dataset_from_npy(sample_dir: str, output_dir: str | None = None) -> dict:
    """从已采集的 .npy 样本构建 XGBoost 用的特征矩阵。

    Args:
        sample_dir: sample_collector 输出目录（含 <code>_samples.npy）
        output_dir: 输出目录，None=sample_dir

    Returns:
        统计 dict
    """
    output_dir = output_dir or sample_dir
    os.makedirs(output_dir, exist_ok=True)

    # 扫描所有样本文件
    sample_files = sorted([f for f in os.listdir(sample_dir)
                           if f.endswith('_samples.npy')])

    X_list = []
    y_list = []
    codes_list = []

    for f in sample_files:
        code = f.replace('_samples.npy', '')
        sample_path = os.path.join(sample_dir, f)
        label_path = os.path.join(sample_dir, f'{code}_labels.npy')
        if not os.path.exists(label_path):
            continue

        try:
            samples = np.load(sample_path)  # (M, 20, 5)
            labels = np.load(label_path)    # (M,)
            if len(samples) == 0:
                continue

            # 逐样本提取 131 维特征
            # samples[i] 形状 (20, 5)，需要转为 DataFrame 才能用 features.py
            feats = np.zeros((len(samples), N_FEATURES), dtype=np.float32)
            for i in range(len(samples)):
                df = pd.DataFrame(samples[i], columns=['open', 'high', 'low', 'close', 'vol'])
                # extract_features 返回 (20, 131)，取最后一行（窗口末根）
                feat_mat = extract_features(df)
                feats[i] = feat_mat[-1]

            X_list.append(feats)
            y_list.append(labels)
            codes_list.append(np.array([code] * len(samples), dtype=object))

            print(f'[xgb_dataset] {code}: {len(samples)} 样本 → {feats.shape}',
                  flush=True)
        except Exception as e:
            print(f'[xgb_dataset] {code} 失败: {e}', flush=True)

    if not X_list:
        raise RuntimeError(f'目录 {sample_dir} 下无有效样本')

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    codes = np.concatenate(codes_list, axis=0)

    # 保存
    np.save(os.path.join(output_dir, 'xgb_X.npy'), X)
    np.save(os.path.join(output_dir, 'xgb_y.npy'), y)
    np.save(os.path.join(output_dir, 'xgb_codes.npy'), codes)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f'[xgb_dataset] 总计: {len(X)} 样本 (正{n_pos}/负{n_neg}), '
          f'特征维度 {X.shape[1]}', flush=True)

    return {
        'total_samples': len(X),
        'positive': n_pos,
        'negative': n_neg,
        'n_features': X.shape[1],
        'codes': len(set(codes)),
    }


def build_xgb_dataset_from_raw(code_list: list[str], start_date: str, end_date: str,
                                output_dir: str) -> dict:
    """从盈湖原始 K 线直接构建 XGBoost 数据集（一步到位，不经过 .npy 中间文件）。

    Args:
        code_list: 标的代码列表
        start_date: 起始日 YYYYMMDD
        end_date: 结束日 YYYYMMDD
        output_dir: 输出目录

    Returns:
        统计 dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # 延迟导入 yinghu_db
    sys.path.insert(0, str(_BACKEND_DIR / 'data'))
    try:
        from data.yinghu_db import get_kline
    except ImportError:
        from yinghu_db import get_kline

    X_list = []
    y_list = []
    codes_list = []

    for idx, code in enumerate(code_list):
        try:
            df = get_kline(code, start_date, end_date)
            if df is None or df.empty:
                continue

            # 调用 sample_collector 生成窗口 + 标签
            samples, labels = collect_samples_for_code(df)
            if len(samples) == 0:
                continue

            # 提取特征
            feats = np.zeros((len(samples), N_FEATURES), dtype=np.float32)
            for i in range(len(samples)):
                df_win = pd.DataFrame(samples[i], columns=['open', 'high', 'low', 'close', 'vol'])
                feat_mat = extract_features(df_win)
                feats[i] = feat_mat[-1]

            X_list.append(feats)
            y_list.append(labels)
            codes_list.append(np.array([code] * len(samples), dtype=object))

            if (idx + 1) % 10 == 0 or idx + 1 == len(code_list):
                print(f'[xgb_raw] 进度: {idx+1}/{len(code_list)}', flush=True)
        except Exception as e:
            print(f'[xgb_raw] {code} 失败: {e}', flush=True)

    if not X_list:
        raise RuntimeError('无有效样本')

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    codes = np.concatenate(codes_list, axis=0)

    np.save(os.path.join(output_dir, 'xgb_X.npy'), X)
    np.save(os.path.join(output_dir, 'xgb_y.npy'), y)
    np.save(os.path.join(output_dir, 'xgb_codes.npy'), codes)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f'[xgb_raw] 总计: {len(X)} 样本 (正{n_pos}/负{n_neg}), '
          f'特征维度 {X.shape[1]}', flush=True)

    return {
        'total_samples': len(X),
        'positive': n_pos,
        'negative': n_neg,
        'n_features': X.shape[1],
        'codes': len(set(codes)),
    }


def load_xgb_dataset(data_dir: str):
    """加载已构建的 XGBoost 数据集。

    Returns:
        (X, y, codes)
    """
    X = np.load(os.path.join(data_dir, 'xgb_X.npy'))
    y = np.load(os.path.join(data_dir, 'xgb_y.npy'))
    codes = np.load(os.path.join(data_dir, 'xgb_codes.npy'), allow_pickle=True)
    return X, y, codes


def split_by_codes(X, y, codes, val_ratio: float = 0.2, seed: int = 42):
    """按标的分组划分训练/验证集。"""
    rng = np.random.default_rng(seed)
    all_codes = sorted(set(codes))
    rng.shuffle(all_codes)
    n_val = max(1, int(len(all_codes) * val_ratio))
    val_codes = set(all_codes[:n_val])

    train_mask = np.array([c not in val_codes for c in codes])
    val_mask = ~train_mask

    return (X[train_mask], y[train_mask], codes[train_mask],
            X[val_mask], y[val_mask], codes[val_mask])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='构建 XGBoost 数据集')
    parser.add_argument('--sample_dir', help='已采集的 .npy 样本目录')
    parser.add_argument('--output_dir', help='输出目录')
    parser.add_argument('--from_raw', action='store_true',
                        help='从盈湖原始数据直接构建（不经 .npy）')
    parser.add_argument('--code_list', help='标的代码列表（逗号分隔），--from_raw 时使用')
    parser.add_argument('--start_date', default='20100101')
    parser.add_argument('--end_date', default='20260727')
    args = parser.parse_args()

    if args.from_raw:
        codes = [c.strip() for c in args.code_list.split(',')] if args.code_list else []
        build_xgb_dataset_from_raw(codes, args.start_date, args.end_date,
                                    args.output_dir or 'backend/ai/data/xgb')
    else:
        build_xgb_dataset_from_npy(args.sample_dir, args.output_dir)
