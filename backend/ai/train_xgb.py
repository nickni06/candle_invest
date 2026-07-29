"""阶段1 XGBoost 训练脚本：纯特征输入，验证 AI 辅助是否比纯 TA-Lib 更好。

特性:
    - 输入: 131 维特征（118 TA-Lib + 13 手工）
    - 输出: 二分类概率（0~1）
    - 按标的分组划分训练/验证（防止数据泄露）
    - 类别不平衡处理: scale_pos_weight
    - 早停: 验证集 F1 连续 patience 轮无提升则停止
    - 特征重要性分析（输出 top-20）

用法:
    # 从已构建的 xgb 数据集训练
    python train_xgb.py --data_dir backend/ai/data/xgb

    # 从盈湖原始数据一步构建 + 训练
    python train_xgb.py --from_raw --code_list 600519.SH,000001.SZ

    # 从 sample_collector 已采集的 .npy 构建 + 训练
    python train_xgb.py --from_npy --sample_dir backend/ai/data/train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.dataset_xgb import (  # noqa: E402
    build_xgb_dataset_from_npy,
    build_xgb_dataset_from_raw,
    load_xgb_dataset,
    split_by_codes,
)
from ai.features import FEATURE_NAMES  # noqa: E402


def compute_metrics(y_true, y_pred, y_prob, threshold: float = 0.5) -> dict:
    """二分类评估指标。"""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        'accuracy': float(accuracy), 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1),
        'auc': float(_compute_auc(y_true, y_prob)),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def _compute_auc(labels, probs):
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        return 0.0
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    cum_tp = np.cumsum(labels_sorted == 1)
    cum_fp = np.cumsum(labels_sorted == 0)
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg
    trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz_fn(tpr, fpr))


def train_xgb(data_dir: str, output_dir: str,
              val_ratio: float = 0.2, seed: int = 42,
              n_estimators: int = 500, max_depth: int = 6,
              learning_rate: float = 0.05, patience: int = 30,
              subsample: float = 0.8, colsample_bytree: float = 0.8) -> dict:
    """训练 XGBoost 模型。

    Args:
        data_dir: xgb 数据集目录（含 xgb_X.npy / xgb_y.npy / xgb_codes.npy）
        output_dir: 模型输出目录
        val_ratio: 验证集标的占比
        seed: 随机种子
        n_estimators: 最大树数
        max_depth: 树最大深度
        learning_rate: 学习率
        patience: 早停耐心值
        subsample: 行采样率
        colsample_bytree: 列采样率

    Returns:
        训练结果 dict
    """
    import xgboost as xgb

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据并按标的划分
    print(f'[xgb_train] 加载数据: {data_dir}')
    X, y, codes = load_xgb_dataset(data_dir)
    X_train, y_train, c_train, X_val, y_val, c_val = split_by_codes(
        X, y, codes, val_ratio=val_ratio, seed=seed)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(1, n_pos)
    print(f'[xgb_train] 训练集: {len(X_train)} (正{n_pos}/负{n_neg}), '
          f'scale_pos_weight={spw:.3f}')
    print(f'[xgb_train] 验证集: {len(X_val)} (正{int((y_val==1).sum())}/'
          f'负{int((y_val==0).sum())})')
    print(f'[xgb_train] 特征维度: {X.shape[1]}')

    # 2. 训练
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error', 'auc'],
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'scale_pos_weight': spw,
        'seed': seed,
        'nthread': os.cpu_count() or 4,
        'verbosity': 0,
    }

    history = {'train_logloss': [], 'val_logloss': [], 'val_f1': [],
               'best_val_f1': 0.0, 'best_iter': -1}
    best_model = None
    no_improve = 0

    print(f'[xgb_train] 开始训练，最多 {n_estimators} 轮，早停耐心 {patience}')
    t0 = time.time()

    # 手动训练循环以便支持 F1 早停（xgb 原生只支持 logloss/error 早停）
    for step in range(1, n_estimators + 1):
        best_model = xgb.train(
            params, dtrain, num_boost_round=step,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=False,
        )
        # 评估
        y_val_prob = best_model.predict(dval)
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        m = compute_metrics(y_val, y_val_pred, y_val_prob)

        history['train_logloss'].append(None)  # xgb 不直接返回
        history['val_logloss'].append(None)
        history['val_f1'].append(m['f1'])

        improved = m['f1'] > history['best_val_f1']
        if improved:
            history['best_val_f1'] = m['f1']
            history['best_iter'] = step
            no_improve = 0
        else:
            no_improve += 1

        if step % 10 == 0 or improved or step == 1:
            print(f'[xgb_train] Step {step:4d}/{n_estimators} | '
                  f'val_f1={m["f1"]:.4f} val_p={m["precision"]:.4f} '
                  f'val_r={m["recall"]:.4f} auc={m["auc"]:.4f} | '
                  f'{"*BEST*" if improved else f"no improve {no_improve}/{patience}"}',
                  flush=True)

        if no_improve >= patience:
            print(f'[xgb_train] 早停：F1 连续 {patience} 轮无提升', flush=True)
            break

    total_time = time.time() - t0
    print(f'[xgb_train] 训练完成，耗时 {total_time:.1f}s，'
          f'最佳 val_f1={history["best_val_f1"]:.4f} @ iter {history["best_iter"]}')

    # 3. 用最佳轮数重新训练并保存
    if history['best_iter'] > 0:
        final_model = xgb.train(
            params, dtrain, num_boost_round=history['best_iter'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=False,
        )
    else:
        final_model = best_model

    # 4. 最终评估
    y_val_prob = final_model.predict(dval)
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    final_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    print(f'\n[xgb_train] ===== 最终验证集评估 =====')
    print(f'  Accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'  Precision:{final_metrics["precision"]:.4f}')
    print(f'  Recall:   {final_metrics["recall"]:.4f}')
    print(f'  F1:       {final_metrics["f1"]:.4f}')
    print(f'  AUC:      {final_metrics["auc"]:.4f}')
    print(f'  TP/FP/FN/TN: {final_metrics["tp"]}/{final_metrics["fp"]}/{final_metrics["fn"]}/{final_metrics["tn"]}')

    # 5. 特征重要性
    importance = final_model.get_score(importance_type='gain')
    importance_sorted = sorted(importance.items(), key=lambda x: -x[1])
    print(f'\n[xgb_train] Top-20 重要特征:')
    for i, (name, score) in enumerate(importance_sorted[:20], 1):
        print(f'  {i:2d}. {name:25s}  {score:.2f}')

    # 6. 保存模型
    model_path = os.path.join(output_dir, 'xgb_model.json')
    final_model.save_model(model_path)
    print(f'[xgb_train] 模型已保存: {model_path}')

    # 7. 保存训练历史
    history['final_metrics'] = final_metrics
    history['feature_importance'] = dict(importance_sorted[:20])
    history['n_features'] = X.shape[1]
    history['n_train'] = len(X_train)
    history['n_val'] = len(X_val)
    history['total_time_sec'] = total_time
    history_path = os.path.join(output_dir, 'xgb_training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'[xgb_train] 训练历史已保存: {history_path}')

    return history


def main():
    parser = argparse.ArgumentParser(description='阶段1 XGBoost 训练')
    parser.add_argument('--data_dir', help='已构建的 xgb 数据集目录')
    parser.add_argument('--output_dir', default='backend/ai/outputs',
                        help='模型输出目录')
    parser.add_argument('--from_raw', action='store_true',
                        help='从盈湖原始数据直接构建（不经 .npy）')
    parser.add_argument('--from_npy', action='store_true',
                        help='从 sample_collector .npy 构建')
    parser.add_argument('--sample_dir', help='sample_collector 输出目录（--from_npy 时使用）')
    parser.add_argument('--code_list', help='标的代码列表（逗号分隔），--from_raw 时使用')
    parser.add_argument('--start_date', default='20100101')
    parser.add_argument('--end_date', default='20260727')
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    # 如果需要先构建数据集
    if args.from_raw:
        codes = [c.strip() for c in args.code_list.split(',')] if args.code_list else []
        data_dir = os.path.join(os.path.dirname(args.output_dir), 'data', 'xgb')
        build_xgb_dataset_from_raw(codes, args.start_date, args.end_date, data_dir)
    elif args.from_npy:
        data_dir = os.path.join(os.path.dirname(args.output_dir), 'data', 'xgb')
        build_xgb_dataset_from_npy(args.sample_dir, data_dir)

    train_xgb(
        data_dir=args.data_dir or data_dir,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()
