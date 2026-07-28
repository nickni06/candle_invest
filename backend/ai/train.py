"""训练脚本：1D-CNN K 线形态识别。

特性:
    - 设备自适应: MPS(Mac) > CUDA > CPU
    - BCEWithLogitsLoss 二分类（保留 sigmoid 概率作为置信度）
    - 输出训练日志: loss 曲线、准确率、精确率、召回率、F1
    - 类别不平衡处理: pos_weight 自动按正负样本比例计算
    - 早停: 验证集 F1 连续 patience 轮无提升则停止
    - 模型保存: .pth（state_dict）+ 训练日志 JSON

用法:
    python train.py --data_dir <样本目录> --epochs 50
    python train.py --data_dir <样本目录> --epochs 50 --batch_size 64 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.cnn_model import CandleCNN, get_device, count_parameters  # noqa: E402
from ai.dataset import CandleDataset, split_by_codes  # noqa: E402


def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray, threshold: float = 0.5) -> dict:
    """计算二分类评估指标。

    Args:
        labels: 真实标签 (N,)
        preds: 预测标签 (N,)，0 或 1
        probs: 预测概率 (N,)，0~1
        threshold: 二分类阈值

    Returns:
        dict: accuracy, precision, recall, f1, auc
    """
    labels = labels.astype(int)
    preds = preds.astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    # AUC（简单实现，无需额外依赖）
    auc = _compute_auc(labels, probs)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def _compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """计算 AUC（ROC 曲线下面积），纯 numpy 实现。"""
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        return 0.0
    # 按 prob 降序排列
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # 累积 TP 和 FP
    cum_tp = np.cumsum(labels_sorted == 1)
    cum_fp = np.cumsum(labels_sorted == 0)
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg
    # 梯形法积分（numpy 2.0 移除 trapz，改用 trapezoid）
    trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    auc = trapz_fn(tpr, fpr)
    return float(auc)


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion, optimizer, device) -> tuple[float, dict]:
    """训练一个 epoch。

    Returns:
        (avg_loss, metrics)
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x).squeeze(-1)  # (B,)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        n_samples += len(y)
        all_labels.append(y.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss = total_loss / max(1, n_samples)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    metrics = compute_metrics(labels, preds, probs)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion, device) -> tuple[float, dict]:
    """在验证集上评估。

    Returns:
        (avg_loss, metrics)
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        n_samples += len(y)
        all_labels.append(y.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    avg_loss = total_loss / max(1, n_samples)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    metrics = compute_metrics(labels, preds, probs)
    return avg_loss, metrics


def train(data_dir: str, output_dir: str,
          epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
          val_ratio: float = 0.2, patience: int = 10,
          seed: int = 42) -> dict:
    """完整训练流程。

    Args:
        data_dir: 样本目录（含 <code>_samples.npy）。
        output_dir: 模型和日志输出目录。
        epochs: 最大训练轮数。
        batch_size: 批大小。
        lr: 学习率。
        val_ratio: 验证集标的占比。
        patience: 早停耐心值（验证 F1 连续无提升轮数）。
        seed: 随机种子。

    Returns:
        训练历史 dict。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f'[train] 设备: {device}')

    # 1. 加载数据
    print(f'[train] 加载数据: {data_dir}')
    train_ds, val_ds = split_by_codes(data_dir, val_ratio=val_ratio, seed=seed)

    if len(train_ds) == 0:
        raise RuntimeError('训练集为空，请先用 sample_collector.py 生成样本')

    # 处理类别不平衡：pos_weight = 负样本数 / 正样本数
    n_pos = int((train_ds.labels == 1).sum())
    n_neg = int((train_ds.labels == 0).sum())
    pos_weight_val = n_neg / max(1, n_pos)
    print(f'[train] 训练集: {len(train_ds)} 样本 (正{n_pos}/负{n_neg})，'
          f'pos_weight={pos_weight_val:.3f}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0) if len(val_ds) > 0 else None

    # 2. 模型
    model = CandleCNN().to(device)
    print(f'[train] 模型参数量: {count_parameters(model):,}')

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # 3. 训练循环
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_metrics': [], 'val_metrics': [],
        'best_val_f1': 0.0, 'best_epoch': -1,
    }
    best_state = None
    no_improve = 0

    print(f'[train] 开始训练，最多 {epochs} 轮，早停耐心 {patience}')
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['train_metrics'].append(train_metrics)

        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_metrics['f1'])
            history['val_metrics'].append(val_metrics)
            scheduler.step(val_metrics['f1'])

            improved = val_metrics['f1'] > history['best_val_f1']
            if improved:
                history['best_val_f1'] = val_metrics['f1']
                history['best_epoch'] = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - t0
            print(f'[train] Epoch {epoch:3d}/{epochs} | '
                  f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} | '
                  f'train_f1={train_metrics["f1"]:.4f} val_f1={val_metrics["f1"]:.4f} | '
                  f'val_p={val_metrics["precision"]:.4f} val_r={val_metrics["recall"]:.4f} '
                  f'auc={val_metrics["auc"]:.4f} | '
                  f'{"*BEST*" if improved else f"no improve {no_improve}/{patience}"} '
                  f'({elapsed:.1f}s)', flush=True)

            if no_improve >= patience:
                print(f'[train] 早停：验证 F1 连续 {patience} 轮无提升', flush=True)
                break
        else:
            elapsed = time.time() - t0
            print(f'[train] Epoch {epoch:3d}/{epochs} | '
                  f'train_loss={train_loss:.4f} train_f1={train_metrics["f1"]:.4f} '
                  f'({elapsed:.1f}s)', flush=True)

    total_time = time.time() - train_start
    print(f'[train] 训练完成，总耗时 {total_time:.1f}s，'
          f'最佳 val_f1={history["best_val_f1"]:.4f} @ epoch {history["best_epoch"]}')

    # 4. 保存最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = os.path.join(output_dir, 'candle_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f'[train] 模型已保存: {model_path}')

    # 保存训练历史
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'[train] 训练历史已保存: {history_path}')

    # 5. 最终测试集评估（用验证集）
    if val_loader is not None:
        final_loss, final_metrics = evaluate(model, val_loader, criterion, device)
        print(f'\n[train] ===== 最终验证集评估 =====')
        print(f'  Loss:     {final_loss:.4f}')
        print(f'  Accuracy: {final_metrics["accuracy"]:.4f}')
        print(f'  Precision:{final_metrics["precision"]:.4f}')
        print(f'  Recall:   {final_metrics["recall"]:.4f}')
        print(f'  F1:       {final_metrics["f1"]:.4f}')
        print(f'  AUC:      {final_metrics["auc"]:.4f}')
        print(f'  TP/FP/FN/TN: {final_metrics["tp"]}/{final_metrics["fp"]}/{final_metrics["fn"]}/{final_metrics["tn"]}')

    return history


def main():
    parser = argparse.ArgumentParser(description='1D-CNN K 线形态识别训练')
    parser.add_argument('--data_dir', required=True, help='样本目录（含 *_samples.npy）')
    parser.add_argument('--output_dir', default='backend/ai/outputs',
                        help='模型和日志输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='最大训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集标的占比')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
