"""PyTorch Dataset：加载按标的分文件存储的 .npy 样本。

存储格式（由 sample_collector 生成）:
    <output_dir>/<code>_samples.npy  # (M, 20, 5) float32
    <output_dir>/<code>_labels.npy   # (M,) int32

Dataset 会扫描目录下所有 *_samples.npy，配对加载并拼接。
支持训练/验证按标的分组划分（避免同一标的的样本同时出现在训练和验证集，防止数据泄露）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.cnn_model import SEQ_LEN, NUM_FEATURES  # noqa: E402


class CandleDataset(Dataset):
    """K 线形态样本数据集。

    从目录加载所有 <code>_samples.npy + <code>_labels.npy，拼成大数组。
    支持按标的代码列表过滤（用于训练/验证划分）。
    """

    def __init__(self, data_dir: str, code_filter: Optional[list[str]] = None):
        """初始化数据集。

        Args:
            data_dir: 样本目录，含 <code>_samples.npy 和 <code>_labels.npy。
            code_filter: 仅加载这些标的的样本；None 表示加载全部。
        """
        self.data_dir = data_dir
        self.samples_list: list[np.ndarray] = []
        self.labels_list: list[np.ndarray] = []
        self.code_sample_counts: dict[str, int] = {}

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'样本目录不存在: {data_dir}')

        # 扫描所有样本文件
        sample_files = sorted([f for f in os.listdir(data_dir)
                               if f.endswith('_samples.npy')])

        loaded_codes = []
        for f in sample_files:
            code = f.replace('_samples.npy', '')
            if code_filter is not None and code not in code_filter:
                continue
            sample_path = os.path.join(data_dir, f)
            label_path = os.path.join(data_dir, f'{code}_labels.npy')
            if not os.path.exists(label_path):
                continue
            try:
                samples = np.load(sample_path)
                labels = np.load(label_path)
                if len(samples) == 0:
                    continue
                self.samples_list.append(samples)
                self.labels_list.append(labels)
                self.code_sample_counts[code] = len(samples)
                loaded_codes.append(code)
            except Exception as e:
                print(f'[dataset] 加载 {code} 失败: {e}', flush=True)

        if not self.samples_list:
            self.samples = np.empty((0, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int32)
        else:
            self.samples = np.concatenate(self.samples_list, axis=0)
            self.labels = np.concatenate(self.labels_list, axis=0)

        self.loaded_codes = loaded_codes
        print(f'[dataset] 加载 {len(loaded_codes)} 个标的，'
              f'共 {len(self.samples)} 个样本 '
              f'(正 {int((self.labels==1).sum())} / 负 {int((self.labels==0).sum())})',
              flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y)。

        x: (5, 20) float32 - 注意转置为 (channels, length) 以适配 Conv1d
        y: 标量 float32 - 0 或 1（BCEWithLogitsLoss 需要 float）
        """
        # 原始存储为 (20, 5)，Conv1d 需要 (5, 20)
        x = torch.from_numpy(self.samples[idx].T.astype(np.float32))
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


def split_by_codes(data_dir: str, val_ratio: float = 0.2,
                   seed: int = 42) -> tuple[CandleDataset, CandleDataset]:
    """按标的分组划分训练/验证集，防止数据泄露。

    Args:
        data_dir: 样本目录。
        val_ratio: 验证集标的占比。
        seed: 随机种子。

    Returns:
        (train_dataset, val_dataset)
    """
    rng = np.random.default_rng(seed)

    # 扫描所有可用标的
    all_codes = []
    for f in os.listdir(data_dir):
        if f.endswith('_samples.npy'):
            code = f.replace('_samples.npy', '')
            label_path = os.path.join(data_dir, f'{code}_labels.npy')
            if os.path.exists(label_path):
                all_codes.append(code)
    all_codes = sorted(all_codes)

    if not all_codes:
        raise FileNotFoundError(f'目录 {data_dir} 下无有效样本文件')

    rng.shuffle(all_codes)
    n_val = max(1, int(len(all_codes) * val_ratio))
    val_codes = set(all_codes[:n_val])
    train_codes = [c for c in all_codes if c not in val_codes]

    print(f'[split] 标的划分: 训练 {len(train_codes)} 只 / 验证 {len(val_codes)} 只')
    train_ds = CandleDataset(data_dir, code_filter=train_codes)
    val_ds = CandleDataset(data_dir, code_filter=list(val_codes))
    return train_ds, val_ds


if __name__ == '__main__':
    # 自测：需先用 sample_collector 生成样本
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(_THIS_DIR / 'data' / 'train')
    if os.path.isdir(data_dir):
        train_ds, val_ds = split_by_codes(data_dir)
        print(f'训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}')
        if len(train_ds) > 0:
            x, y = train_ds[0]
            print(f'单样本: x.shape={x.shape} dtype={x.dtype}, y={y.item()}')
    else:
        print(f'目录不存在: {data_dir}，请先运行 sample_collector.py 生成样本')
