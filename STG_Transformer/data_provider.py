import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Dict


class GasTurbineDataset(Dataset):
    """燃气轮机数据集"""
    def __init__(
        self,
        data_values: np.ndarray,
        seq_len: int = 24,
        pred_len: int = 1,
        target_indices: Optional[List[int]] = None,
        indices: Optional[List[int]] = None
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        # data_values: (T, N)
        self.data_values = data_values
        self.n_nodes = self.data_values.shape[1]
        if target_indices is None:
            self.target_indices = list(range(self.n_nodes))
        else:
            self.target_indices = target_indices
        if indices is None:
            self.indices = list(range(len(self.data_values) - self.seq_len - self.pred_len + 1))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        # x: (seq_len, N), y: (pred_len, n_targets)
        x = self.data_values[idx:idx + self.seq_len]  # (seq_len, n_nodes)
        y = self.data_values[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_indices]

        # 输出对齐模型输入/输出形状
        x = torch.FloatTensor(x).unsqueeze(-1)  # (seq_len, n_nodes, 1)
        y = torch.FloatTensor(y).transpose(0, 1).unsqueeze(-1)  # (n_targets, pred_len, 1)

        return x, y


def create_adjacency_matrix(n_nodes, connectivity='full'):
    """创建邻接矩阵"""
    if connectivity == 'full':
        adj = np.ones((n_nodes, n_nodes))
    elif connectivity == 'chain':
        adj = np.eye(n_nodes)
        for i in range(n_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
    else:
        adj = np.eye(n_nodes)

    return torch.FloatTensor(adj)


def _parse_feature_names(features_path: str) -> Optional[List[str]]:
    if not features_path or not os.path.exists(features_path):
        return None

    names = []
    with open(features_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "-" in line:
                left, right = line.split("-", 1)
                if left.strip().isdigit():
                    line = right.strip()
            match = re.search(r"\(([^)]+)\)", line)
            if match:
                name = match.group(1).strip()
            else:
                lower = line.lower()
                if "compressor decay" in lower:
                    name = "kMc"
                elif "turbine decay" in lower:
                    name = "kMt"
                else:
                    name = re.sub(r"[^A-Za-z0-9_]+", "_", line).strip("_")
                    if not name:
                        name = f"f{len(names) + 1}"
            names.append(name)

    seen = {}
    unique = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")

    return unique if unique else None


def _load_dataframe(
    data_path: str,
    drop_cols: Optional[List[str]] = None,
    features_path: Optional[str] = None
) -> pd.DataFrame:
    if data_path.lower().endswith(".txt"):
        # UCI CBM 的 data.txt 是空格分隔且无表头
        df = pd.read_csv(data_path, sep=r"\s+", header=None)
        if features_path is None:
            features_path = os.path.join(os.path.dirname(data_path), "Features.txt")
        feature_names = _parse_feature_names(features_path)
        if feature_names and len(feature_names) == df.shape[1]:
            df.columns = feature_names
        else:
            df.columns = [f"f{i + 1}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(data_path)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    return df


def get_dataloaders(
    data_path: str,
    batch_size: int = 32,
    seq_len: int = 24,
    pred_len: int = 1,
    train_ratio: float = 0.8,
    drop_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
    features_path: Optional[str] = None,
    num_workers: int = 4,
    drop_constant_cols: bool = False,
    split_mode: str = "sequential",
    split_seed: int = 42,
    split_block_len: int = 1000
) -> Tuple[DataLoader, DataLoader, StandardScaler, Dict[str, object]]:
    """获取训练/验证数据加载器及标准化器"""
    df = _load_dataframe(data_path, drop_cols=drop_cols, features_path=features_path)

    if drop_constant_cols:
        n_train_rows = int(len(df) * train_ratio)
        train_df = df.iloc[:n_train_rows]
        std = train_df.std(numeric_only=True)
        constant_cols = std[std == 0].index.tolist()
        if target_cols:
            constant_cols = [c for c in constant_cols if c not in target_cols]
        if constant_cols:
            df = df.drop(columns=constant_cols)

    if target_cols:
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Target columns not found: {missing}")
        target_indices = [df.columns.get_loc(col) for col in target_cols]
    elif "kMc" in df.columns and "kMt" in df.columns:
        target_indices = [df.columns.get_loc("kMc"), df.columns.get_loc("kMt")]
    elif df.shape[1] >= 2:
        target_indices = [df.shape[1] - 2, df.shape[1] - 1]
    else:
        target_indices = list(range(df.shape[1]))

    # 只用训练集拟合标准化器，避免泄漏
    scaler = StandardScaler()
    total_windows = len(df) - seq_len - pred_len + 1
    if total_windows <= 0:
        raise ValueError("Not enough data for the given seq_len and pred_len.")

    if split_mode == "random_windows":
        indices = np.arange(total_windows)
        rng = np.random.default_rng(split_seed)
        rng.shuffle(indices)
        n_train = int(total_windows * train_ratio)
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:].tolist()

        mask = np.zeros(len(df), dtype=bool)
        window_len = seq_len + pred_len
        for start in train_indices:
            mask[start:start + window_len] = True
        train_df = df.iloc[mask]
        scaler.fit(train_df.values)
        all_values = scaler.transform(df.values)

        train_dataset = GasTurbineDataset(all_values, seq_len, pred_len, target_indices, indices=train_indices)
        val_dataset = GasTurbineDataset(all_values, seq_len, pred_len, target_indices, indices=val_indices)
    elif split_mode == "random_blocks":
        window_len = seq_len + pred_len
        block_len = max(split_block_len, window_len)
        blocks = [(start, min(start + block_len, len(df))) for start in range(0, len(df), block_len)]
        rng = np.random.default_rng(split_seed)
        rng.shuffle(blocks)
        n_train_blocks = int(len(blocks) * train_ratio)
        train_blocks = blocks[:n_train_blocks]
        val_blocks = blocks[n_train_blocks:]

        def _block_indices(blocks_list):
            starts = []
            for start, end in blocks_list:
                last_start = end - window_len
                if last_start < start:
                    continue
                starts.extend(range(start, last_start + 1))
            return starts

        train_indices = _block_indices(train_blocks)
        val_indices = _block_indices(val_blocks)
        if not train_indices or not val_indices:
            raise ValueError("Not enough windows for random_blocks split; increase data or reduce split_block_len.")

        mask = np.zeros(len(df), dtype=bool)
        for start, end in train_blocks:
            mask[start:end] = True
        train_df = df.iloc[mask]
        scaler.fit(train_df.values)
        all_values = scaler.transform(df.values)

        train_dataset = GasTurbineDataset(all_values, seq_len, pred_len, target_indices, indices=train_indices)
        val_dataset = GasTurbineDataset(all_values, seq_len, pred_len, target_indices, indices=val_indices)
    else:
        n_train = int(len(df) * train_ratio)
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:]
        train_values = scaler.fit_transform(train_df.values)
        val_values = scaler.transform(val_df.values)
        train_dataset = GasTurbineDataset(train_values, seq_len, pred_len, target_indices)
        val_dataset = GasTurbineDataset(val_values, seq_len, pred_len, target_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    meta = {
        "columns": df.columns.tolist(),
        "target_indices": target_indices if target_indices is not None else list(range(len(df.columns))),
        "n_nodes": len(df.columns),
        "n_features": 1
    }

    return train_loader, val_loader, scaler, meta
