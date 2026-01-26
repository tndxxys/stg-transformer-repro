#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_adjacency(values, mode, top_k=None, threshold=None):
    n_nodes = values.shape[1]
    if mode == "full":
        adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 1.0)

    if threshold is not None:
        corr[corr < threshold] = 0.0

    if mode == "topk" and top_k is not None and top_k < n_nodes:
        mask = np.zeros_like(corr, dtype=np.float32)
        for i in range(n_nodes):
            idx = np.argsort(-corr[i])[: top_k + 1]
            mask[i, idx] = 1.0
        mask = np.maximum(mask, mask.T)
        corr = corr * mask

    return corr


def write_adj_csv(adj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("from,to,distance\n")
        n = adj.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if adj[i, j] > 0:
                    f.write(f"{i},{j},1\n")


def write_destgnn_config(path: Path, data_dir: Path, n_nodes: int, pred_len: int):
    content = f"""[Data]
adj_filename = {data_dir.as_posix()}/cbm_adj.npy
graph_signal_matrix_filename = {data_dir.as_posix()}/cbm.npz
num_of_vertices = {n_nodes}
points_per_hour = {pred_len}
num_for_predict = {pred_len}
len_input = {pred_len}
dataset_name = CBM


[Training]
use_nni = 3
batch_size = 8
model_name = DeSTGNN
num_of_weeks = 0
num_of_days = 0
num_of_hours = 1
start_epoch = 0
epochs = 120
fine_tune_epochs = 80
learning_rate = 0.001
direction = 2
encoder_input_size = 1
decoder_input_size = 1
dropout = 0.0
kernel_size = 3
num_layers = 4
d_model = 64
nb_head = 8
ScaledSAt = 1
SE = 1
smooth_layer_num = 0
aware_temporal_context = 1
TE = 1
train_random = 1
eval_random = 0
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_matgcn_config(path: Path, data_dir: Path, n_nodes: int, pred_len: int, hours):
    payload = {
        "lr": 0.001,
        "epochs": 80,
        "batch_size": 48,
        "data_split": 0.8,
        "adj_file": f"{data_dir.as_posix()}/cbm_distance.csv",
        "data_file": f"{data_dir.as_posix()}/cbm.npz",
        "saved_dir": "saved/cbm",
        "n_nodes": n_nodes,
        "out_timesteps": pred_len,
        "points_per_hour": pred_len,
        "hours": hours,
        "device_for_data": "cpu",
        "device_for_model": "cuda:0"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare graph baseline inputs for UCI CBM.")
    parser.add_argument("--csv_path", required=True, help="Path to uci_cbm.csv")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction length")
    parser.add_argument("--adj_mode", choices=["corr", "topk", "full"], default="corr")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--hours", type=str, default="1,2,3", help="Hour steps list for MATGCN")
    parser.add_argument("--destgcn_dir", type=str, default="baselines/DeSTGNN_full/data/CBM")
    parser.add_argument("--destgcn_config", type=str, default="baselines/DeSTGNN_full/configurations/CBM.conf")
    parser.add_argument("--matgcn_dir", type=str, default="baselines/matgcn/data/cbm")
    parser.add_argument("--matgcn_config", type=str, default="baselines/matgcn/config/cbm.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    values = df.values.astype(np.float32)
    n_nodes = values.shape[1]

    data = values[:, :, None]  # (T, N, 1)

    destgcn_dir = Path(args.destgcn_dir)
    matgcn_dir = Path(args.matgcn_dir)
    destgcn_dir.mkdir(parents=True, exist_ok=True)
    matgcn_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(destgcn_dir / "cbm.npz", data=data)
    np.savez_compressed(matgcn_dir / "cbm.npz", data=data)

    adj = build_adjacency(values, args.adj_mode, args.top_k, args.threshold)
    np.save(destgcn_dir / "cbm_adj.npy", adj)
    write_adj_csv(adj, matgcn_dir / "cbm_distance.csv")

    write_destgnn_config(Path(args.destgcn_config), destgcn_dir, n_nodes, args.pred_len)
    hours = [int(x) for x in args.hours.split(",") if x.strip()]
    write_matgcn_config(Path(args.matgcn_config), matgcn_dir, n_nodes, args.pred_len, hours)

    print(f"Prepared graph data with {n_nodes} nodes, pred_len={args.pred_len}")


if __name__ == "__main__":
    main()
