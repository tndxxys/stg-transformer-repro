import argparse
import csv
import os
from typing import List, Tuple

import numpy as np


def parse_cols(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_csv_numeric(data_path: str, drop_cols: List[str]) -> Tuple[List[str], np.ndarray]:
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    keep = [i for i, c in enumerate(header) if c not in set(drop_cols)]
    kept_header = [header[i] for i in keep]

    values = []
    for row in rows:
        vals = []
        bad = False
        for i in keep:
            try:
                vals.append(float(row[i]))
            except Exception:
                bad = True
                break
        if not bad:
            values.append(vals)

    if not values:
        raise ValueError("No numeric rows after dropping columns.")

    return kept_header, np.asarray(values, dtype=float)


def squeeze_series(x: np.ndarray) -> np.ndarray:
    # Expected output shape is usually [B, pred_len, 1] or [B, pred_len].
    # Convert to [N] for scalar metric calculation.
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    elif x.ndim == 3 and x.shape[-1] > 1:
        raise ValueError(
            "Array has multiple output channels. Use --target_index to select one channel first."
        )
    return x.reshape(-1)


def select_target_channel(x: np.ndarray, target_index: int) -> np.ndarray:
    if x.ndim < 3:
        return x
    if x.shape[-1] == 1:
        return x
    if target_index < 0 or target_index >= x.shape[-1]:
        raise ValueError(
            f"target_index {target_index} out of bounds for last dim size {x.shape[-1]}"
        )
    return x[..., target_index : target_index + 1]


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    # R2 on 1-D flattened series
    den = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if den == 0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(err ** 2) / den)
    return mse, rmse, mae, r2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction files with one fixed protocol to compare STG/iTransformer fairly."
    )
    parser.add_argument("--data_path", type=str, required=True, help="CSV data path")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to pred.npy")
    parser.add_argument("--true_file", type=str, required=True, help="Path to true.npy")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name, e.g. kMc or kMt")
    parser.add_argument("--drop_cols", type=str, default="date", help="Comma-separated columns to drop")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio used for normalization stats")
    parser.add_argument(
        "--target_index",
        type=int,
        default=0,
        help="If pred/true last dim has multiple channels, pick one target index.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        raise FileNotFoundError(f"pred_file not found: {args.pred_file}")
    if not os.path.exists(args.true_file):
        raise FileNotFoundError(f"true_file not found: {args.true_file}")

    drop_cols = parse_cols(args.drop_cols)
    cols, data = load_csv_numeric(args.data_path, drop_cols=drop_cols)
    if args.target_col not in cols:
        raise ValueError(f"target_col {args.target_col} not in columns: {cols}")
    target_idx = cols.index(args.target_col)

    n = len(data)
    n_train = int(n * args.train_ratio)
    train_target = data[:n_train, target_idx]
    t_std = float(np.std(train_target))
    t_min = float(np.min(train_target))
    t_max = float(np.max(train_target))
    t_range = float(t_max - t_min)
    if t_std == 0:
        raise ValueError(f"Train std is zero for target {args.target_col}.")
    if t_range == 0:
        raise ValueError(f"Train range is zero for target {args.target_col}.")

    pred = np.load(args.pred_file)
    true = np.load(args.true_file)

    pred = select_target_channel(pred, args.target_index)
    true = select_target_channel(true, args.target_index)

    y_pred = squeeze_series(pred)
    y_true = squeeze_series(true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"pred shape {y_pred.shape} != true shape {y_true.shape}")

    mse, rmse, mae, r2 = calc_metrics(y_true, y_pred)
    std_mse = mse / (t_std ** 2)
    std_rmse = rmse / t_std
    std_mae = mae / t_std
    nrmse_range = rmse / t_range

    print("=== Unified Eval (Paper Protocol Helper) ===")
    print(f"target: {args.target_col}")
    print(f"samples: {len(y_true)}")
    print(f"train_ratio: {args.train_ratio}")
    print(f"train_target_std: {t_std:.12g}")
    print(f"train_target_range: {t_range:.12g} ({t_min:.12g} -> {t_max:.12g})")
    print("")
    print(f"raw_mse: {mse:.12g}")
    print(f"raw_rmse: {rmse:.12g}")
    print(f"raw_mae: {mae:.12g}")
    print(f"raw_r2: {r2:.12g}")
    print("")
    print(f"std_mse(train_std): {std_mse:.12g}")
    print(f"std_rmse(train_std): {std_rmse:.12g}")
    print(f"std_mae(train_std): {std_mae:.12g}")
    print(f"nrmse(train_range): {nrmse_range:.12g}")


if __name__ == "__main__":
    main()
