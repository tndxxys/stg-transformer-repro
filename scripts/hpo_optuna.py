#!/usr/bin/env python3
import argparse
import json
import math
import os
import random

import numpy as np
import optuna
import torch

from STG_Transformer.data_provider import get_dataloaders
from STG_Transformer.model import STGTransformer
from STG_Transformer.train import train_epoch, validate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    seed = args.seed + trial.number
    set_seed(seed)

    # Search space
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    d_ff = trial.suggest_categorical("d_ff", [128, 256, 512])
    gnn_layers = trial.suggest_categorical("gnn_layers", [1, 2])
    enc_layers = trial.suggest_categorical("enc_layers", [1, 2])
    top_k = trial.suggest_categorical("top_k", [2, 4, 6, 8])
    freq_ratio = trial.suggest_categorical("freq_ratio", [0.05, 0.1, 0.2])

    device = select_device()

    train_loader, val_loader, scaler, meta = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        drop_cols=args.drop_cols,
        target_cols=args.target_cols,
        features_path=args.features_path,
        num_workers=args.num_workers,
        drop_constant_cols=args.drop_constant_cols,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        split_block_len=args.split_block_len,
    )

    model = STGTransformer(
        n_features=meta["n_features"],
        n_nodes=meta["n_nodes"],
        d_model=d_model,
        n_heads=args.n_heads,
        d_ff=d_ff,
        dropout=dropout,
        pred_len=args.pred_len,
        gnn_layers=gnn_layers,
        enc_layers=enc_layers,
        dec_layers=args.dec_layers,
        freq_ratio=freq_ratio,
        freq_cutoff=args.freq_cutoff,
        top_k=top_k,
        corr_threshold=args.corr_threshold,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    best = math.inf
    patience = 0
    for epoch in range(args.epochs):
        _ = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            meta["target_indices"],
            args.grad_clip,
        )
        val_loss, metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            meta["target_indices"],
            scaler,
        )
        scheduler.step(val_loss)

        std_rmse = metrics["Std_RMSE"]
        trial.report(std_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if std_rmse < best - args.early_stop_delta:
            best = std_rmse
            patience = 0
        else:
            patience += 1
            if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
                break

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for STG-Transformer")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--target_cols", type=lambda s: [c.strip() for c in s.split(",") if c.strip()], default=None)
    parser.add_argument("--drop_cols", type=lambda s: [c.strip() for c in s.split(",") if c.strip()], default=None)
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--freq_cutoff", type=int, default=None)
    parser.add_argument("--corr_threshold", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_delta", type=float, default=0.0)
    parser.add_argument("--drop_constant_cols", action="store_true")
    parser.add_argument("--keep_constant_cols", dest="drop_constant_cols", action="store_false")
    parser.set_defaults(drop_constant_cols=True)
    parser.add_argument("--split_mode", type=str, default="sequential", choices=["sequential", "random_windows", "random_blocks"])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_block_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    target_tag = "all"
    if args.target_cols:
        target_tag = "_".join(args.target_cols)
    if args.study_name is None:
        args.study_name = f"hpo_{target_tag}_s{args.seq_len}_p{args.pred_len}"
    if args.out is None:
        args.out = f"hpo_best_{target_tag}_s{args.seq_len}_p{args.pred_len}.json"

    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, timeout=args.timeout)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best Std RMSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
