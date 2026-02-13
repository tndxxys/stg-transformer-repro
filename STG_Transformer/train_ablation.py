import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data_provider import get_dataloaders
from .model_ablation import STGAblationModel
from .utils import calculate_metrics, inverse_transform_targets


def _select_targets(tensor, target_indices):
    idx = torch.tensor(target_indices, device=tensor.device)
    return tensor.index_select(1, idx)


def _set_seed(seed):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_mode(mode):
    mode = mode.lower()
    mapping = {
        "backbone": (False, False, False),
        "m1": (True, False, False),
        "m1_m2": (True, True, False),
        "m1_m2_m3": (True, True, True),
        "full": (True, True, True),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported ablation_mode: {mode}")
    return mapping[mode]


def _train_epoch(model, dataloader, criterion, optimizer, device, target_indices, grad_clip):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        output = _select_targets(output, target_indices)
        loss = criterion(output, batch_y)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float("inf")
    return total_loss / total_batches


def _evaluate(model, dataloader, criterion, device, target_indices, scaler):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            output = _select_targets(output, target_indices)
            loss = criterion(output, batch_y)
            if torch.isfinite(loss):
                total_loss += loss.item()
                total_batches += 1

            all_preds.append(output.detach().cpu())
            all_trues.append(batch_y.detach().cpu())

    preds_std = torch.cat(all_preds, dim=0).squeeze(-1).permute(0, 2, 1).numpy()
    trues_std = torch.cat(all_trues, dim=0).squeeze(-1).permute(0, 2, 1).numpy()

    std_metrics = calculate_metrics(
        trues_std.reshape(-1, trues_std.shape[-1]),
        preds_std.reshape(-1, preds_std.shape[-1]),
    )

    preds_raw = inverse_transform_targets(scaler, preds_std, target_indices)
    trues_raw = inverse_transform_targets(scaler, trues_std, target_indices)
    raw_metrics = calculate_metrics(
        trues_raw.reshape(-1, trues_raw.shape[-1]),
        preds_raw.reshape(-1, preds_raw.shape[-1]),
    )

    metrics = {
        "MSE": float(raw_metrics["MSE"]),
        "RMSE": float(raw_metrics["RMSE"]),
        "MAE": float(raw_metrics["MAE"]),
        "R2": float(raw_metrics["R2"]),
        "Std_MSE": float(std_metrics["MSE"]),
        "Std_RMSE": float(std_metrics["RMSE"]),
        "Std_MAE": float(std_metrics["MAE"]),
        "Std_R2": float(std_metrics["R2"]),
    }

    avg_loss = float(total_loss / total_batches) if total_batches > 0 else float("inf")
    return avg_loss, metrics, preds_raw, trues_raw, preds_std, trues_std


def _choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    _set_seed(args.seed)
    device = _choose_device()

    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()] if args.target_cols else None
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else None

    train_loader, val_loader, scaler, meta = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        drop_cols=drop_cols,
        target_cols=target_cols,
        features_path=args.features_path,
        num_workers=args.num_workers,
        drop_constant_cols=args.drop_constant_cols,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        split_block_len=args.split_block_len,
    )

    use_m1, use_m2, use_m3 = _resolve_mode(args.ablation_mode)
    model = STGAblationModel(
        n_features=meta["n_features"],
        n_nodes=meta["n_nodes"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pred_len=args.pred_len,
        gnn_layers=args.gnn_layers,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        freq_ratio=args.freq_ratio,
        freq_cutoff=args.freq_cutoff,
        top_k=args.top_k,
        corr_threshold=args.corr_threshold,
        use_m1=use_m1,
        use_m2=use_m2,
        use_m3=use_m3,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    run_name = args.experiment_name
    if not run_name:
        run_name = f"ablation_{args.ablation_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    target_indices = meta["target_indices"]
    best_val_loss = float("inf")
    best_payload = None
    patience_count = 0
    history = []

    for epoch in range(args.epochs):
        train_loss = _train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            target_indices=target_indices,
            grad_clip=args.grad_clip,
        )

        val_loss, metrics, preds_raw, trues_raw, preds_std, trues_std = _evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            target_indices=target_indices,
            scaler=scaler,
        )
        scheduler.step(val_loss)

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "RMSE": float(metrics["RMSE"]),
            "Std_RMSE": float(metrics["Std_RMSE"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, RMSE: {metrics['RMSE']:.4f}, "
            f"Std RMSE: {metrics['Std_RMSE']:.4f}"
        )

        if math.isfinite(val_loss) and (val_loss < best_val_loss - args.early_stop_delta):
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            best_payload = {
                "metrics": metrics,
                "pred_raw": preds_raw,
                "true_raw": trues_raw,
                "pred_std": preds_std,
                "true_std": trues_std,
                "best_epoch": epoch + 1,
                "best_val_loss": float(val_loss),
            }
        else:
            patience_count += 1
            if args.early_stop_patience > 0 and patience_count >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val loss: {best_val_loss:.4f})."
                )
                break

    if best_payload is None:
        raise RuntimeError("Training failed: no valid checkpoint was produced.")

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload["metrics"], f, indent=2)

    np.save(os.path.join(run_dir, "pred.npy"), best_payload["pred_raw"])
    np.save(os.path.join(run_dir, "true.npy"), best_payload["true_raw"])
    np.save(os.path.join(run_dir, "pred_std.npy"), best_payload["pred_std"])
    np.save(os.path.join(run_dir, "true_std.npy"), best_payload["true_std"])

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "run_name": run_name,
        "run_dir": run_dir,
        "device": str(device),
        "ablation_mode": args.ablation_mode,
        "module_switches": {
            "use_m1": use_m1,
            "use_m2": use_m2,
            "use_m3": use_m3,
        },
        "data": {
            "data_path": args.data_path,
            "target_cols": target_cols,
            "drop_cols": drop_cols,
            "drop_constant_cols": args.drop_constant_cols,
            "split_mode": args.split_mode,
            "train_ratio": args.train_ratio,
            "split_seed": args.split_seed,
            "split_block_len": args.split_block_len,
            "columns": meta["columns"],
            "target_indices": meta["target_indices"],
        },
        "training": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_delta": args.early_stop_delta,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "model": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "gnn_layers": args.gnn_layers,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "freq_ratio": args.freq_ratio,
            "freq_cutoff": args.freq_cutoff,
            "top_k": args.top_k,
            "corr_threshold": args.corr_threshold,
        },
        "best_epoch": best_payload["best_epoch"],
        "best_val_loss": best_payload["best_val_loss"],
        "artifacts": {
            "metrics": os.path.join(run_dir, "metrics.json"),
            "pred": os.path.join(run_dir, "pred.npy"),
            "true": os.path.join(run_dir, "true.npy"),
            "pred_std": os.path.join(run_dir, "pred_std.npy"),
            "true_std": os.path.join(run_dir, "true_std.npy"),
            "history": os.path.join(run_dir, "history.json"),
            "checkpoint": os.path.join(run_dir, "best_model.pth"),
        },
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Saved metrics to: {os.path.join(run_dir, 'metrics.json')}")
    print(f"Saved meta to: {os.path.join(run_dir, 'run_meta.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STG ablation variants and export unified artifacts.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--target_cols", type=str, default=None)
    parser.add_argument("--drop_cols", type=str, default="date")
    parser.add_argument("--drop_constant_cols", action="store_true")
    parser.add_argument("--keep_constant_cols", dest="drop_constant_cols", action="store_false")
    parser.set_defaults(drop_constant_cols=True)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--ablation_mode", type=str, default="m1_m2_m3",
                        choices=["backbone", "m1", "m1_m2", "m1_m2_m3", "full"])

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--freq_ratio", type=float, default=0.1)
    parser.add_argument("--freq_cutoff", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--corr_threshold", type=float, default=None)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_mode", type=str, default="sequential",
                        choices=["sequential", "random_windows", "random_blocks"])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_block_len", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--early_stop_delta", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_root", type=str, default="./experiments/ablation_runs")
    parser.add_argument("--experiment_name", type=str, default=None)

    main(parser.parse_args())
