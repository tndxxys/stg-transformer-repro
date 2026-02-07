import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from .model import STGTransformer
from .model_v2 import STGTransformerV2
from .data_provider import get_dataloaders
from .utils import calculate_metrics, inverse_transform_targets


def _select_targets(tensor, target_indices):
    # 从所有节点预测中抽取目标变量对应的节点
    idx = torch.tensor(target_indices, device=tensor.device)
    return tensor.index_select(1, idx)

def _is_finite(tensor):
    return torch.isfinite(tensor).all()

def _tensor_summary(name, tensor):
    if tensor is None:
        return f"{name}: None"
    finite = torch.isfinite(tensor)
    count = tensor.numel()
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    if finite_count > 0:
        t = tensor[finite]
        t_min = t.min().item()
        t_max = t.max().item()
        t_mean = t.mean().item()
        t_std = t.std(unbiased=False).item()
    else:
        t_min = float("nan")
        t_max = float("nan")
        t_mean = float("nan")
        t_std = float("nan")
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"finite={finite_count}/{count} nan={nan_count} inf={inf_count} "
        f"min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g} std={t_std:.6g}"
    )

def _nonfinite_param_report(model, limit=5):
    bad = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            nan_count = int(torch.isnan(param).sum().item())
            inf_count = int(torch.isinf(param).sum().item())
            bad.append(
                f"{name}: shape={tuple(param.shape)} nan={nan_count} inf={inf_count}"
            )
            if len(bad) >= limit:
                break
    if not bad:
        return "all_finite"
    suffix = " ..." if len(bad) >= limit else ""
    return "; ".join(bad) + suffix

def train_epoch(model, dataloader, criterion, optimizer, device, target_indices, grad_clip):
    model.train()
    total_loss = 0
    reported = False

    for batch_x, batch_y in tqdm(dataloader, desc="Training"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        # output: (B, N, pred_len, 1)
        output = model(batch_x)
        if not _is_finite(output):
            print("Warning: non-finite output in training batch; skipping.")
            if not reported:
                print(_tensor_summary("train_batch_x", batch_x))
                print(_tensor_summary("train_output", output))
                print(f"Non-finite params: {_nonfinite_param_report(model)}")
                reported = True
            optimizer.zero_grad()
            continue
        output = _select_targets(output, target_indices)

        loss = criterion(output, batch_y)
        if not torch.isfinite(loss):
            print("Warning: non-finite loss in training batch; skipping.")
            if not reported:
                print(_tensor_summary("train_batch_x", batch_x))
                print(_tensor_summary("train_output", output))
                print(_tensor_summary("train_loss", loss))
                print(f"Non-finite params: {_nonfinite_param_report(model)}")
                reported = True
            optimizer.zero_grad()
            continue
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, target_indices, scaler):
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    reported = False

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 验证阶段不反向传播
            output = model(batch_x)
            if not _is_finite(output):
                print("Warning: non-finite output in validation batch; replacing with zeros.")
                if not reported:
                    print(_tensor_summary("val_batch_x", batch_x))
                    print(_tensor_summary("val_output", output))
                    print(f"Non-finite params: {_nonfinite_param_report(model)}")
                    reported = True
                output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            output = _select_targets(output, target_indices)
            loss = criterion(output, batch_y)

            if torch.isfinite(loss):
                total_loss += loss.item()
            all_preds.append(output.detach().cpu())
            all_trues.append(batch_y.detach().cpu())

    # (B, pred_len, n_targets) 便于指标计算
    preds = torch.cat(all_preds, dim=0).squeeze(-1).permute(0, 2, 1).numpy()
    trues = torch.cat(all_trues, dim=0).squeeze(-1).permute(0, 2, 1).numpy()

    # 标准化尺度指标（便于与论文无量纲指标对齐）
    metrics_std = calculate_metrics(
        trues.reshape(-1, trues.shape[-1]),
        preds.reshape(-1, preds.shape[-1])
    )

    preds = inverse_transform_targets(scaler, preds, target_indices)
    trues = inverse_transform_targets(scaler, trues, target_indices)
    metrics_raw = calculate_metrics(
        trues.reshape(-1, trues.shape[-1]),
        preds.reshape(-1, preds.shape[-1])
    )

    metrics = {
        **metrics_raw,
        "Std_MSE": metrics_std["MSE"],
        "Std_RMSE": metrics_std["RMSE"],
        "Std_MAE": metrics_std["MAE"],
        "Std_R2": metrics_std["R2"],
    }

    return total_loss / len(dataloader), metrics


def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

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
        split_block_len=args.split_block_len
    )

    model_kwargs = dict(
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
    )
    if args.model_variant == "v2":
        model = STGTransformerV2(
            **model_kwargs,
            fusion_layers=args.fusion_layers,
        ).to(device)
    else:
        model = STGTransformer(**model_kwargs).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    writer = SummaryWriter(log_dir=args.log_dir)
    target_indices = meta["target_indices"]

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, target_indices, args.grad_clip)
        val_loss, metrics = validate(model, val_loader, criterion, device, target_indices, scaler)

        scheduler.step(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/MSE', metrics['MSE'], epoch)
        writer.add_scalar('Metrics/MAE', metrics['MAE'], epoch)
        writer.add_scalar('Metrics/RMSE', metrics['RMSE'], epoch)
        writer.add_scalar('MetricsStd/MSE', metrics['Std_MSE'], epoch)
        writer.add_scalar('MetricsStd/MAE', metrics['Std_MAE'], epoch)
        writer.add_scalar('MetricsStd/RMSE', metrics['Std_RMSE'], epoch)

        print(
            f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}, RMSE: {metrics["RMSE"]:.4f}, '
            f'Std RMSE: {metrics["Std_RMSE"]:.4f}'
        )

        if math.isfinite(val_loss) and (val_loss < best_val_loss - args.early_stop_delta):
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), f'{args.save_dir}/best_model.pth')
        else:
            patience += 1
            if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f}).")
                break

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--target_cols', type=str, default=None)
    parser.add_argument('--drop_cols', type=str, default=None)
    parser.add_argument('--features_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--gnn_layers', type=int, default=1)
    parser.add_argument('--enc_layers', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--freq_ratio', type=float, default=0.1)
    parser.add_argument('--freq_cutoff', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--corr_threshold', type=float, default=None)
    parser.add_argument('--model_variant', type=str, default='base', choices=['base', 'v2'])
    parser.add_argument('--fusion_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--early_stop_delta', type=float, default=0.0)
    parser.add_argument('--drop_constant_cols', action='store_true')
    parser.add_argument('--keep_constant_cols', dest='drop_constant_cols', action='store_false')
    parser.set_defaults(drop_constant_cols=True)
    parser.add_argument('--random_split', action='store_true', help='Deprecated; use --split_mode random_windows')
    parser.add_argument('--split_mode', type=str, default=None, choices=['sequential', 'random_windows', 'random_blocks'])
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--split_block_len', type=int, default=1000)

    args = parser.parse_args()
    if args.split_mode is None:
        args.split_mode = 'random_windows' if args.random_split else 'sequential'
    main(args)
