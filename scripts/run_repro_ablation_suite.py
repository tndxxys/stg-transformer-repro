import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd


TSL_MODELS = [
    "DLinear",
    "FiLM",
    "SCINet",
    "TimesNet",
    "iTransformer",
    "Autoformer",
    "Crossformer",
    "PatchTST",
]

ABLATION_MODES = [
    ("backbone", "Backbone"),
    ("m1", "Backbone+M1"),
    ("m1_m2", "Backbone+M1+M2"),
    ("m1_m2_m3", "Backbone+M1+M2+M3"),
]


STG_BEST_PARAMS = {
    "kMc": {
        96: {
            "lr": 0.0005078368289055296,
            "dropout": 0.4346368025701775,
            "weight_decay": 4.8374700814857773e-05,
            "d_model": 128,
            "d_ff": 128,
            "gnn_layers": 1,
            "enc_layers": 1,
            "top_k": 4,
            "freq_ratio": 0.05,
        },
        192: {
            "batch_size": 16,
            "lr": 0.0001505514988046204,
            "dropout": 0.3762362346369281,
            "weight_decay": 3.168904034072666e-05,
            "d_model": 128,
            "d_ff": 256,
            "gnn_layers": 2,
            "enc_layers": 1,
            "top_k": 2,
            "freq_ratio": 0.05,
        },
        336: {
            "lr": 0.0005435844547909818,
            "dropout": 0.43664740065549473,
            "weight_decay": 0.00072233337412830551,
            "d_model": 128,
            "d_ff": 512,
            "gnn_layers": 2,
            "enc_layers": 1,
            "top_k": 2,
            "freq_ratio": 0.1,
        },
    },
    "kMt": {
        96: {
            "lr": 0.0006522383045206187,
            "dropout": 0.10709997850395263,
            "weight_decay": 2.468294818486875e-05,
            "d_model": 128,
            "d_ff": 128,
            "gnn_layers": 2,
            "enc_layers": 1,
            "top_k": 6,
            "freq_ratio": 0.05,
        },
        192: {
            "epochs": 100,
            "early_stop_patience": 15,
            "lr": 0.0008892069371284909,
            "dropout": 0.11642409491508082,
            "weight_decay": 0.00021570362493032383,
            "d_model": 64,
            "d_ff": 256,
            "gnn_layers": 1,
            "enc_layers": 2,
            "top_k": 4,
            "freq_ratio": 0.05,
        },
        336: {
            "lr": 0.0008575084816196847,
            "dropout": 0.13043841410342713,
            "weight_decay": 2.7872184324978292e-05,
            "d_model": 64,
            "d_ff": 128,
            "gnn_layers": 1,
            "enc_layers": 2,
            "top_k": 4,
            "freq_ratio": 0.1,
        },
    },
}


def _now():
    return datetime.now().isoformat()


def _run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metrics_from_arrays(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    err = pred - true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def _extract_target_array(arr, target_idx, n_features):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(-1)
    candidate_axes = [ax for ax, size in enumerate(arr.shape) if size == n_features]
    if candidate_axes:
        arr = np.take(arr, indices=target_idx, axis=candidate_axes[0])
    return arr.reshape(-1)


def _run_command(cmd, log_path, cwd):
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n[{_now()}] CMD: {' '.join(shlex.quote(x) for x in cmd)}\n")
        log_f.flush()
        rc = subprocess.run(cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - start


def _record_path(records_dir, exp_id):
    return os.path.join(records_dir, f"{exp_id}.json")


def _record_is_success(path):
    if not os.path.exists(path):
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    return payload.get("status") == "SUCCESS"


def _write_record(records_dir, payload):
    _write_json(_record_path(records_dir, payload["exp_id"]), payload)


def _prepare_clean_csv(raw_csv, out_csv):
    df = pd.read_csv(raw_csv)
    dropped = []
    if "date" in df.columns:
        numeric_cols = [c for c in df.columns if c != "date"]
    else:
        numeric_cols = list(df.columns)
    nunique = df[numeric_cols].nunique(dropna=False)
    constant_cols = [c for c in numeric_cols if int(nunique[c]) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        dropped.extend(constant_cols)
    df.to_csv(out_csv, index=False)
    feature_cols = [c for c in df.columns if c != "date"]
    return {
        "clean_csv": out_csv,
        "dropped_constant_cols": dropped,
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
    }


def _find_new_result_dir(before_dirs, result_root):
    after_dirs = set(glob.glob(os.path.join(result_root, "*")))
    new_dirs = [d for d in after_dirs if d not in before_dirs and os.path.isdir(d)]
    if not new_dirs:
        return None
    new_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return new_dirs[0]


def _read_metrics_npy(path):
    arr = np.load(path)
    if arr.ndim == 0:
        return None
    flat = arr.reshape(-1)
    if flat.size < 3:
        return None
    # convention: [mae, mse, rmse, ...]
    return {"MSE": float(flat[1]), "MAE": float(flat[0]), "RMSE": float(flat[2])}


def _build_stg_ablation_cmd(args, clean_csv, target, horizon, mode, exp_id):
    hp = dict(STG_BEST_PARAMS[target][horizon])
    cmd = [
        args.python_bin,
        "-m",
        "STG_Transformer.train_ablation",
        "--data_path",
        clean_csv,
        "--seq_len",
        str(horizon),
        "--pred_len",
        str(horizon),
        "--target_cols",
        target,
        "--drop_cols",
        "date",
        "--split_mode",
        "sequential",
        "--train_ratio",
        "0.8",
        "--ablation_mode",
        mode,
        "--output_root",
        os.path.join(args.suite_dir, "artifacts", "stg_ablation"),
        "--experiment_name",
        exp_id,
        "--epochs",
        str(int(hp.pop("epochs", args.stg_epochs))),
        "--early_stop_patience",
        str(int(hp.pop("early_stop_patience", args.stg_patience))),
        "--seed",
        str(args.seed),
    ]
    for key, value in hp.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _build_tsl_cmd(args, clean_csv, clean_root, target, horizon, model, exp_id, c_in):
    return [
        args.python_bin,
        "baselines/Time-Series-Library/run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", exp_id,
        "--model", model,
        "--data", "custom",
        "--root_path", clean_root,
        "--data_path", os.path.basename(clean_csv),
        "--features", "MS",
        "--target", target,
        "--freq", "t",
        "--seq_len", str(horizon),
        "--label_len", str(horizon // 2),
        "--pred_len", str(horizon),
        "--enc_in", str(c_in),
        "--dec_in", str(c_in),
        "--c_out", "1",
        "--train_ratio", "0.8",
        "--val_ratio", "0.2",
        "--test_ratio", "0.0",
        "--no_time_features",
        "--train_epochs", str(args.baseline_epochs),
        "--patience", str(args.baseline_patience),
        "--des", exp_id,
    ]


def _build_timexer_cmd(args, clean_csv, clean_root, target, horizon, exp_id, c_in):
    return [
        args.python_bin,
        "baselines/TimeXer/run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", exp_id,
        "--model", "TimeXer",
        "--data", "custom",
        "--root_path", clean_root,
        "--data_path", os.path.basename(clean_csv),
        "--features", "MS",
        "--target", target,
        "--freq", "t",
        "--seq_len", str(horizon),
        "--label_len", str(horizon // 2),
        "--pred_len", str(horizon),
        "--enc_in", str(c_in),
        "--dec_in", str(c_in),
        "--c_out", "1",
        "--train_epochs", str(args.baseline_epochs),
        "--patience", str(args.baseline_patience),
        "--des", exp_id,
    ]


def _build_wpmixer_cmd(args, clean_csv, clean_root, target, horizon, c_in):
    return [
        args.python_bin,
        "baselines/WPMixer/run_LTF.py",
        "--data", "custom",
        "--root_path", clean_root,
        "--data_path", os.path.basename(clean_csv),
        "--features", "MS",
        "--target", target,
        "--freq", "t",
        "--c_in", str(c_in),
        "--c_out", "1",
        "--seq_len", str(horizon),
        "--pred_len", str(horizon),
        "--train_epochs", str(args.baseline_epochs),
        "--patience", str(args.baseline_patience),
    ]


def _build_graph_prepare_cmd(args, clean_csv, horizon):
    return [
        args.python_bin,
        "scripts/prepare_graph_baselines.py",
        "--csv_path", clean_csv,
        "--pred_len", str(horizon),
        "--drop_constant_cols",
    ]


def _build_destgnn_train_cmd(args):
    return [
        args.python_bin,
        "baselines/DeSTGNN_full/prepareData.py",
        "--config",
        "baselines/DeSTGNN_full/configurations/CBM.conf",
    ], [
        args.python_bin,
        "baselines/DeSTGNN_full/train_DeSTGNN.py",
        "--config",
        "baselines/DeSTGNN_full/configurations/CBM.conf",
        "--cuda",
        str(args.cuda_id),
    ]


def _build_matgcn_train_cmd(args):
    return [
        args.python_bin,
        "baselines/matgcn/main.py",
        "baselines/matgcn/config/cbm.json",
    ]


def _collect_destgnn_npz():
    candidates = glob.glob("baselines/DeSTGNN_full/experiments/CBM/**/output_epoch_*_test.npz", recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _collect_matgcn_saved_dir():
    candidates = glob.glob("baselines/matgcn/saved/cbm/*")
    if not candidates:
        return None
    candidates = [p for p in candidates if os.path.isdir(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _run_stg_ablation(args, records_dir, logs_dir, clean_csv, target, horizon):
    for mode, paper_name in ABLATION_MODES:
        exp_id = f"ablation_{paper_name.replace('+', '_').replace('-', '_')}_{target}_{horizon}"
        record_file = _record_path(records_dir, exp_id)
        if _record_is_success(record_file):
            continue

        cmd = _build_stg_ablation_cmd(args, clean_csv, target, horizon, mode, exp_id)
        log_path = os.path.join(logs_dir, f"{exp_id}.log")
        started = _now()
        rc, duration = _run_command(cmd, log_path, cwd=args.repo_root)

        run_dir = os.path.join(args.suite_dir, "artifacts", "stg_ablation", exp_id)
        metrics_path = os.path.join(run_dir, "metrics.json")
        pred_path = os.path.join(run_dir, "pred.npy")
        true_path = os.path.join(run_dir, "true.npy")

        if rc == 0 and os.path.exists(metrics_path):
            metrics_obj = _read_json(metrics_path)
            metrics = {
                "MSE": _safe_float(metrics_obj.get("Std_MSE", metrics_obj.get("MSE"))),
                "MAE": _safe_float(metrics_obj.get("Std_MAE", metrics_obj.get("MAE"))),
                "RMSE": _safe_float(metrics_obj.get("Std_RMSE", metrics_obj.get("RMSE"))),
            }
            status = "SUCCESS"
            err = None
        else:
            metrics = None
            status = "FAILED"
            err = f"command_return_code={rc}"

        _write_record(records_dir, {
            "exp_id": exp_id,
            "group": "ablation",
            "paper_model": paper_name,
            "runner": "stg_ablation",
            "target": target,
            "horizon": int(horizon),
            "status": status,
            "started_at": started,
            "ended_at": _now(),
            "duration_sec": duration,
            "command": cmd,
            "log_path": log_path,
            "artifacts": {
                "run_dir": run_dir,
                "metrics": metrics_path,
                "pred": pred_path,
                "true": true_path,
            },
            "metrics": metrics,
            "error": err,
        })


def _run_tsl_model(args, records_dir, logs_dir, clean_csv, clean_root, target, horizon, model, c_in):
    exp_id = f"sota_{model}_{target}_{horizon}"
    if _record_is_success(_record_path(records_dir, exp_id)):
        return

    before = set(glob.glob(os.path.join(args.repo_root, "results", "*")))
    cmd = _build_tsl_cmd(args, clean_csv, clean_root, target, horizon, model, exp_id, c_in)
    log_path = os.path.join(logs_dir, f"{exp_id}.log")
    started = _now()
    rc, duration = _run_command(cmd, log_path, cwd=args.repo_root)

    new_dir = _find_new_result_dir(before, os.path.join(args.repo_root, "results"))
    metrics_path = os.path.join(new_dir, "metrics.npy") if new_dir else None
    pred_path = os.path.join(new_dir, "pred.npy") if new_dir else None
    true_path = os.path.join(new_dir, "true.npy") if new_dir else None

    if rc == 0 and metrics_path and os.path.exists(metrics_path):
        metrics = _read_metrics_npy(metrics_path)
        status = "SUCCESS" if metrics else "FAILED"
        err = None if metrics else "metrics_parse_failed"
    else:
        metrics = None
        status = "FAILED"
        err = f"command_return_code={rc}"

    _write_record(records_dir, {
        "exp_id": exp_id,
        "group": "sota",
        "paper_model": model,
        "runner": "time_series_library",
        "target": target,
        "horizon": int(horizon),
        "status": status,
        "started_at": started,
        "ended_at": _now(),
        "duration_sec": duration,
        "command": cmd,
        "log_path": log_path,
        "artifacts": {
            "run_dir": new_dir,
            "metrics": metrics_path,
            "pred": pred_path,
            "true": true_path,
        },
        "metrics": metrics,
        "error": err,
    })


def _run_timexer(args, records_dir, logs_dir, clean_csv, clean_root, target, horizon, c_in):
    exp_id = f"sota_TimeXer_{target}_{horizon}"
    if _record_is_success(_record_path(records_dir, exp_id)):
        return

    before = set(glob.glob(os.path.join(args.repo_root, "results", "*")))
    cmd = _build_timexer_cmd(args, clean_csv, clean_root, target, horizon, exp_id, c_in)
    log_path = os.path.join(logs_dir, f"{exp_id}.log")
    started = _now()
    rc, duration = _run_command(cmd, log_path, cwd=args.repo_root)

    new_dir = _find_new_result_dir(before, os.path.join(args.repo_root, "results"))
    metrics_path = os.path.join(new_dir, "metrics.npy") if new_dir else None
    pred_path = os.path.join(new_dir, "pred.npy") if new_dir else None
    true_path = os.path.join(new_dir, "true.npy") if new_dir else None

    if rc == 0 and metrics_path and os.path.exists(metrics_path):
        metrics = _read_metrics_npy(metrics_path)
        status = "SUCCESS" if metrics else "FAILED"
        err = None if metrics else "metrics_parse_failed"
    else:
        metrics = None
        status = "FAILED"
        err = f"command_return_code={rc}"

    _write_record(records_dir, {
        "exp_id": exp_id,
        "group": "sota",
        "paper_model": "TimeXer",
        "runner": "timexer",
        "target": target,
        "horizon": int(horizon),
        "status": status,
        "started_at": started,
        "ended_at": _now(),
        "duration_sec": duration,
        "command": cmd,
        "log_path": log_path,
        "artifacts": {
            "run_dir": new_dir,
            "metrics": metrics_path,
            "pred": pred_path,
            "true": true_path,
        },
        "metrics": metrics,
        "error": err,
    })


def _run_wpmixer(args, records_dir, logs_dir, clean_csv, clean_root, target, horizon, c_in):
    exp_id = f"sota_WPMixer_{target}_{horizon}"
    if _record_is_success(_record_path(records_dir, exp_id)):
        return

    before = set(glob.glob(os.path.join(args.repo_root, "results", "*")))
    cmd = _build_wpmixer_cmd(args, clean_csv, clean_root, target, horizon, c_in)
    log_path = os.path.join(logs_dir, f"{exp_id}.log")
    started = _now()
    rc, duration = _run_command(cmd, log_path, cwd=args.repo_root)

    new_dir = _find_new_result_dir(before, os.path.join(args.repo_root, "results"))
    metrics_path = os.path.join(new_dir, "metrics.npy") if new_dir else None
    pred_path = os.path.join(new_dir, "pred.npy") if new_dir else None
    true_path = os.path.join(new_dir, "true.npy") if new_dir else None

    if rc == 0 and metrics_path and os.path.exists(metrics_path):
        metrics = _read_metrics_npy(metrics_path)
        status = "SUCCESS" if metrics else "FAILED"
        err = None if metrics else "metrics_parse_failed"
    else:
        metrics = None
        status = "FAILED"
        err = f"command_return_code={rc}"

    _write_record(records_dir, {
        "exp_id": exp_id,
        "group": "sota",
        "paper_model": "WPMixer",
        "runner": "wpmixer",
        "target": target,
        "horizon": int(horizon),
        "status": status,
        "started_at": started,
        "ended_at": _now(),
        "duration_sec": duration,
        "command": cmd,
        "log_path": log_path,
        "artifacts": {
            "run_dir": new_dir,
            "metrics": metrics_path,
            "pred": pred_path,
            "true": true_path,
        },
        "metrics": metrics,
        "error": err,
    })


def _run_destgnn_shared(args, records_dir, logs_dir, clean_csv, horizon, target_map, feature_count):
    pending_targets = []
    for target in target_map:
        exp_id = f"sota_DEST-GNN_{target}_{horizon}"
        if not _record_is_success(_record_path(records_dir, exp_id)):
            pending_targets.append(target)
    if not pending_targets:
        return

    prepare_cmd = _build_graph_prepare_cmd(args, clean_csv, horizon)
    prep_log = os.path.join(logs_dir, f"prepare_graph_h{horizon}.log")
    rc1, _ = _run_command(prepare_cmd, prep_log, cwd=args.repo_root)

    cmd_prepare, cmd_train = _build_destgnn_train_cmd(args)
    log_path = os.path.join(logs_dir, f"sota_DEST-GNN_shared_{horizon}.log")
    started = _now()
    rc2, _ = _run_command(cmd_prepare, log_path, cwd=args.repo_root)
    rc3, duration = _run_command(cmd_train, log_path, cwd=args.repo_root)
    rc = 0 if (rc1 == 0 and rc2 == 0 and rc3 == 0) else 1

    npz_path = _collect_destgnn_npz()
    for target in target_map:
        exp_id = f"sota_DEST-GNN_{target}_{horizon}"
        target_idx = target_map[target]
        if rc == 0 and npz_path and os.path.exists(npz_path):
            blob = np.load(npz_path)
            pred_all = blob["prediction"]
            true_all = blob["data_target_tensor"]
            pred = _extract_target_array(pred_all, target_idx, feature_count)
            true = _extract_target_array(true_all, target_idx, feature_count)
            metrics = _metrics_from_arrays(pred, true)
            status = "SUCCESS"
            err = None
        else:
            metrics = None
            status = "FAILED"
            err = f"command_return_code={rc}"

        _write_record(records_dir, {
            "exp_id": exp_id,
            "group": "sota",
            "paper_model": "DEST-GNN",
            "runner": "destgnn",
            "target": target,
            "horizon": int(horizon),
            "status": status,
            "started_at": started,
            "ended_at": _now(),
            "duration_sec": duration,
            "command": cmd_prepare + ["&&"] + cmd_train,
            "log_path": log_path,
            "artifacts": {"npz": npz_path},
            "metrics": metrics,
            "error": err,
        })


def _run_matgcn_shared(args, records_dir, logs_dir, clean_csv, horizon, target_map, feature_count):
    pending_targets = []
    for target in target_map:
        exp_id = f"sota_MA-T-GCN_{target}_{horizon}"
        if not _record_is_success(_record_path(records_dir, exp_id)):
            pending_targets.append(target)
    if not pending_targets:
        return

    prepare_cmd = _build_graph_prepare_cmd(args, clean_csv, horizon)
    prep_log = os.path.join(logs_dir, f"prepare_graph_matgcn_h{horizon}.log")
    rc1, _ = _run_command(prepare_cmd, prep_log, cwd=args.repo_root)

    cmd = _build_matgcn_train_cmd(args)
    log_path = os.path.join(logs_dir, f"sota_MA-T-GCN_shared_{horizon}.log")
    started = _now()
    rc2, duration = _run_command(cmd, log_path, cwd=args.repo_root)
    rc = 0 if (rc1 == 0 and rc2 == 0) else 1

    saved_dir = _collect_matgcn_saved_dir()
    pred_path = os.path.join(saved_dir, "best_pred.npy") if saved_dir else None
    true_path = os.path.join(saved_dir, "best_true.npy") if saved_dir else None
    metrics_path = os.path.join(saved_dir, "best_metrics.json") if saved_dir else None

    for target in target_map:
        exp_id = f"sota_MA-T-GCN_{target}_{horizon}"
        target_idx = target_map[target]
        if rc == 0 and pred_path and true_path and os.path.exists(pred_path) and os.path.exists(true_path):
            pred_all = np.load(pred_path)
            true_all = np.load(true_path)
            pred = _extract_target_array(pred_all, target_idx, feature_count)
            true = _extract_target_array(true_all, target_idx, feature_count)
            metrics = _metrics_from_arrays(pred, true)
            status = "SUCCESS"
            err = None
        else:
            metrics = None
            status = "FAILED"
            err = f"command_return_code={rc}"

        _write_record(records_dir, {
            "exp_id": exp_id,
            "group": "sota",
            "paper_model": "MA-T-GCN",
            "runner": "matgcn",
            "target": target,
            "horizon": int(horizon),
            "status": status,
            "started_at": started,
            "ended_at": _now(),
            "duration_sec": duration,
            "command": cmd,
            "log_path": log_path,
            "artifacts": {
                "saved_dir": saved_dir,
                "pred": pred_path,
                "true": true_path,
                "metrics": metrics_path,
            },
            "metrics": metrics,
            "error": err,
        })


def _run_report_builder(args):
    cmd = [
        args.python_bin,
        "scripts/build_paper_comparison_report.py",
        "--suite_dir",
        args.suite_dir,
        "--reference_json",
        args.reference_json,
    ]
    log_path = os.path.join(args.suite_dir, "logs", "build_report.log")
    rc, _ = _run_command(cmd, log_path, cwd=args.repo_root)
    return rc


def main():
    parser = argparse.ArgumentParser(description="Run full ablation + SOTA reproduction suite and build paper comparison report.")
    parser.add_argument("--repo_root", type=str, default=os.getcwd())
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--data_path", type=str, default="UCI CBM Dataset/uci_cbm.csv")
    parser.add_argument("--targets", type=str, default="kMc,kMt")
    parser.add_argument("--horizons", type=str, default="96,192,336")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="experiments/repro_suite")
    parser.add_argument("--reference_json", type=str, default="scripts/paper_reference_tables.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--stg_epochs", type=int, default=30)
    parser.add_argument("--stg_patience", type=int, default=15)
    parser.add_argument("--baseline_epochs", type=int, default=30)
    parser.add_argument("--baseline_patience", type=int, default=10)
    args = parser.parse_args()

    args.repo_root = os.path.abspath(args.repo_root)
    if args.run_id is None:
        args.run_id = _run_id()
    args.suite_dir = os.path.join(args.repo_root, args.output_root, args.run_id)

    _ensure_dir(args.suite_dir)
    logs_dir = _ensure_dir(os.path.join(args.suite_dir, "logs"))
    records_dir = _ensure_dir(os.path.join(args.suite_dir, "records"))
    data_dir = _ensure_dir(os.path.join(args.suite_dir, "data"))

    clean_csv = os.path.join(data_dir, "uci_cbm_clean.csv")
    clean_meta = _prepare_clean_csv(os.path.join(args.repo_root, args.data_path), clean_csv)
    feature_cols = clean_meta["feature_cols"]
    target_map = {name: feature_cols.index(name) for name in _parse_csv_list(args.targets)}
    c_in = clean_meta["feature_count"]

    suite_meta = {
        "run_id": args.run_id,
        "timestamp": _now(),
        "repo_root": args.repo_root,
        "python_bin": args.python_bin,
        "raw_data_path": args.data_path,
        "clean_data_path": clean_csv,
        "clean_data_meta": clean_meta,
        "targets": list(target_map.keys()),
        "horizons": _parse_csv_list(args.horizons, int),
        "seed": args.seed,
        "stg_defaults": {
            "epochs": args.stg_epochs,
            "early_stop_patience": args.stg_patience,
        },
        "baseline_defaults": {
            "epochs": args.baseline_epochs,
            "patience": args.baseline_patience,
        },
        "stg_best_params": STG_BEST_PARAMS,
        "ablation_modes": [x[1] for x in ABLATION_MODES],
        "sota_models": TSL_MODELS + ["TimeXer", "WPMixer", "DEST-GNN", "MA-T-GCN"],
        "reference_json": args.reference_json,
    }
    _write_json(os.path.join(args.suite_dir, "suite_meta.json"), suite_meta)

    horizons = _parse_csv_list(args.horizons, int)
    targets = list(target_map.keys())
    clean_root = os.path.dirname(clean_csv)

    for target in targets:
        for horizon in horizons:
            _run_stg_ablation(args, records_dir, logs_dir, clean_csv, target, horizon)
            for model in TSL_MODELS:
                _run_tsl_model(args, records_dir, logs_dir, clean_csv, clean_root, target, horizon, model, c_in)
            _run_timexer(args, records_dir, logs_dir, clean_csv, clean_root, target, horizon, c_in)
            _run_wpmixer(args, records_dir, logs_dir, clean_csv, clean_root, target, horizon, c_in)

    for horizon in horizons:
        _run_destgnn_shared(args, records_dir, logs_dir, clean_csv, horizon, target_map, c_in)
        _run_matgcn_shared(args, records_dir, logs_dir, clean_csv, horizon, target_map, c_in)

    rc = _run_report_builder(args)
    if rc != 0:
        print("Warning: report builder failed. Check logs/build_report.log")
    else:
        print(f"Suite finished. Report: {os.path.join(args.suite_dir, 'ablation_vs_paper.md')}")


if __name__ == "__main__":
    main()
