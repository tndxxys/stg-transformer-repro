import argparse
import csv
import glob
import json
import os
from datetime import datetime


METRICS = ("MSE", "MAE", "RMSE")
TABLE_NAME = {
    ("sota", "kMt"): "Table 2",
    ("sota", "kMc"): "Table 3",
    ("ablation", "kMt"): "Table 4",
    ("ablation", "kMc"): "Table 5",
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _round3(value):
    if value is None:
        return None
    return round(float(value), 3)


def _fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value):
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def _normalize_metrics(metrics_obj):
    if not isinstance(metrics_obj, dict):
        return None
    out = {}
    for metric in METRICS:
        value = None
        if metric in metrics_obj:
            value = metrics_obj.get(metric)
        elif metric.lower() in metrics_obj:
            value = metrics_obj.get(metric.lower())
        out[metric] = _safe_float(value)
    if any(out[m] is None for m in METRICS):
        return None
    return out


def _record_key(group, target, horizon, model):
    return (group, target, int(horizon), model)


def _load_records(records_dir):
    by_key = {}
    if not os.path.isdir(records_dir):
        return by_key
    record_files = sorted(glob.glob(os.path.join(records_dir, "*.json")))
    for path in record_files:
        try:
            rec = _read_json(path)
        except Exception:
            continue
        key = _record_key(
            rec.get("group"),
            rec.get("target"),
            rec.get("horizon"),
            rec.get("paper_model"),
        )
        if None in key:
            continue
        rec["metrics"] = _normalize_metrics(rec.get("metrics"))
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = rec
            continue
        # Keep SUCCESS over FAILED when duplicates exist.
        if prev.get("status") != "SUCCESS" and rec.get("status") == "SUCCESS":
            by_key[key] = rec
            continue
        # Otherwise keep the latest ended_at record.
        prev_end = str(prev.get("ended_at", ""))
        curr_end = str(rec.get("ended_at", ""))
        if curr_end > prev_end:
            by_key[key] = rec
    return by_key


def _paper_rows(reference_json):
    rows = []
    for target, horizons in reference_json.get("sota", {}).items():
        for horizon_str, model_map in horizons.items():
            horizon = int(horizon_str)
            for model, metrics in model_map.items():
                rows.append(
                    {
                        "group": "sota",
                        "target": target,
                        "horizon": horizon,
                        "model": model,
                        "paper_metrics": {m: _safe_float(metrics.get(m)) for m in METRICS},
                    }
                )
    for target, horizons in reference_json.get("ablation", {}).items():
        for horizon_str, model_map in horizons.items():
            horizon = int(horizon_str)
            for model, metrics in model_map.items():
                rows.append(
                    {
                        "group": "ablation",
                        "target": target,
                        "horizon": horizon,
                        "model": model,
                        "paper_metrics": {m: _safe_float(metrics.get(m)) for m in METRICS},
                    }
                )
    rows.sort(key=lambda x: (x["group"], x["target"], x["horizon"], x["model"]))
    return rows


def _resolve_repro_record(records, group, target, horizon, model):
    key = _record_key(group, target, horizon, model)
    rec = records.get(key)
    if rec is not None:
        return rec
    # "Ours" in paper SOTA corresponds to the full STG ablation setting.
    if group == "sota" and model == "Ours":
        return records.get(_record_key("ablation", target, horizon, "Backbone+M1+M2+M3"))
    return None


def _build_rows(reference_json, records):
    long_rows = []
    wide_rows = []
    for ref in _paper_rows(reference_json):
        group = ref["group"]
        target = ref["target"]
        horizon = ref["horizon"]
        model = ref["model"]
        table_name = TABLE_NAME.get((group, target), "Unknown")
        rec = _resolve_repro_record(records, group, target, horizon, model)
        status = "MISSING"
        exp_id = ""
        runner = ""
        log_path = ""
        repro_metrics = None
        if rec is not None:
            status = rec.get("status", "UNKNOWN")
            exp_id = rec.get("exp_id", "")
            runner = rec.get("runner", "")
            log_path = rec.get("log_path", "")
            repro_metrics = rec.get("metrics")
        if status != "SUCCESS" or repro_metrics is None:
            repro_metrics = {m: None for m in METRICS}

        wide = {
            "table": table_name,
            "group": group,
            "target": target,
            "horizon": horizon,
            "model": model,
            "status": status,
            "runner": runner,
            "exp_id": exp_id,
            "log_path": log_path,
        }

        for metric in METRICS:
            paper_value = ref["paper_metrics"].get(metric)
            repro_value = repro_metrics.get(metric)
            delta_abs = None
            delta_rel = None
            better = None
            if paper_value is not None and repro_value is not None:
                delta_abs = repro_value - paper_value
                if abs(paper_value) > 0:
                    delta_rel = delta_abs / abs(paper_value)
                better = repro_value < paper_value

            long_rows.append(
                {
                    "table": table_name,
                    "group": group,
                    "target": target,
                    "horizon": horizon,
                    "model": model,
                    "metric": metric,
                    "paper_value": paper_value,
                    "paper_value_3dp": _round3(paper_value),
                    "repro_value": repro_value,
                    "repro_value_3dp": _round3(repro_value),
                    "delta_abs": delta_abs,
                    "delta_rel": delta_rel,
                    "is_better": better,
                    "status": status,
                    "runner": runner,
                    "exp_id": exp_id,
                    "log_path": log_path,
                }
            )

            wide[f"paper_{metric}"] = paper_value
            wide[f"paper_{metric}_3dp"] = _round3(paper_value)
            wide[f"repro_{metric}"] = repro_value
            wide[f"repro_{metric}_3dp"] = _round3(repro_value)
            wide[f"delta_{metric}"] = delta_abs
            wide[f"delta_rel_{metric}"] = delta_rel
            wide[f"better_{metric}"] = better

        wide_rows.append(wide)
    return long_rows, wide_rows


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_markdown_table(headers, rows):
    if not rows:
        return "_No data_\n"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _section_paper_values(reference_json):
    lines = ["## Paper Values (Tables 2-5)", ""]
    for group, label in (("sota", "SOTA"), ("ablation", "Ablation")):
        source = reference_json.get(group, {})
        for target in sorted(source.keys()):
            table_name = TABLE_NAME.get((group, target), "Unknown")
            lines.append(f"### {table_name} ({label}, target={target})")
            headers = ["Horizon", "Model", "MSE", "MAE", "RMSE"]
            rows = []
            for horizon_str in sorted(source[target].keys(), key=lambda x: int(x)):
                horizon = int(horizon_str)
                model_map = source[target][horizon_str]
                for model in sorted(model_map.keys()):
                    m = model_map[model]
                    rows.append(
                        [
                            str(horizon),
                            model,
                            _fmt(m.get("MSE")),
                            _fmt(m.get("MAE")),
                            _fmt(m.get("RMSE")),
                        ]
                    )
            lines.append(_to_markdown_table(headers, rows))
    return "\n".join(lines)


def _section_repro_values(wide_rows):
    lines = ["## Reproduction Values", ""]
    for table_name in ("Table 2", "Table 3", "Table 4", "Table 5"):
        subset = [r for r in wide_rows if r["table"] == table_name]
        lines.append(f"### {table_name}")
        headers = ["Target", "Horizon", "Model", "Status", "MSE", "MAE", "RMSE"]
        rows = []
        for r in sorted(subset, key=lambda x: (x["target"], x["horizon"], x["model"])):
            rows.append(
                [
                    r["target"],
                    str(r["horizon"]),
                    r["model"],
                    r["status"],
                    _fmt(r.get("repro_MSE")),
                    _fmt(r.get("repro_MAE")),
                    _fmt(r.get("repro_RMSE")),
                ]
            )
        lines.append(_to_markdown_table(headers, rows))
    return "\n".join(lines)


def _section_delta_analysis(long_rows):
    lines = ["## Delta Analysis", ""]
    headers = [
        "Table",
        "Target",
        "Horizon",
        "Model",
        "Metric",
        "Paper",
        "Repro",
        "Delta(abs)",
        "Delta(rel)",
        "Better",
    ]
    rows = []
    for r in sorted(
        long_rows,
        key=lambda x: (x["table"], x["target"], x["horizon"], x["model"], x["metric"]),
    ):
        rows.append(
            [
                r["table"],
                r["target"],
                str(r["horizon"]),
                r["model"],
                r["metric"],
                f"{_fmt(r['paper_value'])} ({_fmt(r['paper_value_3dp'], digits=3)})",
                f"{_fmt(r['repro_value'])} ({_fmt(r['repro_value_3dp'], digits=3)})"
                if r["repro_value"] is not None
                else "NA",
                _fmt(r["delta_abs"]),
                _fmt_pct(r["delta_rel"]),
                "YES" if r["is_better"] is True else ("NO" if r["is_better"] is False else "NA"),
            ]
        )
    lines.append(_to_markdown_table(headers, rows))

    comparable = [r for r in long_rows if r["paper_value"] is not None and r["repro_value"] is not None]
    if comparable:
        better_count = sum(1 for r in comparable if r["is_better"])
        lines.append(
            f"- Comparable metric points: {len(comparable)}; better-than-paper points: {better_count}."
        )
    else:
        lines.append("- No comparable metric points were found.")
    return "\n".join(lines)


def _section_failures(wide_rows):
    failed = [r for r in wide_rows if r.get("status") != "SUCCESS"]
    lines = ["## Exceptions and Failures", ""]
    if not failed:
        lines.append("- No failed items.")
        return "\n".join(lines)
    headers = ["Table", "Target", "Horizon", "Model", "Status", "Runner", "ExpID", "LogPath"]
    rows = []
    for r in sorted(failed, key=lambda x: (x["table"], x["target"], x["horizon"], x["model"])):
        rows.append(
            [
                r["table"],
                r["target"],
                str(r["horizon"]),
                r["model"],
                r["status"],
                r.get("runner", ""),
                r.get("exp_id", ""),
                r.get("log_path", ""),
            ]
        )
    lines.append(_to_markdown_table(headers, rows))
    return "\n".join(lines)


def _collect_devices(suite_dir):
    devices = set()
    for meta_path in glob.glob(os.path.join(suite_dir, "artifacts", "stg_ablation", "*", "run_meta.json")):
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        device = meta.get("device")
        if device:
            devices.add(str(device))
    return sorted(devices)


def _section_setup(suite_dir, suite_meta):
    clean_meta = suite_meta.get("clean_data_meta", {})
    devices = _collect_devices(suite_dir)
    lines = ["## Experiment Setup", ""]
    lines.append(f"- run_id: {suite_meta.get('run_id', os.path.basename(suite_dir))}")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- raw_data_path: {suite_meta.get('raw_data_path', 'NA')}")
    lines.append(f"- clean_data_path: {suite_meta.get('clean_data_path', 'NA')}")
    lines.append(f"- dropped_constant_cols: {clean_meta.get('dropped_constant_cols', [])}")
    lines.append("- split_policy: sequential split, train_ratio=0.8, val_ratio=0.2")
    lines.append(f"- targets: {suite_meta.get('targets', [])}")
    lines.append(f"- horizons: {suite_meta.get('horizons', [])}")
    lines.append(f"- python_bin: {suite_meta.get('python_bin', 'NA')}")
    lines.append(f"- devices_seen: {devices if devices else ['NA']}")
    lines.append("- dependencies: see requirements.txt / baseline repos in current environment")
    return "\n".join(lines)


def _section_conclusion(wide_rows):
    lines = ["## Conclusion and Suggested Next Steps", ""]
    total = len(wide_rows)
    success = sum(1 for r in wide_rows if r.get("status") == "SUCCESS")
    lines.append(f"- completed_items: {success}/{total}")
    metric_points = 0
    better_points = 0
    for row in wide_rows:
        if row.get("status") != "SUCCESS":
            continue
        for metric in METRICS:
            paper_value = row.get(f"paper_{metric}")
            repro_value = row.get(f"repro_{metric}")
            if paper_value is None or repro_value is None:
                continue
            metric_points += 1
            if repro_value < paper_value:
                better_points += 1
    if metric_points > 0:
        lines.append(f"- better_than_paper_metric_points: {better_points}/{metric_points}")
    else:
        lines.append("- better_than_paper_metric_points: NA")
    lines.append("- if failures exist: rerun failed items first, then regenerate this report.")
    lines.append("- if large deltas persist: verify split protocol, normalization domain, and metric domain (raw vs std).")
    return "\n".join(lines)


def _write_markdown(path, suite_dir, suite_meta, reference_json, long_rows, wide_rows):
    parts = [
        "# Ablation vs Paper Report",
        "",
        _section_setup(suite_dir, suite_meta),
        "",
        _section_paper_values(reference_json),
        "",
        _section_repro_values(wide_rows),
        "",
        _section_delta_analysis(long_rows),
        "",
        _section_failures(wide_rows),
        "",
        _section_conclusion(wide_rows),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Build paper comparison CSV/Markdown from repro suite records.")
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument("--reference_json", type=str, default="scripts/paper_reference_tables.json")
    args = parser.parse_args()

    suite_dir = os.path.abspath(args.suite_dir)
    records_dir = os.path.join(suite_dir, "records")
    suite_meta_path = os.path.join(suite_dir, "suite_meta.json")
    if os.path.exists(suite_meta_path):
        suite_meta = _read_json(suite_meta_path)
    else:
        suite_meta = {}

    repo_root = suite_meta.get("repo_root")
    if not repo_root:
        repo_root = os.path.abspath(os.path.join(suite_dir, "..", "..", ".."))
    reference_path = args.reference_json
    if not os.path.isabs(reference_path):
        reference_path = os.path.join(repo_root, reference_path)

    reference_json = _read_json(reference_path)
    records = _load_records(records_dir)
    long_rows, wide_rows = _build_rows(reference_json, records)

    long_csv = os.path.join(suite_dir, "paper_alignment_long.csv")
    wide_csv = os.path.join(suite_dir, "paper_alignment_wide.csv")
    md_path = os.path.join(suite_dir, "ablation_vs_paper.md")

    long_fields = [
        "table",
        "group",
        "target",
        "horizon",
        "model",
        "metric",
        "paper_value",
        "paper_value_3dp",
        "repro_value",
        "repro_value_3dp",
        "delta_abs",
        "delta_rel",
        "is_better",
        "status",
        "runner",
        "exp_id",
        "log_path",
    ]
    wide_fields = [
        "table",
        "group",
        "target",
        "horizon",
        "model",
        "status",
        "runner",
        "exp_id",
        "log_path",
        "paper_MSE",
        "paper_MSE_3dp",
        "repro_MSE",
        "repro_MSE_3dp",
        "delta_MSE",
        "delta_rel_MSE",
        "better_MSE",
        "paper_MAE",
        "paper_MAE_3dp",
        "repro_MAE",
        "repro_MAE_3dp",
        "delta_MAE",
        "delta_rel_MAE",
        "better_MAE",
        "paper_RMSE",
        "paper_RMSE_3dp",
        "repro_RMSE",
        "repro_RMSE_3dp",
        "delta_RMSE",
        "delta_rel_RMSE",
        "better_RMSE",
    ]
    _write_csv(long_csv, long_rows, long_fields)
    _write_csv(wide_csv, wide_rows, wide_fields)

    _write_markdown(md_path, suite_dir, suite_meta, reference_json, long_rows, wide_rows)

    print(f"Saved: {long_csv}")
    print(f"Saved: {wide_csv}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
