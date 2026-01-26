#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_feature_names(features_path: Path):
    names = []
    with features_path.open("r", encoding="utf-8", errors="ignore") as f:
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
    return unique


def main():
    parser = argparse.ArgumentParser(description="Prepare UCI CBM dataset CSV with date column.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data.txt")
    parser.add_argument("--features_path", type=str, required=True, help="Path to Features.txt")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--start", type=str, default="2000-01-01 00:00:00", help="Start datetime")
    parser.add_argument("--freq", type=str, default="min", help="Date frequency (e.g. min, h, d)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    features_path = Path(args.features_path)
    output_path = Path(args.output)

    feature_names = parse_feature_names(features_path)
    df = pd.read_csv(data_path, sep=r"\s+", header=None)
    if len(feature_names) != df.shape[1]:
        raise ValueError(
            f"Feature count mismatch: {len(feature_names)} names vs {df.shape[1]} columns."
        )
    df.columns = feature_names

    date_index = pd.date_range(start=args.start, periods=len(df), freq=args.freq)
    df.insert(0, "date", date_index)

    if "kMc" in df.columns and "kMt" in df.columns:
        front = ["date"]
        tail = ["kMc", "kMt"]
        middle = [c for c in df.columns if c not in front + tail]
        df = df[front + middle + tail]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with shape {df.shape}")


if __name__ == "__main__":
    main()
