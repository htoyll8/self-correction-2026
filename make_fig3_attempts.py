#!/usr/bin/env python3
"""
Figure 3: How many attempts does recovery actually require?

Bar chart of avg_attempts_recovered (conditional on recovery).

Design:
- y-axis: Avg attempts to first correct solution (among recovered tasks)
- x-axis: Model
- grouped bars: Dataset / difficulty (optional split by history)

Usage:
  python3 make_fig3_attempts.py --csv path/to/summary.csv
  python3 make_fig3_attempts.py --csv path/to/summary.csv --out fig3.pdf
  python3 make_fig3_attempts.py --csv path/to/summary.csv --group dataset
  python3 make_fig3_attempts.py --csv path/to/summary.csv --group dataset+difficulty
  python3 make_fig3_attempts.py --csv path/to/summary.csv --group dataset+difficulty --history 0
  python3 make_fig3_attempts.py --csv path/to/summary.csv --group dataset --history all

Notes:
- Excludes rows where avg_attempts_recovered is NA or where recovery_rate is 0/NA.
- If you want to include "0 recovered" as 0 attempts (not recommended), you can change the filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def normalize_dataset(s: str) -> str:
    mapping = {
        "human_eval": "HumanEval",
        "humaneval": "HumanEval",
        "human_eval_java": "HumanEval-Java",
        "mbppplus": "MBPP+",
        "mbpp_plus": "MBPP+",
        "apps": "APPS",
    }
    key = str(s).strip()
    return mapping.get(key, key)


def normalize_model(s: str) -> str:
    mapping = {
        "gpt-4": "GPT-4",
        "gpt4": "GPT-4",
        "gpt-5.1": "GPT-5.1",
        "gpt5.1": "GPT-5.1",
        "claude": "Claude",
        "claude 4.5": "Claude",
    }
    key = str(s).strip()
    return mapping.get(key, key)


def parse_float(x):
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.upper() in {"NA", "NAN"}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        v = float(x)
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v


def make_group_key(row: pd.Series, group_cols: List[str]) -> str:
    parts = []
    for c in group_cols:
        if c not in row:
            continue
        v = row[c]
        if pd.isna(v):
            v = ""
        parts.append(f"{c}={v}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=False,
        default=Path.home() / "Desktop" / "self_corrections_pandas.csv",
        help="Path to CSV with avg_attempts_recovered.",
    )
    ap.add_argument(
        "--out",
        required=False,
        default=None,
        help="Output figure path. Defaults to fig3_avg_attempts_recovered.png next to the CSV.",
    )
    ap.add_argument(
        "--group",
        choices=["dataset", "dataset+difficulty", "dataset+difficulty+history"],
        default="dataset+difficulty",
        help="Which fields to use for grouped bars.",
    )
    ap.add_argument(
        "--history",
        choices=["0", "1", "all"],
        default="all",
        help="Filter to a single history value, or keep all.",
    )
    ap.add_argument(
        "--title",
        default="Recovery cost: attempts required to succeed",
        help="Figure title.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG/JPG outputs.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out is not None else csv_path.with_name("fig3_avg_attempts_recovered.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    needed = {"dataset", "model", "avg_attempts_recovered", "recovery_rate"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Normalize labels
    df = df.copy()
    df["dataset_label"] = df["dataset"].map(normalize_dataset)
    df["model_label"] = df["model"].map(normalize_model)

    # Optional history filter
    if args.history != "all":
        if "history" not in df.columns:
            raise SystemExit("Requested --history filter but 'history' column not found in CSV.")
        df = df[df["history"].astype(str) == args.history].copy()

    # Parse numeric columns and filter to recovered cases
    df["avg_attempts_recovered_f"] = df["avg_attempts_recovered"].apply(parse_float)
    df["recovery_rate_f"] = df["recovery_rate"].apply(parse_float)

    # Keep only rows where recovery actually occurred and attempts are defined
    df = df[(df["avg_attempts_recovered_f"].notna()) & (df["recovery_rate_f"].notna()) & (df["recovery_rate_f"] > 0)].copy()
    if df.empty:
        raise SystemExit("No rows left after filtering to recovered cases (recovery_rate>0 and attempts defined).")

    # Choose grouping columns for bar clusters
    group_cols = ["dataset_label"]
    if args.group in {"dataset+difficulty", "dataset+difficulty+history"}:
        if "difficulty" in df.columns:
            group_cols.append("difficulty")
        else:
            # If difficulty is missing, quietly fall back to dataset-only
            group_cols = ["dataset_label"]

    if args.group == "dataset+difficulty+history":
        if "history" not in df.columns:
            raise SystemExit("Requested dataset+difficulty+history grouping but 'history' column not found.")
        group_cols.append("history")

    # Create group labels
    def group_label(row: pd.Series) -> str:
        # Nice compact label instead of "col=val | col=val"
        parts = []
        parts.append(str(row["dataset_label"]))
        if "difficulty" in group_cols and "difficulty" in row and pd.notna(row["difficulty"]) and str(row["difficulty"]).strip() != "":
            parts.append(str(row["difficulty"]))
        if "history" in group_cols and "history" in row and pd.notna(row["history"]):
            parts.append(f"h={int(row['history'])}" if str(row["history"]).isdigit() else f"h={row['history']}")
        return " · ".join(parts)

    df["group_label"] = df.apply(group_label, axis=1)

    # Order models consistently
    model_order = ["Claude", "GPT-4", "GPT-5.1"]
    present_models = [m for m in model_order if m in set(df["model_label"])]
    if not present_models:
        present_models = sorted(df["model_label"].unique().tolist())

    # Order groups in a paper-friendly way
    dataset_order = ["HumanEval", "MBPP+", "APPS", "HumanEval-Java"]
    # sort by dataset order then by label
    def group_sort_key(glab: str):
        ds = glab.split(" · ")[0]
        try:
            ds_rank = dataset_order.index(ds)
        except ValueError:
            ds_rank = 999
        return (ds_rank, glab)

    group_labels = sorted(df["group_label"].unique().tolist(), key=group_sort_key)

    # Build a matrix of values: rows=groups, cols=models
    values = {g: {m: None for m in present_models} for g in group_labels}
    for _, row in df.iterrows():
        g = row["group_label"]
        m = row["model_label"]
        if g in values and m in values[g]:
            values[g][m] = float(row["avg_attempts_recovered_f"])

    plot_df = pd.DataFrame(
        [{"group": g, "model": m, "avg_attempts": values[g][m]} for g in group_labels for m in present_models]
    )

    # Figure sizing: scale with number of groups
    n_groups = len(group_labels)
    fig_w = max(7.0, 0.7 * n_groups)
    fig_h = 4.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Grouped bar positions
    x = list(range(n_groups))
    bar_w = 0.22 if len(present_models) >= 3 else 0.3
    offsets = [(-1) * bar_w, 0.0, bar_w] if len(present_models) == 3 else [0.0]

    # Map model -> offset index
    for i, m in enumerate(present_models):
        sub = plot_df[plot_df["model"] == m].copy()
        ys = [sub[sub["group"] == g]["avg_attempts"].values[0] for g in group_labels]
        xs = [xi + (i - (len(present_models) - 1) / 2.0) * bar_w for xi in x]
        ax.bar(xs, ys, width=bar_w, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=25, ha="right")
    ax.set_ylabel("Avg attempts to first correct solution\n(among recovered tasks)")
    ax.set_xlabel("Dataset / difficulty")
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.25)

    # Legend above the plot
    ax.legend(loc="upper center", ncol=len(present_models), frameon=False, bbox_to_anchor=(0.5, 1.15))

    fig.tight_layout()

    if out_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")

    print(f"✅ Wrote Figure 3 to: {out_path}")


if __name__ == "__main__":
    main()
