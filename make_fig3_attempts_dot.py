 #!/usr/bin/env python3
"""
Figure 3 (dot-plot version): How many attempts does recovery actually require?

Dot plot of avg_attempts_recovered (conditional on recovery), faceted like Fig 1:
- rows: dataset (HumanEval, MBPP+, APPS, HumanEval-Java)
- cols: history (0 vs 1)
Within each panel:
- x-axis: model (Claude, GPT-4, GPT-5.1)
- y-axis: avg_attempts_recovered
Optionally connect model dots with a thin line to help comparisons.

Usage:
  python3 make_fig3_attempts_dot.py
  python3 make_fig3_attempts_dot.py --csv path/to/summary.csv
  python3 make_fig3_attempts_dot.py --out fig3_attempts.pdf
  python3 make_fig3_attempts_dot.py --history all
  python3 make_fig3_attempts_dot.py --difficulty all
  python3 make_fig3_attempts_dot.py --connect 0

Assumptions:
- Each CSV row is an aggregate for a condition (dataset, difficulty, history, model).
- avg_attempts_recovered is defined only for recovered tasks; we filter to recovery_rate>0 and non-NA attempts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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
        help="Output figure path. Defaults to fig3_avg_attempts_recovered_dot.png next to the CSV.",
    )
    ap.add_argument(
        "--difficulty",
        default=None,
        help="Optional filter: keep only this difficulty (exact match). "
             "If omitted, keeps all difficulties (and will average if multiple exist per cell).",
    )
    ap.add_argument(
        "--history",
        choices=["0", "1", "all"],
        default="all",
        help="Filter to a single history value, or keep all.",
    )
    ap.add_argument(
        "--connect",
        type=int,
        default=1,
        help="If 1, connect model dots within each panel with a line (helps comparisons).",
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
    out_path = Path(args.out) if args.out is not None else csv_path.with_name("fig3_avg_attempts_recovered_dot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    needed = {"dataset", "model", "avg_attempts_recovered", "recovery_rate"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df.copy()
    df["dataset_label"] = df["dataset"].map(normalize_dataset)
    df["model_label"] = df["model"].map(normalize_model)
    df["avg_attempts_recovered_f"] = df["avg_attempts_recovered"].apply(parse_float)
    df["recovery_rate_f"] = df["recovery_rate"].apply(parse_float)

    # Optional difficulty filter
    if args.difficulty is not None:
        if "difficulty" not in df.columns:
            raise SystemExit("Requested --difficulty but 'difficulty' column not found in CSV.")
        df = df[df["difficulty"].astype(str) == str(args.difficulty)].copy()

    # Optional history filter
    if args.history != "all":
        if "history" not in df.columns:
            raise SystemExit("Requested --history filter but 'history' column not found in CSV.")
        df = df[df["history"].astype(str) == args.history].copy()

    # Keep only conditions with actual recovery and defined attempts
    df = df[
        (df["avg_attempts_recovered_f"].notna())
        & (df["recovery_rate_f"].notna())
        & (df["recovery_rate_f"] > 0)
    ].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering to recovered cases (recovery_rate>0 and attempts defined).")

    # If multiple rows per (dataset, history, model) remain (e.g., multiple difficulties),
    # aggregate by mean (simple and transparent).
    facet_cols = ["dataset_label", "model_label"]
    if "history" in df.columns and args.history == "all":
        facet_cols.insert(1, "history")

    grouped = (
        df.groupby(facet_cols, dropna=False)["avg_attempts_recovered_f"]
        .mean()
        .reset_index()
        .rename(columns={"avg_attempts_recovered_f": "avg_attempts"})
    )

    # Determine facet values
    dataset_order = ["HumanEval", "MBPP+", "APPS", "HumanEval-Java"]
    datasets = [d for d in dataset_order if d in set(grouped["dataset_label"])]
    if not datasets:
        datasets = sorted(grouped["dataset_label"].unique().tolist())

    if "history" in grouped.columns:
        # ensure 0 then 1 if both present
        def _hk(x):
            try:
                return int(x)
            except Exception:
                return 999

        histories = sorted(grouped["history"].dropna().unique().tolist(), key=_hk)
    else:
        histories = [None]  # single column

    model_order = ["Claude", "GPT-4", "GPT-5.1"]
    models = [m for m in model_order if m in set(grouped["model_label"])]
    if not models:
        models = sorted(grouped["model_label"].unique().tolist())

    nrows = len(datasets)
    ncols = len(histories)

    fig_w = 4.2 * ncols
    fig_h = 2.5 * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Common x positions for models
    x_pos = list(range(len(models)))

    # Plot
    for r, ds in enumerate(datasets):
        for c, h in enumerate(histories):
            ax = axes[r][c]

            if h is None:
                panel = grouped[grouped["dataset_label"] == ds]
                col_title = None
            else:
                panel = grouped[(grouped["dataset_label"] == ds) & (grouped["history"] == h)]
                col_title = f"history={h}"

            if panel.empty:
                ax.set_axis_off()
                continue

            ys = []
            for m in models:
                hit = panel[panel["model_label"] == m]
                ys.append(float(hit["avg_attempts"].values[0]) if not hit.empty else float("nan"))

            # dots
            ax.plot(x_pos, ys, marker="o", linewidth=0, markersize=5)

            # optional connecting line
            if args.connect == 1:
                ax.plot(x_pos, ys, linewidth=1.2)

            # titles/labels
            if r == 0 and col_title is not None:
                ax.set_title(col_title)
            if c == 0:
                ax.set_ylabel(f"{ds}\nAvg attempts")
            if r == nrows - 1:
                ax.set_xlabel("Model")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=0)
            ax.grid(True, axis="y", alpha=0.25)

    # One shared y label (cleaner than repeating), but keep row dataset labels.
    # If you want a single global y-label, uncomment and remove per-axis ylabels above.
    # fig.text(0.04, 0.5, "Avg attempts to first correct solution (among recovered tasks)", va="center", rotation="vertical")

    fig.suptitle(args.title, y=1.02, fontsize=14)
    fig.tight_layout()

    if out_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")

    print(f"✅ Wrote Figure 3 (dot plot) to: {out_path}")


if __name__ == "__main__":
    main()
