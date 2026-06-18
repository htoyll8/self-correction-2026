#!/usr/bin/env python3
"""
Figure 3 (Main): Recovery tradeoff — cost vs benefit

Scatter:
  x = avg_attempts_recovered  (conditional on recovery)
  y = recovery_rate           (benefit)
Color = model
Marker shape = dataset
Panels = history (0 vs 1)

Usage:
  python3 make_fig3_tradeoff_scatter.py
  python3 make_fig3_tradeoff_scatter.py --csv ~/Desktop/self_corrections_pandas.csv --out fig3.pdf
  python3 make_fig3_tradeoff_scatter.py --difficulty intro
  python3 make_fig3_tradeoff_scatter.py --yunit percent   # force percent display
  python3 make_fig3_tradeoff_scatter.py --yunit fraction  # force fraction display

Notes:
- If recovery_rate is already in [0,1], we treat it as a fraction unless forced otherwise.
- If recovery_rate looks like a percent (e.g., 35, 74.2), we treat it as percent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

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


def infer_yunit(vals) -> str:
    """
    Infer whether recovery_rate is in fraction [0,1] or percent [0,100].
    Heuristic: if max <= 1.0 => fraction else percent.
    """
    vals = [v for v in vals if v is not None]
    if not vals:
        return "fraction"
    mx = max(vals)
    return "fraction" if mx <= 1.0 else "percent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=Path.home() / "Desktop" / "self_corrections_pandas.csv",
        help="Path to CSV with columns: dataset, history, model, recovery_rate, avg_attempts_recovered.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output figure path. Defaults to fig3_tradeoff_scatter.png next to the CSV.",
    )
    ap.add_argument(
        "--difficulty",
        default=None,
        help="Optional filter: keep only this difficulty (exact match). If omitted, keep all.",
    )
    ap.add_argument(
        "--yunit",
        choices=["auto", "fraction", "percent"],
        default="auto",
        help="How to interpret recovery_rate values.",
    )
    ap.add_argument("--title", default="Self-repair tradeoff: recovery benefit vs cost", help="Figure title.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PNG outputs.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_name("fig3_tradeoff_scatter.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    needed = {"dataset", "history", "model", "recovery_rate", "avg_attempts_recovered"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df.copy()
    df["dataset_label"] = df["dataset"].map(normalize_dataset)
    df["model_label"] = df["model"].map(normalize_model)
    df["recovery_rate_f"] = df["recovery_rate"].apply(parse_float)
    df["avg_attempts_f"] = df["avg_attempts_recovered"].apply(parse_float)

    if args.difficulty is not None:
        if "difficulty" not in df.columns:
            raise SystemExit("Requested --difficulty but 'difficulty' column not found.")
        df = df[df["difficulty"].astype(str) == str(args.difficulty)].copy()

    # Keep only rows where x is defined and recovery info exists.
    # avg_attempts_recovered is typically undefined when recovery_rate==0.
    df = df[df["avg_attempts_f"].notna() & df["recovery_rate_f"].notna()].copy()
    if df.empty:
        raise SystemExit("No rows left after filtering to non-NA avg_attempts_recovered and recovery_rate.")

    # Interpret y units
    inferred = infer_yunit(df["recovery_rate_f"].tolist())
    yunit = inferred if args.yunit == "auto" else args.yunit

    if yunit == "fraction":
        df["recovery_y"] = df["recovery_rate_f"].clip(0, 1.0)
        y_label = "Recovery rate"
        y_ticks_as_percent = False
    else:
        df["recovery_y"] = df["recovery_rate_f"].clip(0, 100.0)
        y_label = "Recovery rate (%)"
        y_ticks_as_percent = True

    # Panel split by history (0 vs 1)
    histories = sorted(df["history"].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else 999)
    # Force order 0,1 if present
    histories = [h for h in ["0", "1"] if h in histories] + [h for h in histories if h not in {"0", "1"}]
    if len(histories) == 0:
        raise SystemExit("No history values found.")

    ncols = min(2, len(histories))
    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    # Marker per dataset
    # (Matplotlib marker styles; keep distinct and readable in b/w print)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    dataset_order = ["HumanEval", "MBPP+", "APPS", "HumanEval-Java"]
    datasets = [d for d in dataset_order if d in set(df["dataset_label"])] + [
        d for d in sorted(df["dataset_label"].unique().tolist()) if d not in dataset_order
    ]
    marker_map: Dict[str, str] = {d: marker_cycle[i % len(marker_cycle)] for i, d in enumerate(datasets)}

    # Color by model (let matplotlib pick default colors; don't hardcode)
    model_order = ["Claude", "GPT-4", "GPT-5.1"]
    models = [m for m in model_order if m in set(df["model_label"])] + [
        m for m in sorted(df["model_label"].unique().tolist()) if m not in model_order
    ]

    # Plot each panel
    for ax_i, h in enumerate(histories[:ncols]):
        ax = axes[ax_i]
        subh = df[df["history"].astype(str) == h].copy()
        ax.set_title(f"history={h}")

        # Plot points: iterate models to get consistent legend order
        for m in models:
            subm = subh[subh["model_label"] == m]
            if subm.empty:
                continue

            # For each dataset, plot with its marker (same color for model)
            for d in datasets:
                submd = subm[subm["dataset_label"] == d]
                if submd.empty:
                    continue
                ax.scatter(
                    submd["avg_attempts_f"],
                    submd["recovery_y"],
                    marker=marker_map[d],
                    s=60,
                    alpha=0.9,
                    label=m,  # model legend handled below (deduped)
                )

        ax.set_xlabel("Avg attempts to first correct solution\n(conditional on recovery)")
        ax.grid(True, alpha=0.25)

    # Y label on left only
    axes[0].set_ylabel(y_label)

    # Build legends:
    # 1) model legend (colors)
    # 2) dataset legend (markers)
    # To dedupe model handles, collect from first axis.
    handles, labels = axes[0].get_legend_handles_labels()
    model_handles = []
    model_labels = []
    seen = set()
    for h, lab in zip(handles, labels):
        if lab in models and lab not in seen:
            model_handles.append(h)
            model_labels.append(lab)
            seen.add(lab)

    # Dataset legend: create dummy handles
    dataset_handles = [
        plt.Line2D([0], [0], marker=marker_map[d], linestyle="", markersize=8, label=d)
        for d in datasets
    ]

    # Place legends
    # Model legend above; dataset legend to the right (or below if tight)
    fig.legend(model_handles, model_labels, loc="upper center", ncol=len(model_labels), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.legend(dataset_handles, datasets, loc="center left", frameon=False, bbox_to_anchor=(1.01, 0.5), title="Dataset")

    fig.suptitle(args.title, y=1.12, fontsize=14)
    fig.tight_layout()

    if out_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")

    print(f"✅ Wrote Figure 3 tradeoff scatter to: {out_path}")


if __name__ == "__main__":
    main()
