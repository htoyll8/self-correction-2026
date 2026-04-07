#!/usr/bin/env python3
"""
Figure 1: Self-repair trajectories over iterations.

Creates a faceted line plot of pass_iter_k vs iteration k (0..10),
with lines for each model and panels for:
  - rows: dataset (e.g., human_eval, mbppplus, apps)
  - cols: history (0 vs 1) OR difficulty (if you choose)

Usage:
  python3 make_fig1_trajectories.py --csv path/to/summary.csv --out fig1.png
  python3 make_fig1_trajectories.py --csv path/to/summary.csv --out fig1.pdf --facet history
  python3 make_fig1_trajectories.py --csv path/to/summary.csv --out fig1.png --facet difficulty

Notes:
- Assumes the CSV already contains aggregated pass_iter_0..pass_iter_10 values
  (i.e., "averaged across tasks" upstream).
- Handles percent strings like "85.53%" or numeric fractions like 0.8553.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


PASS_COL_RE = re.compile(r"^pass_iter_(\d+)$")


def parse_pct(x):
    """Parse a value that may be '85.53%', 85.53, 0.8553, or NA/None."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.upper() == "NA" or s.upper() == "NAN":
            return None
        if s.endswith("%"):
            try:
                return float(s[:-1])
            except ValueError:
                return None
        # plain number string
        try:
            v = float(s)
        except ValueError:
            return None
    else:
        try:
            v = float(x)
        except Exception:
            return None

    # Heuristic: if it's in [0,1.2], treat as fraction; else percent already.
    if 0.0 <= v <= 1.2:
        return 100.0 * v
    return v


def find_pass_iter_cols(df: pd.DataFrame) -> List[Tuple[int, str]]:
    cols = []
    for c in df.columns:
        m = PASS_COL_RE.match(str(c))
        if m:
            k = int(m.group(1))
            cols.append((k, c))
    cols.sort(key=lambda t: t[0])
    return cols


def normalize_labels(s: str) -> str:
    # For nicer titles in the figure
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


def pretty_model(s: str) -> str:
    key = str(s).strip()
    mapping = {
        "gpt-4": "GPT-4",
        "gpt4": "GPT-4",
        "gpt-5.1": "GPT-5.1",
        "gpt5.1": "GPT-5.1",
        "claude": "Claude",
        "claude 4.5": "Claude",
    }
    return mapping.get(key, key)


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=False,
        default=Path.home() / "Desktop" / "self_corrections_pandas.csv",
        help="Path to CSV with pass_iter_0..pass_iter_10 columns."
    )
    ap.add_argument(
        "--out",
        required=False,
        default=None,
        help="Output figure path (e.g., fig1.png or fig1.pdf). "
             "Defaults to fig1_self_repair_trajectories.png next to the CSV."
    )
    ap.add_argument(
        "--facet",
        choices=["history", "difficulty"],
        default="history",
        help="Facet columns by 'history' (0/1) or by 'difficulty'. Default: history.",
    )
    ap.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset whitelist (e.g., human_eval,mbppplus,apps).",
    )
    ap.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model whitelist (e.g., claude,gpt-4,gpt-5.1).",
    )
    ap.add_argument(
        "--title",
        default="Self-repair trajectories over iterations",
        help="Figure title.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs (PNG). Ignored by PDF.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)

    csv_path = Path(args.csv)

    if args.out is None:
        out_path = csv_path.with_name("fig1_self_repair_trajectories.png")
    else:
        out_path = Path(args.out)

    # out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"dataset", "model"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Optional filters
    if args.datasets:
        keep = {x.strip() for x in args.datasets.split(",") if x.strip()}
        df = df[df["dataset"].astype(str).str.strip().isin(keep)].copy()

    if args.models:
        keep = {x.strip() for x in args.models.split(",") if x.strip()}
        df = df[df["model"].astype(str).str.strip().isin(keep)].copy()

    pass_cols = find_pass_iter_cols(df)
    if not pass_cols:
        raise SystemExit("No pass_iter_k columns found (expected pass_iter_0..pass_iter_10).")

    # Ensure facet column exists
    facet_col = args.facet
    if facet_col not in df.columns:
        # Provide a helpful fallback instead of crashing mysteriously
        raise SystemExit(
            f"--facet {facet_col} requested but column '{facet_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Melt to long format: one row per (dataset, facet, model, k)
    long_rows = []
    for _, row in df.iterrows():
        dataset = safe_str(row.get("dataset"))
        facet_val = row.get(facet_col)
        model = safe_str(row.get("model"))

        for k, col in pass_cols:
            y = parse_pct(row.get(col))
            if y is None:
                continue
            long_rows.append(
                {
                    "dataset": dataset,
                    facet_col: facet_val,
                    "model": model,
                    "k": k,
                    "pass_rate": y,
                }
            )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise SystemExit("After parsing pass_iter values, no data remained (check NA/formatting).")

    # Clean labels for plotting
    long_df["dataset_label"] = long_df["dataset"].map(normalize_labels)
    long_df["model_label"] = long_df["model"].map(pretty_model)

    # Order datasets to match paper narrative
    dataset_order = ["HumanEval", "MBPP+", "APPS", "HumanEval-Java"]
    present_datasets = [d for d in dataset_order if d in set(long_df["dataset_label"])]
    if not present_datasets:
        present_datasets = sorted(long_df["dataset_label"].unique().tolist())

    # Order facet values
    if facet_col == "history":
        # Make sure 0 then 1
        def _hist_key(x):
            try:
                return int(x)
            except Exception:
                return 999

        facet_vals = sorted(long_df[facet_col].dropna().unique().tolist(), key=_hist_key)
        facet_titles = [f"history={v}" for v in facet_vals]
    else:
        facet_vals = sorted(long_df[facet_col].dropna().unique().tolist(), key=lambda x: str(x))
        facet_titles = [f"{facet_col}={v}" for v in facet_vals]

    # Order models (so lines are consistent across panels)
    model_order = ["Claude", "GPT-4", "GPT-5.1"]
    present_models = [m for m in model_order if m in set(long_df["model_label"])]
    if not present_models:
        present_models = sorted(long_df["model_label"].unique().tolist())

    # Build subplot grid
    nrows = len(present_datasets)
    ncols = max(1, len(facet_vals))

    # Size: tuned so it fits in a paper column/page when saved
    fig_w = 4.2 * ncols
    fig_h = 2.6 * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for r, dlab in enumerate(present_datasets):
        for c, fval in enumerate(facet_vals):
            ax = axes[r][c]

            panel = long_df[(long_df["dataset_label"] == dlab) & (long_df[facet_col] == fval)]

            if panel.empty:
                ax.set_axis_off()
                continue

            for m in present_models:
                sub = panel[panel["model_label"] == m].sort_values("k")
                if sub.empty:
                    continue
                ax.plot(sub["k"], sub["pass_rate"], marker="o", linewidth=1.8, markersize=3, label=m)

            ax.set_xlim(min(long_df["k"]), max(long_df["k"]))
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)

            # Titles/labels
            if r == 0:
                ax.set_title(facet_titles[c])
            if c == 0:
                ax.set_ylabel(f"{dlab}\nPass rate (%)")
            if r == nrows - 1:
                ax.set_xlabel("Iteration")

    # Single legend for the whole figure (top center)
    # Collect handles from first non-empty axis
    handles, labels = None, None
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(present_models), frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(args.title, y=1.05, fontsize=14)
    fig.tight_layout()

    # Save
    if out_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")

    print(f"✅ Wrote Figure 1 to: {out_path}")


if __name__ == "__main__":
    main()
