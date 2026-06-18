#!/usr/bin/env python3
"""
Figure RQ2: Recovery-rate heatmap across models x dataset/difficulty.

Two panels side by side: h=0 (left) and h=1 (right).
Rows = models (Claude, GPT-4, GPT-5.1)
Cols = dataset x difficulty (HumanEval, MBPP+, APPS Intro, APPS Comp., HumanEval-Java)
Cell color = recovery rate (0–1 scale)

Usage:
  python3 make_fig_rq2_heatmap.py
  python3 make_fig_rq2_heatmap.py --csv rq5_group_curve_metrics.csv --out fig_rq2_heatmap.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


DATASET_ORDER = [
    ("human_eval",      ""),
    ("mbppplus",        ""),
    ("apps",            "intro"),
    ("apps",            "competition"),
    ("human_eval_java", ""),
]

DATASET_LABELS = {
    ("human_eval",      ""): "HumanEval",
    ("mbppplus",        ""): "MBPP+",
    ("apps",            "intro"): "APPS\nIntro",
    ("apps",            "competition"): "APPS\nComp.",
    ("human_eval_java", ""): "HumanEval\nJava",
}

MODEL_ORDER  = ["claude", "gpt-4", "gpt-5.1"]
MODEL_LABELS = {"claude": "Claude", "gpt-4": "GPT-4", "gpt-5.1": "GPT-5.1"}


def normalise_model(s: str) -> str:
    s = str(s).strip().lower()
    for k in MODEL_ORDER:
        if s.startswith(k):
            return k
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="rq5_group_curve_metrics.csv")
    ap.add_argument("--out", default="fig_rq2_heatmap.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["model_key"] = df["model"].apply(normalise_model)
    df["difficulty"] = df["difficulty"].fillna("").astype(str).str.strip()
    df["dataset"]    = df["dataset"].astype(str).str.strip()
    df["recovery_rate"] = pd.to_numeric(df["recovery_rate"], errors="coerce")

    histories = [0, 1]
    n_cols = len(DATASET_ORDER)
    n_rows = len(MODEL_ORDER)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 3.2),
        gridspec_kw={"wspace": 0.35},
    )

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for ax_i, h in enumerate(histories):
        ax = axes[ax_i]
        sub = df[df["history"] == h]

        matrix = np.full((n_rows, n_cols), np.nan)
        for ri, m in enumerate(MODEL_ORDER):
            for ci, (ds, diff) in enumerate(DATASET_ORDER):
                row = sub[
                    (sub["model_key"] == m) &
                    (sub["dataset"] == ds) &
                    (sub["difficulty"] == diff)
                ]
                if not row.empty and not pd.isna(row.iloc[0]["recovery_rate"]):
                    matrix[ri, ci] = row.iloc[0]["recovery_rate"]

        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        # Annotate cells
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = matrix[ri, ci]
                if np.isnan(val):
                    txt = "—"
                    color = "grey"
                else:
                    txt = f"{val:.2f}"
                    color = "white" if val > 0.55 else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [DATASET_LABELS[k] for k in DATASET_ORDER],
            fontsize=8,
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER],
            fontsize=9,
        )
        ax.set_title(f"$h={h}$ (no history)" if h == 0 else f"$h={h}$ (full history)",
                     fontsize=10, pad=6)

    # Shared colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
    )
    cbar.set_label("Recovery rate", fontsize=9)

    fig.suptitle("Recovery rate across models and datasets", fontsize=12, y=1.03)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"✅ Wrote RQ2 heatmap to: {out_path}")


if __name__ == "__main__":
    main()
