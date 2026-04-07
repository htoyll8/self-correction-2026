#!/usr/bin/env python3
"""
Figure RQ3: Initial pass rate vs recovery rate (capability-ceiling scatter).

Each point is one (model, dataset, difficulty, history) condition.
x = initial_pass_rate
y = recovery_rate
Color = model
Marker shape = dataset
Panels = h=0 (left) and h=1 (right)

The plot illustrates the capability-ceiling effect: a stronger model has a
higher initial pass rate but a smaller, harder residual failure set — so its
recovery rate can be lower than a weaker model's.

Usage:
  python3 make_fig_rq3_scatter.py
  python3 make_fig_rq3_scatter.py --csv rq5_group_curve_metrics.csv --out fig_rq3_scatter.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


MODEL_ORDER  = ["claude", "gpt-4", "gpt-5.1"]
MODEL_LABELS = {"claude": "Claude", "gpt-4": "GPT-4", "gpt-5.1": "GPT-5.1"}
MODEL_COLORS = {"claude": "#E87722", "gpt-4": "#4C72B0", "gpt-5.1": "#55A868"}

DATASET_ORDER = ["human_eval", "mbppplus", "apps", "human_eval_java"]
DATASET_LABELS = {
    "human_eval":      "HumanEval",
    "mbppplus":        "MBPP+",
    "apps":            "APPS",
    "human_eval_java": "HumanEval-Java",
}
MARKER_MAP = {
    "human_eval":      "o",
    "mbppplus":        "s",
    "apps":            "^",
    "human_eval_java": "D",
}


def normalise_model(s: str) -> str:
    s = str(s).strip().lower()
    for k in MODEL_ORDER:
        if s.startswith(k):
            return k
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="rq5_group_curve_metrics.csv")
    ap.add_argument("--out", default="fig_rq3_scatter.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["model_key"]     = df["model"].apply(normalise_model)
    df["dataset"]       = df["dataset"].astype(str).str.strip()
    df["initial_pass_rate"] = pd.to_numeric(df["initial_pass_rate"], errors="coerce")
    df["recovery_rate"]     = pd.to_numeric(df["recovery_rate"],     errors="coerce")

    # Drop rows missing either axis
    df = df.dropna(subset=["initial_pass_rate", "recovery_rate"])

    histories = [0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, sharex=True)

    for ax_i, h in enumerate(histories):
        ax = axes[ax_i]
        sub = df[df["history"] == h]

        for m in MODEL_ORDER:
            subm = sub[sub["model_key"] == m]
            for ds in DATASET_ORDER:
                submd = subm[subm["dataset"] == ds]
                if submd.empty:
                    continue
                ax.scatter(
                    submd["initial_pass_rate"],
                    submd["recovery_rate"],
                    color=MODEL_COLORS[m],
                    marker=MARKER_MAP.get(ds, "o"),
                    s=80,
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )

        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.axline((0, 0), slope=1, color="grey", linestyle="--",
                  linewidth=0.8, alpha=0.5, label="init = recovery")
        ax.set_xlabel("Initial pass rate", fontsize=10)
        ax.set_title(
            "$h=0$ (no history)" if h == 0 else "$h=1$ (full history)",
            fontsize=10,
        )
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Recovery rate", fontsize=10)

    # Model legend (colors)
    model_handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], marker="o", linestyle="",
                      markersize=8, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]

    # Dataset legend (markers)
    dataset_handles = [
        mlines.Line2D([], [], color="grey", marker=MARKER_MAP[ds], linestyle="",
                      markersize=8, label=DATASET_LABELS[ds])
        for ds in DATASET_ORDER
    ]

    fig.legend(
        model_handles,
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.42, 1.04),
        title="Model",
        fontsize=9,
    )
    fig.legend(
        dataset_handles,
        [DATASET_LABELS[ds] for ds in DATASET_ORDER],
        loc="upper center",
        ncol=len(DATASET_ORDER),
        frameon=False,
        bbox_to_anchor=(0.42, 0.98),
        title="Dataset",
        fontsize=9,
    )

    fig.suptitle(
        "Initial pass rate vs recovery rate",
        fontsize=12, y=1.10,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"✅ Wrote RQ3 scatter to: {out_path}")


if __name__ == "__main__":
    main()
