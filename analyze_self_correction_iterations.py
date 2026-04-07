#!/usr/bin/env python3
"""
Analyze iteration pass rates per (dataset, difficulty, history, model).

Input: a CSV with columns:
  dataset, difficulty, history, model, ... plus iteration pass-rate columns.
Iteration columns may be:
  - percent strings like "55.75%"
  - fractions like 0.6844
  - or percent numbers like 55.75 (we'll infer and convert)

Outputs:
  - iteration_long.csv: long/tidy data (one row per condition x iteration)
  - iteration_summary.csv: per-condition trajectory summaries

Also prints key tables and shows plots.

Usage:
  python analyze_iteration_passrates.py --csv /path/to/your.csv --out_dir results_iters
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


META_COLS = {
    "dataset",
    "difficulty",
    "history",
    "model",
    "initial_pass_rate",
    "recovery_rate",
    "avg_attempts_recovered",
    "initial_pass_rate_recovered",
    "initial_pass_rate_failed",
}


def parse_rate(x):
    """Convert values like '55.75%' / 55.75 / 0.5575 / NA to float in [0,1]."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.upper() in {"NA", "N/A", "NAN", ""}:
        return np.nan

    # Percent string
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return np.nan

    # Plain number string
    try:
        v = float(s)
    except ValueError:
        return np.nan

    # Heuristic: if v > 1.5, it's probably a percent (e.g., 55.75)
    if v > 1.5:
        return v / 100.0
    return v


def detect_iteration_cols(df: pd.DataFrame):
    """
    Detect iteration columns.

    Priority:
      1) columns that look like iter_0 / iter0 / pass_iter_0 / refinement_1 etc.
      2) otherwise, columns not in META_COLS whose values parse as rates in [0,1] for many rows
    Returns: list of (colname, iteration_index)
    """
    cols = list(df.columns)

    # 1) name-based detection
    name_patterns = [
        (re.compile(r"^(?:pass_)?iter[_ ]?(\d+)$", re.I), 1),
        (re.compile(r"^pass_iter[_ ]?(\d+)$", re.I), 1),
        (re.compile(r"^refinement[_ ]?(\d+)$", re.I), 1),
        (re.compile(r"^r(\d+)$", re.I), 1),
    ]
    found = []
    for c in cols:
        for pat, grp in name_patterns:
            m = pat.match(c.strip())
            if m:
                found.append((c, int(m.group(grp))))
                break

    if found:
        return sorted(found, key=lambda t: t[1])

    # 2) content-based detection
    candidates = [c for c in cols if c not in META_COLS]
    scored = []
    for c in candidates:
        # sample up to first 20 rows
        sample = df[c].head(20).map(parse_rate)
        ok = sample.notna().mean()
        # consider it an iteration col if at least half of sampled entries parse
        if ok >= 0.5:
            scored.append(c)

    # If content-based, preserve column order and assign iteration index by position (0..k-1)
    iter_cols = [(c, i) for i, c in enumerate(scored)]
    return iter_cols


def auc_trapz(xs, ys):
    """Area under curve via trapezoid rule, ignoring NaNs."""
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs2, ys2 = xs[mask], ys[mask]
    if len(xs2) < 2:
        return np.nan
    return float(np.trapz(ys2, xs2))


def first_stable_iteration(y, eps=0.01, window=2):
    """
    Return first iteration t where changes are small:
      |y[t+i] - y[t+i-1]| <= eps for 'window' consecutive steps.
    """
    if y is None or len(y) < window + 2:
        return np.nan
    y = np.array(y, dtype=float)
    for t in range(1, len(y) - window):
        ok = True
        for i in range(1, window + 1):
            if np.isnan(y[t + i]) or np.isnan(y[t + i - 1]):
                ok = False
                break
            if abs(y[t + i] - y[t + i - 1]) > eps:
                ok = False
                break
        if ok:
            return int(t)
    return np.nan


def spearman_corr(x, y):
    """Spearman correlation without scipy, using rank + Pearson on ranks."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_xy) < 3:
        return np.nan
    rx = df_xy["x"].rank(method="average")
    ry = df_xy["y"].rank(method="average")
    return float(rx.corr(ry, method="pearson"))


def fit_slope(x, y):
    """Simple linear slope y ~ a + b x, ignoring NaNs."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    b = np.polyfit(x[mask], y[mask], 1)[0]
    return float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path.home() / "Desktop" / "self_corrections_pandas.csv",
        help="Path to self-correction results CSV"
    )
    ap.add_argument("--out_dir", default="iteration_analysis_out", help="Output directory")
    ap.add_argument("--title", default="Iteration pass-rate trajectories", help="Plot title")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = df.replace(["NA", "N/A", "NaN", ""], np.nan)

    # Ensure history numeric (0/1) if present
    if "history" in df.columns:
        df["history"] = pd.to_numeric(df["history"], errors="coerce")

    iter_cols = detect_iteration_cols(df)
    if not iter_cols:
        raise SystemExit(
            "Could not detect iteration columns. "
            "Rename them like pass_iter_0, pass_iter_1, ... or include values as percents."
        )

    # Parse iteration columns into numeric fractions
    for c, _t in iter_cols:
        df[c] = df[c].map(parse_rate)

    # Create long format
    long_rows = []
    for _, row in df.iterrows():
        meta = {k: row.get(k, np.nan) for k in ["dataset", "difficulty", "history", "model"]}
        for c, t in iter_cols:
            long_rows.append({**meta, "iteration": t, "pass_rate": row[c]})
    dfl = pd.DataFrame(long_rows)
    dfl = dfl.sort_values(["dataset", "difficulty", "history", "model", "iteration"]).reset_index(drop=True)

    # Save long data
    long_path = out_dir / "iteration_long.csv"
    dfl.to_csv(long_path, index=False)

    # Per-condition summaries
    group_cols = ["dataset", "difficulty", "history", "model"]
    summaries = []
    for key, g in dfl.groupby(group_cols):
        g = g.sort_values("iteration")
        xs = g["iteration"].to_numpy(dtype=float)
        ys = g["pass_rate"].to_numpy(dtype=float)

        # Metrics
        rho = spearman_corr(xs, ys)                       # monotonic trend
        slope = fit_slope(xs, ys)                         # linear trend (per-iter change)
        auc = auc_trapz(xs, ys)                           # overall mass of curve
        y0 = ys[0] if len(ys) else np.nan                  # initial pass at iter0 if present
        y_max = np.nanmax(ys) if np.any(~np.isnan(ys)) else np.nan
        t_max = int(xs[np.nanargmax(ys)]) if np.any(~np.isnan(ys)) else np.nan
        stable_t = first_stable_iteration(ys, eps=0.01, window=2)

        # “best improvement after initial”
        post = ys[1:] if len(ys) > 1 else np.array([])
        best_post = np.nanmax(post) if len(post) else np.nan
        best_gain = best_post - y0 if (len(post) and not np.isnan(best_post) and not np.isnan(y0)) else np.nan

        summaries.append(
            {
                "dataset": key[0],
                "difficulty": key[1],
                "history": key[2],
                "model": key[3],
                "iters_observed": int(np.sum(~np.isnan(ys))),
                "pass_iter0": y0,
                "pass_max": y_max,
                "iter_at_max": t_max,
                "best_gain_after_iter0": best_gain,
                "spearman_rho(iter, pass)": rho,
                "linear_slope_per_iter": slope,
                "auc_trapz": auc,
                "first_stable_iter(eps=0.01,win=2)": stable_t,
            }
        )

    dfs = pd.DataFrame(summaries).sort_values(group_cols).reset_index(drop=True)
    summary_path = out_dir / "iteration_summary.csv"
    dfs.to_csv(summary_path, index=False)

    # Print key views
    print("\n=== Detected iteration columns ===")
    print([c for c, _ in iter_cols])

    print("\n=== Per-condition summary (sorted) ===")
    with pd.option_context("display.max_columns", 200, "display.width", 140):
        print(dfs)

    print("\n=== Average pass_rate by iteration (overall) ===")
    overall = dfl.groupby("iteration")["pass_rate"].mean().reset_index()
    print(overall)

    print("\n=== Average pass_rate by iteration x history ===")
    by_hist = dfl.groupby(["history", "iteration"])["pass_rate"].mean().reset_index()
    print(by_hist)

    print("\n=== Average pass_rate by iteration x model ===")
    by_model = dfl.groupby(["model", "iteration"])["pass_rate"].mean().reset_index()
    print(by_model)

    # Plots
    # 1) Per-condition trajectories
    plt.figure()
    for key, g in dfl.groupby(group_cols):
        g = g.sort_values("iteration")
        label = f"{key[0]}|{key[1]}|h={int(key[2]) if not pd.isna(key[2]) else key[2]}|{key[3]}"
        plt.plot(g["iteration"], g["pass_rate"], marker="o", label=label)
    plt.title(args.title + " (per condition)")
    plt.xlabel("Iteration (0 = initial)")
    plt.ylabel("Pass rate")
    plt.legend(fontsize=7, ncol=1, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "trajectories_per_condition.png", dpi=200)
    plt.show()

    # 2) Mean curves by history
    plt.figure()
    for h, g in dfl.groupby("history"):
        gg = g.groupby("iteration")["pass_rate"].mean().reset_index().sort_values("iteration")
        plt.plot(gg["iteration"], gg["pass_rate"], marker="o", label=f"history={int(h)}")
    plt.title(args.title + " (mean by history)")
    plt.xlabel("Iteration")
    plt.ylabel("Mean pass rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_by_history.png", dpi=200)
    plt.show()

    # 3) Mean curves by model
    plt.figure()
    for m, g in dfl.groupby("model"):
        gg = g.groupby("iteration")["pass_rate"].mean().reset_index().sort_values("iteration")
        plt.plot(gg["iteration"], gg["pass_rate"], marker="o", label=str(m))
    plt.title(args.title + " (mean by model)")
    plt.xlabel("Iteration")
    plt.ylabel("Mean pass rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_by_model.png", dpi=200)
    plt.show()

    print(f"\nSaved:\n- {long_path}\n- {summary_path}\n- plots in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
