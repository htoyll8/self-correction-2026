#!/usr/bin/env python3
"""
compare_humaneval_python_vs_java.py

Compares HumanEval-Python (dataset == "human_eval") vs HumanEval-Java
(dataset == "human_eval_java") for:
- headline metrics: initial_pass_rate, recovery_rate, avg_attempts_recovered
- iteration dynamics: pass_iter_0..pass_iter_10 (seed dominance, trend, slope)
- effect sizes (Python - Java and Java - Python)
- optional permutation tests (no SciPy/statsmodels required)

USAGE:
  python3 compare_humaneval_python_vs_java.py --path /path/to/your.csv
"""

import argparse
import math
import random
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
ITER_COLS = [f"pass_iter_{i}" for i in range(0, 11)]


def to_num(x):
    """Parse numbers like 0.57, '55.75%', 'NA', '' into float (0-1 scale)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation without SciPy: compute Pearson correlation on ranks.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def linear_slope_per_iter(pass_rates: np.ndarray) -> float:
    """
    Fit a simple line pass_rate ~ a + b * iteration, return b.
    This is "average change in pass_rate per iteration".
    """
    it = np.arange(len(pass_rates), dtype=float)
    mask = ~np.isnan(pass_rates)
    it = it[mask]
    pr = pass_rates[mask]
    if len(pr) < 2:
        return np.nan
    # least squares slope
    it_mean = it.mean()
    pr_mean = pr.mean()
    denom = np.sum((it - it_mean) ** 2)
    if denom == 0:
        return np.nan
    slope = np.sum((it - it_mean) * (pr - pr_mean)) / denom
    return float(slope)


def auc_trapz(pass_rates: np.ndarray) -> float:
    """Trapezoidal AUC across iterations (iteration axis step=1)."""
    mask = ~np.isnan(pass_rates)
    pr = pass_rates[mask]
    if len(pr) < 2:
        return np.nan
    # If missing in the middle, this is a "compressed" AUC; for your data it's typically complete.
    return float(np.trapz(pr, dx=1.0))


def first_stable_iter(pass_rates: np.ndarray, eps: float = 0.01, win: int = 2) -> float:
    """
    First iteration k such that |p[k+1]-p[k]| <= eps and ... for 'win' consecutive steps.
    Returns k (as float for nicer printing), or NaN if never stable.
    """
    pr = pass_rates.copy()
    if np.any(np.isnan(pr)):
        # if missing, just bail out (your Humaneval rows should have values)
        return np.nan
    diffs = np.abs(np.diff(pr))
    # need win consecutive diffs <= eps starting at k
    for k in range(0, len(diffs) - win + 1):
        if np.all(diffs[k : k + win] <= eps):
            return float(k)
    return np.nan


def perm_test_diff_means(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """
    Two-sided permutation test for mean difference (mean(b) - mean(a)).
    Returns (diff, p_perm).
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    diff_obs = float(b.mean() - a.mean())

    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        a2 = pooled[:n_a]
        b2 = pooled[n_a:]
        diff = float(b2.mean() - a2.mean())
        if abs(diff) >= abs(diff_obs):
            more_extreme += 1
    p = (more_extreme + 1) / (n_perm + 1)
    return diff_obs, float(p)


def summarize_group(df: pd.DataFrame) -> pd.Series:
    out = {
        "n": len(df),
        "mean_initial_pass": df["initial_pass_rate"].mean(),
        "median_initial_pass": df["initial_pass_rate"].median(),
        "mean_recovery": df["recovery_rate"].mean(),
        "median_recovery": df["recovery_rate"].median(),
        "mean_attempts_recovered": df["avg_attempts_recovered"].mean(),
        "median_attempts_recovered": df["avg_attempts_recovered"].median(),
    }

    # Seed dominance + best refinement gain (per row)
    best_gain = []
    iter_at_max = []
    for _, r in df.iterrows():
        pr = r[ITER_COLS].to_numpy(dtype=float)
        if np.all(np.isnan(pr)):
            continue
        i0 = pr[0]
        mx = np.nanmax(pr)
        it_mx = int(np.nanargmax(pr))
        best_gain.append(mx - i0)
        iter_at_max.append(it_mx)

    out["frac_best_at_seed"] = float(np.mean([1 if i == 0 else 0 for i in iter_at_max])) if iter_at_max else np.nan
    out["mean_best_gain_after_seed"] = float(np.mean(best_gain)) if best_gain else np.nan
    out["median_best_gain_after_seed"] = float(np.median(best_gain)) if best_gain else np.nan

    # Aggregate iteration curve (mean across rows)
    iter_means = []
    for c in ITER_COLS:
        iter_means.append(df[c].mean())
    out.update({f"mean_{c}": iter_means[i] for i, c in enumerate(ITER_COLS)})

    # Trend metrics on the *mean curve*
    mean_curve = np.array(iter_means, dtype=float)
    out["spearman_rho_iter_pass_mean_curve"] = spearman_rho(np.arange(len(mean_curve), dtype=float), mean_curve)
    out["linear_slope_per_iter_mean_curve"] = linear_slope_per_iter(mean_curve)
    out["auc_trapz_mean_curve"] = auc_trapz(mean_curve)
    out["first_stable_iter_mean_curve"] = first_stable_iter(mean_curve, eps=0.01, win=2)

    return pd.Series(out)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=Path,
        default=Path.home() / "Desktop" / "self_corrections_pandas.csv",
        help="Path to self-correction results CSV"
    )
    ap.add_argument("--n_perm", type=int, default=10000, help="Permutations for permutation tests.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_prefix", type=str, default="humaneval_py_vs_java", help="Prefix for saved plots.")
    args = ap.parse_args()

    df = pd.read_csv(args.path)

    # Clean NA + convert numeric cols
    df = df.replace(["NA", "NaN", ""], np.nan)

    # Convert key numeric cols
    core_cols = [
        "initial_pass_rate",
        "recovery_rate",
        "avg_attempts_recovered",
        "history",
    ] + ITER_COLS

    for c in core_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_num)

    # Filter to language datasets
    df_lang = df[df["dataset"].isin(["human_eval", "human_eval_java"])].copy()

    # Basic sanity
    print("\n=== Rows by dataset ===")
    print(df_lang["dataset"].value_counts(dropna=False))

    # Split
    py = df_lang[df_lang["dataset"] == "human_eval"].copy()
    ja = df_lang[df_lang["dataset"] == "human_eval_java"].copy()

    # Summaries overall + by history + by model
    print("\n=== OVERALL summaries ===")
    overall = df_lang.groupby("dataset").apply(summarize_group)
    print(overall[[
        "n",
        "mean_initial_pass", "median_initial_pass",
        "mean_recovery", "median_recovery",
        "mean_attempts_recovered", "median_attempts_recovered",
        "frac_best_at_seed",
        "mean_best_gain_after_seed", "median_best_gain_after_seed",
        "spearman_rho_iter_pass_mean_curve",
        "linear_slope_per_iter_mean_curve",
        "auc_trapz_mean_curve",
        "first_stable_iter_mean_curve",
    ]])

    print("\n=== Summaries by dataset x history ===")
    by_hist = df_lang.groupby(["dataset", "history"]).apply(summarize_group)
    print(by_hist[[
        "n",
        "mean_initial_pass", "mean_recovery", "mean_attempts_recovered",
        "frac_best_at_seed", "mean_best_gain_after_seed",
        "spearman_rho_iter_pass_mean_curve",
        "linear_slope_per_iter_mean_curve",
    ]])

    print("\n=== Summaries by dataset x model ===")
    by_model = df_lang.groupby(["dataset", "model"]).apply(summarize_group)
    print(by_model[[
        "n",
        "mean_initial_pass", "mean_recovery", "mean_attempts_recovered",
        "frac_best_at_seed", "mean_best_gain_after_seed",
        "spearman_rho_iter_pass_mean_curve",
        "linear_slope_per_iter_mean_curve",
    ]])

    # -------------------------
    # Effect sizes + permutation tests (Python vs Java)
    # -------------------------
    def report_effect(metric: str):
        a = py[metric].to_numpy(dtype=float)
        b = ja[metric].to_numpy(dtype=float)
        diff, p = perm_test_diff_means(a, b, n_perm=args.n_perm, seed=args.seed)
        print(f"\n=== Permutation test: {metric} (diff = mean(Java) - mean(Python)) ===")
        print(f"mean(Python)={np.nanmean(a):.4f}  mean(Java)={np.nanmean(b):.4f}  diff={diff:.4f}  p_perm={p:.4f}  n_py={np.sum(~np.isnan(a))}  n_ja={np.sum(~np.isnan(b))}")

    for m in ["initial_pass_rate", "recovery_rate", "avg_attempts_recovered"]:
        report_effect(m)

    # Seed dominance effect: fraction best at seed (use per-row iter_at_max)
    def per_row_seed_dom(df_sub: pd.DataFrame) -> np.ndarray:
        vals = []
        for _, r in df_sub.iterrows():
            pr = r[ITER_COLS].to_numpy(dtype=float)
            if np.all(np.isnan(pr)):
                continue
            it_mx = int(np.nanargmax(pr))
            vals.append(1.0 if it_mx == 0 else 0.0)
        return np.array(vals, dtype=float)

    seed_py = per_row_seed_dom(py)
    seed_ja = per_row_seed_dom(ja)
    diff, p = perm_test_diff_means(seed_py, seed_ja, n_perm=args.n_perm, seed=args.seed)
    print("\n=== Permutation test: frac_best_at_seed (diff = mean(Java) - mean(Python)) ===")
    print(f"mean(Python)={seed_py.mean() if len(seed_py)>0 else np.nan:.4f}  mean(Java)={seed_ja.mean() if len(seed_ja)>0 else np.nan:.4f}  diff={diff:.4f}  p_perm={p:.4f}  n_py={len(seed_py)}  n_ja={len(seed_ja)}")

    # Best gain after seed: per-row
    def per_row_best_gain(df_sub: pd.DataFrame) -> np.ndarray:
        vals = []
        for _, r in df_sub.iterrows():
            pr = r[ITER_COLS].to_numpy(dtype=float)
            if np.all(np.isnan(pr)):
                continue
            i0 = pr[0]
            mx = np.nanmax(pr)
            vals.append(float(mx - i0))
        return np.array(vals, dtype=float)

    gain_py = per_row_best_gain(py)
    gain_ja = per_row_best_gain(ja)
    diff, p = perm_test_diff_means(gain_py, gain_ja, n_perm=args.n_perm, seed=args.seed)
    print("\n=== Permutation test: best_gain_after_seed (diff = mean(Java) - mean(Python)) ===")
    print(f"mean(Python)={gain_py.mean() if len(gain_py)>0 else np.nan:.4f}  mean(Java)={gain_ja.mean() if len(gain_ja)>0 else np.nan:.4f}  diff={diff:.4f}  p_perm={p:.4f}  n_py={len(gain_py)}  n_ja={len(gain_ja)}")

    # -------------------------
    # Plots: mean pass_rate vs iteration (overall + by model)
    # -------------------------
    def plot_mean_curve(df_sub: pd.DataFrame, title: str, save_name: str):
        means = [df_sub[c].mean() for c in ITER_COLS]
        plt.figure()
        plt.plot(range(0, 11), means, marker="o")
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Mean pass_rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_name, dpi=200)
        plt.close()

    plot_mean_curve(py, "HumanEval-Python: mean pass_rate vs iteration", f"{args.save_prefix}_py_mean_curve.png")
    plot_mean_curve(ja, "HumanEval-Java: mean pass_rate vs iteration", f"{args.save_prefix}_java_mean_curve.png")

    # By model overlay, per language
    def plot_overlay_by_model(df_sub: pd.DataFrame, title: str, save_name: str):
        plt.figure()
        for model, g in df_sub.groupby("model"):
            means = [g[c].mean() for c in ITER_COLS]
            plt.plot(range(0, 11), means, marker="o", label=str(model))
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Mean pass_rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_name, dpi=200)
        plt.close()

    plot_overlay_by_model(py, "HumanEval-Python: mean pass_rate vs iteration (by model)", f"{args.save_prefix}_py_by_model.png")
    plot_overlay_by_model(ja, "HumanEval-Java: mean pass_rate vs iteration (by model)", f"{args.save_prefix}_java_by_model.png")

    # Optional: side-by-side plot (Python vs Java overall)
    plt.figure()
    py_means = [py[c].mean() for c in ITER_COLS]
    ja_means = [ja[c].mean() for c in ITER_COLS]
    plt.plot(range(0, 11), py_means, marker="o", label="Python (HumanEval)")
    plt.plot(range(0, 11), ja_means, marker="o", label="Java (HumanEval-Java)")
    plt.title("Mean pass_rate vs iteration: HumanEval Python vs Java")
    plt.xlabel("Iteration")
    plt.ylabel("Mean pass_rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.save_prefix}_py_vs_java_mean_curve.png", dpi=200)
    plt.close()

    print("\nSaved plots:")
    print(f"- {args.save_prefix}_py_mean_curve.png")
    print(f"- {args.save_prefix}_java_mean_curve.png")
    print(f"- {args.save_prefix}_py_by_model.png")
    print(f"- {args.save_prefix}_java_by_model.png")
    print(f"- {args.save_prefix}_py_vs_java_mean_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
