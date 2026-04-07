#!/usr/bin/env python3
"""
RQ5: How do models differ in their self-correction behavior?

This script compares self-correction behavior across models (e.g., claude, gpt-4, gpt-5.1)
using BOTH:
  (A) aggregate outcomes: initial_pass_rate, recovery_rate, avg_attempts_recovered
  (B) iteration dynamics: pass_iter_0..pass_iter_10

Outputs:
- rq5_out/model_summary.csv
- rq5_out/model_pairwise_diffs.csv
- rq5_out/model_iteration_curves.csv
- rq5_out/model_iteration_dynamics.csv
- rq5_out/model_summary.xlsx (optional; requires openpyxl)
- plots in rq5_out/plots/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
DEFAULT_PATH = os.environ.get(
    "SELF_CORRECTIONS_CSV",
    str(Path.home() / "Desktop" / "self_corrections_pandas.csv"),
)
PATH = DEFAULT_PATH

OUT_DIR = Path("rq5_out")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ITER_PREFIX = "pass_iter_"


# -------------------------
# Helpers
# -------------------------
def _to_float_pass(x) -> float:
    """Converts pass values that may be in [0,1] or '85.53%' form into float in [0,1]."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"na", "nan"}:
            return np.nan
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def detect_iter_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith(ITER_PREFIX)]
    # Sort by numeric suffix
    def key(c: str) -> int:
        try:
            return int(c.split("_")[-1])
        except Exception:
            return 10**9

    cols = sorted(cols, key=key)
    return cols


def trapezoid_auc(curve: np.ndarray) -> float:
    """AUC of pass-rate curve across iterations using trapezoid rule."""
    curve = curve.astype(float)
    if np.all(np.isnan(curve)):
        return np.nan
    # replace isolated NaNs by linear interpolation for AUC stability
    s = pd.Series(curve)
    s = s.interpolate(limit_direction="both")
    return float(np.trapezoid(s.values, dx=1.0))


def spearman_iter_pass(curve: np.ndarray) -> float:
    """Spearman rho between iteration index and pass rate."""
    curve = curve.astype(float)
    if np.sum(~np.isnan(curve)) < 3:
        return np.nan
    it = np.arange(len(curve), dtype=float)
    s = pd.DataFrame({"it": it, "p": curve}).dropna()
    return float(s["it"].corr(s["p"], method="spearman"))


def linear_slope(curve: np.ndarray) -> float:
    """OLS slope of pass_rate ~ iteration (per-iteration change, in absolute probability)."""
    curve = curve.astype(float)
    if np.sum(~np.isnan(curve)) < 3:
        return np.nan
    it = np.arange(len(curve), dtype=float)
    s = pd.DataFrame({"it": it, "p": curve}).dropna()
    x = s["it"].values
    y = s["p"].values
    # slope = cov(x,y)/var(x)
    vx = np.var(x)
    if vx == 0:
        return np.nan
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def first_stable_iter(curve: np.ndarray, eps: float = 0.01, win: int = 2) -> float:
    """
    First iteration t such that |p[t+k]-p[t+k-1]| <= eps for k=1..win.
    Returns NaN if never stable.
    """
    curve = curve.astype(float)
    if np.sum(~np.isnan(curve)) < 3:
        return np.nan
    s = pd.Series(curve).interpolate(limit_direction="both").values
    diffs = np.abs(np.diff(s))
    for t in range(len(diffs) - win + 1):
        if np.all(diffs[t : t + win] <= eps):
            return float(t + 1)  # stability condition holds starting at iteration (t+1)
    return np.nan


def frac_best_at_seed(curve: np.ndarray) -> float:
    """Indicator: is max achieved at iteration 0? (ties count as 1 if seed equals max)."""
    curve = curve.astype(float)
    if np.all(np.isnan(curve)):
        return np.nan
    s = pd.Series(curve).interpolate(limit_direction="both").values
    m = np.max(s)
    return float(s[0] == m)


def best_gain_after_seed(curve: np.ndarray) -> float:
    """max(pass_iter_t) - pass_iter_0"""
    curve = curve.astype(float)
    if np.all(np.isnan(curve)):
        return np.nan
    s = pd.Series(curve).interpolate(limit_direction="both").values
    return float(np.max(s) - s[0])


def summarize_group(g: pd.DataFrame, iter_cols: List[str]) -> pd.Series:
    # Aggregate outcome metrics
    out = {
        "n": len(g),
        "mean_initial_pass_rate": g["initial_pass_rate"].mean(),
        "median_initial_pass_rate": g["initial_pass_rate"].median(),
        "mean_recovery_rate": g["recovery_rate"].mean(),
        "median_recovery_rate": g["recovery_rate"].median(),
        "mean_avg_attempts_recovered": g["avg_attempts_recovered"].mean(),
        "median_avg_attempts_recovered": g["avg_attempts_recovered"].median(),
    }

    # Mean pass curve for the group
    curve = g[iter_cols].mean(axis=0, numeric_only=True).to_numpy(dtype=float)
    out.update(
        {
            "pass_iter0_mean_curve": float(curve[0]) if len(curve) else np.nan,
            "pass_max_mean_curve": float(np.nanmax(curve)) if len(curve) else np.nan,
            "iter_at_max_mean_curve": float(np.nanargmax(curve)) if len(curve) else np.nan,
            "spearman_rho_mean_curve": spearman_iter_pass(curve),
            "linear_slope_per_iter_mean_curve": linear_slope(curve),
            "auc_trapz_mean_curve": trapezoid_auc(curve),
            "first_stable_iter_mean_curve": first_stable_iter(curve, eps=0.01, win=2),
            "frac_best_at_seed_mean_curve": frac_best_at_seed(curve),
            "best_gain_after_seed_mean_curve": best_gain_after_seed(curve),
        }
    )
    return pd.Series(out)


def perm_test_diff_means(
    a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> Tuple[float, float, float]:
    """
    Two-sided permutation test for diff in means: mean(b)-mean(a).
    Returns (mean_a, mean_b, p_perm).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    obs = np.mean(b) - np.mean(a)
    pooled = np.concatenate([a, b])
    n_a = len(a)

    # permutation distribution
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        a_p = pooled[:n_a]
        b_p = pooled[n_a:]
        diff = np.mean(b_p) - np.mean(a_p)
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return (float(np.mean(a)), float(np.mean(b)), float(p))


# -------------------------
# Load + clean
# -------------------------
df = pd.read_csv(PATH)
df = df.replace(["NA", "NaN", ""], np.nan)

# Numeric conversions for main outcome columns
for c in [
    "initial_pass_rate",
    "recovery_rate",
    "avg_attempts_recovered",
    "initial_pass_rate_recovered",
    "initial_pass_rate_failed",
    "history",
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

iter_cols = detect_iter_cols(df)
if iter_cols:
    for c in iter_cols:
        df[c] = df[c].apply(_to_float_pass)

print("\n=== Detected iteration columns ===")
print(iter_cols)

# Keep only rows with a model label
df = df.dropna(subset=["model"])
df["model"] = df["model"].astype(str).str.strip().str.lower()

# -------------------------
# RQ5 core summaries
# -------------------------
print("\n=== Rows by model ===")
print(df["model"].value_counts(dropna=False))

model_summary = df.groupby("model", dropna=False).apply(lambda g: summarize_group(g, iter_cols))
model_summary = model_summary.sort_values("mean_recovery_rate", ascending=False)
print("\n=== Model summary (sorted by mean_recovery_rate) ===")
print(model_summary)

model_summary.to_csv(OUT_DIR / "model_summary.csv", index=True)

# Pairwise permutation tests: recovery_rate + avg_attempts_recovered + curve slope/auc
models = list(model_summary.index)
pair_rows = []
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        m1, m2 = models[i], models[j]
        g1 = df[df["model"] == m1]
        g2 = df[df["model"] == m2]

        for metric in ["initial_pass_rate", "recovery_rate", "avg_attempts_recovered"]:
            mean1, mean2, p = perm_test_diff_means(g1[metric].to_numpy(), g2[metric].to_numpy(), n_perm=10000, seed=0)
            pair_rows.append(
                {
                    "metric": metric,
                    "A": m1,
                    "B": m2,
                    "mean(A)": mean1,
                    "mean(B)": mean2,
                    "diff_mean(B-A)": mean2 - mean1 if (not np.isnan(mean1) and not np.isnan(mean2)) else np.nan,
                    "p_perm": p,
                    "n(A)": int(np.sum(~np.isnan(g1[metric]))),
                    "n(B)": int(np.sum(~np.isnan(g2[metric]))),
                }
            )

pairwise = pd.DataFrame(pair_rows).sort_values(["metric", "p_perm"])
print("\n=== Pairwise permutation tests (A vs B) ===")
print(pairwise)

pairwise.to_csv(OUT_DIR / "model_pairwise_diffs.csv", index=False)

# -------------------------
# Iteration dynamics (curves)
# -------------------------
if iter_cols:
    # Long form for plotting / analysis
    long_rows = []
    for _, row in df.iterrows():
        for k, c in enumerate(iter_cols):
            long_rows.append(
                {
                    "dataset": row.get("dataset", np.nan),
                    "difficulty": row.get("difficulty", np.nan),
                    "history": row.get("history", np.nan),
                    "model": row.get("model", np.nan),
                    "iteration": k,
                    "pass_rate": row.get(c, np.nan),
                }
            )
    it_long = pd.DataFrame(long_rows)
    it_long.to_csv(OUT_DIR / "model_iteration_curves.csv", index=False)

    # Average curve by model
    avg_curve = (
        it_long.groupby(["model", "iteration"], dropna=False)["pass_rate"]
        .mean()
        .reset_index()
        .sort_values(["model", "iteration"])
    )
    print("\n=== Average pass_rate by iteration x model ===")
    print(avg_curve)
    avg_curve.to_csv(OUT_DIR / "avg_pass_by_iter_x_model.csv", index=False)

    # Dynamics metrics computed per model from mean curve (already in model_summary),
    # but also compute on each condition row then average within model for robustness.
    cond_dyn_rows = []
    for _, row in df.iterrows():
        curve = np.array([row.get(c, np.nan) for c in iter_cols], dtype=float)
        cond_dyn_rows.append(
            {
                "dataset": row.get("dataset", np.nan),
                "difficulty": row.get("difficulty", np.nan),
                "history": row.get("history", np.nan),
                "model": row.get("model", np.nan),
                "spearman_rho": spearman_iter_pass(curve),
                "linear_slope_per_iter": linear_slope(curve),
                "auc_trapz": trapezoid_auc(curve),
                "first_stable_iter": first_stable_iter(curve, eps=0.01, win=2),
                "frac_best_at_seed": frac_best_at_seed(curve),
                "best_gain_after_seed": best_gain_after_seed(curve),
            }
        )

    cond_dyn = pd.DataFrame(cond_dyn_rows)
    cond_dyn.to_csv(OUT_DIR / "condition_dynamics.csv", index=False)

    model_dyn = cond_dyn.groupby("model").agg(
        n=("model", "count"),
        spearman_rho_mean=("spearman_rho", "mean"),
        spearman_rho_median=("spearman_rho", "median"),
        linear_slope_mean=("linear_slope_per_iter", "mean"),
        linear_slope_median=("linear_slope_per_iter", "median"),
        auc_mean=("auc_trapz", "mean"),
        auc_median=("auc_trapz", "median"),
        frac_best_at_seed_mean=("frac_best_at_seed", "mean"),
        best_gain_after_seed_mean=("best_gain_after_seed", "mean"),
    ).reset_index()

    print("\n=== Model iteration dynamics (avg over condition curves) ===")
    print(model_dyn)
    model_dyn.to_csv(OUT_DIR / "model_iteration_dynamics.csv", index=False)

    # -------------------------
    # Plots (matplotlib only; no forced colors)
    # -------------------------
    # Average pass curve per model
    plt.figure()
    for m in sorted(df["model"].unique()):
        sub = avg_curve[avg_curve["model"] == m]
        if len(sub) == 0:
            continue
        plt.plot(sub["iteration"], sub["pass_rate"], marker="o", label=m)
    plt.title("RQ5: Average pass rate vs iteration (by model)")
    plt.xlabel("Iteration")
    plt.ylabel("Pass rate")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_curve_by_model.png", dpi=200)
    plt.close()

    # Distribution of slopes per model (boxplot)
    plt.figure()
    order = sorted(df["model"].unique())
    data = [cond_dyn.loc[cond_dyn["model"] == m, "linear_slope_per_iter"].dropna().to_numpy() for m in order]
    if any(len(x) > 0 for x in data):
        plt.boxplot(data, tick_labels=order)
        plt.title("RQ5: Per-condition linear slope per iteration (by model)")
        plt.xlabel("Model")
        plt.ylabel("Linear slope per iter (Δ pass rate)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "slope_boxplot_by_model.png", dpi=200)
    plt.close()

    # Fraction best at seed (bar)
    plt.figure()
    fb = model_dyn.set_index("model")["frac_best_at_seed_mean"].reindex(order)
    plt.bar(fb.index, fb.values)
    plt.title("RQ5: Fraction of conditions where best iteration is the seed (iter 0)")
    plt.xlabel("Model")
    plt.ylabel("Fraction best at seed")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "frac_best_at_seed_by_model.png", dpi=200)
    plt.close()

# -------------------------
# Optional: Excel workbook
# -------------------------
try:
    out_xlsx = OUT_DIR / "model_summary.xlsx"
    with pd.ExcelWriter(out_xlsx) as writer:
        model_summary.reset_index().to_excel(writer, sheet_name="model_summary", index=False)
        pairwise.to_excel(writer, sheet_name="pairwise_tests", index=False)
        if iter_cols:
            avg_curve.to_excel(writer, sheet_name="avg_curve_by_model", index=False)
            model_dyn.to_excel(writer, sheet_name="model_dynamics", index=False)
    print(f"\nWrote: {out_xlsx}")
except Exception as e:
    print("\n[Skipping Excel export] (install openpyxl) Error:", e)

print("\nSaved:")
print(f"- {OUT_DIR / 'model_summary.csv'}")
print(f"- {OUT_DIR / 'model_pairwise_diffs.csv'}")
if iter_cols:
    print(f"- {OUT_DIR / 'avg_pass_by_iter_x_model.csv'}")
    print(f"- {OUT_DIR / 'condition_dynamics.csv'}")
    print(f"- {OUT_DIR / 'model_iteration_dynamics.csv'}")
print(f"- plots in {PLOTS_DIR.resolve()}")
