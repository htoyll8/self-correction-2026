"""
RQ4: Does providing repair history change recovery outcomes?

This script:
1) Loads / cleans your tidy CSV
2) Computes history vs no-history differences for:
   - Effectiveness: recovery_rate
   - Depth (conditional): avg_attempts_recovered
   - (Optional context): initial_pass_rate
3) Runs stratified comparisons (by dataset, difficulty, model)
4) Runs simple nonparametric tests where possible (permutation test; MWU optional)
5) Exports ALL tables to an Excel workbook so you can inspect everything

Input CSV columns expected:
dataset, difficulty, history, model,
initial_pass_rate, recovery_rate, avg_attempts_recovered,
initial_pass_rate_recovered, initial_pass_rate_failed

NOTE:
- Your rows are "experimental conditions" (not per-task observations),
  so statistical tests are illustrative. The key outputs are effect sizes
  (mean/median deltas) and stratified summaries.

Usage:
python3 rq4_history_stats.py

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
PATH = "/Users/ritaholloway/Desktop/self_corrections_pandas.csv"
OUT_XLSX = "/Users/ritaholloway/Desktop/RQ4_history_results.xlsx"

# -------------------------
# Helpers
# -------------------------
def norm_diff(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"intro", "introductory", "beginner", "easy"}:
        return "intro"
    if s in {"interview", "medium"}:
        return "interview"
    if s in {"competition", "hard"}:
        return "competition"
    if s in {"all"}:
        return "all"
    return s

def summarize(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": len(g),
            "mean_initial_pass": g["initial_pass_rate"].mean(),
            "median_initial_pass": g["initial_pass_rate"].median(),
            "mean_recovery": g["recovery_rate"].mean(),
            "median_recovery": g["recovery_rate"].median(),
            "mean_attempts_recovered": g["avg_attempts_recovered"].mean(),
            "median_attempts_recovered": g["avg_attempts_recovered"].median(),
        }
    )

def history_effect_table(
    df_in: pd.DataFrame,
    group_cols: list[str] | None = None,
    metrics: tuple[str, ...] = ("recovery_rate", "avg_attempts_recovered", "initial_pass_rate"),
) -> pd.DataFrame:
    """
    Computes, for each group (or overall if group_cols=None):
      mean/median(metric | history=0)
      mean/median(metric | history=1)
      delta_mean = mean(1) - mean(0)
      delta_median = median(1) - median(0)
      n0, n1
    """
    group_cols = group_cols or []
    rows = []

    grouped = [("", df_in)] if not group_cols else list(df_in.groupby(group_cols, dropna=False))

    for key, g in grouped:
        g0 = g[g["history"] == 0]
        g1 = g[g["history"] == 1]

        # Skip groups missing a condition
        if len(g0) == 0 or len(g1) == 0:
            continue

        for m in metrics:
            v0 = pd.to_numeric(g0[m], errors="coerce").dropna()
            v1 = pd.to_numeric(g1[m], errors="coerce").dropna()

            # For attempts (recovered), NA can appear; handle gracefully
            if len(v0) == 0 or len(v1) == 0:
                continue

            row = {
                "metric": m,
                "n(history=0)": len(v0),
                "n(history=1)": len(v1),
                "mean(history=0)": float(v0.mean()),
                "mean(history=1)": float(v1.mean()),
                "delta_mean(1-0)": float(v1.mean() - v0.mean()),
                "median(history=0)": float(v0.median()),
                "median(history=1)": float(v1.median()),
                "delta_median(1-0)": float(v1.median() - v0.median()),
            }

            # Add group identifiers
            if group_cols:
                if isinstance(key, tuple):
                    for c, kv in zip(group_cols, key):
                        row[c] = kv
                else:
                    row[group_cols[0]] = key

            rows.append(row)

    out = pd.DataFrame(rows)
    # Put group cols first if present
    if group_cols and not out.empty:
        out = out[[*group_cols, "metric", *[c for c in out.columns if c not in (*group_cols, "metric")]]]
    return out

def permutation_test_diff_means(x0: np.ndarray, x1: np.ndarray, n_perm: int = 20000, seed: int = 0) -> dict:
    """
    Two-sided permutation test for difference in means: mean(x1) - mean(x0)
    Returns observed diff and permutation p-value.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]
    if len(x0) == 0 or len(x1) == 0:
        return {"diff": np.nan, "p_perm": np.nan, "n0": len(x0), "n1": len(x1)}

    obs = float(x1.mean() - x0.mean())
    pooled = np.concatenate([x0, x1])
    n0 = len(x0)
    n = len(pooled)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        a = pooled[perm[:n0]]
        b = pooled[perm[n0:]]
        diff = b.mean() - a.mean()
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"diff": obs, "p_perm": p, "n0": len(x0), "n1": len(x1)}

# -------------------------
# 1) Load + clean
# -------------------------
df = pd.read_csv(PATH)
df = df.replace(["NA", "NaN", ""], np.nan)

NUM_COLS = [
    "initial_pass_rate",
    "recovery_rate",
    "avg_attempts_recovered",
    "initial_pass_rate_recovered",
    "initial_pass_rate_failed",
    "history",
]
for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["difficulty"] = df.get("difficulty", np.nan).apply(norm_diff)
df.loc[df["difficulty"].isna(), "difficulty"] = "all"

# Sanity prints
print("\n=== Row counts by history ===")
print(df["history"].value_counts(dropna=False))

print("\n=== Row counts by dataset x history ===")
print(df.groupby(["dataset", "history"]).size())

# -------------------------
# 2) Core summaries
# -------------------------
print("\n=== Overall summary by history ===")
overall_by_history = df.groupby("history", dropna=False).apply(summarize)
print(overall_by_history)

print("\n=== Summary by dataset x history ===")
by_dataset_history = df.groupby(["dataset", "history"], dropna=False).apply(summarize)
print(by_dataset_history)

print("\n=== Summary by difficulty x history ===")
by_diff_history = df.groupby(["difficulty", "history"], dropna=False).apply(summarize)
print(by_diff_history)

print("\n=== Summary by model x history ===")
by_model_history = df.groupby(["model", "history"], dropna=False).apply(summarize)
print(by_model_history)

# -------------------------
# 3) Effect-size tables (history=1 minus history=0)
# -------------------------
print("\n=== Effect sizes (overall): history vs no-history ===")
eff_overall = history_effect_table(df, group_cols=None)
print(eff_overall)

print("\n=== Effect sizes by dataset ===")
eff_by_dataset = history_effect_table(df, group_cols=["dataset"])
print(eff_by_dataset)

print("\n=== Effect sizes by difficulty ===")
eff_by_difficulty = history_effect_table(df, group_cols=["difficulty"])
print(eff_by_difficulty)

print("\n=== Effect sizes by model ===")
eff_by_model = history_effect_table(df, group_cols=["model"])
print(eff_by_model)

print("\n=== Effect sizes by dataset x difficulty ===")
eff_by_dataset_diff = history_effect_table(df, group_cols=["dataset", "difficulty"])
print(eff_by_dataset_diff)

# -------------------------
# 4) Simple permutation tests (illustrative)
#    (a) overall recovery_rate
#    (b) overall avg_attempts_recovered (drops NA)
# -------------------------
x0 = df.loc[df["history"] == 0, "recovery_rate"].dropna().to_numpy()
x1 = df.loc[df["history"] == 1, "recovery_rate"].dropna().to_numpy()
perm_recovery = permutation_test_diff_means(x0, x1)

a0 = df.loc[df["history"] == 0, "avg_attempts_recovered"].dropna().to_numpy()
a1 = df.loc[df["history"] == 1, "avg_attempts_recovered"].dropna().to_numpy()
perm_attempts = permutation_test_diff_means(a0, a1)

perm_tests = pd.DataFrame(
    [
        {"metric": "recovery_rate", **perm_recovery},
        {"metric": "avg_attempts_recovered", **perm_attempts},
    ]
)

print("\n=== Permutation tests (overall; diff = mean(history=1) - mean(history=0)) ===")
print(perm_tests)

# -------------------------
# 5) Plots (matplotlib only)
# -------------------------
# Recovery rate by history
plt.figure()
rec0 = df.loc[df["history"] == 0, "recovery_rate"].dropna()
rec1 = df.loc[df["history"] == 1, "recovery_rate"].dropna()
plt.boxplot([rec0, rec1], tick_labels=["no-history", "history"])
plt.title("RQ4: Recovery rate by repair history")
plt.ylabel("Recovery rate")
plt.show()

# Attempts (recovered) by history
plt.figure()
att0 = df.loc[df["history"] == 0, "avg_attempts_recovered"].dropna()
att1 = df.loc[df["history"] == 1, "avg_attempts_recovered"].dropna()
plt.boxplot([att0, att1], tick_labels=["no-history", "history"])
plt.title("RQ4: Attempts among recovered trajectories by repair history")
plt.ylabel("Avg attempts (recovered)")
plt.show()

# Stratified: APPS only, by difficulty, history
apps = df[df["dataset"].astype(str).str.lower().eq("apps")].copy()
if len(apps) > 0:
    for diff in sorted(apps["difficulty"].unique()):
        sub = apps[apps["difficulty"] == diff]
        if sub["history"].nunique() < 2:
            continue
        plt.figure()
        r0 = sub.loc[sub["history"] == 0, "recovery_rate"].dropna()
        r1 = sub.loc[sub["history"] == 1, "recovery_rate"].dropna()
        plt.boxplot([r0, r1], tick_labels=["no-history", "history"])
        plt.title(f"APPS ({diff}): Recovery rate by repair history")
        plt.ylabel("Recovery rate")
        plt.show()

# -------------------------
# 6) Export everything to Excel (no xlsxwriter dependency)
#    Uses openpyxl if available (most common). If not, you can install it:
#    python3 -m pip install openpyxl
# -------------------------
with pd.ExcelWriter(OUT_XLSX) as writer:
    # Cleaned data
    df.to_excel(writer, sheet_name="cleaned_data", index=False)

    # Core summaries
    overall_by_history.to_excel(writer, sheet_name="summary_overall_history")
    by_dataset_history.to_excel(writer, sheet_name="summary_dataset_history")
    by_diff_history.to_excel(writer, sheet_name="summary_diff_history")
    by_model_history.to_excel(writer, sheet_name="summary_model_history")

    # Effect sizes
    eff_overall.to_excel(writer, sheet_name="effects_overall", index=False)
    eff_by_dataset.to_excel(writer, sheet_name="effects_by_dataset", index=False)
    eff_by_difficulty.to_excel(writer, sheet_name="effects_by_difficulty", index=False)
    eff_by_model.to_excel(writer, sheet_name="effects_by_model", index=False)
    eff_by_dataset_diff.to_excel(writer, sheet_name="effects_dataset_diff", index=False)

    # Permutation tests
    perm_tests.to_excel(writer, sheet_name="perm_tests_overall", index=False)

print(f"\n✅ Wrote RQ4 workbook to: {OUT_XLSX}")

print("\nDone. For RQ4 reporting, focus on:")
print("- summary_overall_history (mean/median recovery + attempts)")
print("- effects_overall (delta history - no-history)")
print("- effects_by_dataset / effects_by_model (heterogeneity)")
print("- perm_tests_overall (illustrative p-values)")
