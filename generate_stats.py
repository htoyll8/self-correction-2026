# 

"""
RQ3: How does task difficulty affect the depth and effectiveness of self-correction?

Assumes your CSV has (at least) these columns:
dataset, difficulty, history, model,
initial_pass_rate, recovery_rate, avg_attempts_recovered,
initial_pass_rate_recovered, initial_pass_rate_failed

Output:
- Cleaned dataframe (difficulty normalized + numeric conversions)
- Summary tables by difficulty (overall + by dataset/model/history)
- Effect sizes (pairwise differences) for key metrics
- Correlations between difficulty and metrics (where ordinal is defined)
- Regressions:
    (1) avg_attempts_recovered ~ difficulty + controls
    (2) recovery_rate ~ difficulty + controls
- Plots:
    boxplot attempts by difficulty
    boxplot recovery by difficulty
    scatter attempts vs recovery colored by difficulty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) Load + clean
# -------------------------
PATH = "/Users/ritaholloway/Desktop/self_corrections_pandas.csv"
df = pd.read_csv(PATH)

# Normalize NA tokens
df = df.replace(["NA", "NaN", ""], np.nan)

# Ensure expected numeric columns are numeric
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


# Normalize difficulty text
# (keeps unknown difficulties as-is)
def _norm_diff(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    # common aliases
    if s in {"intro", "introductory", "beginner", "easy"}:
        return "intro"
    if s in {"interview", "medium"}:
        return "interview"
    if s in {"competition", "hard"}:
        return "competition"
    if s in {"all"}:
        return "all"
    return s


df["difficulty"] = df.get("difficulty", np.nan).apply(_norm_diff)

# If your difficulty column is blank for some datasets (e.g., human_eval),
# you can optionally set it to "all" so it participates in difficulty-grouped summaries.
# Comment out if you prefer to drop blanks.
df.loc[df["difficulty"].isna(), "difficulty"] = "all"

# Define an ordinal scale for difficulty where meaningful
# A/B comparisons make sense.
# NOTE: "all" is not ordinal; we'll treat it as NaN for ordinal analyses.
DIFF_ORD_MAP = {
    "intro": 1,
    "interview": 2,
    "competition": 3,
}
df["difficulty_ord"] = df["difficulty"].map(DIFF_ORD_MAP)

# Quick sanity print
print("\n=== Rows per difficulty ===")
print(df["difficulty"].value_counts(dropna=False))

# -------------------------
# 2) RQ3 metrics definition
# -------------------------
# Depth of self-correction (proxy): avg_attempts_recovered
# Effectiveness (proxies): recovery_rate (primary), initial_pass_rate (context),
#                          optionally init pass rates within recovered/failed
DEPTH_METRIC = "avg_attempts_recovered"
EFFECT_METRIC = "recovery_rate"

# -------------------------
# 3) Summary tables by difficulty
# -------------------------
def summarize(group_df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": len(group_df),
            "mean_initial_pass": group_df["initial_pass_rate"].mean(),
            "mean_recovery": group_df["recovery_rate"].mean(),
            "median_recovery": group_df["recovery_rate"].median(),
            "mean_attempts_recovered": group_df["avg_attempts_recovered"].mean(),
            "median_attempts_recovered": group_df["avg_attempts_recovered"].median(),
        }
    )

print("\n=== Overall summary by difficulty ===")
overall_by_diff = df.groupby("difficulty", dropna=False).apply(summarize).sort_index()
print(overall_by_diff)

print("\n=== Summary by dataset x difficulty ===")
by_dataset_diff = (
    df.groupby(["dataset", "difficulty"], dropna=False)
    .apply(summarize)
    .sort_index()
)
print(by_dataset_diff)

print("\n=== Summary by model x difficulty ===")
by_model_diff = (
    df.groupby(["model", "difficulty"], dropna=False)
    .apply(summarize)
    .sort_index()
)
print(by_model_diff)

print("\n=== Summary by history x difficulty ===")
by_hist_diff = (
    df.groupby(["history", "difficulty"], dropna=False)
    .apply(summarize)
    .sort_index()
)
print(by_hist_diff)

# -------------------------
# 4) Pairwise effect sizes across difficulty levels
# -------------------------
def pairwise_diff(df_in: pd.DataFrame, metric: str, levels=("intro", "interview", "competition")):
    out = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = levels[i], levels[j]
            va = df_in.loc[df_in["difficulty"] == a, metric].dropna()
            vb = df_in.loc[df_in["difficulty"] == b, metric].dropna()
            if len(va) == 0 or len(vb) == 0:
                continue
            out.append(
                {
                    "metric": metric,
                    "A": a,
                    "B": b,
                    "mean(A)": va.mean(),
                    "mean(B)": vb.mean(),
                    "diff_mean(B-A)": vb.mean() - va.mean(),
                    "median(A)": va.median(),
                    "median(B)": vb.median(),
                    "diff_median(B-A)": vb.median() - va.median(),
                    "n(A)": len(va),
                    "n(B)": len(vb),
                }
            )
    return pd.DataFrame(out)

print("\n=== Pairwise differences: recovery_rate ===")
print(pairwise_diff(df, "recovery_rate"))

print("\n=== Pairwise differences: avg_attempts_recovered ===")
print(pairwise_diff(df, "avg_attempts_recovered"))

# -------------------------
# 5) Ordinal correlations: difficulty_ord vs metrics
#    (Only uses rows where difficulty is mapped: intro/interview/competition)
# -------------------------
df_ord = df.dropna(subset=["difficulty_ord"])

if len(df_ord) > 2:
    rho_recovery = df_ord["difficulty_ord"].corr(df_ord["recovery_rate"], method="spearman")
    rho_attempts = df_ord["difficulty_ord"].corr(df_ord["avg_attempts_recovered"], method="spearman")
    print("\n=== Spearman correlations with ordinal difficulty (intro<interview<competition) ===")
    print(f"rho(difficulty_ord, recovery_rate)         = {rho_recovery}")
    print(f"rho(difficulty_ord, avg_attempts_recovered)= {rho_attempts}")
else:
    print("\nNot enough ordinal difficulty rows to compute correlations.")

# Within-dataset ordinal correlations
print("\n=== Within-dataset Spearman correlations (ordinal difficulty) ===")
def within_corr(g: pd.DataFrame):
    g = g.dropna(subset=["difficulty_ord"])
    if len(g) < 3:
        return pd.Series({"rho_recovery": np.nan, "rho_attempts": np.nan, "n": len(g)})
    return pd.Series(
        {
            "rho_recovery": g["difficulty_ord"].corr(g["recovery_rate"], method="spearman"),
            "rho_attempts": g["difficulty_ord"].corr(g["avg_attempts_recovered"], method="spearman"),
            "n": len(g),
        }
    )

print(df.groupby("dataset").apply(within_corr))

# -------------------------
# 6) Controlled regressions (optional but strong for RQ3)
# -------------------------
try:
    import statsmodels.formula.api as smf

    # Use only ordinal difficulties for regressions, to interpret slope per difficulty step.
    reg_df = df_ord.copy()

    # Depth model: attempts for recovered trajectories
    # Note: avg_attempts_recovered is already conditioned on "recovered"; if NA indicates "no recovered",
    # regression will naturally drop those rows.
    depth_model = smf.ols(
        "avg_attempts_recovered ~ difficulty_ord + history + C(model) + C(dataset)",
        data=reg_df,
    ).fit()
    print("\n=== Regression (Depth): avg_attempts_recovered ~ difficulty_ord + controls ===")
    print(depth_model.summary())

    eff_model = smf.ols(
        "recovery_rate ~ difficulty_ord + history + C(model) + C(dataset)",
        data=reg_df,
    ).fit()
    print("\n=== Regression (Effectiveness): recovery_rate ~ difficulty_ord + controls ===")
    print(eff_model.summary())

except Exception as e:
    print("\n[Skipping regressions] statsmodels not available or regression failed:", e)

# -------------------------
# 7) Plots (matplotlib only)
# -------------------------
# Boxplot: attempts by difficulty
plot_df = df.copy()
plot_df = plot_df[plot_df["difficulty"].isin(["intro", "interview", "competition", "all"])]

# Attempts boxplot (focus on ordinal diffs)
ord_levels = ["intro", "interview", "competition"]
attempts_data = [df.loc[df["difficulty"] == d, "avg_attempts_recovered"].dropna() for d in ord_levels]
if any(len(x) > 0 for x in attempts_data):
    plt.figure()
    plt.boxplot(attempts_data, labels=ord_levels)
    plt.title("Depth of self-correction: Attempts (Recovered) by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Avg Attempts (Recovered)")
    plt.show()

# Recovery boxplot
recovery_data = [df.loc[df["difficulty"] == d, "recovery_rate"].dropna() for d in ord_levels]
if any(len(x) > 0 for x in recovery_data):
    plt.figure()
    plt.boxplot(recovery_data, labels=ord_levels)
    plt.title("Effectiveness of self-correction: Recovery Rate by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Recovery Rate")
    plt.show()

# Scatter: attempts vs recovery colored by difficulty (ordinal only)
scatter_df = df[df["difficulty"].isin(ord_levels)].copy()
scatter_df = scatter_df.dropna(subset=["avg_attempts_recovered", "recovery_rate"])
if len(scatter_df) > 0:
    plt.figure()
    for d in ord_levels:
        sub = scatter_df[scatter_df["difficulty"] == d]
        if len(sub) == 0:
            continue
        plt.scatter(sub["avg_attempts_recovered"], sub["recovery_rate"], label=d)
    plt.title("Depth vs Effectiveness by Difficulty")
    plt.xlabel("Avg Attempts (Recovered)")
    plt.ylabel("Recovery Rate")
    plt.legend(title="Difficulty")
    plt.show()

print("\nDone. For RQ3 reporting, focus on:")
print("- mean/median recovery_rate by difficulty")
print("- mean/median avg_attempts_recovered by difficulty")
print("- rho(difficulty_ord, recovery_rate) and rho(difficulty_ord, attempts)")
print("- regression coefficient on difficulty_ord in both models (with p-values)")


# -------------------------
# 8) Export everything to Excel
# -------------------------

OUT_PATH = "/Users/ritaholloway/Desktop/RQ3_difficulty_results.xlsx"

# with pd.ExcelWriter(OUT_PATH, engine="xlsxwriter") as writer:
with pd.ExcelWriter(OUT_PATH) as writer:
    # 1) Cleaned data
    df.to_excel(writer, sheet_name="cleaned_data", index=False)

    # 2) Row counts
    df["difficulty"].value_counts(dropna=False).to_frame(
        name="count"
    ).to_excel(writer, sheet_name="rows_per_difficulty")

    # 3) Summaries
    overall_by_diff.to_excel(writer, sheet_name="overall_by_difficulty")
    by_dataset_diff.to_excel(writer, sheet_name="by_dataset_difficulty")
    by_model_diff.to_excel(writer, sheet_name="by_model_difficulty")
    by_hist_diff.to_excel(writer, sheet_name="by_history_difficulty")

    # 4) Pairwise effects
    pairwise_diff(df, "recovery_rate").to_excel(
        writer, sheet_name="pairwise_recovery", index=False
    )
    pairwise_diff(df, "avg_attempts_recovered").to_excel(
        writer, sheet_name="pairwise_attempts", index=False
    )

    # 5) Ordinal correlations
    ordinal_corr_df = pd.DataFrame(
        {
            "metric": ["recovery_rate", "avg_attempts_recovered"],
            "spearman_rho": [rho_recovery, rho_attempts],
        }
    )
    ordinal_corr_df.to_excel(
        writer, sheet_name="ordinal_correlations", index=False
    )

    # 6) Within-dataset correlations
    within_dataset_corr_df = df.groupby("dataset").apply(within_corr)
    within_dataset_corr_df.to_excel(
        writer, sheet_name="within_dataset_correlations"
    )

    # 7) Regressions (as tables)
    try:
        depth_reg_df = pd.DataFrame(
            {
                "coef": depth_model.params,
                "std_err": depth_model.bse,
                "t": depth_model.tvalues,
                "p_value": depth_model.pvalues,
            }
        )
        depth_reg_df.to_excel(
            writer, sheet_name="regression_depth"
        )

        eff_reg_df = pd.DataFrame(
            {
                "coef": eff_model.params,
                "std_err": eff_model.bse,
                "t": eff_model.tvalues,
                "p_value": eff_model.pvalues,
            }
        )
        eff_reg_df.to_excel(
            writer, sheet_name="regression_effectiveness"
        )
    except Exception:
        pass

print(f"\n✅ RQ3 results written to:\n{OUT_PATH}")
