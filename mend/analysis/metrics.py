"""Single source of truth for every reported number.

Reads one or more canonical per-run tables (data/results_<run_id>.parquet),
concatenates them, and computes all metrics per condition (dataset, model,
refine_mode). Definitions are locked in metrics_definitions.md — change them here
and there only. Run:  python -m mend.analysis.metrics data/results_*.parquet
"""
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITION_COLS = ["dataset", "model", "refine_mode"]


SEED_COLS = ["run_id", "task_id", "seed_idx"]


def load(paths: str | list[str]) -> pd.DataFrame:
    """Load one or more canonical result tables and assert no duplicate program rows."""
    if isinstance(paths, str):
        paths = [paths]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    dups = int(df.duplicated(SEED_COLS + ["attempt"]).sum())
    assert dups == 0, f"{dups} duplicate program rows on (run_id,task_id,seed_idx,attempt); data is corrupted"
    return df


def compute_condition(g: pd.DataFrame) -> dict:
    """Compute all locked metrics for a single condition group."""
    init = g[g.attempt == 0]
    failed = init[~init.passed]
    n_seeds, n_failed = len(init), len(failed)
    recovered = g[(g.attempt >= 1) & (g.passed)][SEED_COLS].drop_duplicates()
    n_rec = len(recovered)
    n_tasks = g["task_id"].nunique()
    solved_tasks = g[g.passed]["task_id"].nunique()
    rounds = {
        "round0_all": float(init.pass_fraction.mean()) if n_seeds else None,
        "round0_failed": float(failed.pass_fraction.mean()) if n_failed else None,
    }
    for k, sub in g[g.attempt >= 1].groupby("attempt"):   # single scan over refinement rows
        rounds[f"round{int(k)}"] = {
            "mean_pass_fraction": float(sub.pass_fraction.mean()),
            "n": int(len(sub)),
        }
    return {
        "n_seeds": int(n_seeds),
        "n_failed_seeds": int(n_failed),
        "n_tasks": int(n_tasks),
        "initial_pass_rate": float(init.passed.mean()) if n_seeds else None,
        "recovered_of_failed": float(n_rec / n_failed) if n_failed else None,
        "recovered_of_all": float(n_rec / n_seeds) if n_seeds else None,
        "task_solve_rate": float(solved_tasks / n_tasks) if n_tasks else None,
        "rounds": rounds,
    }


def compute(df: pd.DataFrame) -> dict:
    """Return {condition_string: metrics} grouped by (dataset, model, refine_mode)."""
    out = {}
    for keys, g in df.groupby(CONDITION_COLS):
        cond = " | ".join(map(str, keys if isinstance(keys, tuple) else (keys,)))
        out[cond] = compute_condition(g)
    return out


def figure(df: pd.DataFrame, path: str) -> None:
    """The headline curve: failed-only mean pass fraction vs refinement round."""
    plt.figure(figsize=(6.2, 4))
    for keys, g in df.groupby(CONDITION_COLS):
        init = g[g.attempt == 0]
        failed = init[~init.passed]
        maxk = int(g.attempt.max())
        xs, ys = [0], [failed.pass_fraction.mean() if len(failed) else float("nan")]
        for k in range(1, maxk + 1):
            sub = g[g.attempt == k]
            xs.append(k)
            ys.append(sub.pass_fraction.mean() if len(sub) else float("nan"))
        label = " | ".join(map(str, keys if isinstance(keys, tuple) else (keys,)))
        plt.plot(xs, [y * 100 for y in ys], marker="o", label=label)
    plt.xlabel("Refinement round  (0 = initial, failed seeds only)")
    plt.ylabel("Mean % of tests passed")
    plt.title("Self-correction: gains front-load, then plateau")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def latex_table(metrics: dict, path: str) -> None:
    """Write a small LaTeX summary table (one row per condition)."""
    def pct(x):
        return f"{x*100:.1f}\\%" if x is not None else "--"
    lines = [
        r"\begin{tabular}{lrrrr}", r"\hline",
        r"Condition & Init.\ pass & Recov.\ (of fail) & Task solve & \# failed seeds \\",
        r"\hline",
    ]
    for cond, m in metrics.items():
        lines.append(
            f"{cond} & {pct(m['initial_pass_rate'])} & {pct(m['recovered_of_failed'])} "
            f"& {pct(m['task_solve_rate'])} & {m['n_failed_seeds']} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    df = load(sys.argv[1:])
    print(json.dumps(compute(df), indent=2))
