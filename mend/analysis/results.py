"""Canonical results write layer.

`to_row` turns one scored Attempt into a canonical row; `finalize` turns a run's JSONL log
into the canonical parquet/CSV plus its derived metrics, figure, and table. This is the
write-side counterpart to mend.analysis.metrics; the row schema here and the definitions
there are kept in sync via metrics_definitions.md.
"""
import json
import os

import pandas as pd

from mend.analysis import metrics
from mend.strategies.base import Attempt
from mend.utils import DATA


def to_row(base: dict, a: Attempt) -> dict:
    """Expand one Attempt into a canonical row by merging in the per-task `base` metadata."""
    return {
        **base,
        "seed_idx": a.seed_idx,
        "attempt": a.attempt,
        "pass_fraction": float(a.pass_fraction),
        "passed": bool(a.passed),
        "program": a.program,
        "feedback": a.feedback,
    }


def finalize(jsonl: str, run_id: str) -> tuple[dict, list[str]]:
    """Build the canonical table and its derived metrics/figure/table from the JSONL log.

    Returns (metrics, artifact_paths); artifact_paths[0] is the parquet source of truth.
    """
    df = pd.read_json(jsonl, lines=True)
    df["task_id"] = df["task_id"].astype(str)
    pq = os.path.join(DATA, f"results_{run_id}.parquet")
    df.to_parquet(pq)
    df.to_csv(pq.replace(".parquet", ".csv"), index=False)

    m = metrics.compute(df)
    mjson = os.path.join(DATA, f"metrics_{run_id}.json")
    with open(mjson, "w") as f:
        json.dump(m, f, indent=2)
    fig = os.path.join(DATA, f"curve_{run_id}.png")
    metrics.figure(df, fig)
    tex = os.path.join(DATA, f"summary_{run_id}.tex")
    metrics.latex_table(m, tex)
    return m, [pq, mjson, fig, tex]
