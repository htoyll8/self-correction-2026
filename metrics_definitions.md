# Metric definitions (single source of truth)

Every reported number is computed by `mend/analysis/metrics.py` from the canonical results table
(`data/results_<run_id>.parquet`). Do not copy numbers from logs or recompute them by
hand. To change a definition, edit it here and in `metrics.py`, then rerun
`python -m mend.analysis.metrics <parquet>` to regenerate `metrics.json`, the tables, and the figures.

## Canonical table: one row per generated program

| column | meaning |
|---|---|
| `run_id` | unique run id (`<timestamp>_<git_sha>`) |
| `git_sha` | code version that produced the row |
| `dataset`, `model`, `refine_mode`, `difficulty` | experimental condition |
| `task_id` | benchmark task |
| `seed_idx` | which of the `np` independent seeds |
| `attempt` | 0 is the initial seed; 1..K are refinement rounds |
| `pass_fraction` | fraction of the task's tests this program passes, in [0,1] |
| `passed` | true when `pass_fraction == 1.0` |
| `n_tests` | number of tests for the task |
| `program` | full source of the program scored at this attempt |
| `feedback` | critique that produced this program (null at attempt 0, the seed) |

A seed trajectory is all rows sharing
`(run_id, dataset, model, refine_mode, task_id, seed_idx)`, ordered by `attempt`.
Refinement rounds (`attempt >= 1`) exist only for seeds that failed at attempt 0, and
stop once a seed passes.

## Definitions (locked)

- Initial pass rate: seeds passing at attempt 0, divided by all seeds.
- Failed seeds: seeds not passing at attempt 0.
- Recovered (of failed): failed seeds that later pass, divided by failed seeds.
- Recovered (of all): recovered failed seeds, divided by all seeds. Always state the
  denominator; never mix the two in one column.
- Task solve rate (any of `np`): tasks where at least one seed passes, divided by all tasks.
- Mean pass_fraction by round (the front-loaded gains curve):
  - `round0_all`: mean `pass_fraction` over all attempt-0 rows.
  - `round0_failed`: mean `pass_fraction` over attempt-0 rows that are failed seeds.
  - `round k>=1`: mean `pass_fraction` over all attempt-`k` rows (the still-unsolved seeds).
  - Report the count behind each round. Read the headline claim (gains front-load, then
    plateau) off the failed-only series: `round0_failed`, then round 1, round 2, and so on.
    One denominator throughout.
- pass@k (if reported): the Chen et al. (2021) estimator over a task's programs. A
  separate quantity from the trajectory metrics, labeled as such. Not computed in the pilot.

## Why this stops the prior inconsistencies

- One table, one `groupby`. Figures cannot disagree.
- Round 0 has two named fields (`round0_all`, `round0_failed`), so "percent passed
  initially" can never be plotted against a failed-only axis. In the old runs, initial
  was 76.8 percent over all seeds but 14.7 percent over failed seeds; mixing them turned
  a round-1 gain (about +4 points) into an apparent 57-point crash.
- "Recovered" carries its denominator in its name.
- Every row is tagged with `run_id` and `git_sha`, so numbers from different runs cannot
  land in one table.
