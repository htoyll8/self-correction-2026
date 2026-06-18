"""Self-correction experiment orchestrator (thin entrypoint).

Wires a dataset x model x strategy x evaluator into the canonical results table
(data/results_<run_id>.parquet, the single source of truth for mend.analysis.metrics) and
logs to a local MLflow store. The science lives in datasets/strategies/evaluators, results
I/O in mend.analysis, and generic plumbing in mend.utils — this file only wires them.

Pilot:  python -m mend.run_experiment --model gpt-4o-mini --dataset humaneval \
          --n_tasks 6 --np 3 --max_attempts 5
"""
import argparse
import json
import os
import time

import mlflow

from mend import datasets, strategies
from mend.analysis import results
from mend.evaluators import make_scorer
from mend.models.llm import Model
from mend.utils import DATA, git_sha, make_run_id, write_rows
from mend.utils.tracking import log_condition_metrics, setup_mlflow


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Self-correction experiment (canonical logging).")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dataset", default="mbppplus", choices=sorted(datasets.LOADERS))
    ap.add_argument("--n_tasks", type=int, default=8)
    ap.add_argument("--np", type=int, default=3)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--refine_mode", default="critique+refine", choices=sorted(strategies.STRATEGIES))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sha = git_sha()
    run_id = make_run_id(sha)
    os.makedirs(DATA, exist_ok=True)

    model = Model(model_name=args.model)
    tasks = datasets.load_tasks(args.dataset, args.n_tasks)
    strategy = strategies.get_strategy(args.refine_mode)
    print(f"[INFO] run_id={run_id} model={args.model} dataset={args.dataset} "
          f"tasks={len(tasks)} np={args.np} max_attempts={args.max_attempts} mode={args.refine_mode}")

    setup_mlflow()
    jsonl = os.path.join(DATA, f"results_{run_id}.jsonl")

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "run_id": run_id, "git_sha": sha, "model": args.model, "dataset": args.dataset,
            "np": args.np, "max_attempts": args.max_attempts, "refine_mode": args.refine_mode,
            "n_tasks": len(tasks),
        })
        with open(jsonl, "w", encoding="utf-8") as jf:
            for i, task in enumerate(tasks, 1):
                scorer = make_scorer(task.setup, task.tests, prelude=task.prelude)
                base = {
                    "run_id": run_id, "git_sha": sha, "dataset": args.dataset, "model": args.model,
                    "refine_mode": args.refine_mode, "difficulty": "na", "task_id": task.task_id,
                    "n_tests": task.n_tests, "np": args.np, "max_attempts": args.max_attempts,
                }
                t0 = time.time()
                attempts = strategy(model, task.description, scorer, args.np, args.max_attempts)
                rows = [results.to_row(base, a) for a in attempts]
                write_rows(jf, rows)
                init_pass = sum(a.passed for a in attempts if a.attempt == 0)
                print(f"[{i}/{len(tasks)}] task {task.task_id}: {len(rows)} programs, "
                      f"{init_pass}/{args.np} seeds passed initially ({time.time()-t0:.0f}s)")

        m, artifacts = results.finalize(jsonl, run_id)
        log_condition_metrics(m)
        for p in artifacts:
            mlflow.log_artifact(p)

    print(f"\n[DONE] wrote {artifacts[0]}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
