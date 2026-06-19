"""Self-correction experiment orchestrator (thin entrypoint).

Wires a dataset x model x strategy x evaluator into the canonical results table
(data/results_<run_id>.parquet, the single source of truth for mend.analysis.metrics) and
logs to a local MLflow store. The science lives in datasets/strategies/evaluators, results
I/O in mend.analysis, and generic plumbing in mend.utils — this file only wires them.

Tasks run concurrently (--workers): the bottleneck is API latency + subprocess scoring, so
threads overlap that I/O. The JSONL writer stays single-threaded (writes happen as futures
complete) so logging is durable and lock-free.

Pilot:  python -m mend.run_experiment --model gpt-4o-mini --dataset humaneval \
          --n_tasks 6 --np 3 --max_attempts 5 --workers 8
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
from tqdm import tqdm

from mend import datasets, strategies
from mend.analysis import results
from mend.datasets.base import Task
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
    ap.add_argument("--workers", type=int, default=8, help="tasks run concurrently (API I/O-bound)")
    return ap.parse_args()


def run_one_task(model: Model, strategy, task: Task, base_template: dict,
                 np_: int, max_attempts: int) -> list[dict]:
    """Run one task end-to-end (seeds + refinement) and return its canonical rows."""
    scorer = make_scorer(task.setup, task.tests, prelude=task.prelude)
    base = {**base_template, "task_id": task.task_id, "n_tests": task.n_tests}
    attempts = strategy(model, task.description, scorer, np_, max_attempts)
    return [results.to_row(base, a) for a in attempts]


def main() -> None:
    args = parse_args()
    sha = git_sha()
    run_id = make_run_id(sha)
    os.makedirs(DATA, exist_ok=True)

    model = Model(model_name=args.model)
    tasks = datasets.load_tasks(args.dataset, args.n_tasks)
    strategy = strategies.get_strategy(args.refine_mode)
    print(f"[INFO] run_id={run_id} model={args.model} dataset={args.dataset} tasks={len(tasks)} "
          f"np={args.np} max_attempts={args.max_attempts} mode={args.refine_mode} workers={args.workers}")

    setup_mlflow()
    jsonl = os.path.join(DATA, f"results_{run_id}.jsonl")
    base_template = {
        "run_id": run_id, "git_sha": sha, "dataset": args.dataset, "model": args.model,
        "refine_mode": args.refine_mode, "difficulty": "na",
        "np": args.np, "max_attempts": args.max_attempts,
    }

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "run_id": run_id, "git_sha": sha, "model": args.model, "dataset": args.dataset,
            "np": args.np, "max_attempts": args.max_attempts, "refine_mode": args.refine_mode,
            "n_tasks": len(tasks), "workers": args.workers,
        })
        t0 = time.time()
        failed: list[str] = []
        seeds_ok = 0
        with open(jsonl, "w", encoding="utf-8") as jf:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {
                    ex.submit(run_one_task, model, strategy, task, base_template, args.np, args.max_attempts): task
                    for task in tasks
                }
                bar = tqdm(as_completed(futures), total=len(tasks), unit="task",
                           desc=f"{args.model} {args.dataset}", disable=None)
                for fut in bar:
                    task = futures[fut]
                    try:
                        rows = fut.result()
                    except Exception as e:  # don't let one task sink a multi-hour run
                        failed.append(task.task_id)
                        bar.write(f"task {task.task_id}: FAILED ({type(e).__name__}: {e})")
                        continue
                    write_rows(jf, rows)  # single-threaded writer: durable, lock-free
                    seeds_ok += sum(1 for r in rows if r["attempt"] == 0 and r["passed"])
                    bar.set_postfix(seeds_ok=seeds_ok, failed=len(failed))
        print(f"[INFO] all tasks done in {time.time()-t0:.0f}s")
        mlflow.log_metric("tasks_failed", len(failed))
        if failed:
            print(f"[WARN] {len(failed)} task(s) failed and are NOT in the table: {', '.join(failed)}")

        m, artifacts = results.finalize(jsonl, run_id)
        log_condition_metrics(m)
        for p in artifacts:
            mlflow.log_artifact(p)

    print(f"\n[DONE] wrote {artifacts[0]}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
