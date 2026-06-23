"""Whack-a-mole / regression analysis.

The canonical table stores only aggregate pass_fraction, not which cases passed. Whack-a-mole
is a per-case event ("case 37 passed at attempt 1, failed at attempt 2"), so this re-scores
each program PER CASE (free: the program text is stored), caches the per-case vectors to
data/percase_<key>.parquet, then walks each seed's refinement trajectory counting, per
consecutive (attempt k-1 -> k) step:
  fix      = fail -> pass
  regress  = pass -> fail   <- the "gradual failure" mechanism

Re-scoring runs each program in the isolated subprocess worker (mend.evaluators.make_cases),
so a crashing/looping candidate dies alone instead of poisoning the run; the spawns are
parallelized with threads. Results cache to parquet; a second run reads the cache instantly.

Usage: python -m mend.analysis.regression_audit data/results_<run>.parquet
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from mend import datasets
from mend.datasets.base import Task
from mend.evaluators.scorer import make_cases

SCORER_VERSION = 2   # bump when per-case scoring changes; invalidates stale caches


@dataclass
class Tally:
    """Running counts over refinement steps, for one round or the whole run."""
    fixes: int = 0
    regressions: int = 0
    steps: int = 0
    steps_with_regress: int = 0   # steps that broke at least one passing case
    net_negative: int = 0         # steps where regressions outnumbered fixes

    def update(self, fixes: int, regressions: int) -> None:
        self.fixes += fixes
        self.regressions += regressions
        self.steps += 1
        if regressions > 0:
            self.steps_with_regress += 1
        if regressions > fixes:
            self.net_negative += 1


def _load_task_map(dataset: str, difficulty: str) -> dict[str, Task]:
    difficulties = (difficulty,) if dataset == "apps" and difficulty != "na" else None
    n_tasks = 5000 if dataset == "apps" else 1000
    return {t.task_id: t for t in datasets.load_tasks(dataset, n_tasks, difficulties=difficulties)}


def _percase_table(df: pd.DataFrame, tasks: dict[str, Task], cache: str) -> pd.DataFrame:
    """Per-case pass vectors for every program in a refining trajectory, cached to parquet."""
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        if not cached.empty and int(cached["scorer_version"].iloc[0]) == SCORER_VERSION:
            print(f"[cache] reading per-case vectors from {os.path.basename(cache)}")
            return cached
        print(f"[cache] {os.path.basename(cache)} is from an older scorer; recomputing")

    refined = df.groupby(["task_id", "seed_idx"]).filter(lambda g: (g.attempt >= 1).any())
    jobs = [(r.task_id, int(r.seed_idx), int(r.attempt), r.program)
            for r in refined.itertuples() if r.task_id in tasks]
    # one cases-scorer per task (tests are fixed); reused across that task's programs
    scorers = {tid: make_cases(t.setup, t.tests, prelude=t.prelude,
                               per_timeout=t.per_timeout, io_mode=t.io_mode)
               for tid, t in tasks.items()}

    def work(job: tuple) -> dict:
        task_id, seed_idx, attempt, program = job
        return {"task_id": task_id, "seed_idx": seed_idx, "attempt": attempt,
                "case_pass": scorers[task_id](program)}

    rows = []
    workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:  # each thread waits on a subprocess
        futures = [ex.submit(work, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="re-scoring", unit="prog"):
            rows.append(fut.result())

    out = pd.DataFrame(rows)
    out["scorer_version"] = SCORER_VERSION
    out.to_parquet(cache)
    print(f"[cache] wrote {len(out)} per-case vectors to {os.path.basename(cache)}")
    return out


def _tally(pc: pd.DataFrame) -> tuple[Tally, dict[int, Tally]]:
    overall = Tally()
    by_round: dict[int, Tally] = defaultdict(Tally)
    for _, g in pc.groupby(["task_id", "seed_idx"]):
        g = g.sort_values("attempt")
        vecs = {int(r.attempt): list(r.case_pass) for r in g.itertuples()}
        attempts = sorted(vecs)
        for a, b in zip(attempts, attempts[1:]):
            va, vb = vecs[a], vecs[b]
            if len(va) != len(vb):
                continue
            fixes = sum(1 for x, y in zip(va, vb) if not x and y)
            regressions = sum(1 for x, y in zip(va, vb) if x and not y)
            overall.update(fixes, regressions)
            by_round[b].update(fixes, regressions)
    return overall, by_round


def _report(label: str, overall: Tally, by_round: dict[int, Tally], out_json: str) -> None:
    payload = {"label": label, "overall": overall.__dict__,
               "by_round": {k: by_round[k].__dict__ for k in sorted(by_round)}}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    s = overall
    print(f"\n=== whack-a-mole: {label} ===")
    print(f"refinement steps examined: {s.steps}")
    if not s.steps:
        return
    print(f"steps that introduced >=1 regression: {s.steps_with_regress} ({s.steps_with_regress/s.steps*100:.1f}%)")
    print(f"steps net-negative (regress > fix):    {s.net_negative} ({s.net_negative/s.steps*100:.1f}%)")
    print(f"total cases fixed:     {s.fixes}")
    print(f"total cases regressed: {s.regressions}")
    print(f"regress/fix ratio:     {s.regressions/s.fixes:.2f}" if s.fixes else "regress/fix ratio: n/a")
    print("\nby round transition (-> round k):  fixes / regress / steps  (avg regress/step)")
    for k in sorted(by_round):
        t = by_round[k]
        print(f"  -> round {k}: {t.fixes:5d} / {t.regressions:5d} / {t.steps:4d}   ({t.regressions/t.steps:.2f})")
    print(f"\nwrote {os.path.basename(out_json)}")


def main(path: str) -> None:
    df = pd.read_parquet(path)
    df["task_id"] = df["task_id"].astype(str)
    dataset, difficulty, model = df.dataset.iloc[0], df.difficulty.iloc[0], df.model.iloc[0]
    key = os.path.basename(path).replace("results_", "").replace(".parquet", "")
    data_dir = os.path.dirname(path)

    tasks = _load_task_map(dataset, difficulty)
    pc = _percase_table(df, tasks, os.path.join(data_dir, f"percase_{key}.parquet"))
    overall, by_round = _tally(pc)
    label = f"{dataset} / {model}" + (f" / {difficulty}" if difficulty != "na" else "")
    _report(label, overall, by_round, os.path.join(data_dir, f"regression_{key}.json"))


if __name__ == "__main__":
    main(sys.argv[1])
