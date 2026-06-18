"""Self-correction rerun: iterative repair with one tidy row per generated program.

Uses the legacy prompts/model interface (mend.models.llm.Model) so the rerun is
methodologically identical to the original study. Writes the canonical table
data/results_<run_id>.parquet (single source of truth for mend/metrics.py) and logs
params/metrics/artifacts to a local MLflow store.

Pilot:  python -m mend.run_experiment --model gpt-4o-mini --dataset mbppplus \
          --n_tasks 8 --np 3 --max_attempts 5
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from datasets import load_dataset

from mend import metrics
from mend.models.llm import Model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
MLFLOW_DB = "sqlite:///" + os.path.join(REPO, "mlflow.db")
MLFLOW_ARTIFACTS = "file:" + os.path.join(REPO, "mlartifacts")
MLFLOW_EXPERIMENT = "self-correction-rerun"
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbpp_worker.py")

WRAPPER_FUNCS = {"set", "sorted", "list", "tuple", "dict", "len", "max", "min", "sum",
                 "all", "any", "abs", "round", "int", "float", "str", "bool", "isclose"}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "nogit"


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:\w+)?\n?(.*?)```", text or "", flags=re.DOTALL)
    return (m[0] if m else (text or "")).strip()


# --- legacy MBPP function-name/arity hint (ported from legacy/main.py) ---
def _split_top_level_args(s: str) -> list[str]:
    args, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def extract_function_signature(assert_lines: list[str]) -> tuple[str | None, int | None]:
    for line in assert_lines:
        lhs = line.split("==")[0]
        i, func_start = 0, None
        while i < len(lhs):
            ch = lhs[i]
            if ch.isalpha() or ch == "_":
                if func_start is None:
                    func_start = i
            else:
                if func_start is not None:
                    name = lhs[func_start:i]
                    if i < len(lhs) and lhs[i] == "(":
                        paren, j = 1, i + 1
                        while j < len(lhs) and paren > 0:
                            if lhs[j] == "(":
                                paren += 1
                            elif lhs[j] == ")":
                                paren -= 1
                            j += 1
                        if paren == 0 and name not in WRAPPER_FUNCS:
                            return name, len(_split_top_level_args(lhs[i + 1:j - 1]))
                    func_start = None
            i += 1
    return None, None


def make_scorer(setup, tests: list[str], per_timeout: int = 5) -> Callable[[str], float]:
    """Score a program by running its asserts in an isolated, killable subprocess
    (mend/mbpp_worker.py). Returns the fraction of tests passed."""
    setup_code = "\n".join(setup) if isinstance(setup, list) else (setup or "")
    tests = list(tests)
    if not tests:
        return lambda code: 0.0
    base = {"setup": setup_code, "tests": tests, "per_timeout": per_timeout}
    overall = per_timeout * len(tests) + 15

    def score(code: str) -> float:
        payload = json.dumps({**base, "program": code})
        proc = subprocess.Popen(
            [sys.executable, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, preexec_fn=os.setsid,
        )
        try:
            out, _ = proc.communicate(payload, timeout=overall)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the whole tree
            except Exception:
                proc.kill()
            return 0.0
        for line in reversed(out.splitlines()):
            if line.startswith("##LCT##"):
                try:
                    r = json.loads(line[len("##LCT##"):])
                    return r["passed"] / r["total"] if r["total"] else 0.0
                except Exception:
                    return 0.0
        return 0.0

    return score


def _row(base: dict, seed_idx: int, attempt: int, frac: float, passed: bool) -> dict:
    return {
        "run_id": base["run_id"], "git_sha": base["git_sha"], "dataset": base["dataset"],
        "model": base["model"], "refine_mode": base["refine_mode"], "difficulty": base["difficulty"],
        "task_id": base["task_id"], "seed_idx": seed_idx, "attempt": attempt,
        "pass_fraction": float(frac), "passed": bool(passed), "n_tests": base["n_tests"],
        "np": base["np"], "max_attempts": base["max_attempts"],
    }


def run_task(model: Model, desc: str, scorer: Callable[[str], float],
             np_: int, max_attempts: int, base: dict) -> list[dict]:
    rows = []
    seeds = [extract_code(s) for s in model.generate(task_description=desc, n=np_, temperature=0.7)]
    for si, seed in enumerate(seeds):
        frac = scorer(seed)
        passed = frac == 1.0
        rows.append(_row(base, si, 0, frac, passed))
        if passed:
            continue
        cur = seed
        for k in range(1, max_attempts + 1):
            feedback = model.generate_feedback(desc, cur, temperature=0)          # legacy critique
            cur = extract_code(model.refine(desc, cur, feedback, temperature=0))   # legacy refine
            frac = scorer(cur)
            passed = frac == 1.0
            rows.append(_row(base, si, k, frac, passed))
            if passed:
                break
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Self-correction rerun (canonical logging).")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dataset", default="mbppplus", choices=sorted(DATASET_SOURCES))
    ap.add_argument("--n_tasks", type=int, default=8)
    ap.add_argument("--np", type=int, default=3)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--refine_mode", default="critique+refine")
    return ap.parse_args()


DATASET_SOURCES = {"mbppplus": "evalplus/mbppplus"}


def load_tasks(dataset: str, n_tasks: int) -> list[dict]:
    """Load the first n_tasks of the benchmark as plain dicts.

    Only datasets in DATASET_SOURCES are wired; fail loudly rather than silently
    running a different dataset than the one labeling the rows.
    """
    if dataset not in DATASET_SOURCES:
        raise ValueError(f"dataset {dataset!r} not supported; choose from {sorted(DATASET_SOURCES)}")
    ds = load_dataset(DATASET_SOURCES[dataset])["test"]
    return list(ds)[:n_tasks]


def build_desc(ex: dict) -> str:
    """Task prompt augmented with the legacy MBPP function-name/arity hint."""
    desc = ex["prompt"]
    fname, pcount = extract_function_signature(ex["test_list"])
    if fname:
        desc += f"\nMAKE SURE THE FUNCTION IS NAMED `{fname}`."
    if pcount is not None:
        desc += f"\nTHE FUNCTION MUST TAKE EXACTLY {pcount} PARAMETER(S)."
    return desc


def write_rows(jf, rows: list[dict]) -> None:
    """Append rows to the durable JSONL log and fsync (a crash keeps finished tasks)."""
    for r in rows:
        jf.write(json.dumps(r) + "\n")
    jf.flush()
    os.fsync(jf.fileno())


def log_condition_metrics(m: dict) -> None:
    """Log the (single-condition) headline scalars to the active MLflow run."""
    mm = next(iter(m.values()), None)
    if mm is None:
        return
    for key in ("initial_pass_rate", "recovered_of_failed", "recovered_of_all", "task_solve_rate"):
        mlflow.log_metric(key, mm[key] or 0.0)
    for rk, rv in mm["rounds"].items():
        val = rv["mean_pass_fraction"] if isinstance(rv, dict) else rv
        if val is not None:
            mlflow.log_metric(rk, val)


def setup_mlflow() -> None:
    """Use a SQLite tracking backend (MLflow's recommended local store) with a
    dedicated artifacts dir; create the experiment on first run."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    try:  # EAFP: create once; swallow only "already exists", re-raise real errors
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS)
    except MlflowException as e:
        if "already exists" not in str(e).lower():
            raise
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def main() -> None:
    args = parse_args()
    sha = git_sha()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + sha
    os.makedirs(DATA, exist_ok=True)
    model = Model(model_name=args.model)
    tasks = load_tasks(args.dataset, args.n_tasks)
    print(f"[INFO] run_id={run_id} model={args.model} tasks={len(tasks)} np={args.np} max_attempts={args.max_attempts}")

    setup_mlflow()
    jsonl = os.path.join(DATA, f"results_{run_id}.jsonl")

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "run_id": run_id, "git_sha": sha, "model": args.model, "dataset": args.dataset,
            "np": args.np, "max_attempts": args.max_attempts, "refine_mode": args.refine_mode,
            "n_tasks": len(tasks),
        })
        with open(jsonl, "w", encoding="utf-8") as jf:
            for i, ex in enumerate(tasks, 1):
                tests = ex["test_list"]
                base = {
                    "run_id": run_id, "git_sha": sha, "dataset": args.dataset, "model": args.model,
                    "refine_mode": args.refine_mode, "difficulty": "na", "task_id": str(ex["task_id"]),
                    "n_tests": len(tests), "np": args.np, "max_attempts": args.max_attempts,
                }
                scorer = make_scorer(ex["test_imports"], tests)
                t0 = time.time()
                rows = run_task(model, build_desc(ex), scorer, args.np, args.max_attempts, base)
                write_rows(jf, rows)
                init_pass = sum(r["passed"] for r in rows if r["attempt"] == 0)
                print(f"[{i}/{len(tasks)}] task {ex['task_id']}: {len(rows)} programs, "
                      f"{init_pass}/{args.np} seeds passed initially ({time.time()-t0:.0f}s)")

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

        log_condition_metrics(m)
        for p in (pq, mjson, fig, tex):
            mlflow.log_artifact(p)

    print(f"\n[DONE] wrote {pq}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
