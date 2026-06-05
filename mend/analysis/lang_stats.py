#!/usr/bin/env python3
"""
RQ5: How does programming language affect self-correction behavior?

This script ingests one or more JSONL result files (each line = one trajectory/run for a task),
extracts the programming language (from fields or filename), and computes *language-stratified*
self-correction metrics.

It is intentionally defensive about schema differences across result dumps.

Outputs:
- stdout tables (per-language)
- optional CSV/JSON summaries

USAGE:
  python rq5_language_effect.py \
    --inputs results/*.jsonl \
    --out_csv rq5_language_summary.csv

If your JSONL has an explicit language field, it will use it.
Otherwise it will try to infer language from filename tokens like:
  humaneval-x_java_..., mbppplus_python_..., apps_js_...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Helpers: robust extraction
# ----------------------------

LANG_ALIASES = {
    "py": "python",
    "python3": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "go": "go",
    "golang": "go",
    "rb": "ruby",
    "ruby": "ruby",
}

FILENAME_LANG_PATTERNS = [
    # common tokens separated by underscores/dashes
    r"(?:^|[_\-])(python|py|java|cpp|c\+\+|js|javascript|ts|typescript|go|golang|ruby|rb)(?:[_\-]|$)",
]


def _norm_lang(s: str) -> Optional[str]:
    if not s:
        return None
    s2 = s.strip().lower()
    return LANG_ALIASES.get(s2, s2)


def infer_language_from_filename(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    for pat in FILENAME_LANG_PATTERNS:
        m = re.search(pat, name)
        if m:
            return _norm_lang(m.group(1))
    return None


def get_first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def extract_task_id(rec: Dict[str, Any], fallback: str) -> str:
    # Common places a task id can show up
    v = get_first(rec, ["task_id", "id", "problem_id", "qid", "uid"])
    if v is not None:
        return str(v)

    # Nested task object
    task = rec.get("task") or rec.get("problem") or rec.get("meta") or {}
    if isinstance(task, dict):
        v2 = get_first(task, ["task_id", "id", "problem_id", "qid", "uid", "name"])
        if v2 is not None:
            return str(v2)

    # fallback stable-ish per line
    return fallback


def extract_language(rec: Dict[str, Any], filename: str) -> str:
    # Common direct fields
    v = get_first(rec, ["language", "lang", "prog_lang", "programming_language"])
    if isinstance(v, str) and v.strip():
        return _norm_lang(v) or "unknown"

    # Nested task/meta
    for container_key in ["task", "problem", "meta", "example"]:
        obj = rec.get(container_key)
        if isinstance(obj, dict):
            v2 = get_first(obj, ["language", "lang", "prog_lang", "programming_language"])
            if isinstance(v2, str) and v2.strip():
                return _norm_lang(v2) or "unknown"

    # Infer from filename
    v3 = infer_language_from_filename(filename)
    return v3 or "unknown"


def extract_attempts(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Attempt to normalize attempts into a list of dicts with keys:
      - passed (bool|None)
      - pass_fraction (float|None in [0,1])
    """
    # Most explicit: attempts list
    attempts = rec.get("attempts") or rec.get("steps") or rec.get("trajectory")
    if isinstance(attempts, list) and attempts:
        out = []
        for a in attempts:
            if isinstance(a, dict):
                passed = a.get("passed")
                if passed is None:
                    passed = a.get("pass") if "pass" in a else None
                pf = a.get("pass_fraction")
                if pf is None:
                    pf = a.get("pass_rate") if "pass_rate" in a else None
                out.append({"passed": passed, "pass_fraction": pf})
            else:
                out.append({"passed": None, "pass_fraction": None})
        return out

    # If only summary fields exist, synthesize minimal attempts
    # e.g. initial_passed/final_passed/num_attempts
    init_passed = get_first(rec, ["initial_passed", "passed_initially", "init_passed"])
    final_passed = get_first(rec, ["final_passed", "passed", "success", "solved"])
    num_attempts = get_first(rec, ["num_attempts", "attempt_count", "attempts_used", "n_attempts"])
    try:
        n = int(num_attempts) if num_attempts is not None else None
    except Exception:
        n = None

    # Make a list length = (1 + repairs) if we can
    if n is not None and n >= 0:
        # Interpret num_attempts as number of *repairs* OR total attempts? unclear.
        # We'll treat it as total attempts including initial if it looks like >=1.
        total = n if n >= 1 else 1
        arr = [{"passed": None, "pass_fraction": None} for _ in range(total)]
        # annotate best guesses
        if init_passed is not None:
            arr[0]["passed"] = bool(init_passed)
        if final_passed is not None:
            arr[-1]["passed"] = bool(final_passed)
        return arr

    # As a last resort, one attempt only
    passed = get_first(rec, ["passed", "success", "solved"])
    pf = get_first(rec, ["pass_fraction", "pass_rate"])
    return [{"passed": passed, "pass_fraction": pf}]


def boolish(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        # treat 0/1 as bool if exact
        if x == 0:
            return False
        if x == 1:
            return True
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "pass", "passed"}:
            return True
        if s in {"false", "f", "no", "n", "fail", "failed"}:
            return False
    return None


def clamp01(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    # some logs use percents
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


# ----------------------------
# Metrics per trajectory
# ----------------------------

@dataclass
class TrajSummary:
    language: str
    task_id: str
    passed_initially: bool
    ever_passed: bool
    attempts_used: int           # number of repair attempts performed (0 if passed initially)
    total_attempts_observed: int # including initial
    initial_pass_fraction: float # fraction passed on initial attempt (0..1)
    best_pass_fraction: float    # max over attempts (0..1)


def summarize_trajectory(rec: Dict[str, Any], filename: str, line_idx: int) -> TrajSummary:
    task_id = extract_task_id(rec, fallback=f"{Path(filename).name}:{line_idx}")
    lang = extract_language(rec, filename)

    attempts = extract_attempts(rec)
    # Normalize per-attempt fields
    passed_list: List[Optional[bool]] = [boolish(a.get("passed")) for a in attempts]
    pf_list: List[Optional[float]] = [clamp01(a.get("pass_fraction")) for a in attempts]

    # Determine pass/fractions robustly
    passed_initially = bool(passed_list[0]) if passed_list[0] is not None else False

    # ever_passed: any attempt passed=True; else if no booleans, infer from best pass fraction == 1
    ever_passed_bool = any(p is True for p in passed_list)
    best_pf = max([pf for pf in pf_list if pf is not None], default=0.0)
    ever_passed = ever_passed_bool or (best_pf >= 1.0 - 1e-12)

    init_pf = pf_list[0] if pf_list[0] is not None else (1.0 if passed_initially else 0.0)

    total_attempts = len(attempts)
    # attempts_used: repairs performed until success (if logged) or total-1 if never passed.
    # If ever passed and we have a passed=True index, use the first success index.
    if ever_passed_bool:
        first_success_idx = next(i for i, p in enumerate(passed_list) if p is True)
        # repairs used = attempts before first success excluding initial
        attempts_used = max(0, first_success_idx)  # since idx=0 => 0 repairs
    elif best_pf >= 1.0 - 1e-12:
        # inferred success at some point, but no passed flags: assume last attempt is success
        attempts_used = max(0, total_attempts - 1)
    else:
        # never succeeded: repairs tried = total_attempts - 1 (could be 0)
        attempts_used = max(0, total_attempts - 1)

    return TrajSummary(
        language=lang,
        task_id=task_id,
        passed_initially=passed_initially,
        ever_passed=ever_passed,
        attempts_used=attempts_used,
        total_attempts_observed=total_attempts,
        initial_pass_fraction=float(init_pf),
        best_pass_fraction=float(best_pf),
    )


# ----------------------------
# Aggregations for RQ5
# ----------------------------

def safe_mean(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]
    return float(mean(xs2)) if xs2 else float("nan")


def safe_median(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]
    return float(median(xs2)) if xs2 else float("nan")


def compute_language_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    df columns expected:
      language, task_id, passed_initially, ever_passed, attempts_used,
      initial_pass_fraction, best_pass_fraction
    """
    rows = []
    for lang, g in df.groupby("language"):
        # trajectory-level
        n_traj = len(g)
        n_success = int(g["passed_initially"].sum())
        n_recovered = int(((~g["passed_initially"]) & (g["ever_passed"])).sum())
        n_failed = int(((~g["ever_passed"])).sum())

        # conditional recovery rate among initially failing trajectories
        denom = n_recovered + n_failed
        recovery_pct = (100.0 * n_recovered / denom) if denom > 0 else float("nan")

        # attempts among failed seeds = among trajectories that did NOT pass initially
        failed_seed = g[~g["passed_initially"]]
        mean_attempts_failed_seed = safe_mean(failed_seed["attempts_used"].tolist())
        med_attempts_failed_seed = safe_median(failed_seed["attempts_used"].tolist())

        # task-level initial pass: need to aggregate per task_id
        # A task is "passed initially" if ANY trajectory for that task passed initially.
        per_task = g.groupby("task_id").agg(
            task_passed_initially=("passed_initially", "max"),
            task_ever_passed=("ever_passed", "max"),
        )
        n_tasks = len(per_task)
        frac_passed_initially = float(per_task["task_passed_initially"].mean()) if n_tasks else float("nan")
        frac_required_repair = 1.0 - frac_passed_initially if n_tasks else float("nan")
        frac_ever_solved = float(per_task["task_ever_passed"].mean()) if n_tasks else float("nan")

        rows.append(
            {
                "language": lang,
                "tasks": n_tasks,
                "trajectories": n_traj,
                "task_frac_passed_initially": frac_passed_initially,
                "task_frac_required_repair": frac_required_repair,
                "task_frac_ever_solved": frac_ever_solved,
                "traj_success_count": n_success,
                "traj_recovered_count": n_recovered,
                "traj_failed_count": n_failed,
                "traj_recovery_pct_among_failed_seeds": recovery_pct,
                "mean_attempts_failed_seeds": mean_attempts_failed_seed,
                "median_attempts_failed_seeds": med_attempts_failed_seed,
                "mean_initial_pass_fraction": safe_mean(g["initial_pass_fraction"].tolist()),
                "median_initial_pass_fraction": safe_median(g["initial_pass_fraction"].tolist()),
                "mean_best_pass_fraction": safe_mean(g["best_pass_fraction"].tolist()),
                "median_best_pass_fraction": safe_median(g["best_pass_fraction"].tolist()),
            }
        )

    out = pd.DataFrame(rows).sort_values(["tasks", "trajectories"], ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL files (glob ok via shell).")
    ap.add_argument("--out_csv", default=None, help="Optional path to write CSV summary.")
    ap.add_argument("--out_json", default=None, help="Optional path to write JSON summary.")
    ap.add_argument("--keep_unknown", action="store_true", help="Keep rows where language=unknown (default drops).")
    args = ap.parse_args()

    traj_rows: List[Dict[str, Any]] = []
    for fp in args.inputs:
        fp = str(fp)
        with open(fp, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                s = summarize_trajectory(rec, filename=fp, line_idx=i)
                traj_rows.append(
                    {
                        "file": Path(fp).name,
                        "language": s.language,
                        "task_id": s.task_id,
                        "passed_initially": bool(s.passed_initially),
                        "ever_passed": bool(s.ever_passed),
                        "attempts_used": int(s.attempts_used),
                        "total_attempts_observed": int(s.total_attempts_observed),
                        "initial_pass_fraction": float(s.initial_pass_fraction),
                        "best_pass_fraction": float(s.best_pass_fraction),
                    }
                )

    df = pd.DataFrame(traj_rows)
    if not args.keep_unknown:
        df = df[df["language"] != "unknown"].copy()

    if df.empty:
        print("No data after filtering. (Maybe language could not be inferred?)")
        print("Try: --keep_unknown or ensure your JSONL includes a 'language' field.")
        return

    summary = compute_language_metrics(df)

    # Pretty print
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print("\n=== RQ5: Language-stratified self-correction metrics ===")
        print(summary)

    if args.out_csv:
        summary.to_csv(args.out_csv, index=False)
        print(f"\nWrote CSV: {args.out_csv}")

    if args.out_json:
        summary.to_json(args.out_json, orient="records", indent=2)
        print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
