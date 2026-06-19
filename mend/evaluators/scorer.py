"""Isolated, killable subprocess scorer.

Runs a candidate program's tests inside mend/evaluators/test_worker.py in its own process
group, so one infinite-loop test can't wedge the parent (a hard SIGKILL backstop on the
whole tree). Returns the fraction of the task's tests passed.
"""
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_worker.py")


def make_scorer(setup: str, tests: list[str], prelude: str = "", per_timeout: int = 5) -> Callable[[str], float]:
    """Build a scorer for one task. `prelude` runs after the program (e.g. bind `candidate`
    for HumanEval). The returned callable scores a candidate program string in [0, 1]."""
    tests = list(tests)
    if not tests:
        return lambda code: 0.0
    base = {"setup": setup or "", "prelude": prelude, "tests": tests, "per_timeout": per_timeout}
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
