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

# Hard wall-clock ceiling per scorer call, independent of test count. Without it, a
# candidate that hangs on every one of N cases burns per_timeout*N seconds (many minutes
# for MBPP+'s ~150-case harnesses). Set generously (5 min) so a slow-but-correct solution
# is never falsely failed; it only fires on genuinely stuck candidates.
MAX_WALL_SECONDS = 300


def make_scorer(setup: str, tests: list, prelude: str = "", per_timeout: int = 5,
                io_mode: str = "function") -> Callable[[str], float]:
    """Build a scorer for one task. `prelude` runs after the program (e.g. bind `candidate`
    for HumanEval). `io_mode` is "function" (assert-based tests) or "stdio" ([input, output]
    pairs run via piped stdin). The returned callable scores a candidate program in [0, 1]."""
    tests = list(tests)
    if not tests:
        return lambda code: 0.0
    overall = min(per_timeout * len(tests) + 15, MAX_WALL_SECONDS)
    # The worker stops at `budget` and prints its partial progress, so hitting the wall cap
    # yields partial credit instead of the 0.0 we'd get if the parent SIGKILLed it before it
    # emitted the sentinel. The margin must exceed one per_timeout: budget is only checked
    # between cases, so a case started just under budget can run a full per_timeout longer and
    # must still finish before the parent's `overall` kill.
    margin = per_timeout + 3
    budget = max(overall - margin, per_timeout)
    base = {"setup": setup or "", "prelude": prelude, "tests": tests,
            "per_timeout": per_timeout, "io_mode": io_mode, "budget": budget}

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
