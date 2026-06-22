"""APPS loader (codeparrot/apps).

APPS problems come in two forms, distinguished by the `input_output` JSON:
- stdio (the vast majority): the program reads stdin and writes stdout. Scored with
  io_mode="stdio" over [input, expected] pairs.
- call-based (a small minority, `fn_name` present): a LeetCode-style `class Solution` with
  a method. Scored with the function-assert path by instantiating Solution and asserting
  the method's return for each case.

Each task carries its `difficulty` (introductory / interview / competition) so the
canonical table can be sliced by difficulty.

Programs are executed at module scope (matching how a model writes a stdin script), so the
rare APPS reference solution that uses a top-level `return` (expecting the official harness's
function wrapping) is not supported; model outputs do not use that idiom.
"""
import json

from mend.datasets.base import Task, take

SOURCE = "codeparrot/apps"


def _as_text(x) -> str:
    """Coerce an APPS stdin/stdout case to text. Most are strings; a few are list-wrapped
    (e.g. ['iloveyou']) — flatten those with newlines so they pipe through stdin cleanly."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_as_text(e) for e in x)
    return str(x)


def _description(ex: dict) -> str:
    """Problem statement plus any starter code the model should complete."""
    desc = ex["question"]
    starter = ex.get("starter_code") or ""
    if starter.strip():
        desc += "\n\nUse this starter code:\n" + starter
    return desc


def _call_tests(io: dict) -> tuple[str, list[str]]:
    """Build (prelude, assert-tests) for a call-based problem. `candidate` is bound to the
    Solution method; each case asserts the method's return, accepting APPS's habit of
    wrapping the expected value in a single-element list."""
    fname = io["fn_name"]
    prelude = f"_sol = Solution()\ncandidate = _sol.{fname}"
    tests = []
    for inp, exp in zip(io["inputs"], io["outputs"]):
        # exp is the expected return, sometimes wrapped as [value]; accept either form.
        tests.append(
            f"__exp = {exp!r}\n"
            f"__got = candidate(*{inp!r})\n"
            f"assert __got == __exp or [__got] == __exp"
        )
    return prelude, tests


def load(n_tasks: int, difficulties: tuple[str, ...] | None = None) -> list[Task]:
    """Return up to `n_tasks` APPS tasks, scanning the split in order until that many are
    collected (the split is ordered interview-first, so reaching introductory/competition
    means scanning past the interview block). `difficulties` filters to a subset of
    {introductory, interview, competition}; problems with no tests are skipped."""
    tasks: list[Task] = []
    skipped_no_tests = 0
    for ex in take(SOURCE, 10**9):  # whole split, lazily; we stop once we have n_tasks
        if len(tasks) >= n_tasks:
            break
        if difficulties and ex["difficulty"] not in difficulties:
            continue
        raw = ex["input_output"]
        io = json.loads(raw) if raw else {}
        if not io.get("inputs"):
            skipped_no_tests += 1  # no tests to score against; skip rather than log a fake pass
            continue
        task_id = str(ex["problem_id"])
        desc = _description(ex)
        if "fn_name" in io:
            prelude, tests = _call_tests(io)
            tasks.append(Task(task_id=task_id, description=desc, setup="", tests=tests,
                              prelude=prelude, difficulty=ex["difficulty"]))
        else:
            cases = [[_as_text(i), _as_text(o)] for i, o in zip(io["inputs"], io["outputs"])]
            tasks.append(Task(task_id=task_id, description=desc, setup="", tests=cases,
                              io_mode="stdio", difficulty=ex["difficulty"]))
    if skipped_no_tests:
        print(f"[apps] skipped {skipped_no_tests} problem(s) with no usable tests")
    return tasks
