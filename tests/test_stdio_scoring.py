"""Pin stdio-mode scoring, including the __name__ == "__main__" guard.

APPS stdin solutions often wrap logic in `def main()` behind `if __name__ == "__main__":`.
Stdio mode sets __name__ to "__main__" so that driver runs; otherwise the program prints
nothing and scores 0. Function mode leaves the guard dormant, since its driver may read
stdin and fail.
"""
from mend.evaluators import make_scorer

DOUBLE_CASES = [["5\n", "10\n"], ["7\n", "14\n"]]


def test_stdio_plain_script():
    prog = "n = int(input())\nprint(n * 2)"
    assert make_scorer("", DOUBLE_CASES, io_mode="stdio")(prog) == 1.0


def test_stdio_main_guard_runs():
    prog = "def main():\n    n = int(input())\n    print(n * 2)\nif __name__ == '__main__':\n    main()"
    assert make_scorer("", DOUBLE_CASES, io_mode="stdio")(prog) == 1.0


def test_stdio_partial_credit():
    prog = "n = int(input())\nprint(n * 2 if n != 7 else 0)"  # passes 5, fails 7
    assert make_scorer("", DOUBLE_CASES, io_mode="stdio")(prog) == 0.5


def test_function_mode_leaves_guard_dormant():
    # A __main__ driver that reads stdin would EOF-fail; it must not run in function mode.
    prog = "def f(x):\n    return x * 2\nif __name__ == '__main__':\n    raise SystemExit(input())"
    assert make_scorer("", ["assert candidate(3) == 6"], prelude="candidate = f",
                       io_mode="function")(prog) == 1.0


if __name__ == "__main__":
    test_stdio_plain_script()
    test_stdio_main_guard_runs()
    test_stdio_partial_credit()
    test_function_mode_leaves_guard_dormant()
    print("ok: stdio scoring pinned (incl. __main__ guard)")
