import coverage
import math
import io
import sys
from termcolor import colored
from contextlib import redirect_stdout


# -------------------------------
# Example buggy program
# -------------------------------
def buggy_sum(nums):
    """Should return sum of all positive numbers."""
    total = 0
    for x in nums:
        if x > 0:
            total += x
        else:
            total -= x  # ❌ BUG: should skip negatives instead of subtracting
    return total


# -------------------------------
# Tests (some pass, some fail)
# -------------------------------
def test_positive_only():
    assert buggy_sum([1, 2, 3]) == 6


def test_with_negatives():
    assert buggy_sum([1, -1, 2]) == 3  # will fail because result = 4


def test_empty():
    assert buggy_sum([]) == 0


TESTS = [
    ("test_positive_only", test_positive_only),
    ("test_with_negatives", test_with_negatives),
    ("test_empty", test_empty),
]


# -------------------------------
# Step 1: Run each test with coverage
# -------------------------------
def run_tests_with_coverage():
    coverage_data = {}

    for name, func in TESTS:
        cov = coverage.Coverage(source=["."])
        cov.start()

        try:
            func()
            result = "PASS"
        except AssertionError:
            result = "FAIL"
        finally:
            cov.stop()
            cov.save()

        data = cov.get_data()
        executed_lines = set()
        for f in data.measured_files():
            if f.endswith(__file__):
                executed_lines |= set(data.lines(f))

        coverage_data[name] = {"result": result, "lines": executed_lines}

        print(f"\n[{name}] {result}")
        print(f"Executed lines: {sorted(executed_lines)}")

    return coverage_data


# -------------------------------
# Step 2: Build spectra
# -------------------------------
def build_sbfl_matrix(coverage_data):
    all_lines = sorted(set.union(*(d["lines"] for d in coverage_data.values())))
    matrix = {}
    total_pass = sum(1 for d in coverage_data.values() if d["result"] == "PASS")
    total_fail = sum(1 for d in coverage_data.values() if d["result"] == "FAIL")

    for line in all_lines:
        passed = sum(
            1 for d in coverage_data.values()
            if d["result"] == "PASS" and line in d["lines"]
        )
        failed = sum(
            1 for d in coverage_data.values()
            if d["result"] == "FAIL" and line in d["lines"]
        )
        matrix[line] = (passed, failed)

    print("\nLine coverage summary:")
    for line, (p, f) in matrix.items():
        print(f"  Line {line}: passed={p}, failed={f}")

    return matrix, total_pass, total_fail


# -------------------------------
# Step 3: Compute Ochiai suspiciousness
# -------------------------------
def ochiai(failed, passed, total_fail, total_pass):
    top = failed
    bottom = math.sqrt(total_fail * (failed + passed))
    return 0 if bottom == 0 else top / bottom


def show_code_context(lines):
    print("\nCode context for suspicious lines:")
    with open(__file__, "r") as f:
        source_lines = f.readlines()
    for line_no in sorted(lines):
        code = source_lines[line_no - 1].rstrip()
        print(f"  {line_no:>3}: {code}")


def rank_lines(matrix, total_pass, total_fail):
    scores = []
    for line, (p, f) in matrix.items():
        score = ochiai(f, p, total_fail, total_pass)
        scores.append((line, score))
    scores.sort(key=lambda x: -x[1])
    print("\nSuspiciousness ranking:")
    for line, s in scores:
        color = "red" if s > 0.7 else "yellow" if s > 0.3 else "green"
        print(colored(f"  Line {line}: {s:.3f}", color))
    return scores


if __name__ == "__main__":
    coverage_data = run_tests_with_coverage()
    matrix, total_pass, total_fail = build_sbfl_matrix(coverage_data)
    rank_lines(matrix, total_pass, total_fail)
    show_code_context(matrix.keys())
