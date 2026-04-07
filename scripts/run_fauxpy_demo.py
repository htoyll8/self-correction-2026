import os
import tempfile
import subprocess
from pathlib import Path
import csv


def run_fauxpy_on_apps_task_with_io(task, family="sbfl"):
    """
    Run FauxPy fault localization on a single APPS task dict.
    task should contain keys:
        - problem_id
        - solutions (list of candidate codes)
        - input_output (dict with 'inputs' and 'outputs' lists)
    """
    task_id = task["problem_id"]
    program_code = task["solutions"][0]
    inputs = task["input_output"]["inputs"]
    outputs = task["input_output"]["outputs"]

    tmpdir = Path(tempfile.mkdtemp(prefix=f"apps_task_{task_id}_"))
    code_dir, tests_dir = tmpdir / "code", tmpdir / "tests"
    code_dir.mkdir()
    tests_dir.mkdir()
    (code_dir / "__init__.py").write_text("")
    (tests_dir / "__init__.py").write_text("")

    # Write the model's generated solution
    (code_dir / "solution.py").write_text(program_code)

    # --- Generate pytest tests safely ---
    test_lines = [
        "import subprocess, pytest, os, sys",
        "from pathlib import Path",
        "ROOT = Path(__file__).resolve().parents[1]",
        "SRC = ROOT / 'code' / 'solution.py'",
        ""
    ]

    test_lines = ["import pytest", "from io import StringIO", "import sys, importlib"]

    for j, (inp, out) in enumerate(zip(inputs, outputs), 1):
        inp_literal = repr(inp.rstrip("\n"))
        out_literal = repr(out.rstrip("\n"))

        test_lines.append("@pytest.mark.timeout(5)")
        test_lines.append(f"def test_case_{j}():")
        test_lines.append(f"    input_data = {inp_literal}")
        test_lines.append("    backup_stdin, backup_stdout = sys.stdin, sys.stdout")
        test_lines.append("    sys.stdin, sys.stdout = StringIO(input_data), StringIO()")
        test_lines.append("    try:")
        test_lines.append("        from code import solution")  # 👈 moved inside test
        test_lines.append("        importlib.reload(solution)   # re-run the code")
        test_lines.append("        pred = sys.stdout.getvalue().strip()")
        test_lines.append("    finally:")
        test_lines.append("        sys.stdin, sys.stdout = backup_stdin, backup_stdout")
        test_lines.append(
            f"    assert pred == {out_literal}.strip(), f\"Expected {out_literal}, got {{pred!r}}\""
        )
        test_lines.append("")

    (tests_dir / "test_solution.py").write_text("\n".join(test_lines))

    # --- Run FauxPy ---
    cmd = [
        "python", "-m", "pytest", str(tests_dir),
        "--src", "code",
        "--family", family,
        "--granularity", "statement",
        "--top-n", "10",
    ]

    print(f"\n🚀 Running FauxPy on APPS task {task_id} ({family})\n")
    result = subprocess.run(cmd, cwd=tmpdir, text=True, capture_output=True)
    print(result.stdout)

    # --- Parse results ---
    results_dir = next(
        (p for p in tmpdir.parent.iterdir()
         if p.name.startswith(tmpdir.name) and p.name.endswith("_results")),
        None,
    )
    if not results_dir:
        print("⚠️ No FauxPy results found.")
        return None

    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        print("⚠️ No result CSVs found.")
        return None

    csv_path = csv_files[0]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return [
        {"line": int(r["Line"]), "score": float(r["Score"]), "file": r["File"]}
        for r in rows
    ]


task = {
  "problem_id": 0,
  "question": "Polycarp has $n$ different binary words. A word is binary if it contains only characters '0' and '1'. For example...",
  "solutions": [
    "for _ in range(int(input())):\n    n = int(input())\n    mass = []\n    pass"
  ],
  "input_output": {
    "inputs": [
      "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n"
    ],
    "outputs": [
      "1\n3 \n-1\n0\n\n2\n1 2 \n"
    ]
  },
  "difficulty": "interview",
  "url": "https://codeforces.com/problemset/problem/1259/D",
  "starter_code": ""
}

results = run_fauxpy_on_apps_task_with_io(task)
print(results)
