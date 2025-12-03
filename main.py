import re
import os
import sys
import json
import time
import signal
import textwrap
import inspect
import argparse
import traceback
import tempfile
import subprocess
from tqdm import tqdm
from io import StringIO
import concurrent.futures
from datetime import datetime
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess, json, tempfile, sys
from pathlib import Path
from threading import Lock
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
coverage_lock = Lock()

from model import Model

TRACE_RUNNER = str(Path(__file__).resolve().parent / "trace_runner.py")


def safe_call(fn, *args, timeout=90):
    """Run a blocking function with a timeout to avoid indefinite hangs."""
    with concurrent.futures.ThreadPoolExecutor() as ex:
        future = ex.submit(fn, *args)
        return future.result(timeout=timeout)


WRAPPER_FUNCS = {
    "set", "sorted", "list", "tuple", "dict",
    "len", "max", "min", "sum", "all", "any"
}


def split_top_level_args(s: str):
    """
    Split argument list on commas at top level only.
    Handles nested parentheses/brackets/braces.
    """
    args = []
    buf = []
    depth = 0

    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1

        if ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    if buf:
        args.append("".join(buf).strip())

    return args


WRAPPER_FUNCS = {"set", "sorted", "list", "tuple"}


def extract_function_signature(assert_lines):
    for line in assert_lines:
        lhs = line.split("==")[0]

        i = 0
        func_start = None

        while i < len(lhs):
            ch = lhs[i]

            if ch.isalpha() or ch == "_":
                if func_start is None:
                    func_start = i
            else:
                if func_start is not None:
                    name = lhs[func_start:i]

                    # check '(' right after name
                    if i < len(lhs) and lhs[i] == "(":
                        # parse args
                        paren = 1
                        j = i + 1
                        while j < len(lhs) and paren > 0:
                            if lhs[j] == "(":
                                paren += 1
                            elif lhs[j] == ")":
                                paren -= 1
                            j += 1

                        if paren == 0:
                            if name not in WRAPPER_FUNCS:
                                raw_args = lhs[i+1 : j-1]
                                args = split_top_level_args(raw_args)
                                return name, len(args)

                    func_start = None
            i += 1

    return None, None


def build_feedback_history(trajectory):
    """
    Construct a full history string of all previous refinement attempts
    for use in history-aware feedback generation.
    Includes the full code for each past attempt (no feedback text).

    Returns None if fewer than 2 refinement attempts.
    """
    attempts = trajectory.get("refinement_attempts", [])

    # Only include history if there are at least 2 attempts
    if len(attempts) < 2:
        return None

    # Exclude the last attempt (the current program)
    past_attempts = attempts[:-1]

    history_lines = []
    for r in past_attempts:
        program_text = r.get("program") or "None"
        history_lines.append(
            f"───────────────────────────────\n"
            f"Attempt {r['attempt']}: pass={r['pass_fraction']*100:.1f}% | passed={r['passed']}\n"
            f"Program:\n{program_text}\n"
        )

    return "\n".join(history_lines)


def build_antiunified_history(model, trajectory):
    print("\n[DEBUG] Entered build_antiunified_history()", flush=True)

    attempts = trajectory.get("refinement_attempts", [])
    if not attempts:
        print("[DEBUG] No attempts found. Returning empty structure.", flush=True)
        return {
            "anti_unified": "",
            "failing_sets": {},
            "overlap": set(),
            "unique": {}
        }

    # Anti-unified structure
    anti_unified = model.generate_antiunified_history(trajectory)

    # Storage
    failing_sets = {}
    overlap = set()
    unique = {}

    # Helper to parse the FAIL detail field
    def parse_failure(detail):
        # detail is like: "Expected '4', Got '3'"
        import re
        m = re.findall(r"Expected:?\s*'?(.*?)'?,?\s*Got:?\s*'?(.*?)'?$", detail)
        if m:
            expected, got = m[0]
            return expected, got
        return None, None

    # ========== INITIAL ATTEMPT ==========
    print("\n[DEBUG] --- Processing initial attempt ---", flush=True)
    init_results = trajectory["initial_test_results"]
    init_inputs = trajectory["initial_test_inputs"]
    init_outputs = trajectory["initial_test_outputs"]

    initial_fail_ids = set()

    for tid, status, detail in init_results:
        if "FAIL" in status:
            initial_fail_ids.add(tid)
            expected, got = parse_failure(detail)

            print(f"\n[FAIL] Initial attempt – Test {tid}", flush=True)
            print("  input:   ", repr(init_inputs[tid-1]), flush=True)
            print("  expected:", repr(init_outputs[tid-1]), flush=True)
            print("  got:     ", repr(got), flush=True)

    failing_sets[0] = initial_fail_ids

    # ========== REFINEMENT ATTEMPTS ==========
    print("\n[DEBUG] --- Processing refinement attempts ---", flush=True)

    for j, attempt in enumerate(attempts, start=1):
        results = attempt["test_results"]
        test_inputs = attempt["test_inputs"]
        test_outputs = attempt["test_outputs"]
        fail_ids = set()

        for tid, status, detail in results:
            if "FAIL" in status:
                fail_ids.add(tid)
                expected, got = parse_failure(detail)
                print(f"\n[FAIL] Attempt {j} – Test {tid}", flush=True)
                print("  input:   ", repr(test_inputs[tid-1]), flush=True)
                print("  expected:", repr(test_outputs[tid-1]), flush=True)
                print("  got:     ", repr(got), flush=True)

        failing_sets[j] = fail_ids

    # ========== COMPUTE OVERLAP ==========

    non_empty = [s for s in failing_sets.values() if s]
    overlap = set.intersection(*non_empty) if non_empty else set()

    unique = {k: v - overlap for k, v in failing_sets.items()}

    return {
        "anti_unified": anti_unified,
        "failing_sets": failing_sets,
        "overlap": overlap,
        "unique": unique
    }


def run_in_subprocess(code_str, input_str, timeout=3):
    """Runs code_str on input_str inside a fully isolated Python subprocess.
       This version guarantees kill of the full process group.
    """
    payload = json.dumps({
        "code": code_str,
        "input": input_str
    })

    # Launch subprocess in its own process group
    proc = subprocess.Popen(
        [sys.executable, TRACE_RUNNER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid  # <-- start new process group so we can kill everything
    )

    try:
        out, err = proc.communicate(payload, timeout=timeout)

    except subprocess.TimeoutExpired:
        # Kill the entire process group (proc + any children)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()

        return {
            "output": None,
            "coverage": [],
            "error": "TIMEOUT"
        }

    # stderr from runner means program-level error
    if err:
        return {
            "output": None,
            "coverage": [],
            "error": "RunnerError:\n" + err
        }

    # JSON decode the coverage + output payload
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {
            "output": None,
            "coverage": [],
            "error": f"Bad JSON from runner: {out}"
        }


def separate_pass_fail_lines(results, coverage):
    fail_lines = []
    pass_lines = []

    for (test_id, status, _) in results:
        lines = coverage.get(test_id, set())
        if "PASS" in status:
            pass_lines.append(lines)
        else:
            fail_lines.append(lines)

    # Union (all lines ever executed in fail/pass tests)
    fail_union = set().union(*fail_lines) if fail_lines else set()
    pass_union = set().union(*pass_lines) if pass_lines else set()

    # Lines *unique* to failing tests
    fail_unique = fail_union - pass_union

    # Lines *unique* to passing tests
    pass_unique = pass_union - fail_union

    return {
        "fail_cases": fail_lines,
        "pass_cases": pass_lines,
        "fail_union": fail_union,
        "pass_union": pass_union,
        "fail_unique": fail_unique,
        "pass_unique": pass_unique,
    }


def code_with_line_numbers(code_str):
    """
    Return a pretty string of the code with line numbers.
    """
    out = []
    out.append("=== Candidate Program (with line numbers) ===")
    for idx, line in enumerate(code_str.splitlines(), start=1):
        out.append(f"{idx:3d}: {line}")
    out.append("===========================================")
    return "\n".join(out)


def format_raw_execution_results(results):
    """
    Format test results exactly like HumanEval/APPS JSON triples.
    Example output:
      [1, "❌ FAIL", "assert candidate(12, 2) == '12'"]
    """
    lines = []
    for tid, status, detail in results:
        # Escape internal quotes if needed
        safe_detail = detail.replace('"', '\\"')
        lines.append(f"[{tid}, \"{status}\", \"{safe_detail}\"]")
    return "\n".join(lines)


def summarize_attempt(program, results, coverage, test_inputs, test_outputs):
    """Return a structured summary of one program attempt."""
    failures = []
    for tid, status, detail in results:
        if "PASS" in status:
            continue

        input_str = test_inputs[tid - 1]
        expected_gold = test_outputs[tid - 1]

        if "FAIL" in status:
            import re
            m = re.findall(r"Expected:? *'?(.*?)'?,? *Got:? *'?(.*?)'?$", detail)
            expected, got = (m[0] if m else ("?", "?"))
        elif "TIMEOUT" in status:
            expected, got = expected_gold, "TIMEOUT"
        else:
            expected, got = expected_gold, "ERROR"

        failures.append({
            "test_id": tid,
            "status": status,
            "input": input_str,
            "expected": expected,
            "got": got,
            "executed_lines": sorted(coverage.get(tid, set())),
        })

    return {
        "program": program,
        "program_with_lines": code_with_line_numbers(program),
        "coverage": coverage,
        "failing_tests": failures,
    }


def cluster_by_expected(failing_tests):
    """
    Cluster test-case dictionaries by their 'expected' field.

    Parameters
    ----------
    failing_tests : list[dict]
        Each dict should include at least:
        - test_id
        - expected
        - input
        - got
        - executed_lines
        - status

    Returns
    -------
    dict[str, list[dict]]
        Mapping: expected_value -> list of test-case dictionaries.
    """

    clusters = {}

    for t in failing_tests:
        exp = t.get("expected", None)

        if exp not in clusters:
            clusters[exp] = []

        clusters[exp].append(t)

    return clusters


def cluster_by_executed_lines(failing_tests):
    """
    failing_tests: list of dicts with fields:
        - test_id
        - input
        - expected
        - got
        - executed_lines (list or set of ints)

    Returns: dict { tuple(sorted_lines) : [test_info, ...] }
    """
    clusters = {}
    for t in failing_tests:
        key = tuple(sorted(t.get("executed_lines", [])))
        clusters.setdefault(key, []).append(t)
    return clusters


def summarize_clusters(clusters):
    parts = []
    for expected_val, tests in clusters.items():
        parts.append(f"### Expected Output = {expected_val!r} ({len(tests)} tests)\n")
        for t in tests[:8]:  # show up to 8 examples
            parts.append(
                f"- Test {t['test_id']}: got {t['got']!r}, input={t['input']!r}"
            )
        parts.append("")  # blank line
    return "\n".join(parts)


def summarize_exec_clusters(clusters):
    """
    clusters: dict from cluster key to list of test dicts
    Returns a big formatted string.
    """
    parts = []
    for exec_lines, tests in clusters.items():
        header = f"=== Executed lines: {list(exec_lines)} | {len(tests)} tests ==="
        block = [header]
        for t in tests[:6]:  # sample first 6
            block.append(
                f"  • Test {t['test_id']} | expected={t['expected']!r}, got={t['got']!r}, input={t['input']!r}"
            )
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def summarize_full_history_for_prompt(history):
    """
    Return a clean human-readable text block for the prompt.
    Includes:
      - failing-test counts
      - full program text with line numbers for each attempt
      - delta analysis across attempts
    """

    lines = []

    # =====================================================
    # 1. HIGH-LEVEL SUMMARY
    # =====================================================
    lines.append("### Previous Attempts Summary\n")
    for idx, h in enumerate(history):
        num_fail = len(h["failing_tests"])
        lines.append(f"Attempt {idx}: {num_fail} failing tests")

    # =====================================================
    # 2. PROGRAM WITH LINE NUMBERS PER ATTEMPT
    # =====================================================
    lines.append("\n### Program Versions Across Attempts\n")

    for idx, h in enumerate(history):
        prog = h["program_with_lines"]
        lines.append(f"\n--- Attempt {idx} Program ---\n{prog}")

    # =====================================================
    # 3. DELTA SUMMARY (fixed/new/persisting)
    # =====================================================
    lines.append("\n### Changes Across Attempts\n")

    deltas = compute_deltas(history)
    for d in deltas:
        lines.append(f"- From attempt {d['prev_attempt']} → {d['attempt']}:")
        lines.append(f"    • Fixed tests: {d['became_fixed']}")
        lines.append(f"    • Newly failing: {d['newly_failed']}")
        lines.append(f"    • Persisting failures: {d['persisted']}")

    return "\n".join(lines)


def compute_deltas(history):
    """
    Compare failing tests across all pairs of attempts.
    Returns a nice dict structure summarizing what changed.
    """
    deltas = []

    for t in range(1, len(history)):
        prev = history[t-1]["failing_tests"]
        curr = history[t]["failing_tests"]

        prev_ids = {f["test_id"] for f in prev}
        curr_ids = {f["test_id"] for f in curr}

        became_fixed = sorted(prev_ids - curr_ids)
        newly_failed = sorted(curr_ids - prev_ids)
        persisted = sorted(curr_ids & prev_ids)

        deltas.append({
            "attempt": t,
            "prev_attempt": t-1,
            "became_fixed": became_fixed,
            "newly_failed": newly_failed,
            "persisted": persisted,
        })

    return deltas


def run_self_repair(task, model, test_suite, np=5,
                    nf=1, nr=1, mode="critique+refine"):
    """
    task: natural-language description of the problem
    model: code-generation LLM interface
    test_suite: callable that returns True if program passes all tests
    np: number of initial seeds
    nf: number of feedback messages per failed program
    nr: number of repairs per feedback
    max_attempts: total number of repair retries per seed
    mode: "critique+refine" or "direct"
    """

    trajectories = []
    success = False

    # ---------- Stage 1: Initial generation ----------
    seeds = model.generate(task_description=task, n=np, temperature=0.7)

    for i, seed in enumerate(seeds, 1):
        seed_code = extract_code(seed)
        frac_passed, results, coverage = test_suite(seed_code)
        print(f"Seed {i} → passed {frac_passed*100:.1f}% of tests")

        fully_passed = frac_passed == 1.0

        trajectory = {
            "seed_index": i,
            "initial_program": seed_code,
            "initial_passed": fully_passed,
            "initial_pass_fraction": frac_passed,
            "initial_test_results": results,
            "feedback_repairs": []  # <--- store all nf × nr pairs here
        }
        trajectories.append(trajectory)

        if fully_passed:
            success = True
            continue

        current_program = seed_code

        # ---------- Stage 2: Iterative repair ----------
        for f_i in range(nf):
            feedback = None
            if mode == "critique+refine":
                feedback = model.generate_feedback(task, current_program, temperature=0)

            for r_i in range(nr):
                repair_num = r_i + 1
                print(f"  Feedback {f_i+1}, Repair {repair_num} → ", end="")

                if mode == "critique+refine":
                    refined = model.refine(task, current_program, feedback, temperature=0)
                elif mode == "direct":
                    refined = model.refine(task, current_program, temperature=0)
                else:
                    raise ValueError("Unknown refinement mode")

                code = extract_code(refined)
                frac_passed, results = test_suite(code)
                fully_passed = frac_passed == 1.0
                print(f"✅ success ({frac_passed*100:.1f}%)" if fully_passed else f"failed ({frac_passed*100:.1f}%)")

                trajectory["feedback_repairs"].append({
                    "feedback_index": f_i + 1,
                    "repair_index": repair_num,
                    "feedback": feedback,
                    "program": code,
                    "pass_fraction": frac_passed,
                    "test_results": results,
                    "passed": fully_passed
                })

                current_program = code
                if fully_passed:
                    success = True
                    print(f"  ✅ Seed {i} succeeded after feedback {f_i+1}, repair {r_i+1}")
                    break
            if fully_passed:
                break

    # ---------- Stage 3: Return aggregated results ----------
    total_programs = np * (1 + nf * nr)

    return {
        "task_id": task,
        "success": success,
        "trajectories": trajectories,
        "k": total_programs
    }


def run_self_repair_iterative(
        task,
        model,
        test_suite,
        np=5,
        max_attempts=10,
        mode="critique+refine",
        use_coverage=False):
    """
    Perform iterative self-repair with multiple independent seeds (np),
    refined in parallel using threads.
    """
    start_time = time.time()
    print(f"\n[DEBUG {time.strftime('%H:%M:%S')}] Starting run_self_repair_iterative "
          f"with {np} seeds, mode={mode}, max_attempts={max_attempts}", flush=True)

    trajectories = []
    overall_success = False
    attempt_history = defaultdict(list)

    print(f"Task at top: {task}", flush=True)

    # ---------- Stage 1: Generate seeds ----------
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] Generating {np} initial seeds...", flush=True)
    try:
        seeds = model.generate(task_description=task, n=np, temperature=0.7)
        print(f"[DEBUG {time.strftime('%H:%M:%S')}] Generated {len(seeds)} seeds successfully.", flush=True)
    except Exception as e:
        print(f"[ERROR {time.strftime('%H:%M:%S')}] Seed generation failed: {e}", flush=True)
        return {"task_id": task, "success": False, "trajectories": [], "k": 0}

    def process_seed(i, seed):
        """Process one seed end-to-end."""
        seed_start = time.time()
        print(f"\n[DEBUG {time.strftime('%H:%M:%S')}] Processing Seed {i}...", flush=True)
        seed_code = extract_code(seed)

        try:
            if use_coverage:
                frac_passed, results, coverage, test_inputs, test_outputs = test_suite(seed_code)
            else:
                frac_passed, results = test_suite(seed_code)
                coverage = None
                test_inputs = None
                test_outputs = None
        except Exception as e:
            print(f"[ERROR {time.strftime('%H:%M:%S')}] Test suite crashed for seed {i}: {e}", flush=True)
            return None, False

        if use_coverage:
            initial_summary = summarize_attempt(
                program=seed_code,
                results=results,
                coverage=coverage,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
            )
            print(f"DEBUG: Initial summary: {initial_summary}", flush=True)
            attempt_history[i].append(initial_summary)
        else:
            print(f"DEBUG: Skipping summarize_attempt (use_coverage=False)", flush=True)

        fully_passed = frac_passed == 1.0

        trajectory = {
            "seed_index": i,
            "initial_program": seed_code,
            "initial_passed": fully_passed,
            "initial_pass_fraction": frac_passed,
            "initial_test_results": results,
            "refinement_attempts": []
        }

        if fully_passed:
            print(f"[INFO {time.strftime('%H:%M:%S')}] Seed {i} ✅ passed all tests initially ({frac_passed*100:.1f}%)", flush=True)
            return trajectory, True

        print(f"[DEBUG {time.strftime('%H:%M:%S')}] Seed {i} failed initially ({frac_passed*100:.1f}%), starting refinement loop...", flush=True)
        current_program = seed_code
        attempt_count = 0
        seed_success = False

        while attempt_count < max_attempts and not seed_success:
            attempt_count += 1
            print(f"[DEBUG {time.strftime('%H:%M:%S')}]   Seed {i}, Attempt {attempt_count} starting...", flush=True)
            feedback = None

            if mode in ("critique+refine",
                        "critique+history+refine",
                        "critique+execresults+refine",
                        "critique+antiunify+refine",
                        "critique+antiunify+execclusters+refine"):
                try:
                    print(f"[TRACE {time.strftime('%H:%M:%S')}]   → Generating feedback...", flush=True)

                    if mode == "critique+refine":
                        # Standard: feedback only based on current program
                        feedback_input = current_program

                    elif mode == "critique+execresults+refine":
                        # Determine latest test results (initial or refined)
                        if trajectory["refinement_attempts"]:
                            last_results = trajectory["refinement_attempts"][-1]["test_results"]
                            current_pass = trajectory["refinement_attempts"][-1]["pass_fraction"] * 100
                        else:
                            last_results = trajectory["initial_test_results"]
                            current_pass = trajectory["initial_pass_fraction"] * 100

                        raw_exec_text = format_raw_execution_results(last_results)

                        feedback_input = (
                            f"Task description:\n{task}\n\n"
                            f"Current program (pass={current_pass:.1f}%):\n"
                            f"{current_program}\n\n"
                            f"### Execution results (raw):\n"
                            f"{raw_exec_text}\n\n"
                            f"Based on these exact test results, refine the program."
                        )

                    elif mode == "critique+history+refine":
                        history_context = build_feedback_history(trajectory)

                        history_block = (
                            f"Summary of previous attempts:\n{history_context}\n\n"
                            if history_context
                            else ""
                        )

                        current_pass = trajectory["refinement_attempts"][-1]["pass_fraction"] * 100

                        feedback_input = (
                            f"Task description:\n{task}\n\n"
                            f"{history_block}"
                            f"Current program to critique "
                            f"(pass={current_pass:.1f}%):\n"
                            f"{current_program}"
                        )

                    elif mode == "critique+antiunify+refine":
                        history = attempt_history[i]
                        # print(f"History: {history}")

                        # Current attempt = last entry
                        cur_attempt = history[-1]
                        failing_tests = cur_attempt["failing_tests"]
                        program_with_lines = cur_attempt["program_with_lines"]

                        # Cluster failing tests
                        clusters = cluster_by_expected(failing_tests)

                        # Produce nice text summary
                        failing_tests_summary = summarize_clusters(clusters)

                        current_pass = trajectory["refinement_attempts"][-1]["pass_fraction"] * 100

                        # ---- Build final feedback prompt ----
                        feedback_input = (
                            f"Task description:\n{task}\n\n"
                            f"Current program to critique (pass={current_pass:.1f}%):\n"
                            f"{program_with_lines}\n\n"
                            f"Summary of failing tests:\n{failing_tests_summary}\n"
                        )

                    elif mode == "critique+antiunify+execclusters+refine":
                        history = attempt_history[i]

                        # --- Full history (deltas + summaries) ---
                        history_summary_text = summarize_full_history_for_prompt(history)

                        # --- Current attempt info ---
                        cur_attempt = history[-1]
                        failing_tests = cur_attempt["failing_tests"]
                        program_with_lines = cur_attempt["program_with_lines"]

                        # Cluster failing tests
                        clusters = cluster_by_expected(failing_tests)

                        # Produce nice text summary
                        cluster_summary = summarize_clusters(clusters)

                        current_pass = trajectory["refinement_attempts"][-1]["pass_fraction"] * 100

                        feedback_input = (
                            f"Task description:\n{task}\n\n"

                            f"### Full Attempt History\n"
                            f"{history_summary_text}\n\n"

                            f"### Current Program (pass={current_pass:.1f}%)\n"
                            f"{program_with_lines}\n\n"

                            f"### Failing Tests (Clustered by Input)\n"
                            f"{cluster_summary}\n"
                        )

                    print(f"Task before input: {task}")
                    print(f"Feedback input: {feedback_input}", flush=True)
                    feedback = safe_call(model.generate_feedback, task, feedback_input, 0, timeout=90)
                    print(f"[TRACE {time.strftime('%H:%M:%S')}]   ← Feedback received ({len(feedback) if feedback else 0} chars)", flush=True)

                except Exception as e:
                    print(f"[WARN  {time.strftime('%H:%M:%S')}]   Feedback generation timeout/error: {e}", flush=True)
                    feedback = None

            try:
                print(f"[TRACE {time.strftime('%H:%M:%S')}]   → Refining program...", flush=True)
                if mode in (
                    "critique+refine",
                    "critique+history+refine",
                    "critique+antiunify+refine",
                    "critique+antiunify+execclusters+refine"
                ):
                    refined = safe_call(model.refine, task, current_program, feedback, 0, timeout=120)
                elif mode == "direct":
                    refined = safe_call(model.refine, task, current_program, 0, timeout=120)
                else:
                    raise ValueError(f"Unknown refinement mode: {mode}")
                print(f"[TRACE {time.strftime('%H:%M:%S')}]   ← Refinement done.", flush=True)
            except Exception as e:
                print(f"[WARN  {time.strftime('%H:%M:%S')}]   Refinement timeout/error: {e}", flush=True)
                continue

            code = extract_code(refined)

            try:
                print(f"[TRACE {time.strftime('%H:%M:%S')}]   → Running test suite on refined code...", flush=True)
                if use_coverage:
                    frac_passed, results, coverage, test_inputs, test_outputs = test_suite(code)
                else:
                    frac_passed, results = test_suite(code)
                    coverage = None
                    test_inputs = None
                    test_outputs = None

                fully_passed = frac_passed == 1.0

                if use_coverage:
                    attempt_summary = summarize_attempt(
                        program=code,
                        results=results,
                        coverage=coverage,
                        test_inputs=test_inputs,
                        test_outputs=test_outputs,
                    )
                    attempt_history[i].append(attempt_summary)
                else:
                    print(f"[DEBUG] Skipping summarize_attempt (use_coverage=False)", flush=True)

                print(f"[TRACE {time.strftime('%H:%M:%S')}]   ← Test suite finished (pass={frac_passed*100:.1f}%)", flush=True)
            except Exception as e:
                print(f"[ERROR {time.strftime('%H:%M:%S')}]   Test suite crash: {e}", flush=True)
                continue

            print(f"{'✅ success' if fully_passed else '❌ fail'} ({frac_passed*100:.1f}%)")

            trajectory["refinement_attempts"].append({
                "attempt": attempt_count,
                "feedback": feedback,
                "program": code,
                "pass_fraction": frac_passed,
                "test_results": results,
                "passed": fully_passed
            })
            print(f"[TRACE] Logged attempt {attempt_count} (pass={frac_passed*100:.1f}%, passed={fully_passed})", flush=True)
            print(f"[TRACE] → Total attempts logged so far: {len(trajectory['refinement_attempts'])}", flush=True)

            current_program = code
            if fully_passed:
                seed_success = True
                print(f"[SUCCESS {time.strftime('%H:%M:%S')}] ✅ Seed {i} succeeded after {attempt_count} attempts "
                      f"({time.time()-seed_start:.1f}s elapsed)", flush=True)

        if not seed_success:
            print(f"[FAIL {time.strftime('%H:%M:%S')}] ❌ Seed {i} exhausted {max_attempts} attempts "
                  f"({time.time()-seed_start:.1f}s elapsed)", flush=True)
            return trajectory, False

        return trajectory, seed_success

    # ---------- Stage 2: Run seeds in parallel ----------
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] Launching thread pool with {min(np,5)} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=min(np, 5)) as executor:
        futures = [executor.submit(process_seed, i, seed) for i, seed in enumerate(seeds, start=1)]

        SEED_TIMEOUT = 120  # seconds per seed

        # for fut in as_completed(futures, timeout=np * SEED_TIMEOUT):
        #     try:
        #         trajectory, success = fut.result(timeout=SEED_TIMEOUT)
        #         if trajectory:
        #             trajectories.append(trajectory)
        #             overall_success = overall_success or success
        #     except concurrent.futures.TimeoutError:
        #         print(f"[ERROR] Seed thread timed out after {SEED_TIMEOUT} seconds — marking as failure.")
        #         # You must mark something to keep system consistent:
        #         trajectories.append({
        #             "seed_index": -1,
        #             "initial_passed": False,
        #             "initial_pass_fraction": 0.0,
        #             "initial_program": "(timeout)",
        #             "refinement_attempts": []
        #         })
        #     except Exception as e:
        #         print(f"[ERROR] Seed thread crashed: {e}")

        for fut in futures:
            try:
                # Per-seed timeout, not global timeout
                trajectory, success = fut.result(timeout=SEED_TIMEOUT)
                if trajectory:
                    trajectories.append(trajectory)
                    overall_success = overall_success or success

            except concurrent.futures.TimeoutError:
                print(f"[ERROR] Seed thread timed out after {SEED_TIMEOUT}s — marking as failure.")
                trajectories.append({
                    "seed_index": -1,
                    "initial_passed": False,
                    "initial_pass_fraction": 0.0,
                    "initial_program": "(timeout)",
                    "refinement_attempts": []
                })

            except Exception as e:
                print(f"[ERROR] Seed thread crashed: {e}")

    # ---------- Stage 3: Aggregate results ----------
    total_programs = sum(1 + len(t["refinement_attempts"]) for t in trajectories)
    print(f"[SUMMARY {time.strftime('%H:%M:%S')}] Finished task after {time.time()-start_time:.1f}s | "
          f"{len(trajectories)} trajectories, total programs: {total_programs}, success={overall_success}", flush=True)

    return {
        "task_id": task,
        "success": overall_success,
        "trajectories": trajectories,
        "k": total_programs
    }


def extract_code(text: str) -> str:
    """
    Extracts the Python code block from an LLM output string.
    If no ``` fences are found, returns the whole string.
    """
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        return text.strip()


def make_mbpp_test_suite(setup_code: str, test_list: list[str]):
    """
    Executes MBPP test cases and reports fractional correctness and detailed outcomes.
    Returns (fraction_passed, results_list).
    """
    def test_suite(program_code: str):
        env = {}
        try:
            # Load imports and helper code
            exec(setup_code or "", env)
            # Define the candidate program
            exec(program_code, env)
        except Exception as e:
            print(f"[Setup failed] {type(e).__name__}: {e}")
            return 0.0, [("setup", f"⚠️ ERROR: {type(e).__name__}: {e}", "<program setup>")]

        passed, total = 0, len(test_list)
        results = []

        # Execute each test independently
        for i, test in enumerate(test_list, 1):
            try:
                exec(test, env)
                passed += 1
                results.append((i, "✅ PASS", test))
            except AssertionError as e:
                results.append((i, "❌ FAIL", test))
                print("DEBUG: ASSERTION FAILED:", e)
            except Exception as e:
                tb = traceback.format_exception_only(type(e), e)[0].strip()
                results.append((i, f"⚠️ ERROR: {tb}", test))

        frac_passed = passed / total if total > 0 else 0.0
        return frac_passed, results

    return test_suite


def make_apps_test_suite(inputs, outputs, timeout_seconds=5):

    def run(program_code: str):
        passed = 0
        results = []
        coverage_map = {}

        for j, (inp, out) in enumerate(zip(inputs, outputs), start=1):

            r = run_in_subprocess(program_code, inp, timeout=timeout_seconds)

            model_out = r["output"]
            cov = set(r["coverage"])
            err = r["error"]

            coverage_map[j] = cov

            if err is not None:
                results.append((j, f"ERROR: {err}", inp.strip()))
                continue

            gold = out.strip()
            if str(model_out) == gold:
                passed += 1
                results.append((j, "PASS", inp.strip()))
            else:
                results.append((j, "FAIL", f"Expected {gold!r}, Got {model_out!r}"))

        score = passed / len(inputs)
        # return score, results, coverage_map, inputs, outputs
        return score, results

    return run


def make_humaneval_test_suite(tests: str, entry_point: str, language="python", timeout_seconds=10):
    print(f"[DEBUG] make_humaneval_test_suite initialized with language={language}")

    # -----------------------------
    # Helper: Timeout for Python code
    # -----------------------------
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Execution timed out")

    # -----------------------------
    # PYTHON
    # -----------------------------
    def run_python(program_code: str):
        def extract_asserts(tests: str):
            lines = [l.rstrip() for l in tests.splitlines()]
            asserts = []
            i = 0
            while i < len(lines):
                line = lines[i].lstrip()

                if line.startswith("assert "):
                    stmt = line
                    open_brackets = stmt.count("[") - stmt.count("]")

                    # continue consuming lines until brackets are balanced
                    while open_brackets > 0:
                        i += 1
                        if i >= len(lines):
                            break
                        next_line = lines[i].rstrip()
                        stmt += " " + next_line
                        open_brackets += next_line.count("[") - next_line.count("]")

                    asserts.append(stmt)

                i += 1

            return asserts

        print("\n=== DEBUG: Starting run_python ===")
        print("Program code received:\n", program_code)

        env = {}
        try:
            print("DEBUG: Executing program_code...")
            exec(program_code, env)
            print("DEBUG: exec() complete. env keys:", list(env.keys()))

            if entry_point not in env:
                return 0.0, [("[Error]", f"Function '{entry_point}' not defined")]

            print(f"DEBUG: Entry point '{entry_point}' FOUND.")
            candidate = env[entry_point]

            assert_lines = extract_asserts(tests)
            print(f"DEBUG: Extracted {len(assert_lines)} assert lines:")
            for line in assert_lines:
                print("   ASSERT:", line)

            total, passed, results = len(assert_lines), 0, []

            for i, assert_line in enumerate(assert_lines, 1):
                print(f"\nDEBUG: Running assertion #{i}: {assert_line}")
                try:
                    exec(assert_line, {**env, "candidate": candidate})
                    passed += 1
                    print("DEBUG: Assertion PASSED")
                    results.append((i, "✅ PASS", assert_line))
                except TimeoutException:
                    print("DEBUG: TIMEOUT")
                    results.append((i, f"TIMEOUT (> {timeout_seconds}s)", assert_line))
                except AssertionError as e:
                    print("DEBUG: ASSERTION FAILED:", e)
                    results.append((i, "❌ FAIL", assert_line))
                except Exception as e:
                    print("DEBUG: ERROR during assertion:", type(e).__name__, str(e))
                    tb = traceback.format_exception_only(type(e), e)[0].strip()
                    results.append((i, f"⚠️ ERROR: {tb}", assert_line))

            score = passed / total if total else 0.0
            print(f"\nDEBUG: Final score = {score} ({passed}/{total})")
            return score, results

        except Exception as e:
            print("DEBUG: FATAL ERROR during program execution:", e)
            return 0.0, [("[Fatal]", str(e))]

    # -----------------------------
    # C++
    # -----------------------------
    def run_cpp(program_code: str):
        print("\n[DEBUG] Running C++ test suite")

        program_code = program_code.strip()
        first_line = program_code.splitlines()[0].strip().lower()
        if first_line in {"cpp", "c++"}:
            print(f"[DEBUG] Removing leading language tag: {first_line}")
            program_code = "\n".join(program_code.splitlines()[1:])

        program_code = re.sub(
            r'\bint\s+main\s*\([^)]*\)\s*\{(?:[^{}]|\{[^{}]*\})*\}',
            '',
            program_code,
            flags=re.DOTALL
        )

        def wrap_asserts_with_check(test_code: str) -> str:
            test_code = re.sub(r'\bassert\s*\((.*?)\);', r'CHECK(\1);', test_code)
            header = r"""
#include <iostream>
int passed_tests = 0;
int total_tests = 0;
#define CHECK(expr) do { \
    ++total_tests; \
    if (expr) { ++passed_tests; } \
    else { std::cerr << "❌ FAIL: " #expr << std::endl; } \
} while(0)
"""
            footer = r"""
std::cout << "PASS FRACTION: "
          << (100.0 * passed_tests / total_tests)
          << "%" << std::endl;
return 0;
}
"""
            test_code = re.sub(r'\}\s*$', footer, test_code.strip())
            return header + "\n" + test_code

        instrumented_tests = wrap_asserts_with_check(tests)
        full_code = f"{program_code}\n\n{instrumented_tests}"

        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "solution.cpp")
            bin_path = os.path.join(tmpdir, "solution.out")

            with open(cpp_path, "w") as f:
                f.write(full_code)

            compile_cmd = ["g++", "-std=c++17", cpp_path, "-o", bin_path]
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)

            if compile_result.returncode != 0:
                return 0.0, [("❌ COMPILE ERROR", compile_result.stderr.strip())]

            try:
                start = time.time()
                run_result = subprocess.run(
                    [bin_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                elapsed = time.time() - start
            except subprocess.TimeoutExpired:
                return 0.0, [("TIMEOUT", f"Execution exceeded {timeout_seconds}s")]

            match = re.search(r'PASS FRACTION:\s*([\d.]+)%', run_result.stdout)
            pass_fraction = float(match.group(1)) / 100.0 if match else (
                1.0 if run_result.returncode == 0 else 0.0
            )

            if run_result.returncode == 0:
                return pass_fraction, [("✅ PASS", run_result.stdout.strip() or "All tests passed.")]
            else:
                return pass_fraction, [("❌ RUNTIME ERROR", run_result.stderr.strip() or run_result.stdout.strip())]

    # -----------------------------
    # JAVA
    # -----------------------------
    def run_java(program_code: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "Main.java")
            test_code = (
                "public class Main {\n"
                f"{program_code}\n"
                "public static void main(String[] args) {\n"
                f"{tests}\nSystem.out.println(\"ALL TESTS PASSED\");\n}}\n"
            )
            with open(src_path, "w") as f:
                f.write(test_code)

            compile_result = subprocess.run(
                ["javac", src_path], capture_output=True, text=True
            )
            if compile_result.returncode != 0:
                return 0.0, [("❌ COMPILE ERROR", compile_result.stderr.strip())]

            try:
                run_result = subprocess.run(
                    ["java", "-cp", tmpdir, "Main"],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
            except subprocess.TimeoutExpired:
                return 0.0, [("TIMEOUT", f"Execution exceeded {timeout_seconds}s")]

            if run_result.returncode == 0:
                return 1.0, [("✅ PASS", run_result.stdout.strip())]
            else:
                return 0.0, [("❌ RUNTIME ERROR", run_result.stderr.strip())]

    # -----------------------------
    # TEST SUITE DISPATCHER
    # -----------------------------
    def test_suite(program_code: str):
        if language == "python":
            return run_python(program_code)
        elif language == "cpp":
            return run_cpp(program_code)
        elif language == "java":
            return run_java(program_code)
        else:
            print(f"[Warning] Language {language} not supported yet.")
            return 0.0, []

    return test_suite


def main():
    parser = argparse.ArgumentParser(description="Run self-repair on MBPP or HumanEval.")
    parser.add_argument(
        "--dataset",
        choices=["mbpp", "humaneval", "humaneval-x", "apps"],
        required=True
    )
    parser.add_argument("--np", type=int, default=5, help="Number of seeds")
    parser.add_argument("--nf", type=int, default=1, help="Feedback per failed program")
    parser.add_argument("--nr", type=int, default=1, help="Repairs per feedback")
    parser.add_argument("--max_attempts", type=int, default=10, help="Maximum refinement iterations per seed")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit tasks for debugging")
    parser.add_argument(
        "--language",
        choices=["python", "cpp", "java", "go", "js"],
        default="python",
        help="Language subset for HumanEval-X (ignored for MBPP and standard HumanEval)."
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Specific task IDs to run, e.g. HumanEval/16 HumanEval/35"
    )
    parser.add_argument(
        "--mode", choices=["standard", "iterative"],
        default="standard",
        help="Choose repair strategy: 'standard' (nf/nr loops) or 'iterative' (rolling self-correction)"
    )
    parser.add_argument(
        "--difficulty",
        choices=["introductory", "interview", "competition", "all"],
        default="interview",
        help="Filter APPS dataset by difficulty level"
    )
    parser.add_argument(
        "--refine_mode",
        choices=[
            "direct",
            "critique+refine",
            "critique+history+refine",
            "critique+antiunify+refine",
            "critique+execresults+refine",
            "critique+antiunify+execclusters+refine"
        ],
        default="critique+refine",
        help=(
            "Refinement strategy to use in iterative mode: "
            "'direct' (no feedback), 'critique+refine' (feedback on current code), "
            "or 'critique+history+refine' (feedback using full past trajectory)."
        )
    )
    parser.add_argument(
        "--start_task_id",
        type=int, default=None,
        help="Start running MBPP from this task_id onward.")

    args = parser.parse_args()

    model = Model(model_name=args.model_name, temperature=args.temperature)

    # ---------- Prepare output file early ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"results/results_{args.model_name}_{args.dataset}_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")

    def write_result(result):
        """Write one result to the file immediately and flush."""
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())  # ensure data is physically written to disk

    # ---------- Select dataset ----------
    if args.dataset == "mbpp":
        ds = load_dataset("Muennighoff/mbpp", "sanitized")["test"]
        selected = ds if args.task_ids is None else [ex for ex in ds if str(ex["task_id"]) in args.task_ids]
        print(f"Loaded MBPP with {len(selected)} test tasks.")

        for i, ex in enumerate(tqdm(selected, desc="Running MBPP tasks")):
            task_id = ex["task_id"]

            # --- Skip until start_task_id ---
            print(f"Start index: {args.start_task_id}")
            print(f"i: {i}")
            if args.start_task_id is not None and i < args.start_task_id:
                continue

            if args.max_tasks and i >= args.max_tasks:
                break

            desc = ex["prompt"]
            setup = ex["test_imports"]
            tests = ex["test_list"]
            ref_code = ex.get("code", "")

            func_name, param_count = extract_function_signature(tests)
            print(f"Tests: {tests}", flush=True)
            print(f"Func name: {func_name}", flush=True)

            if func_name:
                desc += f"\nMAKE SURE THE FUNCTION IS NAMED `{func_name}`."

            if param_count is not None:
                desc += f"\nTHE FUNCTION MUST TAKE EXACTLY {param_count} PARAMETER(S)."

            test_suite = make_mbpp_test_suite(setup, tests)
            if args.mode == "iterative":
                result = run_self_repair_iterative(
                    task=desc, model=model, test_suite=test_suite,
                    np=args.np, max_attempts=args.max_attempts, mode=args.refine_mode
                )
            else:
                result = run_self_repair(
                    task=desc, model=model, test_suite=test_suite,
                    np=args.np, nf=args.nf, nr=args.nr
                )
            result["dataset"] = "mbpp"
            result["task_id"] = task_id
            write_result(result)

    elif args.dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval")["test"]
        selected = ds if args.task_ids is None else [ex for ex in ds if ex["task_id"] in args.task_ids]
        print(f"Loaded HumanEval with {len(selected)} test tasks.")

        for i, ex in enumerate(tqdm(selected, desc="Running HumanEval tasks")):
            # --- Skip until start_task_id ---
            print(f"Start index: {args.start_task_id}")
            print(f"i: {i}")
            if args.start_task_id is not None and i < args.start_task_id:
                continue

            if args.max_tasks and i >= args.max_tasks:
                break
            task_id, prompt, tests, entry_point = (
                ex["task_id"], ex["prompt"], ex["test"], ex["entry_point"]
            )
            test_suite = make_humaneval_test_suite(tests, entry_point)
            if args.mode == "iterative":
                result = run_self_repair_iterative(
                    task=prompt, model=model, test_suite=test_suite,
                    np=args.np, max_attempts=args.max_attempts, mode=args.refine_mode
                )
            else:
                result = run_self_repair(
                    task=prompt, model=model, test_suite=test_suite,
                    np=args.np, nf=args.nf, nr=args.nr
                )
            result["dataset"] = "humaneval"
            result["task_id"] = task_id
            write_result(result)

    elif args.dataset == "humaneval-x":
        ds = load_dataset("THUDM/humaneval-x", args.language)["test"]
        selected = ds if args.task_ids is None else [ex for ex in ds if ex["task_id"] in args.task_ids]
        print(f"Loaded HumanEval-X ({args.language}) with {len(selected)} test tasks.")

        for i, ex in enumerate(tqdm(selected, desc=f"Running HumanEval-X ({args.language}) tasks")):
            if args.max_tasks and i >= args.max_tasks:
                break
            task_id, prompt, declaration, ref_code, tests = (
                ex["task_id"], ex["prompt"], ex["declaration"],
                ex["canonical_solution"], ex["test"]
            )

            test_suite = make_humaneval_test_suite(tests, declaration, language=args.language)

            if args.mode == "iterative":
                result = run_self_repair_iterative(
                    task=prompt, model=model, test_suite=test_suite,
                    np=args.np, max_attempts=args.max_attempts, mode=args.refine_mode
                )
            else:
                result = run_self_repair(
                    task=prompt, model=model, test_suite=test_suite,
                    np=args.np, nf=args.nf, nr=args.nr
                )

            result["dataset"] = f"humaneval-x-{args.language}"
            result["task_id"] = task_id
            write_result(result)

    elif args.dataset == "apps":
        print("[INFO] Loading APPS dataset split: test[:5000]...", flush=True)
        ds = load_dataset("codeparrot/apps", split="test[:5000]")
        print(f"[INFO] Loaded raw dataset with {len(ds)} entries.", flush=True)

        if args.difficulty != "all":
            print(f"[INFO] Filtering for difficulty = '{args.difficulty}'...", flush=True)
            ds = ds.filter(lambda ex: ex.get("difficulty", "") == args.difficulty)
        else:
            print("[INFO] Using all difficulty levels.", flush=True)

        print("[INFO] Selecting up to 200 samples...", flush=True)
        ds = ds.select(range(min(200, len(ds))))
        print(f"[INFO] After selection: {len(ds)} total examples.", flush=True)

        selected = ds if args.task_ids is None else [
            ex for ex in ds if str(ex["problem_id"]) in args.task_ids
        ]
        print(f"[INFO] Final number of tasks to run: {len(selected)}", flush=True)

        for i, ex in enumerate(tqdm(selected, desc="Running APPS tasks")):
            if args.max_tasks and i >= args.max_tasks:
                print(f"[INFO] Reached max_tasks limit ({args.max_tasks}), stopping early.", flush=True)
                break

            problem_id = ex["problem_id"]
            prompt = ex["question"]
            difficulty = ex.get("difficulty", "unknown")

            print(f"\n[INFO] --- Task {i+1}/{len(selected)} ---", flush=True)
            print(f"[INFO] Problem ID: {problem_id}, Difficulty: {difficulty}", flush=True)

            # Parse solutions and I/O pairs safely
            try:
                solutions = json.loads(ex.get("solutions", "[]"))
            except Exception:
                solutions = []

            try:
                io_pairs = json.loads(ex.get("input_output", "{}"))
                inputs, outputs = io_pairs.get("inputs", []), io_pairs.get("outputs", [])
                print(f"[DEBUG] Parsed {len(inputs)} input/output pairs.", flush=True)
            except Exception as e:
                print(f"[WARN] Could not parse I/O for {problem_id}: {e}", flush=True)
                inputs, outputs = [], []

            # Limit excessive test cases for speed
            # if len(inputs) > 30:
            #     print(f"[WARN] Truncating {len(inputs)} → 30 test pairs for problem {problem_id}")
            #     inputs, outputs = inputs[:30], outputs[:30]

            # Skip invalid or empty examples
            if not inputs or not outputs:
                print(f"[Warning] Skipping problem {problem_id} (no I/O pairs)")
                continue

            # Create APPS-style test suite (runs programs with stdin/stdout)
            print("[INFO] Creating APPS test suite with timeout=10s...", flush=True)
            test_suite = make_apps_test_suite(inputs, outputs, timeout_seconds=10)

            # Run repair depending on mode
            print(f"[INFO] Starting {'iterative' if args.mode == 'iterative' else 'standard'} repair...", flush=True)
            try:
                if args.mode == "iterative":
                    result = run_self_repair_iterative(
                        task=prompt,
                        model=model,
                        test_suite=test_suite,
                        np=args.np,
                        max_attempts=args.max_attempts,
                        mode=args.refine_mode
                    )
                else:
                    result = run_self_repair(
                        task=prompt,
                        model=model,
                        test_suite=test_suite,
                        np=args.np,
                        nf=args.nf,
                        nr=args.nr,
                    )
                print(f"[INFO] Completed repair for problem {problem_id}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception during repair of {problem_id}: {e}", flush=True)
                continue

            result.update({
                "dataset": "apps",
                "problem_id": problem_id,
                "difficulty": difficulty,
            })
            print(f"[INFO] Writing results for {problem_id}...", flush=True)
            write_result(result)

    f.close()
    print(f"\n✅ Completed {args.dataset}. Results streamed to {out_path}")


if __name__ == "__main__":
    main()
