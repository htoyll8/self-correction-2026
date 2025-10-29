import re
import os
import json
import time
import signal
import argparse
import traceback
import tempfile
import subprocess
import traceback
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

from model import Model


def extract_function_name_from_code(code: str) -> str | None:
    """
    Extracts the function name from a reference MBPP code snippet.
    Example:
      'def similar_elements(test_tup1, test_tup2):' -> 'similar_elements'
    """
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
    return match.group(1) if match else None


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
        frac_passed, results = test_suite(seed_code)
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


def run_self_repair_iterative(task, model, test_suite, np=5, max_attempts=10, mode="critique+refine"):
    """
    Perform iterative self-repair with multiple independent seeds (np).
    Each seed is refined iteratively until success or reaching max_attempts.

    Args:
        task: str — natural-language description of the problem
        model: Model — code-generation LLM interface
        test_suite: callable(program_code) -> (frac_passed, results)
        np: int — number of initial seed programs
        max_attempts: int — maximum repair attempts per seed
        mode: str — "critique+refine" (generate feedback before each fix) or "direct" (refine without feedback)

    Returns:
        dict summarizing all seed trajectories.
    """

    trajectories = []
    overall_success = False

    # ---------- Stage 1: Initial generation ----------
    seeds = model.generate(task_description=task, n=np, temperature=0.7)

    for i, seed in enumerate(seeds, start=1):
        seed_code = extract_code(seed)
        frac_passed, results = test_suite(seed_code)
        print(f"Seed {i} → passed {frac_passed*100:.1f}% of tests")

        fully_passed = frac_passed == 1.0
        trajectory = {
            "seed_index": i,
            "initial_program": seed_code,
            "initial_passed": fully_passed,
            "initial_pass_fraction": frac_passed,
            "initial_test_results": results,
            "refinement_attempts": []
        }
        trajectories.append(trajectory)

        if fully_passed:
            overall_success = True
            continue

        current_program = seed_code
        attempt_count = 0
        seed_success = False

        # ---------- Stage 2: Iterative refinement ----------
        while attempt_count < max_attempts and not seed_success:
            attempt_count += 1
            print(f"  Seed {i}, Attempt {attempt_count} → ", end="")

            feedback = None
            if mode == "critique+refine":
                feedback = model.generate_feedback(task, current_program, temperature=0)

            if mode == "critique+refine":
                refined = model.refine(task, current_program, feedback, temperature=0)
            elif mode == "direct":
                refined = model.refine(task, current_program, temperature=0)
            else:
                raise ValueError("Unknown refinement mode")

            code = extract_code(refined)
            frac_passed, results = test_suite(code)
            fully_passed = frac_passed == 1.0

            print(f"✅ success ({frac_passed*100:.1f}%)"
                  if fully_passed else f"failed ({frac_passed*100:.1f}%)")

            trajectory["refinement_attempts"].append({
                "attempt": attempt_count,
                "feedback": feedback,
                "program": code,
                "pass_fraction": frac_passed,
                "test_results": results,
                "passed": fully_passed
            })

            current_program = code
            if fully_passed:
                seed_success = True
                overall_success = True
                print(f"  ✅ Seed {i} succeeded after {attempt_count} attempts")

        if not seed_success:
            print(f"  ❌ Seed {i} exhausted {max_attempts} attempts without success")

    # ---------- Stage 3: Return aggregated results ----------
    total_programs = sum(1 + len(t["refinement_attempts"]) for t in trajectories)

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
            except AssertionError:
                results.append((i, "❌ FAIL", test))
            except Exception as e:
                tb = traceback.format_exception_only(type(e), e)[0].strip()
                results.append((i, f"⚠️ ERROR: {tb}", test))

        frac_passed = passed / total if total > 0 else 0.0
        return frac_passed, results

    return test_suite


def make_apps_test_suite(inputs, outputs, timeout_seconds=10):
    """
    Returns a callable test function that runs an APPS-style program.
    Each input is fed via stdin, and stdout is compared to expected output.
    """
    def run(program_code: str):
        results, passed = [], 0

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.py")
            with open(src_path, "w") as f:
                f.write(program_code)

            for j, (inp, out) in enumerate(zip(inputs, outputs), 1):
                try:
                    proc = subprocess.run(
                        ["python3", src_path],
                        input=inp.encode(),
                        capture_output=True,
                        timeout=timeout_seconds,
                    )
                    pred = proc.stdout.decode().strip()
                    gold = out.strip()
                    if pred == gold:
                        passed += 1
                        results.append((j, "✅ PASS", inp.strip()))
                    else:
                        results.append((j, "❌ FAIL", f"Expected: {gold!r}, Got: {pred!r}"))
                except subprocess.TimeoutExpired:
                    results.append((j, f"TIMEOUT (> {timeout_seconds}s)", inp.strip()))
                except Exception as e:
                    results.append((j, f"⚠️ ERROR: {e}", inp.strip()))

        total = len(inputs)
        return (passed / total if total else 0.0), results

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
        env = {}
        try:
            exec(program_code, env)
            if entry_point not in env:
                return 0.0, [("[Error]", f"Function '{entry_point}' not defined")]
            candidate = env[entry_point]

            assert_lines = [
                line.strip() for line in tests.splitlines()
                if line.strip().startswith("assert ")
            ]
            total, passed, results = len(assert_lines), 0, []

            signal.signal(signal.SIGALRM, timeout_handler)

            for i, assert_line in enumerate(assert_lines, 1):
                try:
                    signal.alarm(timeout_seconds)
                    exec(assert_line, {**env, "candidate": candidate})
                    signal.alarm(0)
                    passed += 1
                    results.append((i, "✅ PASS", assert_line))
                except TimeoutException:
                    results.append((i, f"TIMEOUT (> {timeout_seconds}s)", assert_line))
                    signal.alarm(0)
                except AssertionError:
                    results.append((i, "❌ FAIL", assert_line))
                    signal.alarm(0)
                except Exception as e:
                    tb = traceback.format_exception_only(type(e), e)[0].strip()
                    results.append((i, f"⚠️ ERROR: {tb}", assert_line))
                    signal.alarm(0)

            signal.alarm(0)
            return (passed / total if total else 0.0), results

        except Exception as e:
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
    parser.add_argument("--language", choices=["python", "cpp", "java", "go", "js"], default="python",
                        help="Language subset for HumanEval-X (ignored for MBPP and standard HumanEval).")
    parser.add_argument("--task_ids", nargs="+", default=None,
                        help="Specific task IDs to run, e.g. HumanEval/16 HumanEval/35")
    parser.add_argument("--mode", choices=["standard", "iterative"], default="standard",
                        help="Choose repair strategy: 'standard' (nf/nr loops) or 'iterative' (rolling self-correction)")
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
            if args.max_tasks and i >= args.max_tasks:
                break

            task_id, desc, setup, tests, ref_code = (
                ex["task_id"],
                ex["prompt"],
                ex["test_imports"],
                ex["test_list"],
                ex.get("code", "")
            )

            func_name = extract_function_name_from_code(ref_code)
            if func_name:
                desc += f"\n\nThe function should be named `{func_name}`."

            test_suite = make_mbpp_test_suite(setup, tests)
            if args.mode == "iterative":
                result = run_self_repair_iterative(
                    task=desc, model=model, test_suite=test_suite,
                    np=args.np, max_attempts=args.max_attempts, mode="critique+refine"
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
            if args.max_tasks and i >= args.max_tasks:
                break
            task_id, prompt, tests, entry_point = (
                ex["task_id"], ex["prompt"], ex["test"], ex["entry_point"]
            )
            test_suite = make_humaneval_test_suite(tests, entry_point)
            if args.mode == "iterative":
                result = run_self_repair_iterative(
                    task=prompt, model=model, test_suite=test_suite,
                    np=args.np, max_attempts=args.max_attempts, mode="critique+refine"
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
                    np=args.np, max_attempts=args.max_attempts, mode="critique+refine"
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
        # ds = load_dataset("codeparrot/apps", split="test")
        ds = load_dataset("codeparrot/apps", split="test[:200]")
        selected = ds if args.task_ids is None else [
            ex for ex in ds if str(ex["problem_id"]) in args.task_ids
        ]
        print(f"Loaded APPS with {len(selected)} test problems.")

        for i, ex in enumerate(tqdm(selected, desc="Running APPS tasks")):
            if args.max_tasks and i >= args.max_tasks:
                break

            problem_id = ex["problem_id"]
            prompt = ex["question"]
            difficulty = ex.get("difficulty", "unknown")

            # Parse solutions and I/O pairs safely
            try:
                solutions = json.loads(ex.get("solutions", "[]"))
            except Exception:
                solutions = []

            try:
                io_pairs = json.loads(ex.get("input_output", "{}"))
                inputs, outputs = io_pairs.get("inputs", []), io_pairs.get("outputs", [])
            except Exception:
                inputs, outputs = [], []

            # Skip invalid or empty examples
            if not inputs or not outputs:
                print(f"[Warning] Skipping problem {problem_id} (no I/O pairs)")
                continue

            # Create APPS-style test suite (runs programs with stdin/stdout)
            test_suite = make_apps_test_suite(inputs, outputs, timeout_seconds=10)

            # Run repair depending on mode
            if args.mode == "iterative":
                result = run_self_repair_iterative(
                    task=prompt,
                    model=model,
                    test_suite=test_suite,
                    np=args.np,
                    max_attempts=args.max_attempts,
                    mode="critique+refine",
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

            result.update({
                "dataset": "apps",
                "problem_id": problem_id,
                "difficulty": difficulty,
            })
            write_result(result)

        f.close()
    print(f"\n✅ Completed {args.dataset}. Results streamed to {out_path}")


if __name__ == "__main__":
    main()
