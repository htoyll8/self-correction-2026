import re
import json
import argparse
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


def run_self_repair(task, model, test_suite, np=5, max_attempts=10,
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
        passed = test_suite(seed_code)
        print(f"Seed {i} → passed? {passed}")

        trajectory = {
            "seed_index": i,
            "initial_program": seed_code,
            "initial_passed": passed,
            "attempts": []
        }
        trajectories.append(trajectory)

        if passed:
            success = True
            continue

        current_program = seed_code
        attempt_count = 0  # total repairs tried for this seed
        seed_success = False

        # ---------- Stage 2: Iterative repair ----------
        while attempt_count < max_attempts and not seed_success:
            for f_i in range(nf):
                if attempt_count >= max_attempts:
                    break

                feedback = None
                if mode == "critique+refine":
                    feedback = model.generate_feedback(task, current_program, temperature=0)

                for r_i in range(1, nr + 1):
                    if attempt_count >= max_attempts:
                        break

                    attempt_count += 1
                    print(f"  Attempt {attempt_count} (feedback {f_i+1}, repair {r_i}) → ", end="")

                    print(f"Task in iterative refinement: {task}")

                    if mode == "critique+refine":
                        refined = model.refine(task, current_program, feedback, temperature=0)
                    elif mode == "direct":
                        refined = model.refine(task, current_program, temperature=0)
                    else:
                        raise ValueError("Unknown refinement mode")

                    code = extract_code(refined)
                    passed = test_suite(code)

                    trajectory["attempts"].append({
                        "attempt": attempt_count,
                        "feedback_index": f_i + 1,
                        "repair_index": r_i,
                        "feedback": feedback,
                        "program": code,
                        "passed": passed
                    })

                    print("✅ success" if passed else "failed")

                    if passed:
                        seed_success = True
                        success = True
                        print(f"  ✅ Seed {i} succeeded after {attempt_count} attempts")
                        break

                    current_program = code  # next repair builds on last attempt

                if seed_success:
                    break

    # ---------- Stage 3: Return aggregated results ----------
    total_programs = sum(1 + len(t["attempts"]) for t in trajectories)

    return {
        "task_id": task,
        "success": success,
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
    def test_suite(program_code: str) -> bool:
        """
        Executes the candidate program and returns True if all tests pass.
        """
        env = {}
        try:
            # Run imports (e.g., 'import math')
            exec(setup_code or "", env)
            # Define the generated program
            exec(program_code, env)
            # Run all test assertions
            for test in test_list:
                exec(test, env)
            return True
        except Exception as e:
            print(f"[Test failed] {type(e).__name__}: {e}")
            return False

    return test_suite


def make_humaneval_test_suite(tests: str, entry_point: str):
    """
    Returns a callable test function that runs the HumanEval test string.
    """
    def test_suite(program_code: str) -> bool:
        env = {}
        try:
            # 1. Define the functions in env
            exec(program_code, env)

            # 2. Ensure the expected function is defined
            if entry_point not in env:
                return False

            # 3. Define the check() function from the test string
            exec(tests, env)

            # 4. Call check(candidate)
            env["check"](env[entry_point])
            return True
        except Exception:
            return False

    return test_suite


def main():
    parser = argparse.ArgumentParser(description="Run self-repair on MBPP or HumanEval.")
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
    parser.add_argument("--np", type=int, default=5, help="Number of seeds")
    parser.add_argument("--nf", type=int, default=1, help="Feedback per failed program")
    parser.add_argument("--nr", type=int, default=1, help="Repairs per feedback")
    parser.add_argument("--max_attempts", type=int, default=10, help="Maximum refinement iterations per seed")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit tasks for debugging")
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,   # if not provided, we’ll run all tasks
        help="Specific task IDs to run, e.g. HumanEval/16 HumanEval/35"
    )
    args = parser.parse_args()

    model = Model(model_name=args.model_name, temperature=args.temperature)
    results = []

    if args.dataset == "mbpp":
        ds = load_dataset("Muennighoff/mbpp", "sanitized")["test"]
        selected = ds if args.task_ids is None else [ex for ex in ds if str(ex["task_id"]) in args.task_ids]
        print(f"Loaded MBPP with {len(selected)} test tasks.")
        for i, ex in enumerate(tqdm(selected, desc="Running MBPP tasks")):
            if args.max_tasks and i >= args.max_tasks:
                break
            task_id, desc, setup, tests, ref_code  = (
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
            result = run_self_repair(
                task=desc,
                model=model,
                test_suite=test_suite,
                np=args.np,
                nf=args.nf,
                nr=args.nr,
                max_attempts=args.max_attempts
            )
            result["dataset"] = "mbpp"
            result["task_id"] = task_id
            results.append(result)

    elif args.dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval")["test"]
        selected = ds if args.task_ids is None else [ex for ex in ds if ex["task_id"] in args.task_ids]
        print(f"Loaded HumanEval with {len(selected)} test tasks.")
        for i, ex in enumerate(tqdm(selected, desc="Running HumanEval tasks")):
            if args.max_tasks and i >= args.max_tasks:
                break
            task_id, prompt, tests, entry_point = (
                    ex["task_id"],
                    ex["prompt"],
                    ex["test"],
                    ex["entry_point"]
                )
            test_suite = make_humaneval_test_suite(tests, entry_point)
            result = run_self_repair(
                task=prompt,
                model=model,
                test_suite=test_suite,
                np=args.np,
                nf=args.nf,
                nr=args.nr,
                max_attempts=args.max_attempts
            )
            result["dataset"] = "humaneval"
            result["task_id"] = task_id
            results.append(result)

    # ---------- Save results ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"results/results_{args.model_name}_{args.dataset}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n\n")

    print(f"\n✅ Completed {len(results)} tasks from {args.dataset}.")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
