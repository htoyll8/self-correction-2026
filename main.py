import re
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

from model import Model


def run_self_repair(task, model, test_suite, np=5, max_iters=10,
                    nf=1, nr=1, mode="critique+refine"):
    """
    task: natural-language description of the problem
    model: code-generation LLM interface
    test_suite: callable that returns True if program passes all tests
    np: number of initial seeds
    nf: number of feedback messages per failed program
    nr: number of repairs per feedback
    max_iters: maximum repair depth per seed
    mode: "critique+refine" or "direct"
    """

    trajectories = []  # all generated code (for pass@k accounting)
    success = False

    # ---------- Stage 1: Initial generation ----------
    seeds = model.generate(task_description=task, n=np, temperature=0.7)  # diversity via temp>0

    for i, seed in enumerate(seeds, 1):
        # Evaluate initial seed
        seed_code = extract_code(seed)
        passed = test_suite(seed_code)
        print(f"Seed {i} → passed? {passed}")

        trajectory = {
            "seed_index": i,
            "initial_program": seed_code,
            "initial_passed": passed,
            "iterations": []
        }

        trajectories.append(trajectory)

        if passed:
            success = True
            continue  # no repair needed for this seed

        # ---------- Stage 2: Iterative self-repair ----------
        current_program = seed
        for iteration in range(1, max_iters + 1):
            iteration_record = {"iteration": iteration, "feedback_groups": []}
            print(f"  Starting iteration {iteration} for seed {i}")

            feedback_group = {}

            if mode == "critique+refine":
                feedback = model.generate_feedback(task, current_program, temperature=0)
                feedback_group["feedback"] = feedback
            else:
                feedback_group["feedback"] = None

            feedback_group["repairs"] = []

            # retry up to nr times *based on the previous repair
            for attempt in range(1, nr + 1):
                refined = (
                    model.refine(task, current_program, feedback, temperature=0)
                    if mode == "critique+refine"
                    else model.refine(task, current_program, temperature=0)
                )
                code = extract_code(refined)
                passed = test_suite(code)
                feedback_group["repairs"].append({
                    "attempt": attempt,
                    "program": code,
                    "passed": passed
                })
                print(f"    Attempt {attempt} → passed? {passed}")

                if passed:
                    success = True
                    break  # stop retrying; program fixed
                else:
                    # update current_program to the latest failed repair
                    current_program = code

            iteration_record["feedback_groups"].append(feedback_group)
            trajectories[-1]["iterations"].append(iteration_record)

            if success:
                print(f"  ✅ Success at iteration {iteration}")
                break

    # ---------- Stage 3: Return aggregated results ----------
    total_programs = sum(
        1 + sum(
            len(fg["repairs"])
            for it in t["iterations"]
            for fg in it["feedback_groups"]
        )
        for t in trajectories
    )

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
            exec(setup_code, env)
            # Define the generated program
            exec(program_code, env)
            # Run all test assertions
            for test in test_list:
                exec(test, env)
            return True
        except Exception:
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
            task_id, desc, setup, tests = (
                ex["task_id"],
                ex["prompt"],
                ex["test_imports"],
                ex["test_list"],
            )
            test_suite = make_mbpp_test_suite(setup, tests)
            result = run_self_repair(
                task=desc,
                model=model,
                test_suite=test_suite,
                np=args.np,
                nf=args.nf,
                nr=args.nr
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
                np=5,
                nf=1,
                nr=1
            )
            result["dataset"] = "humaneval"
            result["task_id"] = task_id
            results.append(result)

    # ---------- Save results ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"results/results_{args.model_name}_{args.dataset}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            json.dump(r, f, indent=2, ensure_ascii=False)
            f.write("\n\n")

    print(f"\n✅ Completed {len(results)} tasks from {args.dataset}.")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
