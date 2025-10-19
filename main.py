import argparse
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

    all_programs = []  # all generated code (for pass@k accounting)
    success = False

    # ---------- Stage 1: Initial generation ----------
    seeds = model.generate(task_description=task, n=np, temperature=0.7)  # diversity via temp>0
    all_programs.extend(seeds)

    for seed in seeds:
        # Evaluate initial seed
        if test_suite(seed):
            success = True
            continue  # no repair needed for this seed

    # ---------- Stage 3: Return aggregated results ----------
    return {
        "task_id": task,
        "success": success,              # True if any seed/repair passed
        "all_programs": all_programs,    # for pass@k counting
        "k": len(all_programs)           # total samples = np + np*nf*nr (approx)
    }


def load_mbpp_task(task_index=0):
    ds = load_dataset("Muennighoff/mbpp", "sanitized")
    example = ds["test"][task_index]
    task_description = example["prompt"]
    setup_code = example["test_imports"]
    tests = example["test_list"]
    return example["task_id"], task_description, setup_code, tests


def load_humaneval_task(task_index=0):
    ds = load_dataset("openai/openai_humaneval")
    example = ds["test"][task_index]
    task_id = example["task_id"]
    prompt = example["prompt"]
    tests = example["test"]
    entry_point = example["entry_point"]
    return task_id, prompt, tests, entry_point


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
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument("--np", type=int, default=5, help="Number of seeds")
    parser.add_argument("--nf", type=int, default=1, help="Feedback per failed program")
    parser.add_argument("--nr", type=int, default=1, help="Repairs per feedback")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model = Model(model_name="gpt-4o-mini", temperature=0)

    if args.dataset == "mbpp":
        task_id, task_description, setup_code, tests = load_mbpp_task(args.task_index)
        test_suite = make_mbpp_test_suite(setup_code, tests)
        result = run_self_repair(
            task=task_description,
            model=model,
            test_suite=test_suite,
            np=5,
            nf=1,
            nr=1
        )

    elif args.dataset == "humaneval":
        task_id, prompt, tests, entry_point = load_humaneval_task()
        test_suite = make_humaneval_test_suite(tests, entry_point)
        result = run_self_repair(
            task=prompt,
            model=model,
            test_suite=test_suite,
            np=5,
            nf=1,
            nr=1
        )

    print(f"\nDataset: {args.dataset}")
    print(f"Model: {model.model_name}")
    print(f"Task ID: {task_id}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
