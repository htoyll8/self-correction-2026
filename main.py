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

        current_program = seed

    #     # ---------- Stage 2: Iterative self-repair ----------
    #     for iteration in range(max_iters):

    #         if mode == "critique+refine":
    #             feedback = model.generate_feedback(task, current_program, temperature=0)
    #             candidates = [model.refine(task, current_program, feedback, temperature=0)
    #                           for _ in range(nr)]
    #         elif mode == "direct":
    #             candidates = [model.refine(task, current_program, temperature=0)
    #                           for _ in range(nr)]
    #         else:
    #             raise ValueError("Unknown refinement mode")

    #         all_programs.extend(candidates)

    #         # Evaluate all repair candidates
    #         for candidate in candidates:
    #             if test_suite(candidate):
    #                 success = True
    #                 break  # stop repairing this seed if success
    #         if success:
    #             break  # exit iteration loop once a pass is found

    #         # choose next program to repair (greedy or best according to some heuristic)
    #         current_program = select_next_candidate(candidates) # TODO: Define function.

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


# Load one MBPP task
task_id, task_description, setup_code, tests = load_mbpp_task(0)
model = Model(model_name="gpt-4o-mini", temperature=0)
test_suite = make_mbpp_test_suite(setup_code, tests)
result = run_self_repair(
    task=task_description,
    model=model,
    test_suite=test_suite,
    np=5,
    nf=1,
    nr=1
)
print(f"Result: {result}")


# # Load one humaneval-x task
# task_id, prompt, tests, entry_point = load_humaneval_task()
# model = Model(model_name="gpt-4o-mini", temperature=0)
# result = run_self_repair(
#     task=prompt,
#     model=model,
#     test_suite=tests,
#     np=5,
#     nf=1,
#     nr=1
# )
# print(f"Result: {result}")
