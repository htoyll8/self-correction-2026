import ast
import json
import difflib
import numpy as np
from zss import Node, simple_distance
from datasets import load_dataset

# ====== Load APPS dataset (test split) ======
print("📦 Loading APPS dataset...")
apps = load_dataset("codeparrot/apps", split="test")

# Build lookup table: {normalized question start → [solutions]}
solution_lookup = {}
for ex in apps:
    q_text = ex["question"].strip()
    sols_raw = ex.get("solutions", "")
    sols = []
    if sols_raw:
        try:
            sols = json.loads(sols_raw)
        except json.JSONDecodeError:
            sols = []
    if sols:
        solution_lookup[q_text[:80]] = sols

print(f"✅ Loaded {len(solution_lookup)} problems with valid solutions")


def ast_to_zss(node):
    """Convert Python AST node into zss-compatible Node recursively."""
    if not isinstance(node, ast.AST):
        # Leaf node (string, number, etc.)
        return Node(repr(node))

    # Use class name as label, e.g., 'FunctionDef', 'If', etc.
    root = Node(node.__class__.__name__)

    # Iterate through all fields of this AST node
    for field_name, value in ast.iter_fields(node):
        if isinstance(value, list):
            # Lists of child nodes
            for item in value:
                if isinstance(item, ast.AST):
                    root.addkid(ast_to_zss(item))
                else:
                    root.addkid(Node(repr(item)))
        elif isinstance(value, ast.AST):
            root.addkid(ast_to_zss(value))
        else:
            # Primitive attribute
            root.addkid(Node(repr(value)))

    return root


def normalized_ast_distance(code_a, code_b):
    """Compute normalized tree edit distance between two code snippets."""
    try:
        a_tree = ast_to_zss(ast.parse(code_a))
        b_tree = ast_to_zss(ast.parse(code_b))
        print(f"a_tree: {a_tree}")
        print(f"b_tree: {b_tree}")
        dist = simple_distance(a_tree, b_tree)
        norm = dist / (len(ast.dump(ast.parse(code_a))) + len(ast.dump(ast.parse(code_b))) + 1)
        return 1 - norm  # similarity (1 = identical)
    except Exception as e:
        print(f"Exception: {e}")
        return None  # skip unparsable code


# ====== Helper: fuzzy match ======
def match_solution(task_id_text):
    for q_start, sols in solution_lookup.items():
        if task_id_text.strip().startswith(q_start[:40]):
            return sols
    return None


# ====== Compute Similarities ======
input_path = "results/results_gpt-4o-mini_apps_2025-11-11_20-06-13.jsonl"
# input_path = "results/results_gpt-5-2025-08-07_apps_2025-11-02_16-25-51.jsonl"s

# ====== Compute AST-based Similarities ======
iteration_sims = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

with open(input_path, "r") as infile:
    for line in infile:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        task_id = data.get("task_id", "")
        expected_solutions = match_solution(task_id)
        if not expected_solutions:
            continue
        expected = expected_solutions[0].strip()

        for traj in data.get("trajectories", []):
            # --- Seed (iteration 0) ---
            init_prog = traj.get("initial_program", "")
            if init_prog:
                ratio = normalized_ast_distance(expected, init_prog)
                if ratio is not None:
                    iteration_sims[0].append(ratio)

            # --- Attempts (1–5) ---
            for attempt in traj.get("refinement_attempts", []):
                n = attempt.get("attempt", None)
                if n is None or n > 5:
                    continue
                prog = attempt.get("program", "")
                if prog:
                    ratio = normalized_ast_distance(expected, prog)
                    print(f"Ratio: {ratio}")
                    if ratio is not None:
                        iteration_sims[n].append(ratio)

# ====== Aggregate and Print ======
print("\n=== AST-based Similarity per Iteration (Seed → Attempt 5) ===")
for i in range(6):
    sims = iteration_sims[i]
    if sims:
        mean = np.mean(sims)
        std = np.std(sims)
        print(f"Iteration {i}: {mean:.3f} ± {std:.3f}  (n={len(sims)})")
    else:
        print(f"Iteration {i}: no data")


# iteration_sims = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

# with open(input_path, "r") as infile:
#     for line in infile:
#         try:
#             data = json.loads(line)
#         except json.JSONDecodeError:
#             continue

#         task_id = data.get("task_id", "")
#         expected_solutions = match_solution(task_id)
#         if not expected_solutions:
#             continue
#         expected = expected_solutions[0].strip()

#         for traj in data.get("trajectories", []):
#             # --- Seed (iteration 0) ---
#             init_prog = traj.get("initial_program", "")
#             if init_prog:
#                 ratio = difflib.SequenceMatcher(None, expected, init_prog).ratio()
#                 iteration_sims[0].append(ratio)

#             # --- Attempts (1–5) ---
#             for attempt in traj.get("refinement_attempts", []):
#                 n = attempt.get("attempt", None)
#                 if n is None or n > 5:
#                     continue
#                 prog = attempt.get("program", "")
#                 if prog:
#                     ratio = difflib.SequenceMatcher(None, expected, prog).ratio()
#                     iteration_sims[n].append(ratio)

# # ====== Aggregate and Print ======
# print("\n=== Average Similarity per Iteration (Seed → Attempt 5) ===")
# for i in range(6):
#     sims = iteration_sims[i]
#     if sims:
#         mean = np.mean(sims)
#         std = np.std(sims)
#         print(f"Iteration {i}: {mean:.3f} ± {std:.3f}  (n={len(sims)})")
#     else:
#         print(f"Iteration {i}: no data")




# # ====== Diff + Similarity Analysis ======
# input_path = "results/results_gpt-4o-mini_apps_2025-11-11_20-06-13.jsonl"
# output_path = "feedback_and_programs.txt"

# with open(input_path, "r") as infile, open(output_path, "w") as outfile:
#     for line in infile:
#         try:
#             data = json.loads(line)
#         except json.JSONDecodeError:
#             continue

#         task_id = data.get("task_id", "UNKNOWN_TASK")
#         outfile.write(f"\n{'='*100}\nTASK: {task_id}\n{'='*100}\n")

#         # Match expected solution(s)
#         expected_solutions = match_solution(task_id)
#         if not expected_solutions:
#             outfile.write("\n(No matching expected solution found)\n")
#             continue

#         # We'll just use the first ground truth for comparison
#         expected = expected_solutions[0].strip().splitlines()

#         for traj in data.get("trajectories", []):
#             seed_idx = traj.get("seed_index", "UNKNOWN")
#             outfile.write(f"\n### SEED {seed_idx}\n")

#             # Initial program
#             init_prog = traj.get("initial_program", "")
#             if init_prog:
#                 pred_lines = init_prog.strip().splitlines()
#                 diff = difflib.unified_diff(expected, pred_lines, lineterm="")
#                 ratio = difflib.SequenceMatcher(None, "\n".join(expected), "\n".join(pred_lines)).ratio()
#                 outfile.write("\n[Initial Program vs Expected Solution]\n")
#                 outfile.write(f"Similarity: {ratio:.3f}\n")
#                 outfile.write("\n".join(diff) + "\n")

#             # Refinement attempts
#             for attempt in traj.get("refinement_attempts", []):
#                 attempt_num = attempt.get("attempt", "UNKNOWN")
#                 program = attempt.get("program", "").strip().splitlines()
#                 feedback = attempt.get("feedback", "").strip()

#                 diff = difflib.unified_diff(expected, program, lineterm="")
#                 ratio = difflib.SequenceMatcher(None, "\n".join(expected), "\n".join(program)).ratio()

#                 outfile.write(f"\n--- Attempt {attempt_num} ---\n")
#                 outfile.write(f"Similarity: {ratio:.3f}\n")
#                 outfile.write("[Feedback]\n" + feedback + "\n")
#                 outfile.write("[Diff vs Expected Solution]\n")
#                 outfile.write("\n".join(diff) + "\n")

# print(f"✅ Wrote diffs and similarity ratios to {output_path}")
