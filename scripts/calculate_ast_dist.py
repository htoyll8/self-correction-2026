import ast
import zss
import json
import time
import numpy as np
from collections import Counter
from datasets import load_dataset


# ==========================================================
#  AST Wrapper for zss
# ==========================================================
class ASTNode:
    """Light wrapper to adapt Python AST nodes for zss."""
    def __init__(self, node):
        self.node = node
        self.children = []
        self.label = node.__class__.__name__

        # Collect children recursively
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.children.append(ASTNode(item))
            elif isinstance(value, ast.AST):
                self.children.append(ASTNode(value))

    # --- static helpers required by zss ---
    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label


def label_distance(a, b):
    return 0 if a == b else 1


# ==========================================================
#  Extract edit operation counts between two ASTs
# ==========================================================
def extract_ast_edit_operations(code_a, code_b):
    """Return counts of insert/remove/update operations between two ASTs."""
    try:
        tree_a = ASTNode(ast.parse(code_a))
        tree_b = ASTNode(ast.parse(code_b))
    except Exception:
        return None  # skip unparsable code

    try:
        dist, ops = zss.simple_distance(
            tree_a,
            tree_b,
            ASTNode.get_children,
            ASTNode.get_label,
            label_distance,
            return_operations=True
        )
    except Exception:
        return None

    op_counts = Counter()
    for op in ops:
        # --- Handle both tuple-based and object-based returns ---
        if isinstance(op, tuple):
            tag = op[0]
        else:
            # Newer zss.Operation object
            tag = getattr(op, "type", None) or getattr(op, "op", None)

        if tag == "insert":
            op_counts["insert"] += 1
        elif tag == "remove":
            op_counts["delete"] += 1
        elif tag == "update":
            op_counts["relabel"] += 1

    return op_counts, len(ops)


def normalized_ast_similarity(code_a, code_b, max_nodes=400):
    """Compute normalized structural similarity between two code snippets."""
    try:
        a_ast = ast.parse(code_a)
        b_ast = ast.parse(code_b)
        # skip huge programs for speed
        if len(list(ast.walk(a_ast))) > max_nodes or len(list(ast.walk(b_ast))) > max_nodes:
            return None

        tree_a = ASTNode(a_ast)
        tree_b = ASTNode(b_ast)

        dist = zss.simple_distance(
            tree_a, tree_b,
            ASTNode.get_children,
            ASTNode.get_label,
            label_distance
        )

        len_a = sum(1 for _ in ast.walk(a_ast))
        len_b = sum(1 for _ in ast.walk(b_ast))
        norm = dist / (len_a + len_b + 1)
        return 1 - norm  # similarity (1.0 = identical)
    except Exception as e:
        # Uncomment for debugging: print(f"[AST ERROR] {e}")
        print(f"[AST ERROR] {e}")
        return None


# ==========================================================
#  Load APPS dataset (for expected ground-truth solutions)
# ==========================================================
print("📦 Loading APPS dataset...")
apps = load_dataset("codeparrot/apps", split="test")

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


def match_solution(task_id_text):
    """Fuzzy prefix match between task text and APPS question."""
    for q_start, sols in solution_lookup.items():
        if task_id_text.strip().startswith(q_start[:40]):
            return sols
    return None


# ==========================================================
#  Extract AST edit operations (with debug info)
# ==========================================================
def extract_ast_edit_operations(code_a, code_b, debug_prefix=""):
    """Return counts of insert/remove/update operations between two ASTs."""
    try:
        tree_a = ASTNode(ast.parse(code_a))
        tree_b = ASTNode(ast.parse(code_b))
    except Exception as e:
        print(f"[{debug_prefix}] ⚠️ AST parse error: {e}")
        return None  # skip unparsable code

    try:
        dist, ops = zss.simple_distance(
            tree_a,
            tree_b,
            ASTNode.get_children,
            ASTNode.get_label,
            label_distance,
            return_operations=True
        )
    except Exception as e:
        print(f"[{debug_prefix}] ❌ zss error: {e}")
        return None

    # --- Debug: show one operation to inspect structure ---
    if ops:
        op0 = ops[0]
        print(f"[{debug_prefix}] Example op type: {type(op0)}")
        if not isinstance(op0, tuple):
            print(f"[{debug_prefix}] Operation attributes: {vars(op0)}")

    op_counts = Counter()
    for op in ops:
        tag = None

        # 1️⃣ Tuple-based (older API)
        if isinstance(op, tuple):
            tag = op[0]

        # 2️⃣ Operation object (current API)
        elif isinstance(op, zss.compare.Operation):
            # Try to get type field directly
            val = getattr(op, "type", None)

            if isinstance(val, str):
                tag = val  # e.g. "insert", "remove", "update"
            elif isinstance(val, int):
                # Fallback numeric mapping
                tag = {0: "insert", 1: "delete", 2: "relabel"}.get(val, f"unknown_{val}")
            else:
                # Last resort: infer from repr string
                rep = repr(op).lower()
                if "insert" in rep:
                    tag = "insert"
                elif "remove" in rep or "delete" in rep:
                    tag = "delete"
                elif "update" in rep or "relabel" in rep:
                    tag = "relabel"
                else:
                    tag = "unknown"

        # 3️⃣ Fallback
        else:
            tag = getattr(op, "op", None) or getattr(op, "operation", None) or "unknown"

        op_counts[tag] += 1

    print(f"[{debug_prefix}] Summary: {op_counts} / total {len(ops)}")
    return op_counts, len(ops)


# ==========================================================
#  Debug-augmented iteration loop
# ==========================================================
input_path = "results/results_gpt-4o-mini_apps_2025-11-11_18-55-57.jsonl"
iteration_ops = {i: Counter() for i in range(6)}
iteration_counts = Counter()

print(f"🔍 Reading experiment results from {input_path}...")

with open(input_path, "r") as infile:
    for line_no, line in enumerate(infile, start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"[line {line_no}] ⚠️ JSON decode error, skipping")
            continue

        task_id = data.get("task_id", "")
        expected_solutions = match_solution(task_id)
        if not expected_solutions:
            continue
        expected = expected_solutions[0].strip()

        print(f"\n=== [Task {task_id[:30]}...] ===")

        for traj_idx, traj in enumerate(data.get("trajectories", [])):
            print(f"  → Processing trajectory {traj_idx}...")

            # Seed (iteration 0)
            init_prog = traj.get("initial_program", "")
            if init_prog:
                debug_tag = f"T{line_no}-Seed"
                res = extract_ast_edit_operations(expected, init_prog, debug_prefix=debug_tag)
                if res:
                    op_counts, total_ops = res
                    iteration_ops[0].update(op_counts)
                    iteration_counts[0] += total_ops

            # Refinement attempts (1–5)
            for attempt in traj.get("refinement_attempts", []):
                n = attempt.get("attempt", None)
                if n is None or n > 5:
                    continue
                prog = attempt.get("program", "")
                if prog:
                    debug_tag = f"T{line_no}-Iter{n}"
                    res = extract_ast_edit_operations(expected, prog, debug_prefix=debug_tag)
                    if res:
                        op_counts, total_ops = res
                        iteration_ops[n].update(op_counts)
                        iteration_counts[n] += total_ops

print("\n=== Structural Edit Composition per Iteration (Insert/Delete/Relabel Ratios) ===")
for i in range(6):
    total = iteration_counts[i]
    if total == 0:
        print(f"Iteration {i}: no data")
        continue
    inserts = iteration_ops[i]["insert"] / total
    deletes = iteration_ops[i]["delete"] / total
    relabels = iteration_ops[i]["relabel"] / total
    print(
        f"Iteration {i}: "
        f"insert={inserts:.2f}, delete={deletes:.2f}, relabel={relabels:.2f} "
        f"(n_ops={total})"
    )
# ==========================================================
#  Optional: Visualization
# ==========================================================
try:
    import matplotlib.pyplot as plt

    iters = sorted(iteration_ops.keys())
    inserts = [iteration_ops[i]['insert']/iteration_counts[i] if iteration_counts[i] else 0 for i in iters]
    deletes = [iteration_ops[i]['delete']/iteration_counts[i] if iteration_counts[i] else 0 for i in iters]
    relabels = [iteration_ops[i]['relabel']/iteration_counts[i] if iteration_counts[i] else 0 for i in iters]

    plt.stackplot(iters, inserts, deletes, relabels, labels=['insert', 'delete', 'relabel'])
    plt.xlabel('Iteration')
    plt.ylabel('Proportion of Edit Ops')
    plt.title('AST Edit Composition per Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()

except ImportError:
    print("Matplotlib not installed; skipping visualization.")

# # ========== Compute AST Similarity Across Iterations ==========
# # input_path = "results/results_gpt-5-2025-08-07_apps_2025-11-02_16-25-51.jsonl"
# input_path = "results/results_gpt-4o-mini_apps_2025-11-03_09-21-10.jsonl"
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
#             # ---- Seed (iteration 0) ----
#             init_prog = traj.get("initial_program", "")
#             if init_prog:
#                 ratio = normalized_ast_similarity(expected, init_prog)
#                 if ratio is not None:
#                     iteration_sims[0].append(ratio)

#             # ---- Refinement attempts (1–5) ----
#             for attempt in traj.get("refinement_attempts", []):
#                 n = attempt.get("attempt", None)
#                 if n is None or n > 5:
#                     continue
#                 prog = attempt.get("program", "")
#                 if prog:
#                     ratio = normalized_ast_similarity(expected, prog)
#                     if ratio is not None:
#                         iteration_sims[n].append(ratio)


# # ========== Aggregate Results ==========
# print("\n=== AST-Based Structural Similarity per Iteration (Seed → Attempt 5) ===")
# for i in range(6):
#     sims = iteration_sims[i]
#     if sims:
#         mean = np.mean(sims)
#         std = np.std(sims)
#         print(f"Iteration {i}: {mean:.3f} ± {std:.3f}  (n={len(sims)})")
#     else:
#         print(f"Iteration {i}: no data")
