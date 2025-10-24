import json
import math
import numpy as np
from collections import defaultdict


def pass_at_k(n, c, k):
    """Compute unbiased pass@k following Chen et al. (Codex paper)."""
    if c == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def analyze_repair_results(path, k_values=(1, 5, 10), show_feedback=False):
    all_objects = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            all_objects.append(data)

    print(f"Loaded {len(all_objects)} tasks")

    num_tasks = len(set(obj["task_id"] for obj in all_objects))
    initial_passes, repairs = [], []
    pass_k_totals = {k: [] for k in k_values}
    iteration_passes = defaultdict(list)
    all_feedback = []

    for obj in all_objects:
        trajectories = obj["trajectories"]
        n = len(trajectories)
        c = sum(
            t["initial_passed"] or any(r["passed"] for r in t.get("feedback_repairs", []))
            for t in trajectories
        )

        # --- Pass@k per task ---
        for k in k_values:
            pass_k_totals[k].append(pass_at_k(n, c, k))

        for traj in trajectories:
            iteration_passes[0].append(traj.get("initial_pass_fraction", 0.0))
            initial_passes.append(traj["initial_passed"])

            # --- Handle feedback/repairs ---
            for idx, repair in enumerate(traj.get("feedback_repairs", []), start=1):
                iteration_passes[idx].append(repair.get("pass_fraction", 0.0))

                feedback = repair.get("feedback")
                if feedback:
                    all_feedback.append({
                        "task_id": obj["task_id"],
                        "seed_index": traj.get("seed_index"),
                        "feedback_index": repair.get("feedback_index"),
                        "repair_index": repair.get("repair_index"),
                        "feedback": feedback.strip()
                    })

            if not traj["initial_passed"]:
                repairs.append(len(traj.get("feedback_repairs", [])))

    # --- Summary stats ---
    frac_initial_passed = np.mean(initial_passes)
    frac_required_repair = 1 - frac_initial_passed
    mean_repairs = np.mean(repairs) if repairs else 0
    median_repairs = np.median(repairs) if repairs else 0

    print("\n📊 Summary Metrics")
    print(f"Unique tasks: {num_tasks}")
    print(f"Fraction passed initially: {frac_initial_passed:.2f}")
    print(f"Fraction required repair: {frac_required_repair:.2f}")
    print(f"Mean repairs (failed seeds): {mean_repairs:.2f}")
    print(f"Median repairs (failed seeds): {median_repairs:.2f}")

    print("\n✅ Pass@k:")
    for k in k_values:
        mean_passk = np.mean(pass_k_totals[k])
        print(f"  pass@{k:<2}: {mean_passk:.3f}")

    print("\n📈 Average Percentage Passed per Iteration:")
    for iteration in sorted(iteration_passes.keys()):
        avg_frac = np.mean(iteration_passes[iteration])
        print(f"  Iteration {iteration:<2}: {avg_frac*100:.2f}% ({len(iteration_passes[iteration])} programs)")

    # --- Optional: print feedback neatly ---
    if show_feedback:
        print("\n🧠 Extracted Feedback Messages:")
        for fb in all_feedback:
            print("=" * 80)
            print(f"Task {fb['task_id']} | Seed {fb['seed_index']} | Feedback #{fb['feedback_index']} | Repair #{fb['repair_index']}")
            print("-" * 80)
            print(fb['feedback'])
            print()

    return all_feedback


if __name__ == "__main__":
    path = "results/results_gpt-4o-mini_mbpp_2025-10-24_11-53-40.jsonl"
    analyze_repair_results(path, show_feedback=True)
