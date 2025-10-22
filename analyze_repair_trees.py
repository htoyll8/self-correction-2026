import json
import math
import numpy as np


def pass_at_k(n, c, k):
    """Compute unbiased pass@k following Chen et al. (Codex paper)."""
    if c == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def analyze_repair_results(path, k_values=(1, 5, 10)):
    all_objects = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            data = json.loads(line)
            all_objects.append(data)

    print(f"Loaded {len(all_objects)} tasks")

    # Number of unique tasks
    num_tasks = len(set(obj["task_id"] for obj in all_objects))

    # Fractions
    initial_passes = []
    repairs = []

    # For pass@k
    pass_k_totals = {k: [] for k in k_values}

    for obj in all_objects:
        trajectories = obj["trajectories"]
        n = len(trajectories)  # number of seeds for this task
        c = sum(t["initial_passed"] or any(a["passed"] for a in t["attempts"])
                for t in trajectories)

    # Compute pass@k for this task
        for k in k_values:
            pass_k_totals[k].append(pass_at_k(n, c, k))

        # Track initial pass/fail and repair counts
        for traj in trajectories:
            initial_passes.append(traj["initial_passed"])
            if not traj["initial_passed"]:
                repairs.append(len(traj["attempts"]))

    frac_initial_passed = np.mean(initial_passes)
    frac_required_repair = 1 - frac_initial_passed
    mean_attempts = np.mean(repairs) if repairs else 0
    median_attempts = np.median(repairs) if repairs else 0

    print("\n📊 Summary Metrics")
    print(f"Unique tasks: {num_tasks}")
    print(f"Fraction passed initially: {frac_initial_passed:.2f}")
    print(f"Fraction required repair: {frac_required_repair:.2f}")
    print(f"Mean attempts (failed seeds): {mean_attempts:.2f}")
    print(f"Median attempts (failed seeds): {median_attempts:.2f}")

    print("\n✅ Pass@k:")
    for k in k_values:
        mean_passk = np.mean(pass_k_totals[k])
        print(f"  pass@{k:<2}: {mean_passk:.3f}")


if __name__ == "__main__":
    path = "results/results_gpt-4o-mini_humaneval_2025-10-22_11-36-03.jsonl"
    analyze_repair_results(path)
