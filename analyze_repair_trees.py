import json
import math
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def pass_at_k(n, c, k):
    """Compute unbiased pass@k following Chen et al. (Codex paper)."""
    if c == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def analyze_repair_results(path, k_values=(1, 5, 10), show_feedback=False, extract_keywords=True):
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

    # ---------- Aggregate all feedback and pass@k ----------
    for obj in all_objects:
        trajectories = obj["trajectories"]
        n = len(trajectories)

        # --- Detect which schema we’re dealing with ---
        mode = "iterative" if any("refinement_attempts" in t for t in trajectories) else "standard"

        # --- Compute success counts (c) for pass@k ---
        if mode == "standard":
            c = sum(
                t["initial_passed"] or any(r["passed"] for r in t.get("feedback_repairs", []))
                for t in trajectories
            )
        else:  # iterative mode
            c = sum(
                t["initial_passed"] or any(r["passed"] for r in t.get("refinement_attempts", []))
                for t in trajectories
            )

        for k in k_values:
            pass_k_totals[k].append(pass_at_k(n, c, k))

        # --- Aggregate iteration-level pass fractions ---
        for traj in trajectories:
            iteration_passes[0].append(traj.get("initial_pass_fraction", 0.0))
            initial_passes.append(traj["initial_passed"])

            if mode == "standard":
                attempts = traj.get("feedback_repairs", [])
            else:
                attempts = traj.get("refinement_attempts", [])

            for idx, repair in enumerate(attempts, start=1):
                iteration_passes[idx].append(repair.get("pass_fraction", 0.0))
                feedback = repair.get("feedback")
                if feedback:
                    all_feedback.append({
                        "task_id": obj["task_id"],
                        "seed_index": traj.get("seed_index"),
                        "iteration": idx,
                        "feedback": feedback.strip()
                    })

            if not traj["initial_passed"]:
                repairs.append(len(attempts))

    # ---------- Summary Stats ----------
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

    # ---------- Per-iteration performance ----------
    print("\n📈 Average Percentage Passed per Iteration:")
    for iteration in sorted(iteration_passes.keys()):
        avg_frac = np.mean(iteration_passes[iteration])
        label = "Initial" if iteration == 0 else f"Refinement {iteration}"
        print(f"  {label:<12}: {avg_frac*100:.2f}% ({len(iteration_passes[iteration])} programs)")

    # ---------- Feedback Summary ----------
    if show_feedback and all_feedback:
        print("\n🧠 Extracted Feedback Messages:")
        for fb in all_feedback[:10]:  # show only first 10 for brevity
            print("=" * 80)
            print(f"Task {fb['task_id']} | Seed {fb['seed_index']} | Iteration {fb['iteration']}")
            print("-" * 80)
            print(fb['feedback'])
            print()

    # ---------- Optional Keyword Extraction ----------
    if extract_keywords and all_feedback:
        texts = [f["feedback"] for f in all_feedback]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20)
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()

        print("\n🧩 Top Keywords and Phrases in Feedback:")
        top_indices = np.argsort(scores)[::-1][:20]
        for i, idx in enumerate(top_indices, start=1):
            print(f"  {i:>2}. {terms[idx]:<30} ({scores[idx]:.2f})")

    return all_feedback


if __name__ == "__main__":
    path = "results/results_gpt-4o-mini_mbpp_2025-10-24_13-36-04.jsonl"
    analyze_repair_results(path, show_feedback=False)
