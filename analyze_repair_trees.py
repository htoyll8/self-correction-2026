import json
import math
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def is_monotonic_non_decreasing(xs, tol=1e-12):
    """
    Returns True if xs is monotonic non-decreasing: xs[i+1] >= xs[i] (within tol).
    """
    return all((xs[i+1] + tol) >= xs[i] for i in range(len(xs) - 1))


def get_pass_fraction_sequence(traj):
    """
    Returns list of pass fractions in iteration order:
      [initial_pass_fraction, iter1_pass_fraction, iter2_pass_fraction, ...]
    Supports both schemas: refinement_attempts (iterative) and feedback_repairs (standard).
    """
    attempts = traj.get("refinement_attempts")
    if attempts is None:
        attempts = traj.get("feedback_repairs", [])

    seq = [traj.get("initial_pass_fraction", 0.0)]
    seq += [a.get("pass_fraction", 0.0) for a in attempts]
    return seq


def is_recovered(traj):
    """
    Recovered = didn't pass initially, but passed at some later attempt.
    Supports both schemas: refinement_attempts (iterative) and feedback_repairs (standard).
    """
    attempts = traj.get("refinement_attempts")
    if attempts is None:
        attempts = traj.get("feedback_repairs", [])

    return (not traj.get("initial_passed", False)) and any(a.get("passed", False) for a in attempts)


def pass_at_k(n, c, k):
    """Compute unbiased pass@k following Chen et al. (Codex paper)."""
    if c == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def print_pass_rates(jsonl_path):
    """
    Reads a JSONL file where each line is a dict containing:
      task_id, success, trajectories (list of seeds)
    and prints a clean per-iteration pass rate summary.
    """

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            task_id = record.get("task_id", f"(unknown-task-{line_num})")
            trajectories = record.get("trajectories", [])

            print("\n" + "="*80)
            print(f"TASK: {task_id[:120]}{'...' if len(task_id)>120 else ''}")
            print("="*80)

            if not trajectories:
                print("No trajectories found.\n")
                continue

            # Each trajectory corresponds to one seed
            for traj in trajectories:
                seed_idx = traj.get("seed_index")
                print(f"\n--- Seed {seed_idx} ---")

                # iteration 0 = initial program
                init_pass = traj.get("initial_pass_fraction", None)
                if init_pass is not None:
                    print(f"Iter 0: {init_pass*100:.1f}%")

                # subsequent refinement attempts
                refinements = traj.get("refinement_attempts", [])
                if not refinements:
                    print("(no refinement attempts)\n")
                    continue

                for r in refinements:
                    iter_id = r.get("attempt")
                    frac = r.get("pass_fraction", 0.0)
                    print(f"Iter {iter_id}: {frac*100:.1f}%")


def monotonicity_report_for_recovered(all_objects, tol=1e-12, restrict_to_through_first_pass=True, show_examples=5):
    """
    Checks whether recovered trajectories have pass_fraction sequences that are monotonic non-decreasing.

    If restrict_to_through_first_pass is True, we only check up to the FIRST iteration where passed=True
    (inclusive). This avoids penalizing later iterations that might regress after already succeeding.
    """
    total_recovered = 0
    mono_recovered = 0
    non_mono_examples = []

    for obj in all_objects:
        for traj in obj.get("trajectories", []):
            if not is_recovered(traj):
                continue

            total_recovered += 1

            attempts = traj.get("refinement_attempts")
            if attempts is None:
                attempts = traj.get("feedback_repairs", [])

            seq = get_pass_fraction_sequence(traj)

            if restrict_to_through_first_pass:
                # find first attempt index (1-based in seq) where passed=True
                first_pass_pos = None
                for i, a in enumerate(attempts, start=1):
                    if a.get("passed", False):
                        first_pass_pos = i
                        break
                if first_pass_pos is not None:
                    seq = seq[: first_pass_pos + 1]  # include initial (0) and first pass position

            mono = is_monotonic_non_decreasing(seq, tol=tol)
            if mono:
                mono_recovered += 1
            elif len(non_mono_examples) < show_examples:
                non_mono_examples.append({
                    "task_id": obj.get("task_id"),
                    "seed_index": traj.get("seed_index"),
                    "seq": seq,
                })

    print("\n📈 Monotonicity check (Recovered trajectories)")
    print(f"  restrict_to_through_first_pass: {restrict_to_through_first_pass}")
    print(f"  tol: {tol:g}")
    print(f"  Recovered count: {total_recovered}")
    if total_recovered == 0:
        return

    frac = mono_recovered / total_recovered
    print(f"  Monotonic non-decreasing: {mono_recovered}/{total_recovered} ({frac*100:.2f}%)")

    if non_mono_examples:
        print("\n  Examples of NON-monotonic recovered sequences:")
        for ex in non_mono_examples:
            print(f"    - Task {ex['task_id']} | Seed {ex['seed_index']} | seq={ex['seq']}")


def compare_success_vs_failure(all_objects):
    """Compare properties of successful, failed, and recovered repair trajectories."""

    success_pass_fracs, fail_pass_fracs, recovered_pass_fracs = [], [], []
    success_attempts, fail_attempts, recovered_attempts = [], [], []
    success_initials, fail_initials, recovered_initials = [], [], []

    for obj in all_objects:
        for traj in obj.get("trajectories", []):
            refinements = traj.get("refinement_attempts", []) or traj.get("feedback_repairs", [])
            passed = traj["initial_passed"] or any(r.get("passed") for r in refinements)
            recovered = (not traj["initial_passed"]) and any(r.get("passed") for r in refinements)

            pass_fracs = [traj.get("initial_pass_fraction", 0.0)] + [
                r.get("pass_fraction", 0.0) for r in refinements
            ]
            n_attempts = len(refinements)
            init_pf = traj.get("initial_pass_fraction", 0.0)

            if recovered:
                recovered_pass_fracs.extend(pass_fracs)
                recovered_attempts.append(n_attempts)
                recovered_initials.append(init_pf)
            elif passed:
                success_pass_fracs.extend(pass_fracs)
                success_attempts.append(n_attempts)
                success_initials.append(init_pf)
            else:
                fail_pass_fracs.extend(pass_fracs)
                fail_attempts.append(n_attempts)
                fail_initials.append(init_pf)

    def summarize(name, pass_fracs, attempts, initial_pfs):
        if not pass_fracs:
            print(f"{name}: No data")
            return
        print(f"\n{name}")
        print(f"  Initial pass rate (mean):   {np.mean(initial_pfs) * 100:.2f}%")
        print(f"  Initial pass rate (median): {np.median(initial_pfs) * 100:.2f}%")
        print(f"  Avg pass fraction: {np.mean(pass_fracs) * 100:.2f}%")
        print(f"  Median pass fraction: {np.median(pass_fracs) * 100:.2f}%")
        print(f"  Avg attempts: {np.mean(attempts):.2f}")
        print(f"  Median attempts: {np.median(attempts):.2f}")
        print(f"  Count: {len(attempts)}")

    summarize("✅ Successful trajectories (passed initially)", success_pass_fracs, success_attempts, success_initials)
    summarize("🔁 Recovered trajectories (failed initially but later succeeded)", recovered_pass_fracs, recovered_attempts, recovered_initials)
    summarize("❌ Failed trajectories (never passed)", fail_pass_fracs, fail_attempts, fail_initials)


def is_failed(traj):
    """
    Failed = never passed (not initially, and no later attempt passed).
    Supports both schemas: refinement_attempts (iterative) and feedback_repairs (standard).
    """
    attempts = traj.get("refinement_attempts")
    if attempts is None:
        attempts = traj.get("feedback_repairs", [])

    return (not traj.get("initial_passed", False)) and (not any(a.get("passed", False) for a in attempts))


def monotonicity_report_for_failed(
    all_objects,
    tol=1e-12,
    check_full_sequence=True,
    show_examples=5,
):
    """
    Checks whether FAILED trajectories have pass_fraction sequences that are monotonic non-decreasing.

    - If check_full_sequence=True: check across all iterations (initial + all attempts).
      (This is usually what you want for failures since there is no 'first pass'.)
    """
    total_failed = 0
    mono_failed = 0
    non_mono_examples = []

    for obj in all_objects:
        for traj in obj.get("trajectories", []):
            if not is_failed(traj):
                continue

            total_failed += 1
            seq = get_pass_fraction_sequence(traj)
            # (check_full_sequence flag is here mostly for symmetry / future tweaks)
            if not check_full_sequence:
                seq = seq  # no-op for now

            mono = is_monotonic_non_decreasing(seq, tol=tol)
            if mono:
                mono_failed += 1
            elif len(non_mono_examples) < show_examples:
                non_mono_examples.append({
                    "task_id": obj.get("task_id"),
                    "seed_index": traj.get("seed_index"),
                    "seq": seq,
                })

    print("\n📉 Monotonicity check (Failed trajectories)")
    print(f"  tol: {tol:g}")
    print(f"  Failed count: {total_failed}")
    if total_failed == 0:
        return

    frac = mono_failed / total_failed
    print(f"  Monotonic non-decreasing: {mono_failed}/{total_failed} ({frac*100:.2f}%)")

    if non_mono_examples:
        print("\n  Examples of NON-monotonic failed sequences:")
        for ex in non_mono_examples:
            print(f"    - Task {ex['task_id']} | Seed {ex['seed_index']} | seq={ex['seq']}")


def analyze_repair_results(path, k_values=(1, 5, 10), show_feedback=True, extract_keywords=True):
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
    failed_only_passes = defaultdict(list)
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

            # if not traj["initial_passed"]:
            #     repairs.append(len(attempts))

            if traj.get("initial_pass_fraction", 0.0) < 1.0:
                failed_only_passes[0].append(traj.get("initial_pass_fraction", 0.0))
                for idx, attempt in enumerate(attempts, start=1):
                    failed_only_passes[idx].append(attempt.get("pass_fraction", 0.0))
                repairs.append(len(attempts))

        task_variances = []  # store per-task variance of seed performance

        for obj in all_objects:
            trajectories = obj["trajectories"]
            seed_passes = []

            for traj in trajectories:
                # compute best pass fraction across all iterations for this seed
                pass_fracs = [traj.get("initial_pass_fraction", 0.0)]
                pass_fracs += [r.get("pass_fraction", 0.0) for r in traj.get("refinement_attempts", [])]
                seed_passes.append(max(pass_fracs))

            if len(seed_passes) > 1:
                var = np.var(seed_passes)
                task_variances.append(var)

    if task_variances:
        print(f"  Mean variance:   {np.mean(task_variances):.4f}")
        print(f"  Median variance: {np.median(task_variances):.4f}")
        print(f"  Min variance:    {np.min(task_variances):.4f}")
        print(f"  Max variance:    {np.max(task_variances):.4f}")
        print(f"  Tasks analyzed:  {len(task_variances)}")
    else:
        print("  No tasks had ≥ 2 seeds; variance undefined.")

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

    # ---------- Average % passed per iteration (failed seeds only) ----------
    print("\n📊 Average Percentage of Tests Passed (Failed Seeds Only):")
    for iteration in sorted(failed_only_passes.keys()):
        avg_frac = np.mean(failed_only_passes[iteration]) if failed_only_passes[iteration] else 0
        label = "Initial" if iteration == 0 else f"Refinement {iteration}"
        print(f"  {label:<12}: {avg_frac*100:.2f}% ({len(failed_only_passes[iteration])} programs)")

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

    compare_success_vs_failure(all_objects)

    monotonicity_report_for_recovered(
        all_objects,
        tol=1e-12,
        restrict_to_through_first_pass=True,  # <-- usually what people want
        # show_examples=5
    )
    monotonicity_report_for_failed(all_objects, tol=1e-12, show_examples=5)

    return all_feedback


if __name__ == "__main__":
    # path = "results/results_gpt-4o-mini_apps_2025-11-11_18-55-57.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-14_18-56-09.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-11-14_19-22-10.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-11-14_19-37-32.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-11-14_20-19-22.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-11-14_21-24-26_25_critique_refine.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-20_13-57-05.jsonl"
    # # path = "results/results_gpt-4o-mini_apps_2025-11-20_14-18-53.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-20_14-34-27.jsonl"
    # # path = "results/results_gpt-4o-mini_apps_2025-11-20_16-29-31.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-20_19-09-31.jsonl"
    # path = "results_gpt-4o-mini_apps_2025-11-20_19-34-44_expected_output_clusters.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-20_19-34-44_expected_output_clusters.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-11-14_20-19-22_history.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-21_09-30-08.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-21_15-09-14.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-22_18-28-52.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-23_09-09-22.jsonl"
    # path = "results/results_gpt-4o-mini_humaneval_2025-11-24_11-21-00.jsonl"
    # path = "results/results_gpt-4o-mini_humaneval_2025-11-24_11-29-07.jsonl"
    # path = "results/results_gpt-4o-mini_humaneval_2025-11-24_11-45-16.jsonl"
    # path = "results/results_gpt-4o-mini_apps_2025-11-22_18-28-52-history-introductory-60.jsonl"
    path = "results/results_gpt-4o-mini_humaneval_2025-11-24_13-32-23.jsonl"
    path = "results/results_gpt-4o-mini_humaneval_2025-11-24_13-40-34.jsonl"
    
    
    path = "results/results_gpt-3.5-turbo-0125_humaneval_2025-11-24_20-04-52.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-11-26_16-43-03.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-11-26_23-47-04.jsonl"
    path = "results/results_gpt-4o-mini_mbpp_2025-11-28_13-31-25.jsonl"
    path = "results/results_gpt-4o-mini_mbpp_2025-11-28_16-57-59.jsonl"
    path = "results/results_gpt-4o-mini_mbpp_2025-11-28_22-32-36.jsonl"
    path = "results/results_gpt-4o-mini_mbpp_2025-11-28_22-54-26.jsonl"
    path = "results/results_gpt-4o-mini_mbpp_2025-11-28_22-54-26-critique-refine-134.jsonl"
    # path = "results/results_gpt-5.1-2025-11-13_mbpp_2025-12-01_14-47-36.jsonl"
    # path = "results/results_gpt-4o-mini_mbpp_2025-12-01_09-54-00.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_mbpp_2025-12-01_20-36-32.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_mbpp_2025-12-01_20-36-32-critique-refine-84.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_mbpp_2025-12-02_00-08-12-critque-history-refine-54.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_humaneval_2025-12-02_15-14-05-critque-history-refine-76.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_humaneval_2025-12-02_15-14-05-critque-history-refine-76.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_humaneval_2025-11-24_20-04-52.jsonl"
    path = "results/results_gpt-4o-mini_humaneval_2025-12-03_12-34-46.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_humaneval_2025-12-03_15-15-36.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-04_19-03-18.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-04_19-24-03.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_apps_2025-12-06_19-58-12.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_humaneval_2025-12-07_18-04-17-critque-history-164.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_apps_2025-12-07_19-36-35.jsonl"
    path = "results/results_gpt-3.5-turbo-0125_apps_2025-12-08_14-21-14-critque-history-intro-60.jsonl"
    path = "results/results_gpt-4o-mini_mbppplus_2025-12-17_17-26-14.jsonl"
    path = "results/results_gpt-4_mbppplus_2025-12-17_17-38-51-100-final.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_critique-refine_2025-12-19_10-59-23.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_critique-refine_2025-12-19_10-55-58.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_introductory_critique-refine_2025-12-19_14-16-44.jsonl"
    path = "results/results_gpt-4_humaneval-x_interview_critique-refine_2025-12-21_19-52-17.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-12-05_12-31-12.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_interview_critique-refine_2025-12-20_16-31-14.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-08_18-58-22-critque-history-intro.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-11_20-47-39-critque-history-competitive-60.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_interview_critique-refine_2025-12-20_16-31-14.jsonl"
    path = "results_claude-sonnet-4-5-20250929_apps_interview_critique-history-refine_2025-12-20_16-30-34.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_interview_critique-history-refine_2025-12-20_16-30-34.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_humaneval_interview_critique-refine_2025-12-21_15-40-59.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-04_22-26-05-critque-refine-intro-60.jsonl"
    path = "results/results_gpt-4_apps_introductory_critique-history-refine_2025-12-21_20-26-44.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-08_18-58-22-critque-history-intro.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_competition_critique-refine_2025-12-23_21-59-53.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-04_22-26-05-critque-refine-intro-60.jsonl"
    path = "results/results_gpt-4_apps_introductory_critique-history-refine_2025-12-21_20-26-44.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_apps_competition_critique-refine_2025-12-23_21-59-53.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-12-13_00-19-57-critque-refine-competition-60.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_mbppplus_introductory_critique-refine_2025-12-27_03-48-56.jsonl"
    path = "results/results_claude-sonnet-4-5-20250929_mbppplus_introductory_critique-refine_2026-01-27_21-09-59.jsonl"

    # MBPP
    # MISSING: path = "results/results_claude-sonnet-4-5-20250929_mbppplus_introductory_critique-refine_2025-12-27_03-48-56.jsonl"
    # MISSING: path = "results/results_gpt-4_mbppplus_critique-refine_2025-12-18_23-01-14.jsonl"
    # MISSING: path = "results/results_gpt-5.1-2025-11-13_mbppplus_critique-refine_2025-12-18_23-12-43.jsonl"
    # MISSING: path = "results/results_gpt-4_mbppplus_2025-12-18_17-23-11.jsonl"
    # MISSING: results/results_claude-sonnet-4-5-20250929_mbppplus_introductory_critique-history-refine_2025-12-27_15-41-24.jsonl
    # MISSING: results_gpt-5.1-2025-11-13_mbppplus_critique-history-refine_2025-12-18_23-11-52.jsonl

    # HUMANEVAL
    # path = "results/results_claude-sonnet-4-5-20250929_humaneval_interview_critique-refine_2025-12-21_15-40-59.jsonl"
    # path = "results/results_gpt-4o-mini_humaneval_2025-11-24_14-31-03-critique-refine.jsonl"
    # path = "results/results_gpt-5.1-2025-11-13_humaneval_2025-11-24_18-22-34-critique-refine.jsonl"
    # path = "results/results_gpt-4o-mini_humaneval_2025-12-03_12-34-46-critque-history-164.jsonl"
    # path = "results/results_gpt-5.1-2025-11-13_humaneval_2025-12-03_15-15-36-critque-history-164.jsonl"
    # MISSING: results/results_claude-sonnet-4-5-20250929_humaneval_interview_critique-history-refine_2025-12-21_15-43-18.jsonl

    # HUMANEVAL-JAVA
    # MISSING: path = "results/results_gpt-4_humaneval-x_interview_critique-refine_2025-12-22_01-27-01.jsonl"
    # MISSING: path = "results/results_gpt-4_humaneval-x_interview_critique-history-refine_2025-12-22_20-34-00.jsonl"
    # MISSING: results/results_gpt-5.1-2025-11-13_humaneval-x_interview_critique-refine_2025-12-22_01-24-55.jsonl
    # MISSING: results/results_claude-sonnet-4-5-20250929_humaneval-x_interview_critique-refine_2025-12-23_20-19-06.jsonl
    # MISSING: results/results_gpt-5.1-2025-11-13_humaneval-x_interview_critique-history-refine_2025-12-22_20-35-57.jsonl
    # MISSING: results/results_claude-sonnet-4-5-20250929_humaneval-x_interview_critique-history-refine_2025-12-23_20-19-39.jsonl

    # APPS: INTRODUCTORY
    # path = "results/results_claude-sonnet-4-5-20250929_apps_introductory_critique-refine_2025-12-19_14-16-44-20-final.jsonl"
    # path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-04_22-26-05-critque-refine-intro-60.jsonl"
    # path = "results/results_gpt-4_apps_introductory_critique-history-refine_2025-12-21_20-26-44.jsonl"
    # path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-08_18-58-22-critque-history-intro.jsonl"
    # MISSING: results/results_gpt-4_apps_introductory_critique-refine_2025-12-19_22-13-27.jsonl
    # MISSING: path = "results/results_claude-sonnet-4-5-20250929_apps_introductory_critique-history-refine_2025-12-19_19-20-42.jsonl"

    # APPS: COMPETITION
    # MISSING: path = "results/results_claude-sonnet-4-5-20250929_apps_competition_critique-refine_2025-12-23_21-59-53.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-12-13_00-19-57-critque-refine-competition-60.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-13_00-25-20-critque-refine-competitve-60.jsonl"
    path = "results/results_gpt-4o-mini_apps_2025-12-05_12-31-12.jsonl"
    path = "results/results_gpt-5.1-2025-11-13_apps_2025-12-11_20-47-39-critque-history-competitive-60.jsonl"
    # # MISSING: results/results_claude-sonnet-4-5-20250929_apps_competition_critique-history-refine_2025-12-19_17-56-17.jsonl

    analyze_repair_results(path, show_feedback=False)
    # print_pass_rates(path)
