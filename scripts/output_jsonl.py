import json
from pathlib import Path

def summarize_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    output_path = path.with_suffix(".txt")

    with open(path, "r") as f, open(output_path, "w") as out:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            out.write("=" * 80 + "\n")
            out.write(f"🧩 Task ID: {obj.get('task_id')}\n")
            out.write(f"✅ Success: {obj.get('success')}\n")
            out.write("-" * 80 + "\n")

            for traj in obj.get("trajectories", []):
                out.write(f"Seed {traj['seed_index']}:\n")
                out.write(f"  Initial pass: {'✅' if traj['initial_passed'] else '❌'} "
                          f"({traj['initial_pass_fraction'] * 100:.1f}%)\n")
                out.write("  └─ Test Results:\n")
                for i, (idx, status, assertion) in enumerate(traj.get("initial_test_results", []), start=1):
                    out.write(f"     {i:02d}. {status}: {assertion}\n")

                # Show refinement attempts, if any
                attempts = traj.get("refinement_attempts", [])
                if not attempts:
                    out.write("  No refinement attempts.\n\n")
                    continue

                for att in attempts:
                    out.write(f"\n  🔁 Attempt {att['attempt']}:\n")
                    feedback = att.get("feedback", "").strip()
                    if feedback:
                        out.write("    💬 Feedback:\n")
                        for line in feedback.splitlines():
                            out.write(f"      {line}\n")
                    out.write(f"    Pass Fraction: {att.get('pass_fraction', 0) * 100:.1f}%\n")

                    test_results = att.get("test_results", [])
                    if test_results:
                        out.write("    └─ Test Results:\n")
                        for i, (idx, status, assertion) in enumerate(test_results, start=1):
                            out.write(f"       {i:02d}. {status}: {assertion}\n")

                out.write("\n")

    print(f"✅ Done. Human-readable summary written to: {output_path}")


if __name__ == "__main__":
    summarize_jsonl("results/results_gpt-4o-mini_humaneval_2025-10-27_03-52-26.jsonl")
