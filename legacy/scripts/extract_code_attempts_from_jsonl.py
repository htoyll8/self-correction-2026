import json

# Path to your JSONL file
file_path = "results/results_gpt-4o-mini_apps_2025-11-10_21-10-46.jsonl"

attempts = []

with open(file_path, "r") as f:
    for line in f:
        if not line.strip():
            continue  # skip blank lines
        data = json.loads(line)

        # Each JSONL record corresponds to a single task
        for traj in data.get("trajectories", []):
            # Initial program
            if "initial_program" in traj:
                attempts.append({
                    "task_id": data.get("task_id"),
                    "seed_index": traj.get("seed_index"),
                    "type": "initial",
                    "program": traj["initial_program"]
                })
            
            # Refinement attempts
            for ref in traj.get("refinement_attempts", []):
                attempts.append({
                    "task_id": data.get("task_id"),
                    "seed_index": traj.get("seed_index"),
                    "type": f"refinement_{ref.get('attempt')}",
                    "program": ref.get("program")
                })

# Write all extracted programs to a text file
with open("extracted_programs.txt", "w") as out:
    for i, attempt in enumerate(attempts, 1):
        out.write(f"# --- Attempt {i} ---\n")
        out.write(f"# Task: {attempt['task_id']}\n")
        out.write(f"# Seed: {attempt['seed_index']} | Type: {attempt['type']}\n\n")
        out.write(attempt["program"].strip() + "\n\n")

print(f"✅ Extracted {len(attempts)} program attempts to 'extracted_programs.txt'")
