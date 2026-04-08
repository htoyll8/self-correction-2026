#!/usr/bin/env python3
"""
Run this on the GCP VM to create ~/run_experiments.sh

    python3 make_run_experiments.py
"""

script = """#!/bin/bash

SESSION="exp1"
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n "mbpp-claude"

tmux send-keys -t $SESSION:0 "cd ~/self-correction-2026 && export ANTHROPIC_VERTEX_PROJECT=dafny-sketcher && export ANTHROPIC_VERTEX_REGION=us-east5 && python3 main.py --dataset mbppplus --np 5 --max_attempts 10 --model_name claude-sonnet-4-5 --mode iterative --refine_mode critique+refine --max_tasks 100 > logs/claude_mbppplus_critique_refine_h0.log 2>&1" Enter

tmux new-window -t $SESSION -n "mbpp-gpt4"
tmux send-keys -t $SESSION:1 "cd ~/self-correction-2026 && python3 main.py --dataset mbppplus --np 5 --max_attempts 10 --model_name gpt-4-0613 --mode iterative --refine_mode critique+refine --max_tasks 100 > logs/gpt4_mbppplus_critique_refine_h0.log 2>&1" Enter

tmux new-window -t $SESSION -n "apps-claude"
tmux send-keys -t $SESSION:2 "cd ~/self-correction-2026 && export ANTHROPIC_VERTEX_PROJECT=dafny-sketcher && export ANTHROPIC_VERTEX_REGION=us-east5 && python3 main.py --dataset apps --difficulty competition --np 5 --max_attempts 10 --model_name claude-sonnet-4-5 --mode iterative --refine_mode critique+refine > logs/claude_apps_comp_critique_refine_h0.log 2>&1" Enter

tmux new-window -t $SESSION -n "apps-gpt4"
tmux send-keys -t $SESSION:3 "cd ~/self-correction-2026 && python3 main.py --dataset apps --difficulty competition --np 5 --max_attempts 10 --model_name gpt-4-0613 --mode iterative --refine_mode critique+refine > logs/gpt4_apps_comp_critique_refine_h0.log 2>&1" Enter

tmux new-window -t $SESSION -n "apps-gpt51"
tmux send-keys -t $SESSION:4 "cd ~/self-correction-2026 && python3 main.py --dataset apps --difficulty competition --np 5 --max_attempts 10 --model_name gpt-5.1-2025-11-13 --mode iterative --refine_mode critique+refine > logs/gpt51_apps_comp_critique_refine_h0.log 2>&1" Enter

echo "All 5 experiments launched in tmux session '$SESSION'"
echo "Attach with: tmux attach -t $SESSION"
"""

import os

out_path = os.path.expanduser("~/run_experiments.sh")
with open(out_path, "w") as f:
    f.write(script)
os.chmod(out_path, 0o755)
print(f"Written to {out_path}")
print("Now run: bash ~/run_experiments.sh")
